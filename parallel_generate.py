import argparse
import json
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from parallel.columnar import (
    build_flat_linear_layout,
    build_columnar_causal_mask,
    build_incremental_causal_mask,
    set_rope_pos2d,
)


def load_prompts(args) -> List[str]:
    if args.prompts:
        return list(args.prompts)
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            prompts: List[str] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    prompts.append(str(record.get("prompt", record.get("question", ""))))
            return prompts
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        "请介绍一下自己。",
        "推荐几本好书。",
        "未来的科技趋势是什么？",
        "如何理解大语言模型？",
    ]


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        inference_rope_scaling=args.inference_rope_scaling,
    )

    ckpt_path = Path(args.model_path) if args.model_path else Path(args.out_dir) / f"full_sft_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(args.device).eval(), tokenizer


def apply_chat_template(tokenizer, text: str, enable_chat: bool) -> List[int]:
    if enable_chat:
        conversation = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = text
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def sample_token(logits: torch.Tensor, args) -> torch.Tensor:
    if args.do_sample:
        logits = logits / max(args.temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        if args.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= args.top_p
            mask[..., 0] = True
            filtered_probs = sorted_probs * mask
            filtered_probs = filtered_probs / filtered_probs.sum()
            next_index = torch.multinomial(filtered_probs, 1)
            token_id = sorted_indices[next_index]
        else:
            token_id = torch.multinomial(probs, 1)
    else:
        token_id = torch.argmax(logits, dim=-1, keepdim=True)
    return token_id.view(1)


def columnar_generate(model, branch_inputs: Sequence[Sequence[int]], args, tokenizer) -> List[str]:
    """
    列式并行生成 - 每轮为所有分支同时生成一个token

    关键点:
    1. 所有分支在同一列时间生成token
    2. 同列的token可以看到之前所有列,但看不到同列其他分支
    3. 使用批量forward,一次处理所有活跃分支
    """
    device = next(model.parameters()).device
    branches = []
    for tokens in branch_inputs:
        branches.append({"input_ids": list(tokens), "answer_offset": len(tokens)})
    samples = [{"main": "", "branches": branches}]

    layout = build_flat_linear_layout(
        tokenizer,
        samples,
        device=device,
        pad_to=None,
    )

    set_rope_pos2d(model, layout.pos2d)

    input_ids = layout.input_ids
    attn_mask = layout.attention_mask
    pos1d = layout.pos1d
    time_ids = layout.time_ids

    with torch.no_grad():
        # 初始prefill
        column_mask = build_columnar_causal_mask(time_ids, attn_mask).to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=column_mask,
            position_ids=pos1d,
            pos2d=layout.pos2d,
            use_cache=True,
        )

        metadata = layout.metadata[0]
        branch_positions = metadata.branch_positions
        branch_start_y = metadata.branch_start_y
        branch_lengths = metadata.branch_lengths[:]
        branch_pos1d_end = metadata.branch_pos1d_end[:]

        branch_count = len(branch_inputs)
        branch_generated: List[List[int]] = [[] for _ in range(branch_count)]

        # 获取每个分支最后一个位置的logits
        branch_logits = [outputs.logits[0, idx].detach() for idx in branch_pos1d_end]

        time_list = time_ids[0, : attn_mask.sum().item()].tolist()
        past = outputs.past_key_values

        stop_flags = [False] * branch_count

        # 增量生成循环
        for step_idx in range(args.max_new_tokens):
            if all(stop_flags):
                break

            # 1. 为所有活跃分支采样新token
            new_tokens = []
            new_times = []
            new_pos2d = []
            new_pos1d = []
            active_branch_indices = []

            for branch_idx in range(branch_count):
                if stop_flags[branch_idx]:
                    continue

                # 采样
                logits_branch = branch_logits[branch_idx]
                next_token = sample_token(logits_branch, args).to(device)
                token_id = next_token.item()

                # 检查是否结束
                if token_id == tokenizer.eos_token_id:
                    stop_flags[branch_idx] = True
                    branch_generated[branch_idx].append(token_id)
                    continue

                # 记录生成的token
                branch_generated[branch_idx].append(token_id)
                new_tokens.append(token_id)
                active_branch_indices.append(branch_idx)

                # 计算该分支新token的时间坐标
                new_time = branch_start_y[branch_idx] + branch_lengths[branch_idx]
                branch_lengths[branch_idx] += 1
                new_times.append(new_time)

                # 计算2D位置
                new_pos2d.append([branch_positions[branch_idx], new_time])

                # 计算1D位置(线性序列中的位置)
                position_index = len(time_list)
                new_pos1d.append(position_index)

            # 如果没有活跃分支,退出
            if not new_tokens:
                break

            # 2. 批量forward所有新token
            # 构造输入张量
            batch_input_ids = torch.tensor([new_tokens], dtype=torch.long, device=device)  # [1, num_active]
            batch_pos1d = torch.tensor([new_pos1d], dtype=torch.long, device=device)  # [1, num_active]
            batch_pos2d = torch.tensor([new_pos2d], dtype=torch.long, device=device)  # [1, num_active, 2]

            # 更新time_list
            time_list.extend(new_times)

            # 构造增量mask
            # 每个新token可以看到所有历史token (time < 当前token的time)
            num_new = len(new_tokens)
            total_len = len(time_list)

            # 构造一个 [1, 1, num_new, total_len] 的mask
            incremental_mask = torch.full(
                (1, 1, num_new, total_len),
                fill_value=torch.finfo(torch.float32).min,
                device=device
            )

            for i, new_time in enumerate(new_times):
                for j, hist_time in enumerate(time_list):
                    if hist_time < new_time or j == total_len - num_new + i:
                        # 可以看到更早的时间,或者自己
                        incremental_mask[0, 0, i, j] = 0.0

            # Forward
            set_rope_pos2d(model, batch_pos2d)
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=incremental_mask,
                position_ids=batch_pos1d,
                past_key_values=past,
                pos2d=batch_pos2d,
                use_cache=True,
            )

            # 更新past
            past = outputs.past_key_values

            # 更新每个分支的logits
            for i, branch_idx in enumerate(active_branch_indices):
                branch_logits[branch_idx] = outputs.logits[0, i].detach()
                branch_pos1d_end[branch_idx] = new_pos1d[i]

        # 解码结果
        results: List[str] = []
        for generated in branch_generated:
            if tokenizer.eos_token_id in generated:
                eos_index = generated.index(tokenizer.eos_token_id)
                generated = generated[:eos_index]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())
        return results


def main():
    parser = argparse.ArgumentParser(description="MiniMind 列式并行推理")
    parser.add_argument("--prompts", nargs="*", help="命令行直接提供的问题")
    parser.add_argument("--prompts_file", type=str, default="", help="包含多个问题的文本或 JSONL")
    parser.add_argument("--branches_per_sample", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--inference_rope_scaling", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--chat_template", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    prompts = load_prompts(args)

    for start in range(0, len(prompts), args.branches_per_sample):
        batch_prompts = prompts[start : start + args.branches_per_sample]
        branch_inputs = [apply_chat_template(tokenizer, p, args.chat_template) for p in batch_prompts]
        responses = columnar_generate(model, branch_inputs, args, tokenizer)
        for prompt, response in zip(batch_prompts, responses):
            print("\n=== Prompt ===")
            print(prompt)
            print("--- Response ---")
            print(response if response else "(空响应)")


if __name__ == "__main__":
    main()
