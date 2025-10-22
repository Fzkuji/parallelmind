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
        branch_logits = [outputs.logits[0, idx].detach() for idx in branch_pos1d_end]

        time_list = time_ids[0, : attn_mask.sum().item()].tolist()
        past = outputs.past_key_values

        stop_flags = [False] * branch_count

        for _ in range(args.max_new_tokens):
            if all(stop_flags):
                break

            tokens_this_round: List[torch.Tensor] = []
            for branch_idx in range(branch_count):
                if stop_flags[branch_idx]:
                    tokens_this_round.append(torch.tensor([tokenizer.pad_token_id], device=device))
                    continue
                logits_branch = branch_logits[branch_idx]
                next_token = sample_token(logits_branch, args).to(device)
                token_id = next_token.item()
                if token_id == tokenizer.eos_token_id:
                    stop_flags[branch_idx] = True
                branch_generated[branch_idx].append(token_id)
                tokens_this_round.append(next_token)

            for branch_idx, token_tensor in enumerate(tokens_this_round):
                if stop_flags[branch_idx]:
                    continue
                new_time = branch_start_y[branch_idx] + branch_lengths[branch_idx]
                branch_lengths[branch_idx] += 1
                time_list.append(new_time)
                position_index = len(time_list) - 1

                pos2d_token = torch.tensor(
                    [[[branch_positions[branch_idx], new_time]]],
                    device=device,
                    dtype=layout.pos2d.dtype,
                )
                position_ids_token = torch.tensor([[position_index]], device=device, dtype=pos1d.dtype)
                increment_mask = build_incremental_causal_mask(time_list, device)

                set_rope_pos2d(model, pos2d_token)
                outputs = model(
                    input_ids=token_tensor.view(1, 1),
                    attention_mask=increment_mask,
                    position_ids=position_ids_token,
                    past_key_values=past,
                    pos2d=pos2d_token,
                    use_cache=True,
                )
                past = outputs.past_key_values
                branch_logits[branch_idx] = outputs.logits[0, -1].detach()
                branch_pos1d_end[branch_idx] = position_index

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
