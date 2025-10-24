import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from parallel.columnar import (
    build_flat_linear_layout,
    build_columnar_causal_mask,
    build_incremental_causal_mask,
    set_rope_pos2d,
)


def load_prompts(args) -> List[Any]:
    if args.prompts:
        return list(args.prompts)
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            prompts: List[Any] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if args.mode == "pretrain":
                        prompts.append(record)
                    else:
                        prompts.append(str(record.get("prompt", record.get("question", ""))))
            return prompts
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    default_prompts: List[Any] = [
        "请介绍一下自己。",
        "推荐几本好书。",
        "未来的科技趋势是什么？",
        "如何理解大语言模型？",
    ]
    return default_prompts


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

    if args.model_path:
        ckpt_path = Path(args.model_path)
    else:
        suffix = "_moe" if args.use_moe else ""
        prefix = "pretrain" if args.mode == "pretrain" else "full_sft"
        ckpt_path = Path(args.out_dir) / f"{prefix}_{args.hidden_size}{suffix}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(args.device).eval(), tokenizer


def apply_chat_template(tokenizer, text: str, enable_chat: bool) -> List[int]:
    """
    构造输入格式，需要和训练数据格式一致

    训练数据格式: <|im_start|>问题内容<|im_end|>
    所以推理时也应该用相同格式，只给问题部分，让模型生成回答
    """
    if enable_chat:
        # 匹配训练格式: <|im_start|>用户问题<|im_end|>
        # 注意：不包含回答部分，让模型自己生成
        prompt_text = f"<|im_start|>{text}"
    else:
        # 不用模板，直接是纯文本
        prompt_text = text
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def normalize_pretrain_branches(item: Any) -> List[str]:
    if isinstance(item, dict):
        if "branches" in item and isinstance(item["branches"], Sequence) and not isinstance(item["branches"], (str, bytes)):
            return [str(x) for x in item["branches"] if str(x).strip()]
        if "text" in item:
            text = str(item["text"])
            if "||" in text:
                return [seg.strip() for seg in text.split("||") if seg.strip()]
            return [text]
        if "prompt" in item:
            return [str(item["prompt"])]
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if str(x).strip()]
    text = str(item)
    if "||" in text:
        return [seg.strip() for seg in text.split("||") if seg.strip()]
    return [text]


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


def print_streaming_update(branch_texts: List[str], prompts: List[str], is_final: bool = False):
    """
    实时更新多个branch的生成进度

    Args:
        branch_texts: 每个branch当前生成的文本
        prompts: 每个branch的原始prompt
        is_final: 是否是最终结果
    """
    # 清空之前的输出（移动到输出区域的起始位置）
    num_branches = len(branch_texts)
    if not is_final:
        # 向上移动光标到输出区域开始
        sys.stdout.write(f"\033[{num_branches + 1}A")  # 上移 n+1 行
        sys.stdout.write("\r")  # 回到行首

    # 打印每个branch的当前状态
    for idx, (prompt, text) in enumerate(zip(prompts, branch_texts)):
        if is_final:
            # 最终结果，完整打印
            print(f"\n=== Branch {idx} ===")
            print(f"Prompt: {prompt}")
            print(f"Response: {text}")
        else:
            # 流式更新，单行显示
            # 截断过长的文本用于显示
            max_display_len = 80
            display_text = text if len(text) <= max_display_len else text[:max_display_len] + "..."
            sys.stdout.write(f"\033[K")  # 清除当前行
            sys.stdout.write(f"Branch {idx}: {display_text}\n")

    if not is_final:
        sys.stdout.write("\033[K")  # 清除分隔行
        sys.stdout.write("---\n")
        sys.stdout.flush()


def columnar_generate(model, branch_inputs: Sequence[Sequence[int]], args, tokenizer, placeholders: Sequence[bool] | None = None) -> List[str]:
    """
    列式并行生成 - 所有分支在同一列时间同步生成

    关键点:
    1. 所有分支在同一列时间生成token (真正的并行)
    2. 同列的token互不可见 (通过mask控制)
    3. 批量forward, 一次处理所有活跃分支
    """
    device = next(model.parameters()).device
    debug = getattr(args, 'debug', False)
    streaming = getattr(args, 'streaming', False)
    branches = []
    for tokens in branch_inputs:
        branches.append({"input_ids": list(tokens), "answer_offset": len(tokens)})
    samples = [{"main": "", "branches": branches}]

    layout = build_flat_linear_layout(
        tokenizer,
        samples,
        device=device,
        pad_to=None,
        align_to="left",  # 与训练保持一致，使用左对齐
    )

    if debug:
        print(f"\n=== Layout Info ===")
        print(f"input_ids shape: {layout.input_ids.shape}")
        print(f"pos2d shape: {layout.pos2d.shape}")
        print(f"time_ids shape: {layout.time_ids.shape}")
        print(f"Metadata: branches={len(layout.metadata[0].branch_ids)}, "
              f"branch_positions={layout.metadata[0].branch_positions}, "
              f"branch_start_y={layout.metadata[0].branch_start_y}")
        seq_len = layout.attention_mask.sum().item()
        print(f"Actual sequence length: {seq_len}")
        if seq_len > 0:
            print(f"Time range: {layout.time_ids[0, :seq_len].min().item()} ~ {layout.time_ids[0, :seq_len].max().item()}")
        else:
            print("Time range: (empty)")

    set_rope_pos2d(model, layout.pos2d)

    input_ids = layout.input_ids
    attn_mask = layout.attention_mask
    pos1d = layout.pos1d
    time_ids = layout.time_ids

    # 获取模型参数dtype用于mask
    param_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        # 打印输入tensor的布局
        if debug:
            print(f"\n=== Input Tokens Tensor ===")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"input_ids tensor:\n{input_ids}")

            # 打印每个token对应的文本
            seq_len = int(attn_mask.sum().item())
            tokens_list = input_ids[0, :seq_len].tolist()
            print(f"\nToken IDs (valid {seq_len} tokens): {tokens_list}")

            # 逐个解码显示
            print(f"\nToken details:")
            for idx in range(input_ids.size(1)):
                token_id = input_ids[0, idx].item()
                is_valid = attn_mask[0, idx].item() > 0.5
                time_val = time_ids[0, idx].item() if idx < time_ids.size(1) else -1
                pos2d_val = layout.pos2d[0, idx].tolist() if idx < layout.pos2d.size(1) else [-1, -1]

                if is_valid:
                    token_text = tokenizer.decode([token_id])
                    print(f"  [{idx:3d}] id={token_id:5d} time={time_val:3d} pos2d={pos2d_val} | '{token_text}'")
                else:
                    print(f"  [{idx:3d}] <PAD> (id={token_id})")

            # 显示attention mask
            print(f"\nAttention mask (1=valid, 0=padding):")
            mask_str = "".join(["1" if attn_mask[0, i].item() > 0.5 else "0" for i in range(min(80, attn_mask.size(1)))])
            if attn_mask.size(1) > 80:
                mask_str += "..."
            print(f"  {mask_str}")

        # 初始prefill
        column_mask = build_columnar_causal_mask(time_ids, attn_mask).to(device, dtype=param_dtype)
        outputs = model(
            input_ids=input_ids,
            attention_mask=column_mask,
            position_ids=pos1d,
            pos2d=layout.pos2d,
            use_cache=True,
        )

        metadata = layout.metadata[0]
        branch_positions = metadata.branch_positions  # [branch_id * 32, ...]
        branch_start_y = metadata.branch_start_y  # 每个分支的起始time

    branch_count = len(branch_inputs)
    placeholder_flags = list(placeholders) if placeholders is not None else [False] * branch_count
        branch_generated: List[List[int]] = [[] for _ in range(branch_count)]

        # 记录当前序列
        seq_len = int(attn_mask.sum().item())
        time_list = time_ids[0, :seq_len].tolist()
        past = outputs.past_key_values

        # 为每个branch维护独立的当前time（独立生成模式）
        # 每个branch从自己问题结束后的time继续
        branch_current_times = []
        for branch_idx in range(branch_count):
            # 找到该branch的最大time
            branch_pos = branch_positions[branch_idx]
            branch_times = [time_list[i] for i in range(seq_len)
                           if layout.pos2d[0, i, 0].item() == branch_pos]
            if branch_times:
                # 该branch下一个生成的time
                branch_current_times.append(max(branch_times) + 1)
            else:
                branch_current_times.append(0)

        if debug:
            print(f"\n=== Generation Start (Independent Mode) ===")
            print(f"Branch count: {branch_count}")
            print(f"branch_pos1d_end: {metadata.branch_pos1d_end}")
            print(f"branch_current_times: {branch_current_times}")

        stop_flags = placeholder_flags[:]
        eos_id = tokenizer.eos_token_id

        # Streaming初始化
        if streaming:
            # 初始化显示区域
            original_prompts = []
            for idx, tokens in enumerate(branch_inputs):
                if placeholder_flags[idx]:
                    original_prompts.append("(empty branch)")
                else:
                    original_prompts.append(tokenizer.decode(list(tokens), skip_special_tokens=True))
            branch_texts = ["" for _ in range(branch_count)]
            print("\n" + "=" * 80)
            print("开始生成...")
            print("=" * 80)
            # 预留显示空间
            for i in range(branch_count):
                print(f"Branch {i}: ")
            print("---")

        # 增量生成循环 - 列同步
        for step_idx in range(args.max_new_tokens):
            if all(stop_flags):
                break

            # 1. 为当前列的所有活跃分支采样新token
            new_tokens = []
            new_pos2d = []
            new_pos1d = []
            active_branch_indices = []

            # 使用上一轮的logits采样 (第一轮从prefill的输出获取)
            if step_idx == 0:
                # 第一轮: 从prefill的logits采样
                branch_logits = []
                for branch_idx in range(branch_count):
                    if stop_flags[branch_idx]:
                        branch_logits.append(None)
                        continue
                    pos1d_idx = metadata.branch_pos1d_end[branch_idx]
                    if pos1d_idx >= 0 and pos1d_idx < outputs.logits.size(1):
                        branch_logits.append(outputs.logits[0, pos1d_idx])
                        if debug:
                            print(f"Branch {branch_idx}: pos1d_idx={pos1d_idx}, logits_shape={outputs.logits.shape}")
                    else:
                        branch_logits.append(None)
                        if debug:
                            print(f"Branch {branch_idx}: INVALID pos1d_idx={pos1d_idx}")

            for branch_idx in range(branch_count):
                if stop_flags[branch_idx]:
                    continue

                # 获取logits
                if step_idx == 0:
                    logits = branch_logits[branch_idx]
                    if logits is None:
                        stop_flags[branch_idx] = True
                        continue
                else:
                    # 从上一轮的outputs中获取对应分支的logits
                    # outputs.logits的顺序与active_branch_indices一致
                    try:
                        logits_idx = active_branch_indices_prev.index(branch_idx)
                        logits = outputs.logits[0, logits_idx]
                    except (ValueError, IndexError):
                        stop_flags[branch_idx] = True
                        continue

                # 采样
                next_token = sample_token(logits, args).to(device)
                token_id = next_token.item()

                if debug and step_idx == 0:
                    decoded = tokenizer.decode([token_id])
                    is_eos = (eos_id is not None and token_id == eos_id)
                    print(f"Branch {branch_idx} first token: {token_id} -> '{decoded}' (EOS={is_eos}, eos_id={eos_id})")

                # 检查是否结束
                if eos_id is not None and token_id == eos_id:
                    stop_flags[branch_idx] = True
                    branch_generated[branch_idx].append(token_id)
                    if debug:
                        print(f"  -> Branch {branch_idx} hit EOS immediately at step {step_idx}")
                    continue

                # 记录生成的token
                branch_generated[branch_idx].append(token_id)
                new_tokens.append(token_id)
                active_branch_indices.append(branch_idx)

                # Streaming更新
                if streaming:
                    # 解码当前branch的所有生成内容
                    branch_texts[branch_idx] = tokenizer.decode(
                        branch_generated[branch_idx],
                        skip_special_tokens=True
                    )

                # 计算2D位置 (每个分支使用自己的time!)
                branch_time = branch_current_times[branch_idx]
                new_pos2d.append([branch_positions[branch_idx], branch_time])

                # 计算1D位置
                position_index = len(time_list) + len(new_pos1d)
                new_pos1d.append(position_index)

            # 如果没有活跃分支,退出
            if not new_tokens:
                break

            # 保存当前轮的active_branch_indices供下一轮使用
            active_branch_indices_prev = active_branch_indices[:]

            # 2. 批量forward所有新token
            num_new = len(new_tokens)
            batch_input_ids = torch.tensor([new_tokens], dtype=torch.long, device=device)  # [1, num_active]
            batch_pos1d = torch.tensor([new_pos1d], dtype=torch.long, device=device)  # [1, num_active]
            batch_pos2d = torch.tensor([new_pos2d], dtype=torch.long, device=device)  # [1, num_active, 2]

            # 更新time_list (每个token使用自己branch的time)
            new_times = [new_pos2d[i][1] for i in range(num_new)]
            time_list.extend(new_times)
            total_len = len(time_list)

            # 更新每个branch的当前time
            for branch_idx in active_branch_indices:
                branch_current_times[branch_idx] += 1

            # 3. 构造因果mask (基于time的因果关系)
            # 关键: 只能看到time < 当前token的time 的所有token
            incremental_mask = torch.full(
                (1, 1, num_new, total_len),
                fill_value=torch.finfo(torch.float32).min,
                device=device,
                dtype=param_dtype
            )

            for i in range(num_new):
                current_time = new_times[i]
                for j, hist_time in enumerate(time_list):
                    # 可以看到time更早的token
                    if hist_time < current_time:
                        incremental_mask[0, 0, i, j] = 0.0
                    # 可以看到自己
                    elif j == total_len - num_new + i:
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

            # 更新状态
            past = outputs.past_key_values

            # Streaming显示更新
            if streaming:
                print_streaming_update(branch_texts, original_prompts, is_final=False)

        # 解码结果
        if debug:
            print(f"\n=== Decoding Results ===")
            for idx, generated in enumerate(branch_generated):
                print(f"Branch {idx} generated {len(generated)} tokens: {generated[:10]}{'...' if len(generated) > 10 else ''}")

        results: List[str] = []
        for idx, generated in enumerate(branch_generated):
            if eos_id is not None and eos_id in generated:
                eos_index = generated.index(eos_id)
                generated_slice = generated[:eos_index]
                if debug:
                    print(f"Branch {idx}: Found EOS at position {eos_index}, slicing to {len(generated_slice)} tokens")
            else:
                generated_slice = generated
            text = tokenizer.decode(generated_slice, skip_special_tokens=True)
            results.append(text.strip())

        # Streaming最终显示
        if streaming:
            print_streaming_update(results, original_prompts, is_final=True)

        for idx, is_placeholder in enumerate(placeholder_flags):
            if is_placeholder:
                results[idx] = ""

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
    parser.add_argument("--mode", choices=["sft", "pretrain"], default="sft", help="推理模式：sft(微调) 或 pretrain(策划师预训练)")
    parser.add_argument("--debug", action="store_true", help="启用调试输出")
    parser.add_argument("--streaming", action="store_true", help="启用流式生成显示")
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    prompts = load_prompts(args)

    if args.mode == "pretrain":
        prompt_groups: List[List[str]] = []
        buffer: List[str] = []
        for item in prompts:
            branches = normalize_pretrain_branches(item)
            if not branches:
                continue

            is_simple_string = not isinstance(item, (dict, list, tuple)) and "||" not in str(item)

            if is_simple_string and len(branches) == 1:
                buffer.append(branches[0])
                if len(buffer) == args.branches_per_sample:
                    prompt_groups.append(buffer)
                    buffer = []
            else:
                if buffer:
                    prompt_groups.append(buffer)
                    buffer = []
                if len(branches) <= args.branches_per_sample:
                    prompt_groups.append(branches)
                else:
                    for start in range(0, len(branches), args.branches_per_sample):
                        prompt_groups.append(branches[start : start + args.branches_per_sample])

        if buffer:
            prompt_groups.append(buffer)
    else:
        str_prompts = [str(p) for p in prompts]
        prompt_groups = [
            str_prompts[start : start + args.branches_per_sample]
            for start in range(0, len(str_prompts), args.branches_per_sample)
        ]

    for group in prompt_groups:
        if not group:
            continue

        placeholder_flags: List[bool] = []
        if args.mode == "pretrain":
            branch_inputs: List[List[int]] = []
            for branch in group:
                text = str(branch)
                if text.strip() == "":
                    placeholder_flags.append(True)
                    branch_inputs.append([])
                else:
                    placeholder_flags.append(False)
                    ids = tokenizer(text, add_special_tokens=False).input_ids
                    branch_inputs.append(ids[: args.max_prompt_length])
        else:
            branch_inputs = []
            for branch in group:
                text = str(branch)
                if text.strip() == "":
                    placeholder_flags.append(True)
                    ids = apply_chat_template(tokenizer, " ", args.chat_template)
                    branch_inputs.append(ids[: args.max_prompt_length])
                else:
                    placeholder_flags.append(False)
                    ids = apply_chat_template(tokenizer, text, args.chat_template)
                    branch_inputs.append(ids[: args.max_prompt_length])

        if not branch_inputs:
            continue

        responses = columnar_generate(model, branch_inputs, args, tokenizer, placeholders=placeholder_flags)

        # 如果不是streaming模式，打印结果（streaming模式已经在generate函数内打印了）
        if not args.streaming:
            for prompt, response in zip(group, responses):
                print("\n=== Prompt ===")
                print(prompt)
                print("--- Response ---")
                print(response if response else "(空响应)")


if __name__ == "__main__":
    main()
