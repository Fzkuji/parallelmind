import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

# 设置路径，使脚本能从scripts目录运行
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch

from src.model.columnar import (
    build_flat_linear_layout,
    build_columnar_causal_mask,
    build_incremental_causal_mask,
    set_rope_pos2d,
)
from scripts.utils.chat_format import (
    apply_chat_template,
    ensure_pretrain_template,
    normalize_pretrain_branches,
)
from scripts.utils.layout_debug import dump_branch_layout, dump_generated_sequences
from scripts.utils.model_loader import load_model
from scripts.utils.prompts import load_prompts


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


def print_streaming_update(branch_texts: List[str], prompts: List[str], is_final: bool = False, num_branches: int = 0):
    """
    实时更新多个branch的生成进度 - 每个branch占一行

    Args:
        branch_texts: 每个branch当前生成的文本
        prompts: 每个branch的原始prompt
        is_final: 是否是最终结果
        num_branches: branch总数（用于控制光标移动）
    """
    if is_final:
        # 最终结果：换行后打印
        print("\n" + "=" * 80)
        print("生成完成")
        print("=" * 80)
        for idx, (prompt, text) in enumerate(zip(prompts, branch_texts)):
            print(f"\n【Branch {idx}】")
            prompt_display = prompt[:80] + "..." if len(prompt) > 80 else prompt
            print(f"Prompt: {prompt_display}")
            print(f"Response: {text}")
    else:
        # 流式更新：多行实时刷新
        # 先清屏，然后重新打印所有行
        sys.stdout.write("\033[2J\033[H")  # 清屏并移动到左上角

        # 打印标题
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.write(f"生成中... (共{num_branches}个branches)\n")
        sys.stdout.write("=" * 80 + "\n\n")

        # 打印每个branch的当前状态
        for idx, (text, prompt) in enumerate(zip(branch_texts, prompts)):
            # 显示prompt（截断到合理长度）
            prompt_display = prompt[:60] + "..." if len(prompt) > 60 else prompt
            sys.stdout.write(f"Branch {idx}: {prompt_display}\n")

            # 显示生成的内容（支持自动换行，不省略）
            display_text = text.replace('\n', '↵')  # 保留原始换行标记

            if len(display_text) > 0:
                # 手动分行显示，每行最多70个字符
                line_width = 70
                lines = []
                for i in range(0, len(display_text), line_width):
                    lines.append(display_text[i:i+line_width])

                # 显示字数和内容
                sys.stdout.write(f"  ({len(text):3d}字)\n")
                for line in lines:
                    sys.stdout.write(f"  {line}\n")
            else:
                sys.stdout.write(f"  (  0字) <等待生成...>\n")

            sys.stdout.write("\n")  # branch之间空一行

        # 刷新输出
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

    # 默认总是使用interleave（与训练默认值一致），除非明确指定不要
    use_interleave = getattr(args, "interleave_branches", True)
    requested_align = getattr(args, "align_to", None)
    if requested_align is not None:
        align_mode = requested_align
    else:
        align_mode = "right" if (args.mode == "sft" and not use_interleave) else "left"

    layout = build_flat_linear_layout(
        tokenizer,
        samples,
        device=device,
        pad_to=None,
        align_to=align_mode,
        interleave_branches=use_interleave,
        branch_stride=args.branch_stride,
    )

    if getattr(args, "print_layout", False):
        dump_branch_layout(layout, tokenizer, max_tokens=getattr(args, "layout_max_tokens", 256))

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

    if getattr(model, "_uses_pos2d", False):
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
        column_mask = build_columnar_causal_mask(time_ids, attn_mask)
        column_mask = column_mask.to(device=device, dtype=param_dtype)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": column_mask,
            "use_cache": True,
        }
        if not getattr(model, "_is_hf", False):
            forward_kwargs["position_ids"] = pos1d
            forward_kwargs["pos2d"] = layout.pos2d

        outputs = model(**forward_kwargs)

        metadata = layout.metadata[0]
        branch_positions = metadata.branch_positions  # [branch_id * 32, ...]
        branch_start_y = metadata.branch_start_y  # 每个分支的起始time

        branch_count = len(branch_inputs)
        placeholder_flags = list(placeholders) if placeholders is not None else [False] * branch_count
        branch_generated: List[List[int]] = [[] for _ in range(branch_count)]
        branch_generated_meta: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(branch_count)]

        # 记录当前序列
        seq_len = int(attn_mask.sum().item())
        time_list = time_ids[0, :seq_len].tolist()
        past = outputs.past_key_values

        # 为每个branch维护独立的当前time（独立生成模式）
        # 每个branch从自己问题结束后的time继续
        branch_current_times = []
        branch_last_abs_idx = []
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
            branch_last_abs_idx.append(metadata.branch_pos1d_end[branch_idx])

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
            print(f"开始生成 ({branch_count} branches)...")
            print("=" * 80)
            # 预留多行空间用于实时更新
            for idx in range(branch_count):
                print(f"Branch {idx}: 初始化中...")
            streaming_update_count = 0  # 记录更新次数

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
                if getattr(args, "print_topk", False):
                    topk_k = max(1, int(getattr(args, "topk_k", 5)))
                    probs = torch.softmax(logits, dim=-1)
                    topk_vals, topk_idx = torch.topk(probs, k=topk_k, dim=-1)
                    decoded = [tokenizer.decode([tid.item()]).replace("\n", "\\n") for tid in topk_idx]
                    topk_str = " | ".join(
                        f"{tid.item():5d}:{p.item():.4f}:'{txt}'" for tid, p, txt in zip(topk_idx, topk_vals, decoded)
                    )
                    print(f"[topk] step={step_idx} branch={branch_idx} -> {topk_str}")

                next_token = sample_token(logits, args).to(device)
                token_id = next_token.item()

                if debug and step_idx == 0:
                    decoded = tokenizer.decode([token_id])
                    is_eos = (eos_id is not None and token_id == eos_id)
                    print(f"Branch {branch_idx} first token: {token_id} -> '{decoded}' (EOS={is_eos}, eos_id={eos_id})")

                # 检查是否结束
                abs_index = len(time_list) + len(new_pos1d)
                branch_time = branch_current_times[branch_idx]
                source_abs_idx = branch_last_abs_idx[branch_idx]

                if eos_id is not None and token_id == eos_id:
                    stop_flags[branch_idx] = True
                    branch_generated[branch_idx].append(token_id)
                    branch_generated_meta[branch_idx].append(
                        (branch_positions[branch_idx], branch_time, abs_index, source_abs_idx)
                    )
                    branch_last_abs_idx[branch_idx] = abs_index
                    if debug:
                        print(f"  -> Branch {branch_idx} hit EOS immediately at step {step_idx}")
                    continue

                # 记录生成的token
                branch_generated[branch_idx].append(token_id)
                branch_generated_meta[branch_idx].append(
                    (branch_positions[branch_idx], branch_time, abs_index, source_abs_idx)
                )
                new_tokens.append(token_id)
                active_branch_indices.append(branch_idx)
                branch_last_abs_idx[branch_idx] = abs_index

                # Streaming更新
                if streaming:
                    # 解码当前branch的所有生成内容
                    branch_texts[branch_idx] = tokenizer.decode(
                        branch_generated[branch_idx],
                        skip_special_tokens=True
                    )

                # 计算2D/1D位置
                new_pos2d.append([branch_positions[branch_idx], branch_time])
                new_pos1d.append(abs_index)

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
            mask_fill = torch.tensor(-1e4, dtype=param_dtype, device=device)
            incremental_mask = torch.full(
                (1, 1, num_new, total_len),
                fill_value=mask_fill.item(),
                device=device,
                dtype=param_dtype,
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
            if getattr(model, "_uses_pos2d", False):
                set_rope_pos2d(model, batch_pos2d)

            forward_kwargs = {
                "input_ids": batch_input_ids,
                "attention_mask": incremental_mask,
                "past_key_values": past,
                "use_cache": True,
            }
            if not getattr(model, "_is_hf", False):
                forward_kwargs["position_ids"] = batch_pos1d
                forward_kwargs["pos2d"] = batch_pos2d

            outputs = model(**forward_kwargs)

            # 更新状态
            past = outputs.past_key_values

            # Streaming显示更新
            if streaming:
                streaming_update_count += 1
                print_streaming_update(branch_texts, original_prompts, is_final=False, num_branches=branch_count)

        # 解码结果
        if debug:
            print(f"\n=== Decoding Results ===")
            for idx, generated in enumerate(branch_generated):
                print(f"Branch {idx} generated {len(generated)} tokens: {generated[:10]}{'...' if len(generated) > 10 else ''}")

        results: List[str] = []
        branch_token_meta: List[List[Tuple[int, int, int, int]]] = []
        for idx, generated in enumerate(branch_generated):
            if eos_id is not None and eos_id in generated:
                eos_index = generated.index(eos_id)
                generated_slice = generated[:eos_index]
                branch_generated_meta[idx] = branch_generated_meta[idx][:eos_index]
                if debug:
                    print(f"Branch {idx}: Found EOS at position {eos_index}, slicing to {len(generated_slice)} tokens")
            else:
                generated_slice = generated
            text = tokenizer.decode(generated_slice, skip_special_tokens=True)
            results.append(text.strip())

            # 提取位置元信息
            valid_len = len(generated_slice)
            branch_token_meta.append(branch_generated_meta[idx][:valid_len])

        # Streaming最终显示
        if streaming:
            print_streaming_update(results, original_prompts, is_final=True)

        if debug:
            for idx, meta in enumerate(branch_token_meta):
                if not meta:
                    print(f"Branch {idx} token positions: []")
                    continue
                snippet = " ".join(f"(pos={b},t={t},abs={a},src_abs={s})" for b, t, a, s in meta[:32])
                if len(meta) > 32:
                    snippet += " ..."
                print(f"Branch {idx} token positions ({len(meta)} tokens): {snippet}")

        for idx, is_placeholder in enumerate(placeholder_flags):
            if is_placeholder:
                results[idx] = ""

        if getattr(args, "print_layout", False):
            dump_generated_sequences(branch_generated_meta, branch_generated, tokenizer)

        return results


def main():
    parser = argparse.ArgumentParser(description="MiniMind 列式并行推理")
    parser.add_argument("--prompts", nargs="*", help="命令行直接提供的问题")
    parser.add_argument("--prompts_file", type=str, default="", help="包含多个问题的文本或 JSONL")
    parser.add_argument("--data_path", type=str, default="", help="prompts_file 的别名，便于从数据集加载")
    parser.add_argument("--branches_per_sample", type=int, default=4)
    parser.add_argument("--max_branches_per_sample", type=int, default=None)
    parser.add_argument("--min_branches_per_sample", type=int, default=1)
    parser.add_argument("--branch_stride", type=int, default=128, help="分支在 pos2d 中的步长")
    parser.add_argument("--batch_by_samples", action="store_true")
    parser.add_argument("--max_total_tokens", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--hf_base_model", type=str, default="", help="HuggingFace 模型名称/路径（使用 HF + LoRA 推理时指定）")
    parser.add_argument("--lora_path", type=str, default="", help="LoRA 权重路径（HF 模式必填）")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5)
    parser.add_argument("--no_patch_rope", action="store_true")
    parser.add_argument("--hf_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--inference_rope_scaling", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--print_topk", action="store_true", help="打印每个分支当前步的 top-k 备选token")
    parser.add_argument("--topk_k", type=int, default=5, help="打印的top-k数量")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--chat_template", action="store_true")
    parser.add_argument("--mode", choices=["sft", "pretrain"], default="sft", help="推理模式：sft(微调) 或 pretrain(策划师预训练)")
    parser.add_argument("--debug", action="store_true", help="启用调试输出")
    parser.add_argument("--print_layout", action="store_true", help="打印列式布局的 token 时间/branch 信息")
    parser.add_argument("--layout_max_tokens", type=int, default=256, help="打印布局时的最大token数")
    parser.add_argument(
        "--interleave_branches",
        action="store_true",
        dest="interleave_branches",
        help="启用多分支交错排列（默认开启，需与训练设置一致）",
    )
    parser.add_argument(
        "--no_interleave_branches",
        action="store_false",
        dest="interleave_branches",
        help="禁用分支交错排列（仅在训练同样禁用时使用）",
    )
    parser.set_defaults(interleave_branches=True)
    parser.add_argument(
        "--align_to",
        type=str,
        choices=["left", "right"],
        default=None,
        help="列式布局对齐方式（默认根据模式自动选择）",
    )
    parser.add_argument("--streaming", action="store_true", help="启用流式生成显示")
    parser.add_argument("--pe", type=str, choices=['rope', 'fpe'], default=None, help="位置编码类型（不指定则自动检测）")
    parser.add_argument("--out_path", type=str, default="", help="生成结果保存为 JSONL（分布式会自动按 rank 拆分）")
    args = parser.parse_args()

    if args.data_path and not args.prompts_file:
        args.prompts_file = args.data_path

    if not args.hf_base_model and args.model_path and not Path(args.model_path).exists():
        args.hf_base_model = args.model_path
        args.model_path = ""

    if args.hf_base_model:
        args.chat_template = True

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

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

    total_groups = len(prompt_groups)
    prompt_groups = prompt_groups[rank::world_size]

    out_writer = None
    if args.out_path:
        base, ext = os.path.splitext(args.out_path)
        ext = ext or ".jsonl"
        out_path = args.out_path if world_size == 1 else f"{base}_rank{rank}{ext}"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_writer = open(out_path, "w", encoding="utf-8")

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
                    norm_text = ensure_pretrain_template(text)
                    ids = tokenizer(norm_text, add_special_tokens=False).input_ids
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
                if out_writer:
                    out_writer.write(
                        json.dumps({"prompt": str(prompt), "response": response}, ensure_ascii=False) + "\n"
                    )
        else:
            if out_writer:
                for prompt, response in zip(group, responses):
                    out_writer.write(
                        json.dumps({"prompt": str(prompt), "response": response}, ensure_ascii=False) + "\n"
                    )

    if out_writer:
        out_writer.close()


if __name__ == "__main__":
    main()
