#!/usr/bin/env python3
"""
HuggingFace 模型 + LoRA 权重推理脚本
支持 2D RoPE 和 Parallel 生成
"""
import os
import sys
import argparse
import torch
import types
from typing import List, Optional

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_lora import apply_lora, load_lora
from parallel.columnar import (
    patch_model_with_interleaved_2d_rope,
    set_rope_pos2d,
    _find_rotary_holder,
)


def _prepare_pos2d(input_ids: torch.Tensor) -> torch.Tensor:
    """
    为单条文本准备 pos2d（单个 branch + 线性 time id）

    Args:
        input_ids: [batch_size, seq_len]

    Returns:
        pos2d: [batch_size, seq_len, 2]，其中 [:, :, 0] 是 branch_id（全0），[:, :, 1] 是 time_id
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # time_ids: 0, 1, 2, ..., seq_len-1
    time_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # branch_ids: 全部为 0（单个 branch）
    branch_ids = torch.zeros_like(time_ids)

    # 堆叠成 [batch, seq, 2]
    pos2d = torch.stack([branch_ids, time_ids], dim=-1)

    return pos2d


def _set_prompt_pos2d(model, input_ids: torch.Tensor):
    """为整条 prompt 设置 2D RoPE 的位置编码"""
    pos2d = _prepare_pos2d(input_ids)
    set_rope_pos2d(model, pos2d)


def _inject_pos2d_hook(model):
    """重写 prepare_inputs_for_generation，保证增量生成也会携带 pos2d"""
    if getattr(model, "_pos2d_hook_injected", False):
        return

    if not hasattr(model, "prepare_inputs_for_generation"):
        return

    model._orig_prepare_inputs_for_generation = model.prepare_inputs_for_generation

    def _prepare_inputs_for_generation(self, input_ids, **kwargs):
        inputs = self._orig_prepare_inputs_for_generation(input_ids, **kwargs)

        position_ids = inputs.get("position_ids")
        if position_ids is None:
            seq_len = inputs["input_ids"].size(-1)
            position_ids = torch.arange(seq_len, device=inputs["input_ids"].device).unsqueeze(0)
            inputs["position_ids"] = position_ids

        branch_ids = torch.zeros_like(position_ids)
        pos2d = torch.stack([branch_ids, position_ids], dim=-1)
        set_rope_pos2d(self, pos2d)

        return inputs

    model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)
    model._pos2d_hook_injected = True


def auto_pair_indices(model, ratio: float):
    """计算 2D RoPE 的频率对索引"""
    holder = _find_rotary_holder(model)
    inv_freq = getattr(holder.rotary_emb, "inv_freq", None)
    if inv_freq is None:
        raise RuntimeError("未找到 rotary_emb.inv_freq，无法应用 2D RoPE。")
    freq_count = inv_freq.numel()
    pair_count = max(1, min(freq_count, int(round(freq_count * ratio))))
    start = max(1, freq_count - pair_count + 1)
    return list(range(start, freq_count + 1))


def load_model_with_lora(
    base_model: str,
    lora_path: str,
    lora_rank: int = 8,
    rope_2d_ratio: float = 0.5,
    patch_rope: bool = True,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """
    加载 HuggingFace 模型并应用 LoRA 权重

    Args:
        base_model: HuggingFace 模型名称或路径
        lora_path: LoRA 权重文件路径
        lora_rank: LoRA 的秩
        rope_2d_ratio: 2D RoPE 的 branch 频率比例
        patch_rope: 是否应用 2D RoPE
        device: 设备
        dtype: 数据类型
    """
    print(f"=" * 80)
    print(f"加载基础模型: {base_model}")
    print(f"=" * 80)

    # 设置数据类型
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    # 应用 2D RoPE
    if patch_rope:
        print(f"应用 2D RoPE（ratio={rope_2d_ratio}）...")
        pair_indices = auto_pair_indices(model, rope_2d_ratio)
        patch_model_with_interleaved_2d_rope(model, pair_indices)
        print(f"✓ 2D RoPE 已应用（{len(pair_indices)} 个频率对用于 branch 维度）")
        _inject_pos2d_hook(model)
        model._uses_pos2d = True
    else:
        model._uses_pos2d = False

    # 应用 LoRA
    print(f"应用 LoRA（rank={lora_rank}）...")
    apply_lora(model, rank=lora_rank)

    # 加载 LoRA 权重
    if lora_path and os.path.exists(lora_path):
        print(f"加载 LoRA 权重: {lora_path}")
        load_lora(model, lora_path)
        print("✓ LoRA 权重加载成功")
    else:
        raise FileNotFoundError(f"LoRA 权重文件不存在: {lora_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"=" * 80)
    print(f"模型加载完成！总参数量: {total_params / 1e6:.2f}M")
    print(f"=" * 80)

    return model, tokenizer, patch_rope


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: str = "cuda",
):
    """
    生成文本

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度
        top_p: nucleus sampling
        top_k: top-k sampling
        repetition_penalty: 重复惩罚
        device: 设备
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    print(f"\n{'=' * 80}")
    print(f"输入提示:")
    print(f"{'-' * 80}")
    print(prompt)
    print(f"{'=' * 80}\n")

    if getattr(model, "_uses_pos2d", False):
        _set_prompt_pos2d(model, input_ids)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_tokens = outputs[0]
    generated_tokens = output_tokens[input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"{'=' * 80}")
    print(f"生成结果:")
    print(f"{'-' * 80}")
    print(response)
    print(f"{'=' * 80}\n")

    return response


def interactive_chat(
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """交互式对话"""
    print(f"\n{'=' * 80}")
    print("交互式对话模式（输入 'quit' 或 'exit' 退出）")
    print(f"{'=' * 80}\n")

    conversation_history = []

    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not user_input:
                continue

            # 构建对话历史
            conversation_history.append({"role": "user", "content": user_input})

            # 应用 chat template
            prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(device)

            if getattr(model, "_uses_pos2d", False):
                _set_prompt_pos2d(model, input_ids)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            output_tokens = outputs[0]
            generated_tokens = output_tokens[input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            print(f"Assistant: {response}\n")

            # 添加到对话历史
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 模型 + LoRA 推理")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 权重路径")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5, help="2D RoPE ratio")
    parser.add_argument("--no_patch_rope", action="store_true", help="不应用 2D RoPE")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # 推理模式
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "generate"], help="推理模式")
    parser.add_argument("--prompt", type=str, default=None, help="单次生成的提示（mode=generate）")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")

    args = parser.parse_args()

    # 加载模型
    model, tokenizer, patch_rope = load_model_with_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        rope_2d_ratio=args.rope_2d_ratio,
        patch_rope=not args.no_patch_rope,
        device=args.device,
        dtype=args.dtype,
    )

    # 推理
    if args.mode == "chat":
        interactive_chat(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
    elif args.mode == "generate":
        if args.prompt is None:
            print("错误: generate 模式需要指定 --prompt")
            return

        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )


if __name__ == "__main__":
    main()
