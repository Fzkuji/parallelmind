#!/usr/bin/env python3
"""
调试推理问题 - 检查为什么生成的回复是空的
"""
import os
import sys
import torch
import argparse

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from scripts.inference_hf_lora import load_model_with_lora


def debug_generation(
    base_model: str,
    lora_path: str,
    lora_rank: int = 8,
    rope_2d_ratio: float = 0.5,
    device: str = "cuda",
):
    """调试生成过程"""

    print("\n" + "=" * 80)
    print("开始调试推理过程")
    print("=" * 80 + "\n")

    # 加载模型
    model, tokenizer, patch_rope = load_model_with_lora(
        base_model=base_model,
        lora_path=lora_path,
        lora_rank=lora_rank,
        rope_2d_ratio=rope_2d_ratio,
        device=device,
    )

    # 测试prompt
    prompt = "你好"
    print(f"测试 Prompt: {prompt}\n")

    # 编码
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    print(f"Input length: {input_ids.shape[-1]}\n")

    # 检查 pos2d 设置
    if getattr(model, "_uses_pos2d", False):
        print("✓ 模型使用 pos2d")
        from scripts.inference_hf_lora import _set_prompt_pos2d
        _set_prompt_pos2d(model, input_ids)
        print("✓ pos2d 已设置\n")
    else:
        print("⚠️  模型不使用 pos2d\n")

    # 生成（带详细输出）
    print("开始生成...\n")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,  # 明确设置生成长度
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # 分析输出
    output_ids = outputs.sequences[0]
    print(f"Output IDs shape: {output_ids.shape}")
    print(f"Output IDs: {output_ids}")
    print(f"Output tokens: {tokenizer.convert_ids_to_tokens(output_ids)}")
    print(f"Output length: {output_ids.shape[-1]}\n")

    # 提取生成的部分
    input_len = input_ids.shape[-1]
    generated_ids = output_ids[input_len:]

    print(f"Generated IDs (new tokens only): {generated_ids}")
    print(f"Generated tokens: {tokenizer.convert_ids_to_tokens(generated_ids)}")
    print(f"Generated length: {len(generated_ids)}\n")

    # 解码完整输出
    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Full decoded text:\n{repr(full_text)}\n")

    # 解码生成部分
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated text only:\n{repr(generated_text)}\n")

    # 不跳过特殊token的解码
    generated_text_with_special = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"Generated text (with special tokens):\n{repr(generated_text_with_special)}\n")

    # 检查是否立即生成了EOS
    if len(generated_ids) == 0:
        print("❌ 问题：没有生成任何新token！")
        print("   可能原因：")
        print("   1. max_new_tokens 设置为 0")
        print("   2. 立即生成了 EOS token（但这不应该发生）")
        print("   3. 模型推理有问题")
    elif len(generated_ids) == 1 and generated_ids[0].item() == tokenizer.eos_token_id:
        print("❌ 问题：立即生成了 EOS token")
        print("   可能原因：")
        print("   1. LoRA 权重有问题")
        print("   2. 训练不充分")
        print("   3. 模型崩溃")
    elif generated_text.strip() == "":
        print("❌ 问题：生成了token但decode后是空的")
        print("   可能原因：")
        print("   1. 生成的全是特殊token")
        print("   2. tokenizer decode有问题")
        print(f"   生成的原始token IDs: {generated_ids}")
    else:
        print(f"✓ 成功生成内容：{generated_text}")

    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80 + "\n")

    # 额外检查：验证LoRA是否正确加载
    print("=" * 80)
    print("检查 LoRA 层")
    print("=" * 80 + "\n")

    lora_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_layers.append(name)
            print(f"✓ {name}")
            if hasattr(module.lora, 'A'):
                print(f"  A shape: {module.lora.A.weight.shape}")
            if hasattr(module.lora, 'B'):
                print(f"  B shape: {module.lora.B.weight.shape}")

    if not lora_layers:
        print("❌ 没有找到 LoRA 层！")
        print("   可能原因：")
        print("   1. LoRA 权重加载失败")
        print("   2. apply_lora 没有正确执行")
    else:
        print(f"\n✓ 找到 {len(lora_layers)} 个 LoRA 层")


def main():
    parser = argparse.ArgumentParser(description="调试 HuggingFace + LoRA 推理")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    debug_generation(
        base_model=args.base_model,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        rope_2d_ratio=args.rope_2d_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()
