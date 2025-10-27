#!/usr/bin/env python3
"""
裁剪FPE checkpoint的max_positions大小

用途：
- 把训练时用32768的模型裁剪到512
- 减小checkpoint大小，加快推理速度

使用：
    python resize_fpe_checkpoint.py \
      --input out/fpe_stage2/pretrain_512.pth \
      --output out/fpe_stage2/pretrain_512_resized.pth \
      --new_max_positions 512
"""

import argparse
import torch
from pathlib import Path


def resize_fpe_checkpoint(input_path: str, output_path: str, new_max_positions: int):
    """裁剪FPE checkpoint"""

    print(f"加载checkpoint: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')

    # 检查是否包含FPE参数
    fpe_key = 'model.fourier_pe.pe'
    if fpe_key not in state_dict:
        print("❌ 这不是一个FPE模型的checkpoint")
        return False

    # 获取原始大小
    old_pe = state_dict[fpe_key]
    old_max_positions, d_model = old_pe.shape

    print(f"原始大小: [{old_max_positions}, {d_model}]")
    print(f"目标大小: [{new_max_positions}, {d_model}]")

    if new_max_positions > old_max_positions:
        print(f"❌ 新大小({new_max_positions})不能大于原始大小({old_max_positions})")
        return False

    if new_max_positions == old_max_positions:
        print("✓ 大小相同，无需修改")
        return False

    # 裁剪PE参数（只保留前new_max_positions行）
    new_pe = old_pe[:new_max_positions, :]
    state_dict[fpe_key] = new_pe

    print(f"✓ 已裁剪 {fpe_key}: [{old_max_positions}, {d_model}] -> [{new_max_positions}, {d_model}]")

    # 计算节省的空间
    saved_params = (old_max_positions - new_max_positions) * d_model
    saved_mb = saved_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"✓ 节省参数: {saved_params:,} ({saved_mb:.2f} MB)")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"✓ 已保存到: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="裁剪FPE checkpoint的max_positions")
    parser.add_argument("--input", required=True, help="输入checkpoint路径")
    parser.add_argument("--output", required=True, help="输出checkpoint路径")
    parser.add_argument("--new_max_positions", type=int, default=512,
                        help="新的max_positions大小（默认512）")
    args = parser.parse_args()

    success = resize_fpe_checkpoint(args.input, args.output, args.new_max_positions)

    if success:
        print("\n使用方法:")
        print(f"python parallel_generate.py \\")
        print(f"  --model_path {args.output} \\")
        print(f"  --pe fpe \\")
        print(f"  --mode pretrain \\")
        print(f"  --prompts \"测试问题\"")


if __name__ == "__main__":
    main()
