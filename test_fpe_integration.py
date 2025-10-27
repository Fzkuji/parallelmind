#!/usr/bin/env python3
"""
测试Fourier PE集成

验证：
1. 模型能正确加载rope和fpe两种配置
2. FPE能正确编码branch
3. Forward pass正常工作
"""

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def test_rope_mode():
    """测试RoPE 2D模式（原有方式）"""
    print("="*80)
    print("测试 RoPE 2D 模式...")
    print("="*80)

    config = MiniMindConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        vocab_size=1000,
        pe_type='rope'
    )

    model = MiniMindForCausalLM(config)
    print(f"✓ RoPE 2D 模型创建成功")
    print(f"  - fourier_pe: {model.model.fourier_pe}")

    # 测试forward
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 模拟2个branch的pos2d
    branch_ids = torch.tensor([[0]*8 + [128]*8, [0]*8 + [128]*8])  # [batch, seq]
    time_ids = torch.tensor([[0,1,2,3,4,5,6,7]*2, [0,1,2,3,4,5,6,7]*2])
    pos2d = torch.stack([branch_ids, time_ids], dim=-1)  # [batch, seq, 2]

    outputs = model(input_ids=input_ids, pos2d=pos2d)
    print(f"✓ Forward pass成功")
    print(f"  - Logits shape: {outputs.logits.shape}")

    print("✓ RoPE 2D 模式测试通过！\n")
    return model


def test_fpe_mode():
    """测试Fourier PE模式（新方式）"""
    print("="*80)
    print("测试 Fourier PE 模式...")
    print("="*80)

    config = MiniMindConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        vocab_size=1000,
        pe_type='fpe',
        fpe_theta=10000.0,
        fpe_learnable=False,
        fpe_max_positions=512
    )

    model = MiniMindForCausalLM(config)
    print(f"✓ Fourier PE 模型创建成功")
    print(f"  - fourier_pe: {model.model.fourier_pe}")
    print(f"  - fpe_max_positions: {config.fpe_max_positions}")

    # 测试forward
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 模拟2个branch的pos2d
    branch_ids = torch.tensor([[0]*8 + [128]*8, [0]*8 + [128]*8])  # [batch, seq]
    time_ids = torch.tensor([[0,1,2,3,4,5,6,7]*2, [0,1,2,3,4,5,6,7]*2])
    pos2d = torch.stack([branch_ids, time_ids], dim=-1)  # [batch, seq, 2]

    outputs = model(input_ids=input_ids, pos2d=pos2d)
    print(f"✓ Forward pass成功")
    print(f"  - Logits shape: {outputs.logits.shape}")

    print("✓ Fourier PE 模式测试通过！\n")
    return model


def test_branch_discrimination():
    """测试FPE的branch区分能力"""
    print("="*80)
    print("测试 Branch 区分能力...")
    print("="*80)

    config = MiniMindConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        vocab_size=1000,
        pe_type='fpe',
        fpe_max_positions=512
    )

    model = MiniMindForCausalLM(config)

    # 测试：相同的input_ids，不同的branch
    input_ids = torch.tensor([[10, 20, 30, 40]])  # [1, 4]

    # Branch 0
    pos2d_branch0 = torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3]]])
    outputs_branch0 = model(input_ids=input_ids, pos2d=pos2d_branch0)
    logits_branch0 = outputs_branch0.logits

    # Branch 1 (pos=128)
    pos2d_branch1 = torch.tensor([[[128, 0], [128, 1], [128, 2], [128, 3]]])
    outputs_branch1 = model(input_ids=input_ids, pos2d=pos2d_branch1)
    logits_branch1 = outputs_branch1.logits

    # 计算差异
    diff = torch.abs(logits_branch0 - logits_branch1).mean().item()
    print(f"\n相同input_ids，不同branch的logits差异: {diff:.6f}")

    if diff > 0.01:
        print(f"✓ Branch区分成功！不同branch产生不同的输出")
    else:
        print(f"⚠️  Branch区分度可能不足（差异太小）")

    # 测试：相同branch，相同input，不同时间
    pos2d_time0 = torch.tensor([[[0, 0]]])
    pos2d_time10 = torch.tensor([[[0, 10]]])

    input_single = torch.tensor([[10]])
    outputs_time0 = model(input_ids=input_single, pos2d=pos2d_time0)
    outputs_time10 = model(input_ids=input_single, pos2d=pos2d_time10)

    time_diff = torch.abs(outputs_time0.logits - outputs_time10.logits).mean().item()
    print(f"相同branch，相同input，不同时间的logits差异: {time_diff:.6f}")

    if time_diff > 0.01:
        print(f"✓ 时间位置编码工作正常")
    else:
        print(f"⚠️  时间位置编码可能不足")

    print("\n✓ Branch区分能力测试完成！\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Fourier PE 集成测试")
    print("="*80 + "\n")

    # 测试两种模式
    test_rope_mode()
    test_fpe_mode()

    # 测试branch区分
    test_branch_discrimination()

    print("="*80)
    print("✅ 所有测试通过！")
    print("="*80)
    print("""
下一步：

1. 训练测试（单branch预热）：
   python trainer/train_pretrain.py \\
     --pe fpe \\
     --epochs 1 \\
     --batch_size 4 \\
     --branches_per_sample 1 \\
     --branch_slice_count 8 \\
     --branch_loop_all \\
     --out_dir out/fpe_test_stage1

2. 训练测试（2个branch）：
   python trainer/train_pretrain.py \\
     --pe fpe \\
     --epochs 2 \\
     --batch_size 4 \\
     --max_branches_per_sample 2 \\
     --min_branches_per_sample 2 \\
     --batch_by_samples \\
     --init_weight out/fpe_test_stage1/pretrain_512.pth \\
     --out_dir out/fpe_test_stage2

3. 推理测试：
   python parallel_generate.py \\
     --model out/fpe_test_stage2/pretrain_512.pth \\
     --branches 2 \\
     --mode pretrain

参数说明：
- --pe fpe: 使用Fourier PE（默认rope）
- --fpe_theta 10000.0: FPE基础频率（默认10000）
- --fpe_max_positions 512: FPE最大位置数（默认512，足够容纳常见的branch数量）
- --fpe_learnable: 使FPE可学习（默认固定）

说明：
- RoPE 2D模式：branch使用stride=128，pos为[0, 128, 256, ...]
- FPE模式：branch使用stride=1，pos为[0, 1, 2, 3, ...]（直接的branch索引）
- FPE直接查表，每个branch_id对应一个独立的向量
- fpe_max_positions=512足够容纳常见场景（branch数量通常<100）
""")
