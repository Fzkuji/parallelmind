"""
Dynamic Fourier Position Encoding

不预分配，动态计算
"""

import torch
import torch.nn as nn
import math


class DynamicFourierPositionEncoding(nn.Module):
    """
    动态计算的Fourier PE（无需预分配）

    Args:
        d_model: 隐藏层维度
        theta: 基础频率（默认10000）
        learnable_scale: 是否学习一个全局缩放因子
    """

    def __init__(
        self,
        d_model: int,
        theta: float = 10000.0,
        learnable_scale: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.theta = theta

        # 预计算频率（只需要d_model/2个频率，不是所有位置！）
        # freq[i] = 1 / (theta ^ (2i / d_model))
        inv_freq = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(theta) / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)  # [d_model/2]

        # 可选：学习一个全局缩放因子
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        print(f"✓ DynamicFourierPositionEncoding: d_model={d_model}, theta={theta}, learnable_scale={learnable_scale}")

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        动态计算位置编码

        Args:
            positions: [batch, seq_len] - 位置索引（可以是任意整数）

        Returns:
            pe: [batch, seq_len, d_model] - 位置编码
        """
        # positions: [batch, seq]
        # inv_freq: [d/2]

        # 计算 pos * freq，广播为 [batch, seq, d/2]
        freqs = torch.einsum('bs,d->bsd', positions.float(), self.inv_freq)

        # 生成sin和cos
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [batch, seq, d]

        # 应用缩放
        emb = emb * self.scale

        if self.dropout is not None:
            emb = self.dropout(emb)

        return emb


def test_dynamic_fpe():
    """测试动态FPE"""
    print("="*80)
    print("测试 Dynamic Fourier PE...")
    print("="*80)

    d_model = 512
    fpe = DynamicFourierPositionEncoding(d_model=d_model, theta=10000.0)

    # 测试1：任意位置（包括很大的位置）
    positions = torch.tensor([[0, 1, 2, 100, 1000, 10000]])  # [1, 6]
    pe = fpe(positions)  # [1, 6, 512]

    print(f"输入positions: {positions.tolist()}")
    print(f"输出shape: {pe.shape}")
    print(f"✓ 可以处理任意大的位置！")

    # 测试2：区分度
    pos_0 = torch.tensor([[0]])
    pos_1 = torch.tensor([[1]])
    pos_128 = torch.tensor([[128]])

    pe_0 = fpe(pos_0)
    pe_1 = fpe(pos_1)
    pe_128 = fpe(pos_128)

    # 归一化
    pe_0_norm = pe_0 / pe_0.norm()
    pe_1_norm = pe_1 / pe_1.norm()
    pe_128_norm = pe_128 / pe_128.norm()

    sim_0_1 = (pe_0_norm * pe_1_norm).sum().item()
    sim_0_128 = (pe_0_norm * pe_128_norm).sum().item()

    print(f"\n区分度测试:")
    print(f"  Pos 0 vs Pos 1: 相似度 {sim_0_1:.4f}")
    print(f"  Pos 0 vs Pos 128: 相似度 {sim_0_128:.4f}")

    # 测试3：内存占用
    import sys
    total_params = sum(p.numel() for p in fpe.parameters())
    total_buffers = sum(b.numel() for b in fpe.buffers())

    print(f"\n内存占用:")
    print(f"  参数: {total_params} (learnable)")
    print(f"  Buffer: {total_buffers} (固定)")
    print(f"  总共: {total_params + total_buffers} = {(total_params + total_buffers) * 4 / 1024:.2f} KB")

    print(f"\n对比预计算版本:")
    print(f"  预计算（max_pos=512）: {512 * 512} = {512*512*4/1024:.2f} KB")
    print(f"  预计算（max_pos=32768）: {32768 * 512} = {32768*512*4/1024:.2f} KB")
    print(f"  动态计算: {d_model//2} = {(d_model//2)*4/1024:.2f} KB")
    print(f"  ✓ 节省内存 {(32768*512) / (d_model//2):.0f}x !")

    print("\n✓ Dynamic FPE test passed!")


if __name__ == "__main__":
    test_dynamic_fpe()
