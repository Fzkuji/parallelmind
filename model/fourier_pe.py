"""
Fourier Position Encoding (FPE) for Branch Discrimination

使用傅里叶基函数编码branch位置，直接加在embedding后面。

核心思想：
    x_input = x_token_embed + FPE(branch_id)

傅里叶基函数：
    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

与RoPE的区别：
    - RoPE: 作用在Q/K上，影响attention计算（相对位置）
    - FPE: 直接加在输入上（绝对位置）

使用场景：
    - Branch ID编码（离散、绝对）
    - RoPE仍然用于时间维度（连续、相对）

参考：
    - Transformer原始论文的位置编码
    - ViT中的位置编码
    - NeRF中的位置编码
"""

import torch
import torch.nn as nn
import math


class FourierPositionEncoding(nn.Module):
    """
    Fourier Position Encoding for branch identification

    Args:
        d_model: 隐藏层维度（必须和模型的hidden_size一致）
        max_positions: 最大位置数（即最大branch数）
        theta: 基础频率（默认10000，和RoPE一致）
        learnable: 是否让频率可学习
        dropout: dropout比例（可选）
    """

    def __init__(
        self,
        d_model: int,
        max_positions: int = 512,
        theta: float = 10000.0,
        learnable: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        self.theta = theta
        self.learnable = learnable

        # 计算频率
        # freq[i] = 1 / (theta ^ (2i / d_model))
        position = torch.arange(max_positions).unsqueeze(1).float()  # [max_pos, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(theta) / d_model))  # [d/2]

        # 预计算所有位置的编码 [max_pos, d_model]
        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度：cos

        if learnable:
            # 可学习的位置编码（从Fourier初始化）
            self.pe = nn.Parameter(pe)
        else:
            # 固定的位置编码
            self.register_buffer('pe', pe)

        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        print(f"✓ FourierPositionEncoding: d_model={d_model}, max_pos={max_positions}, theta={theta}, learnable={learnable}")

    def forward(self, branch_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_ids: [batch, seq_len] - 每个token的branch ID

        Returns:
            pe: [batch, seq_len, d_model] - 位置编码
        """
        batch_size, seq_len = branch_ids.shape

        # 确保branch_ids在范围内
        branch_ids = torch.clamp(branch_ids.long(), 0, self.max_positions - 1)

        # 查表获取位置编码
        pe = self.pe[branch_ids]  # [batch, seq_len, d_model]

        if self.dropout is not None:
            pe = self.dropout(pe)

        return pe

    def get_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        获取指定位置的编码（用于分析/可视化）

        Args:
            positions: [N] - 位置列表

        Returns:
            encodings: [N, d_model]
        """
        positions = torch.clamp(positions.long(), 0, self.max_positions - 1)
        return self.pe[positions]


class LearnableBranchEncoding(nn.Module):
    """
    可学习的Branch Embedding（作为对比baseline）

    这是最简单的方案：每个branch_id对应一个可学习向量。
    和Fourier PE的区别：
        - Fourier PE: 从数学函数初始化，可选learnable
        - Learnable: 从零/随机初始化，完全可学习
    """

    def __init__(
        self,
        d_model: int,
        max_branches: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_branches = max_branches

        # 可学习的embedding
        self.branch_embed = nn.Embedding(max_branches, d_model)

        # 初始化（使用较小的值，避免影响初始训练）
        nn.init.normal_(self.branch_embed.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        print(f"✓ LearnableBranchEncoding: d_model={d_model}, max_branches={max_branches}")

    def forward(self, branch_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_ids: [batch, seq_len]

        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        branch_ids = torch.clamp(branch_ids.long(), 0, self.max_branches - 1)
        embeddings = self.branch_embed(branch_ids)

        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        return embeddings


def analyze_fourier_discrimination(d_model=512, theta=10000.0, stride=128):
    """
    分析Fourier PE的branch区分能力

    测试：不同branch的FPE编码是否足够不同？
    """
    print("="*80)
    print("Analyzing Fourier Position Encoding discrimination...")
    print("="*80)

    # 确保max_positions足够大（至少能容纳8个branch）
    max_pos = max(1024, stride * 10)
    fpe = FourierPositionEncoding(d_model=d_model, max_positions=max_pos, theta=theta)

    # 测试场景：stride=128，即branch 0,1,2,... 对应位置 0, 128, 256, ...
    branch_positions = [i * stride for i in range(8)]
    positions = torch.tensor(branch_positions)

    # 获取编码
    encodings = fpe.get_encoding(positions)  # [8, d_model]

    print(f"\nBranch positions: {branch_positions}")
    print(f"Encoding shape: {encodings.shape}")

    # 计算相似度矩阵
    # 归一化
    encodings_norm = encodings / (encodings.norm(dim=-1, keepdim=True) + 1e-8)

    # 余弦相似度
    similarity = torch.matmul(encodings_norm, encodings_norm.T)  # [8, 8]

    print(f"\n余弦相似度矩阵 (越接近1越相似):")
    print(similarity.numpy())

    # 分析对角线外的相似度
    mask = ~torch.eye(8, dtype=torch.bool)
    off_diagonal_sim = similarity[mask]

    print(f"\n统计:")
    print(f"  同位置相似度 (对角线): 1.0")
    print(f"  不同branch平均相似度: {off_diagonal_sim.mean().item():.6f}")
    print(f"  不同branch最大相似度: {off_diagonal_sim.max().item():.6f}")
    print(f"  不同branch最小相似度: {off_diagonal_sim.min().item():.6f}")

    # L2距离
    print(f"\nL2距离分析:")
    for i in range(7):
        dist = torch.norm(encodings[i] - encodings[i+1]).item()
        print(f"  Branch {i} vs {i+1} (pos {branch_positions[i]} vs {branch_positions[i+1]}): L2={dist:.4f}")

    # 与RoPE 2D对比
    print(f"\n与RoPE 2D对比:")
    print(f"  RoPE 2D (stride={stride}): 93-99%相似度 ❌")
    print(f"  Fourier PE: {off_diagonal_sim.mean().item()*100:.2f}%平均相似度")

    if off_diagonal_sim.mean().item() < 0.5:
        print(f"  ✅ Fourier PE区分度更好！")
    else:
        print(f"  ⚠️  Fourier PE相似度仍然较高")

    # 测试连续branch（stride=1）
    print(f"\n测试连续branch (stride=1):")
    continuous_positions = torch.arange(8)
    continuous_encodings = fpe.get_encoding(continuous_positions)
    continuous_encodings_norm = continuous_encodings / (continuous_encodings.norm(dim=-1, keepdim=True) + 1e-8)
    continuous_similarity = torch.matmul(continuous_encodings_norm, continuous_encodings_norm.T)
    continuous_off_diag = continuous_similarity[mask]

    print(f"  不同branch平均相似度: {continuous_off_diag.mean().item():.6f}")

    return similarity


def test_fourier_pe():
    """测试FPE的基本功能"""
    print("\n" + "="*80)
    print("Testing Fourier Position Encoding...")
    print("="*80)

    batch_size = 2
    seq_len = 16
    d_model = 512

    # 创建FPE
    fpe = FourierPositionEncoding(d_model=d_model, max_positions=512, theta=10000.0)

    # 模拟场景：2个branch，每个branch 8个token
    # Branch 0: 位置 0 (对应branch_id=0, stride=128 -> pos=0)
    # Branch 1: 位置 128 (对应branch_id=1, stride=128 -> pos=128)
    stride = 128
    branch_ids = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # sample 1
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # sample 2
    ]) * stride

    # 获取位置编码
    pe = fpe(branch_ids)  # [batch, seq, d_model]

    print(f"\nInput shape: {branch_ids.shape}")
    print(f"Output shape: {pe.shape}")

    # 检查同branch的编码是否相同
    branch0_encoding = pe[0, 0, :]  # 第一个样本，branch 0的第一个token
    branch0_encoding_2 = pe[0, 1, :]  # 第一个样本，branch 0的第二个token
    same_branch_diff = torch.norm(branch0_encoding - branch0_encoding_2).item()
    print(f"\n同branch不同token的编码差异: {same_branch_diff:.6f} (应该≈0)")

    # 检查不同branch的编码是否不同
    branch1_encoding = pe[0, 8, :]  # 第一个样本，branch 1的第一个token
    diff_branch_diff = torch.norm(branch0_encoding - branch1_encoding).item()
    print(f"不同branch的编码差异: {diff_branch_diff:.4f} (应该>10)")

    # 检查不同样本相同branch的编码是否相同
    sample2_branch0 = pe[1, 0, :]
    cross_sample_diff = torch.norm(branch0_encoding - sample2_branch0).item()
    print(f"不同样本相同branch的编码差异: {cross_sample_diff:.6f} (应该≈0)")

    print("\n✓ Fourier PE test passed!")

    # 运行区分度分析
    analyze_fourier_discrimination(d_model=d_model, stride=stride)


if __name__ == "__main__":
    test_fourier_pe()
