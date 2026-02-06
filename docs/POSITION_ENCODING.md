# 位置编码指南

## 两种方案

| 方案 | Branch 编码 | Time 编码 | 区分度 |
|------|-------------|-----------|--------|
| RoPE 2D | 旋转（相对） | 旋转（相对） | 93-99% 相似 |
| Fourier PE | 查表（绝对） | 旋转（相对） | 31% 相似 |

## RoPE 2D（默认）

将部分频率对分配给 branch 维度，其余分配给 time 维度。

```bash
python src/training/train_pretrain.py \
  --pe rope \
  --rope_2d_ratio 0.5
```

- `rope_2d_ratio=0.5`：一半频率用于 branch，一半用于 time
- Branch 位置 = `branch_id × 128`

## Fourier PE

Branch 使用绝对位置编码（查表），Time 使用 1D RoPE。

```bash
python src/training/train_pretrain.py \
  --pe fpe
```

**优势：**
- 更强的 branch 区分度（31% vs 93-99%）
- 架构解耦（branch 和 time 分离）

**训练流程（推荐）：**

```bash
# Stage 1: 单分支预热
python src/training/train_pretrain.py \
  --pe fpe \
  --epochs 2 \
  --branches_per_sample 1 \
  --out_dir out/fpe_stage1

# Stage 2: 多分支训练
python src/training/train_pretrain.py \
  --pe fpe \
  --epochs 3 \
  --max_branches_per_sample 2 \
  --init_weight out/fpe_stage1/pretrain_512.pth \
  --out_dir out/fpe_stage2
```

## 参数对比

| 参数 | RoPE 2D | FPE |
|------|---------|-----|
| `--pe` | rope | fpe |
| `--rope_2d_ratio` | 0.5 | - |
| `--branch_stride` | 128（自动） | 1（自动） |

## 选择建议

- **RoPE 2D**：默认方案，大多数场景
- **Fourier PE**：需要强 branch 区分度时
