# ParallelMind 实验脚本

本目录包含 ParallelMind 项目的各类实验脚本。

## 目录结构

```
scripts/
├── README.md
└── parameters/          # 超参数消融实验
    ├── run_ablation_branch.sh     # 训练分支数实验
    ├── run_ablation_heads.sh      # Attention Head 数量实验
    └── run_ablation_headdim.sh    # Attention Head 维度实验
```

---

## 超参数消融实验 (`parameters/`)

测试不同超参数配置对并行解码性能的影响。

### 实验概览

| 实验 | 脚本 | 测试变量 | 目的 |
|------|------|----------|------|
| 训练分支数 | `run_ablation_branch.sh` | 训练分支配置 × rope_2d_ratio | 验证动态分支训练的泛化性 |
| Head 数量 | `run_ablation_heads.sh` | num_heads × rope_2d_ratio | Scaling law 测试 |
| Head 维度 | `run_ablation_headdim.sh` | head_dim × rope_2d_ratio | 2D RoPE 精度影响 |

### 核心参数

**训练分支配置**（每步约 32k tokens）：
- 固定分支: `1,1` (baseline)
- 动态分支: `1,3` / `1,7` / `1,15`

**2D RoPE 比例**：`0, 0.25, 0.5, 0.75, 1.0`
- `0`: 纯 1D RoPE（无分支位置编码）
- `1.0`: 全部频率用于 2D 编码

**评估分支**：`1, 2, 4, 8, 16, 24, 32`

### 运行方式

```bash
# 正常运行（支持断点续训）
./scripts/parameters/run_ablation_branch.sh

# 强制重跑所有实验
./scripts/parameters/run_ablation_branch.sh --force
```

### 输出

```
scripts/logs/ablation/
├── results.csv        # 结构化结果（用于绘图）
├── train.log          # 完整训练日志
├── loss_records.txt   # Loss 记录
├── errors.txt         # 错误记录
└── completed.txt      # 已完成实验标记
```

### 功能特性

1. **断点续训**：自动跳过已完成的训练和评估
2. **OOM 重试**：自动减半 batch size，最多重试 3 次
3. **模型复用**：统一输出目录格式，避免重复训练
4. **信号处理**：中断时自动清理 GPU 进程

---

## 预期结论

1. **rope_2d_ratio=0** 在单分支评估时表现最好（纯 1D RoPE）
2. **rope_2d_ratio=1.0** 在多分支评估时 loss 上升最慢
3. **动态分支训练** 比固定分支训练具有更好的泛化性
4. **更多 attention heads** 带来更好的性能（scaling law）
5. **更大 head_dim** 提供更精细的 2D RoPE 控制

---

## 依赖

实验脚本调用以下核心模块：

- `src/training/train_pretrain.py` - 预训练脚本
- `src/inference/eval_loss.py` - 评估脚本
- `dataset/pretrain_512.jsonl` - 训练数据
- `dataset/pretrain_hq_split.jsonl` - 评估数据
