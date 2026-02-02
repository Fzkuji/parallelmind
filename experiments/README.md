# ParallelMind 消融实验

本目录包含三组消融实验，用于测试不同超参数对模型性能的影响。

## 实验概览

| 实验 | 脚本 | 变量 | 实验数 |
|------|------|------|--------|
| 训练分支数 | `run_ablation_branch.sh` | 训练分支配置 × rope_ratio | 54 |
| Attention Head 数量 | `run_ablation_heads.sh` | num_heads × rope_ratio × 分支配置 | 216 |
| Attention Head 维度 | `run_ablation_headdim.sh` | head_dim × rope_ratio × 分支配置 | 162 |

---

## 实验 1: 训练分支数测试

测试不同训练分支配置对模型在各评估分支数下性能的影响。

### 配置

- **固定参数**: hidden_size=512, num_heads=8, head_dim=64
- **rope_2d_ratio**: 0, 0.25, 0.5, 0.75, 0.875, 1.0
- **固定分支训练**: 1, 2, 4, 8, 16
- **动态分支训练**: 1-3, 1-7, 1-15, 1-31
- **评估分支**: 1, 2, 4, 8, 16, 24, 32

### 运行

```bash
./experiments/scripts/run_ablation_branch.sh           # 正常运行
./experiments/scripts/run_ablation_branch.sh --force   # 强制重跑
```

### 输出

```
experiments/logs/ablation/
├── results.csv        # CSV: rope_ratio,train_branch,eval_branch,loss,ppl
└── ...
```

---

## 实验 2: Attention Head 数量测试（Scaling Law）

固定 head_dim=64，测试不同 attention head 数量对性能的影响。

### 配置

| num_heads | hidden_size | 说明 |
|-----------|-------------|------|
| 4 | 256 | 小模型 |
| 8 | 512 | 基准 |
| 16 | 1024 | 大模型 |
| 24 | 1536 | 更大模型 |

- **rope_2d_ratio**: 0, 0.25, 0.5, 0.75, 0.875, 1.0
- **训练分支**: 同实验 1
- **评估分支**: 1, 2, 4, 8, 16, 24, 32

### 运行

```bash
./experiments/scripts/run_ablation_heads.sh           # 正常运行
./experiments/scripts/run_ablation_heads.sh --force   # 强制重跑
```

### 输出

```
experiments/logs/ablation_heads/
├── results.csv        # CSV: num_heads,hidden_size,rope_ratio,train_branch,eval_branch,loss,ppl
└── ...
```

---

## 实验 3: Attention Head 维度测试

固定 hidden_size=512，测试不同 head 维度对性能的影响。

### 配置

| head_dim | num_heads | freq_count | ratio 精度 |
|----------|-----------|------------|------------|
| 32 | 16 | 16 | 6.25% |
| 64 | 8 | 32 | 3.125% |
| 128 | 4 | 64 | 1.5625% |

- **rope_2d_ratio**: 0, 0.25, 0.5, 0.75, 0.875, 1.0（所有 head_dim 都支持）
- **训练分支**: 同实验 1
- **评估分支**: 1, 2, 4, 8, 16, 24, 32

### 运行

```bash
./experiments/scripts/run_ablation_headdim.sh           # 正常运行
./experiments/scripts/run_ablation_headdim.sh --force   # 强制重跑
```

### 输出

```
experiments/logs/ablation_headdim/
├── results.csv        # CSV: head_dim,num_heads,rope_ratio,train_branch,eval_branch,loss,ppl
└── ...
```

---

## 通用功能

### 断点续训

所有脚本支持中断后继续：

1. **自动跳过已完成的训练**：检测模型文件是否存在
2. **自动跳过已完成的评估**：记录在 `completed.txt` 中
3. **OOM 自动重试**：自动减半 batch size，最多重试 3 次
4. **中断后直接重新运行**：只会继续未完成的部分

### 使用场景

```bash
# 场景 1: 训练中断，直接重新运行继续
./experiments/scripts/run_ablation_branch.sh

# 场景 2: 某个实验 OOM，调整配置后继续
# 脚本会自动跳过已完成的，尝试继续失败的

# 场景 3: 想要重新跑所有实验
./experiments/scripts/run_ablation_branch.sh --force
```

### Batch Size 计算

保持每 batch 约 8 个文本：

**固定分支**:
| 训练分支 | batch_size | 文本数 |
|---------|------------|--------|
| 1 | 8 | 8×1=8 |
| 2 | 4 | 4×2=8 |
| 4 | 2 | 2×4=8 |
| 8 | 1 | 1×8=8 |
| 16 | 1 | 1×16=16 |

**动态分支**:
| 范围 | 平均分支 | batch_size | 平均文本数 |
|------|---------|------------|-----------|
| 1-3 | 2 | 4 | 4×2=8 |
| 1-7 | 4 | 2 | 2×4=8 |
| 1-15 | 8 | 1 | 1×8=8 |
| 1-31 | 16 | 1 | 1×16=16 |

---

## 输出目录命名

### 实验 1 (Branch)
```
out/{hidden_size}-h{num_heads}-r{rope_ratio}-b{branch_config}
```
示例: `out/512-h8-r05-bfixed4`, `out/512-h8-r075-b1-15`

### 实验 2 (Heads)
```
out/{hidden_size}-h{num_heads}-r{rope_ratio}-b{branch_config}
```
示例: `out/1024-h16-r05-bfixed4`, `out/1536-h24-r075-b1-15`

### 实验 3 (HeadDim)
```
out/{hidden_size}-d{head_dim}-h{num_heads}-r{rope_ratio}-b{branch_config}
```
示例: `out/512-d32-h16-r05-bfixed4`, `out/512-d128-h4-r075-b1-15`

---

## 绘图

### Python 示例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 实验 1: 按 rope_ratio 分组
df = pd.read_csv('experiments/logs/ablation/results.csv')
for ratio in df['rope_ratio'].unique():
    subset = df[df['rope_ratio'] == ratio]
    for train_branch in subset['train_branch'].unique():
        data = subset[subset['train_branch'] == train_branch]
        plt.plot(data['eval_branch'], data['loss'], label=f'train={train_branch}')
    plt.xlabel('Eval Branches')
    plt.ylabel('Loss')
    plt.title(f'rope_2d_ratio = {ratio}')
    plt.legend()
    plt.savefig(f'branch_ratio_{ratio}.png')
    plt.clf()

# 实验 2: 按 num_heads 和 rope_ratio 分组
df = pd.read_csv('experiments/logs/ablation_heads/results.csv')
for num_heads in df['num_heads'].unique():
    for ratio in df['rope_ratio'].unique():
        subset = df[(df['num_heads'] == num_heads) & (df['rope_ratio'] == ratio)]
        for train_branch in subset['train_branch'].unique():
            data = subset[subset['train_branch'] == train_branch]
            plt.plot(data['eval_branch'], data['loss'], label=f'train={train_branch}')
        plt.xlabel('Eval Branches')
        plt.ylabel('Loss')
        plt.title(f'num_heads={num_heads}, rope_ratio={ratio}')
        plt.legend()
        plt.savefig(f'heads_{num_heads}_ratio_{ratio}.png')
        plt.clf()

# 实验 3: 按 head_dim 和 rope_ratio 分组
df = pd.read_csv('experiments/logs/ablation_headdim/results.csv')
for head_dim in df['head_dim'].unique():
    for ratio in df['rope_ratio'].unique():
        subset = df[(df['head_dim'] == head_dim) & (df['rope_ratio'] == ratio)]
        for train_branch in subset['train_branch'].unique():
            data = subset[subset['train_branch'] == train_branch]
            plt.plot(data['eval_branch'], data['loss'], label=f'train={train_branch}')
        plt.xlabel('Eval Branches')
        plt.ylabel('Loss')
        plt.title(f'head_dim={head_dim}, rope_ratio={ratio}')
        plt.legend()
        plt.savefig(f'headdim_{head_dim}_ratio_{ratio}.png')
        plt.clf()
```

---

## 文件结构

```
experiments/
├── README.md
├── logs/
│   ├── ablation/                    # 实验 1: 训练分支数
│   │   ├── train.log
│   │   ├── loss_records.txt
│   │   ├── results.csv
│   │   ├── errors.txt
│   │   └── completed.txt
│   ├── ablation_heads/              # 实验 2: Head 数量
│   │   └── ...
│   └── ablation_headdim/            # 实验 3: Head 维度
│       └── ...
└── scripts/
    ├── run_ablation_branch.sh       # 实验 1
    ├── run_ablation_heads.sh        # 实验 2
    └── run_ablation_headdim.sh      # 实验 3
```

单模型评估可直接使用 `scripts/eval_loss.py`。

---

## 预期结果

### 假设

1. **`rope_2d_ratio=0`** 在单分支评估时应该最好（纯 1D RoPE）
2. **`rope_2d_ratio=1.0`** 在多分支评估时 loss 上升最慢
3. **固定分支训练** 在对应分支数评估时表现最好
4. **动态分支训练** 具有更好的泛化性
5. **更多 attention heads** 应该带来更好的性能（scaling law）
6. **更大 head_dim** 可能有更精细的 rope_2d_ratio 控制
