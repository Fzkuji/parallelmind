# 训练指南

## HuggingFace + LoRA 训练

### 快速开始

```bash
torchrun --nproc_per_node 8 src/training/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --data_path dataset/pretrain_hq_split.jsonl \
  --lora_rank 8 \
  --batch_size 4 \
  --batch_by_samples \
  --max_branches_per_sample 16 \
  --rope_2d_ratio 0.5 \
  --epochs 3 \
  --ddp
```

### 数据格式（Parallel JSONL）

```json
{
  "main": "主分支文本...",
  "branches": ["分支1...", "分支2...", "分支3..."]
}
```

或简化格式：
```json
{"text": "主分支文本..."}
```

## 核心参数

### 模型相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | - | HuggingFace 模型路径 |
| `--lora_rank` | 8 | LoRA rank |
| `--rope_2d_ratio` | 0.5 | 2D RoPE 比例 |

### 数据相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | - | 训练数据路径 |
| `--max_seq_len` | 1024 | 单分支最大长度 |
| `--branches_per_sample` | 8 | 固定分支数 |
| `--max_branches_per_sample` | - | 动态模式最大分支数 |
| `--min_branches_per_sample` | 1 | 动态模式最小分支数 |

### 训练相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--accumulation_steps` | 1 | 梯度累积 |
| `--dtype` | bfloat16 | 训练精度 |

## 不同规模配置

### 小模型（0.5B）快速测试

```bash
python src/training/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --lora_rank 8 \
  --batch_size 8 \
  --epochs 1
```

### 中等模型（1.5B）

```bash
torchrun --nproc_per_node 4 src/training/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_rank 16 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --ddp
```

### 大模型（7B+）

```bash
torchrun --nproc_per_node 8 src/training/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_rank 32 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --dtype bfloat16 \
  --ddp
```

## 输出

- 训练中：`out/lora/{name}_{model}.pth`
- 最终：`out/lora/{name}_{model}_final.pth`

## 常见问题

**Q: 显存不足**

1. 减小 `--batch_size`
2. 增加 `--accumulation_steps`
3. 减小 `--max_branches_per_sample`
4. 使用 `--dtype bfloat16`

**Q: LoRA rank 如何选择？**

- 小模型（<1B）：rank=4-8
- 中等模型（1-3B）：rank=8-16
- 大模型（>3B）：rank=16-32

**Q: rope_2d_ratio 如何选择？**

- 默认 0.5 是平衡选择
- 分支多、序列短：0.6-0.7
- 分支少、序列长：0.3-0.4
