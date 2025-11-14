# HuggingFace + LoRA 并行推理指南

## 概述

`scripts/parallel_inference_hf_lora.py` 提供了 HuggingFace + LoRA + 2D RoPE 的并行/批量推理功能，支持：

- ✅ 多 GPU 分布式推理（DDP）
- ✅ Parallel 数据格式（multi-branch）
- ✅ 2D RoPE 自动处理
- ✅ LoRA 权重加载
- ✅ 批量高效推理

## 快速开始

### 单 GPU 推理

```bash
python scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_parallel_lora_hf_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --patch_rope \
  --data_path dataset/pretrain_hq_split.jsonl \
  --out_path out/infer_results.jsonl \
  --batch_size 16 \
  --batch_by_samples \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 1 \
  --max_new_tokens 512
```

### 多 GPU 分布式推理

```bash
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_parallel_lora_hf_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --patch_rope \
  --data_path dataset/pretrain_hq_split.jsonl \
  --out_path out/infer_results.jsonl \
  --batch_size 16 \
  --batch_by_samples \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 1 \
  --max_new_tokens 512
```

## 参数说明

### 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base_model` | HuggingFace 模型路径 | Qwen/Qwen2.5-14B-Instruct |
| `--data_path` | 输入数据路径（Parallel JSONL） | dataset/pretrain_hq_split.jsonl |
| `--out_path` | 输出结果路径（JSONL） | out/infer_results.jsonl |

### LoRA 参数（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_path` | None | LoRA 权重路径（不指定则只用基础模型） |
| `--lora_rank` | 8 | LoRA rank（必须与训练时一致） |

### 2D RoPE 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patch_rope` | True | 应用 2D RoPE |
| `--no_patch_rope` | - | 禁用 2D RoPE |
| `--rope_2d_ratio` | 0.5 | Branch 维度频率比例（必须与训练时一致） |

### Parallel 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--branches_per_sample` | 4 | 固定 branch 数量 |
| `--max_branches_per_sample` | None | 动态模式：最大 branch 数 |
| `--min_branches_per_sample` | 1 | 动态模式：最小 branch 数 |
| `--batch_by_samples` | False | 按样本数计算 batch |
| `--batch_size` | 16 | 批次大小 |
| `--max_total_tokens` | 0 | 最大 token 数（0=动态） |
| `--branch_stride` | 128 | Branch 位置步长 |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_new_tokens` | 512 | 最大生成 token 数 |
| `--temperature` | 0.7 | 温度 |
| `--top_p` | 0.9 | Nucleus sampling |

### 系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | cuda | 设备（单 GPU 模式） |
| `--dtype` | bfloat16 | 数据类型 |
| `--num_workers` | 1 | DataLoader workers |
| `--max_samples` | None | 限制处理的样本数量 |

## 数据格式

### 输入格式（Parallel JSONL）

每行一个 JSON 对象，包含 `main` 和 `branches`：

```json
{
  "main": "主分支内容...",
  "branches": [
    "分支1内容...",
    "分支2内容...",
    "分支3内容..."
  ]
}
```

或简化格式（text 会被当作 main）：

```json
{
  "text": "主分支内容..."
}
```

### 输出格式（JSONL）

```json
{
  "input": "输入文本...",
  "output": "生成的回复..."
}
```

每行一个结果。

## 工作流程

```
加载 Parallel 数据集
    ↓
创建 DataLoader（自动分片到多 GPU）
    ↓
每个 GPU 处理一部分数据
    ↓
生成（自动处理 pos2d）
    ↓
收集所有 GPU 的结果
    ↓
保存到输出文件
```

## 性能优化

### 1. 多 GPU 加速

使用 `torchrun` 启动多 GPU 推理：

```bash
# 8 卡推理
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/xxx.pth \
  --data_path dataset/test.jsonl \
  --out_path out/results.jsonl \
  --batch_size 8 \
  --batch_by_samples
```

每张卡会自动处理数据的一部分，最后合并结果。

### 2. 批次大小调整

- **显存充足**：增大 `--batch_size` 提高吞吐量
- **显存不足**：减小 `--batch_size`
- **使用 batch_by_samples**：让 batch_size 表示样本数，自动换算 branch 数

```bash
# 显存充足：大 batch
--batch_size 32 --batch_by_samples

# 显存不足：小 batch
--batch_size 4 --batch_by_samples
```

### 3. 动态 branch 模式

使用动态 branch 数可以更高效利用显存：

```bash
--max_branches_per_sample 16 \
--min_branches_per_sample 1 \
--batch_by_samples
```

### 4. 数据预处理

- 提前过滤掉过长的样本
- 使用 `--max_samples` 在测试时限制数据量

## 常见问题

### Q1: 如何验证结果正确性？

```bash
# 查看输出文件
head -n 5 out/infer_results.jsonl

# 统计生成的结果数量
wc -l out/infer_results.jsonl
```

### Q2: 多 GPU 时如何查看每张卡的进度？

每张卡会打印自己的日志，包含 `rank=X`。查看日志确认所有卡都在工作。

### Q3: 如何处理超长样本？

设置 `--max_total_tokens` 限制序列长度：

```bash
--max_total_tokens 2048
```

或在数据预处理时过滤。

### Q4: 不使用 LoRA 可以吗？

可以，不指定 `--lora_path` 即可：

```bash
python scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --data_path dataset/test.jsonl \
  --out_path out/results.jsonl
```

### Q5: 如何禁用 2D RoPE？

添加 `--no_patch_rope`：

```bash
python scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --data_path dataset/test.jsonl \
  --out_path out/results.jsonl \
  --no_patch_rope
```

## 与单条推理的对比

| 特性 | parallel_inference_hf_lora.py | inference_hf_lora.py |
|------|------------------------------|----------------------|
| **用途** | 批量推理 | 单条/交互式推理 |
| **多 GPU** | ✅ 支持 DDP | ❌ 不支持 |
| **输入** | JSONL 文件 | 命令行 prompt |
| **输出** | JSONL 文件 | 终端输出 |
| **适用场景** | 大规模数据推理 | 测试、交互 |

## 完整示例

### 示例 1：14B 模型 8 卡推理

```bash
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_14b_lora_final.pth \
  --lora_rank 32 \
  --rope_2d_ratio 0.5 \
  --data_path dataset/test_10k.jsonl \
  --out_path out/test_10k_results.jsonl \
  --batch_size 8 \
  --batch_by_samples \
  --max_branches_per_sample 12 \
  --min_branches_per_sample 2 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --dtype bfloat16
```

### 示例 2：0.5B 模型单卡测试

```bash
python scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --data_path dataset/test_100.jsonl \
  --out_path out/test_100_results.jsonl \
  --batch_size 16 \
  --batch_by_samples \
  --max_branches_per_sample 8 \
  --max_new_tokens 256 \
  --max_samples 100
```

### 示例 3：不使用 LoRA，纯基础模型推理

```bash
torchrun --nproc_per_node 4 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --no_patch_rope \
  --data_path dataset/sft_test.jsonl \
  --out_path out/sft_results.jsonl \
  --batch_size 16 \
  --batch_by_samples \
  --max_branches_per_sample 4
```

## 性能参考

| 模型大小 | GPU 数量 | Batch Size | 吞吐量（samples/s） |
|---------|---------|------------|-------------------|
| 0.5B | 1 | 16 | ~50 |
| 1.5B | 1 | 8 | ~25 |
| 7B | 4 | 4 | ~15 |
| 14B | 8 | 8 | ~20 |

*实际吞吐量取决于硬件、序列长度、branch 数量等因素*

## 故障排查

### 显存溢出

```bash
# 减小 batch_size
--batch_size 4

# 减小 max_branches_per_sample
--max_branches_per_sample 4

# 使用更小的模型
--base_model Qwen/Qwen2-0.5B-Instruct
```

### DDP 初始化失败

确保使用 `torchrun` 而不是直接 `python`：

```bash
# ✓ 正确
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py ...

# ✗ 错误
python scripts/parallel_inference_hf_lora.py ...
```

### 结果数量不匹配

检查是否有样本被跳过：

```bash
# 查看日志中的 "数据集加载完成" 和 "生成完成" 的数量
```

## 总结

- ✅ 支持 HuggingFace + LoRA + 2D RoPE
- ✅ 多 GPU 分布式推理
- ✅ 自动处理 pos2d
- ✅ Parallel 数据格式
- ✅ 高效批量推理

更多详细信息请参考：
- [推理指南](INFERENCE_GUIDE.md)
- [训练文档](TRAIN_HF_LORA_USAGE.md)
- [快速开始](QUICK_START_LORA.md)
