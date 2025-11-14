# HuggingFace + LoRA 快速开始

## 🚀 一分钟上手

### 训练

```bash
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --data_path dataset/pretrain_hq_split.jsonl \
  --lora_rank 8 \
  --batch_size 4 \
  --batch_by_samples \
  --max_branches_per_sample 16 \
  --min_branches_per_sample 1 \
  --rope_2d_ratio 0.5 \
  --epochs 3 \
  --ddp
```

### 推理

```bash
# 交互式对话
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --mode chat

# 单次生成
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 8 \
  --mode generate \
  --prompt "你好，请介绍一下你自己"
```

## 📋 核心参数速查

### 训练参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--base_model` | - | HF 模型路径 | Qwen/Qwen2-xxx |
| `--lora_rank` | 8 | LoRA 秩 | 小模型:8, 中模型:16, 大模型:32 |
| `--rope_2d_ratio` | 0.5 | Branch 维度频率比例 | 0.3-0.7 |
| `--batch_size` | 16 | 批次大小 | 根据显存调整 |
| `--batch_by_samples` | False | 按样本数计数 | 建议启用 |
| `--max_branches_per_sample` | None | 最大分支数 | 8-16 |
| `--min_branches_per_sample` | 1 | 最小分支数 | 1-4 |
| `--epochs` | 3 | 训练轮数 | 1-5 |

### 推理参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--lora_rank` | 8 | **必须与训练一致** | 同训练 |
| `--rope_2d_ratio` | 0.5 | **必须与训练一致** | 同训练 |
| `--mode` | chat | chat/generate | chat |
| `--temperature` | 0.7 | 温度 | 事实问答:0.2, 创意写作:0.8 |
| `--max_new_tokens` | 512 | 最大生成长度 | 100-2048 |

## ⚠️ 重要提示

### ✅ 已自动处理（无需关心）

- ✅ **pos2d 自动注入**：推理脚本已自动处理 2D RoPE 的 pos2d
- ✅ **prepare_inputs_for_generation 重写**：自动在每次生成前调用 `set_rope_pos2d`
- ✅ **增量生成**：每步自动更新 pos2d

### ⚠️ 必须注意

1. **参数一致性**：
   - `--lora_rank` 训练和推理必须一致
   - `--rope_2d_ratio` 训练和推理必须一致
   - 训练用了 `--patch_rope`，推理也必须用（默认启用）

2. **数据格式**：
   - 训练数据必须是 Parallel 格式（`main` + `branches`）
   - 不支持标准 SFT 格式

3. **显存优化**：
   - 显存不足时减小 `--batch_size`
   - 使用 `--accumulation_steps` 补偿
   - 使用 `--dtype bfloat16`

## 📊 不同模型配置

### Qwen2-0.5B（快速测试）

```bash
# 训练
torchrun --nproc_per_node 4 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_rank 8 \
  --batch_size 8 \
  --batch_by_samples \
  --max_branches_per_sample 16 \
  --min_branches_per_sample 1

# 推理
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 8 \
  --mode chat
```

### Qwen2.5-1.5B（生产推荐）

```bash
# 训练
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_rank 16 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --batch_by_samples \
  --max_branches_per_sample 12 \
  --min_branches_per_sample 2 \
  --rope_2d_ratio 0.5

# 推理
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 16 \
  --rope_2d_ratio 0.5 \
  --mode chat
```

### Qwen2.5-7B（大模型）

```bash
# 训练（需要大显存）
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_rank 32 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --batch_by_samples \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 2 \
  --rope_2d_ratio 0.5 \
  --dtype bfloat16

# 推理
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 32 \
  --rope_2d_ratio 0.5 \
  --mode chat \
  --dtype bfloat16
```

## 🔧 常见问题快速修复

### Q1: `RuntimeError: extra_pos2d is not set`

**原因**：使用了旧版推理代码或自己写的代码没有设置 pos2d

**解决**：
- ✅ 使用 `scripts/inference_hf_lora.py`（已自动处理）
- ✅ 或参考 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) 手动实现

### Q2: `size mismatch for ...`

**原因**：`lora_rank` 不一致

**解决**：确保推理时的 `--lora_rank` 与训练时完全一致

### Q3: 显存溢出 `CUDA out of memory`

**解决方案**（按顺序尝试）：
1. 减小 `--batch_size`（从 8 → 4 → 2 → 1）
2. 增加 `--accumulation_steps`（1 → 2 → 4）
3. 减小 `--max_branches_per_sample`（16 → 12 → 8）
4. 使用 `--dtype bfloat16`

### Q4: 生成结果质量差

**调整生成参数**：
```bash
# 事实问答（更确定）
--temperature 0.2 --top_p 0.8 --repetition_penalty 1.2

# 创意写作（更随机）
--temperature 0.8 --top_p 0.95 --repetition_penalty 1.0

# 代码生成（很确定）
--temperature 0.1 --top_p 0.8 --repetition_penalty 1.1
```

### Q5: 训练速度慢

**优化方案**：
- ✅ 使用 `--ddp` 多 GPU 训练
- ✅ 使用 `--dtype bfloat16` 混合精度
- ✅ 调整 `--num_workers`（通常 2-4）
- ✅ 确保数据在 SSD 上

## 📚 详细文档

- 🔍 [推理详细指南](INFERENCE_GUIDE.md)
- 📖 [训练使用文档](TRAIN_HF_LORA_USAGE.md)
- 🛠️ [pos2d 实现总结](POS2D_IMPLEMENTATION_SUMMARY.md)
- 📘 [主文档](../README.md)

## 🧪 测试验证

```bash
# pos2d 单元测试
python scripts/test_pos2d_handling.py

# 完整系统验证
python scripts/validate_inference_setup.py

# 快速推理测试
bash scripts/test_inference.sh Qwen/Qwen2-0.5B-Instruct out/lora/hf_lora_hf_final.pth 8
```

## 💡 最佳实践

### 1. 开发流程

```
1. 准备 Parallel 格式数据（main + branches）
   ↓
2. 小模型快速测试（Qwen2-0.5B, 1 epoch）
   ↓
3. 验证推理效果
   ↓
4. 扩大到生产模型（Qwen2.5-1.5B+, 3 epochs）
   ↓
5. 调优超参数（lora_rank, rope_2d_ratio, temperature）
```

### 2. 参数选择

| 场景 | lora_rank | rope_2d_ratio | batch_size | epochs |
|------|-----------|---------------|------------|--------|
| 快速实验 | 8 | 0.5 | 8 | 1 |
| 生产训练 | 16 | 0.5 | 4 | 3 |
| 大模型 | 32 | 0.5 | 1-2 | 3-5 |

### 3. 数据准备

```python
# Parallel 数据格式示例
{
  "main": "主分支内容...",
  "branches": [
    "分支1内容...",
    "分支2内容...",
    "分支3内容..."
  ]
}

# 或简化格式
{
  "text": "主分支内容..."
}
```

### 4. Python API 调用

```python
from scripts.inference_hf_lora import load_model_with_lora, generate_text

# 加载模型（自动处理 pos2d）
model, tokenizer, patch_rope = load_model_with_lora(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_path="out/lora/hf_lora_hf_final.pth",
    lora_rank=8,
    rope_2d_ratio=0.5,
)

# 生成（自动处理 pos2d）
response = generate_text(model, tokenizer, "你好")
print(response)
```

## 🎯 总结

- ✅ pos2d 已自动处理，开箱即用
- ✅ 支持 HuggingFace 所有 CausalLM 模型
- ✅ LoRA 高效微调，显存友好
- ✅ 2D RoPE 支持并行数据
- ✅ 完整的训练和推理流程
- ✅ 详细的文档和测试

有问题请查看 [详细文档](INFERENCE_GUIDE.md) 或提 Issue！
