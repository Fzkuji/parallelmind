# train_hf_lora.py 使用指南

## 功能说明

`train_hf_lora.py` 用于在 HuggingFace 预训练模型上进行 **Parallel 数据微调**，使用 **2D RoPE** 和 **LoRA** 技术。

## 快速开始

### 基本用法

```bash
python trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "dataset/parallel_data.jsonl" \
    --lora_rank 8 \
    --branches_per_sample 8 \
    --batch_size 16 \
    --epochs 3
```

### 使用动态 branch 模式

```bash
python trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data_path "dataset/parallel_data.jsonl" \
    --lora_rank 16 \
    --max_branches_per_sample 16 \
    --min_branches_per_sample 4 \
    --batch_size 8 \
    --epochs 3
```

### 多 GPU 分布式训练

```bash
torchrun --nproc_per_node 4 trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data_path "dataset/parallel_data.jsonl" \
    --lora_rank 16 \
    --batch_size 8 \
    --ddp
```

## 数据格式

### Parallel 数据格式 (JSONL)

每行一个 JSON 对象：

```json
{
    "main": "主分支文本内容...",
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
    "text": "主分支文本内容..."
}
```

## 核心参数

### 必填参数

| 参数 | 说明 |
|------|------|
| `--base_model` | HuggingFace 模型名称或本地路径 |

### 数据相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | dataset/lora_medical.jsonl | Parallel 数据文件路径 |
| `--max_seq_len` | 1024 | 单个 branch 的最大长度 |
| `--branches_per_sample` | 8 | 固定 branch 数量 |
| `--max_branches_per_sample` | None | 动态模式：最大 branch 数 |
| `--min_branches_per_sample` | 1 | 动态模式：最小 branch 数 |
| `--max_parallel_samples` | None | 限制数据集样本数量 |

### LoRA 相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_rank` | 8 | LoRA 的秩 |
| `--lora_name` | hf_lora | LoRA 权重文件名 |
| `--load_lora` | None | 继续训练已有 LoRA |

### 2D RoPE 相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patch_rope` | True | 应用 2D RoPE |
| `--no_patch_rope` | - | 禁用 2D RoPE |
| `--rope_2d_ratio` | 0.5 | Branch 维度频率比例 |
| `--branch_stride` | 128 | Branch 位置步长 |

### 训练相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--accumulation_steps` | 1 | 梯度累积步数 |
| `--dtype` | bfloat16 | 训练精度 |
| `--grad_clip` | 1.0 | 梯度裁剪 |

### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out_dir` | out | 输出目录 |
| `--log_interval` | 50 | 日志打印间隔 |
| `--save_interval` | 500 | 模型保存间隔 |
| `--use_wandb` | False | 使用 W&B 记录 |
| `--ddp` | False | 使用分布式训练 |

## 工作流程

```
加载 HF 模型
    ↓
应用 2D RoPE
    ↓
添加 LoRA 层
    ↓
冻结非 LoRA 参数
    ↓
加载 Parallel 数据
    ↓
训练
    ↓
保存 LoRA 权重
```

## 输出文件

- **训练中保存**：`out/lora/{lora_name}_{model_tag}.pth`（每 `save_interval` 步）
- **训练结束**：`out/lora/{lora_name}_{model_tag}_final.pth`

## 推理使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_lora import apply_lora, load_lora
from parallel.columnar import patch_model_with_interleaved_2d_rope

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 应用 2D RoPE（使用与训练时相同的 ratio）
from trainer.train_hf_lora import auto_pair_indices
pair_indices = auto_pair_indices(model, ratio=0.5)
patch_model_with_interleaved_2d_rope(model, pair_indices)

# 加载 LoRA（必须先 apply_lora）
apply_lora(model, rank=8)
load_lora(model, "out/lora/hf_lora_hf_final.pth")

# 推理
model.eval()
# ... 进行推理
```

## 常见配置示例

### 小模型快速测试

```bash
python trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "dataset/test.jsonl" \
    --lora_rank 4 \
    --branches_per_sample 4 \
    --batch_size 32 \
    --epochs 1 \
    --max_seq_len 256
```

### 中等模型生产训练

```bash
torchrun --nproc_per_node 4 trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data_path "dataset/full_data.jsonl" \
    --lora_rank 16 \
    --branches_per_sample 8 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --epochs 3 \
    --use_wandb \
    --ddp
```

### 大模型内存优化

```bash
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "dataset/full_data.jsonl" \
    --lora_rank 32 \
    --max_branches_per_sample 12 \
    --min_branches_per_sample 4 \
    --batch_size 2 \
    --accumulation_steps 16 \
    --dtype bfloat16 \
    --ddp
```

## 性能优化建议

1. **显存不足**：
   - 减小 `--batch_size`
   - 增加 `--accumulation_steps`
   - 减小 `--max_seq_len` 或 `--branches_per_sample`
   - 使用 `--dtype bfloat16`

2. **训练速度**：
   - 使用 `--ddp` 多 GPU 训练
   - 使用混合精度 `--dtype bfloat16`
   - 合理设置 `--num_workers`

3. **LoRA rank 选择**：
   - 小模型（< 1B）：rank=4-8
   - 中等模型（1-3B）：rank=8-16
   - 大模型（> 3B）：rank=16-32

4. **rope_2d_ratio 选择**：
   - 默认 0.5 是平衡选择
   - Branch 多、序列短：提高到 0.6-0.7
   - Branch 少、序列长：降低到 0.3-0.4

## 与其他脚本的区别

| 脚本 | 模型来源 | 数据格式 | LoRA | 用途 |
|------|---------|---------|------|------|
| train_pretrain.py | 从头训练 | Parallel | ❌ | 预训练 |
| **train_hf_lora.py** | **HuggingFace** | **Parallel** | **✅** | **并行微调** |
| train_hf_parallel_lora.py | HuggingFace | Parallel | ✅ | 备选版本 |

## 注意事项

1. **数据格式**：必须使用 Parallel 格式（含 main/branches 或 text 字段）
2. **模型兼容性**：支持所有 `AutoModelForCausalLM` 模型
3. **RoPE 修改**：会自动检测并修改模型的 RoPE 为 2D 版本
4. **LoRA 权重**：只保存 LoRA 层权重，推理时需要重新应用
5. **继续训练**：使用 `--load_lora` 加载已有权重继续训练

## 故障排查

### 错误：找不到 rotary_emb

如果遇到 "未找到 rotary_emb.inv_freq" 错误，说明该模型不支持自动 2D RoPE 修改。使用 `--no_patch_rope` 禁用。

### 显存溢出

按以下顺序尝试：
1. 减小 batch_size
2. 减小 max_seq_len 或 branches_per_sample
3. 增加 accumulation_steps
4. 使用动态 branch 模式降低平均长度

### 训练速度慢

检查：
1. 是否使用了 bfloat16
2. 是否启用了多 GPU（--ddp）
3. num_workers 是否设置合理
4. 数据集是否在 SSD 上
