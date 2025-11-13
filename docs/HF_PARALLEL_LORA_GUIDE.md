# HuggingFace 模型 + Parallel 数据 + LoRA 微调指南

## 功能说明

`train_hf_parallel_lora.py` 实现了以下功能：

1. ✅ **加载 HuggingFace 预训练模型**（支持任意 AutoModelForCausalLM 模型）
2. ✅ **修改 RoPE 为 2D RoPE**（支持 branch 和 time 的二维位置编码）
3. ✅ **添加 LoRA 层**（低秩微调，节省显存）
4. ✅ **使用 Parallel 数据格式**（支持多分支并行训练）
5. ✅ **支持分布式训练**（DDP）
6. ✅ **支持流式数据集**（HuggingFace streaming）

## 快速开始

### 1. 基本用法（本地数据）

```bash
python trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "dataset/sft_data_parallel.jsonl" \
    --lora_rank 8 \
    --rope_2d_ratio 0.5 \
    --batch_size 16 \
    --epochs 3 \
    --max_seq_len 512 \
    --branches_per_sample 8
```

### 2. 使用 HuggingFace 数据集

```bash
python trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --hf_dataset "HuggingFaceFW/fineweb-edu" \
    --hf_subset "sample-10BT" \
    --use_streaming \
    --lora_rank 8 \
    --batch_size 8 \
    --epochs 1
```

### 3. 分布式训练（多 GPU）

```bash
torchrun --nproc_per_node 4 trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "dataset/sft_data_parallel.jsonl" \
    --lora_rank 8 \
    --batch_size 8 \
    --ddp
```

## 数据格式要求

### Parallel 数据格式（JSONL）

每行一个 JSON 对象：

```json
{
    "main": "这是主分支的文本内容...",
    "branches": [
        "这是分支1的内容...",
        "这是分支2的内容...",
        "这是分支3的内容..."
    ]
}
```

或者简化格式（自动将 text 作为 main）：

```json
{
    "text": "这是主分支的文本内容..."
}
```

## 关键参数说明

### 模型相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | HuggingFace 模型名称或本地路径 | **必填** |
| `--tokenizer_path` | Tokenizer 路径（不填则与 base_model 一致） | None |
| `--lora_rank` | LoRA 的秩（rank），控制参数量 | 8 |
| `--load_lora` | 加载已有 LoRA 权重继续训练 | None |

### RoPE 2D 相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--patch_rope` | 是否应用 2D RoPE | True |
| `--no_patch_rope` | 禁用 2D RoPE | - |
| `--rope_2d_ratio` | Branch 维度使用的频率对比例（0.0-1.0） | 0.5 |
| `--stride` | Branch 位置的步长 | 128 |

### 数据相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 本地 JSONL 文件路径 | None |
| `--hf_dataset` | HuggingFace 数据集名称 | None |
| `--hf_subset` | HuggingFace 数据集子集 | None |
| `--use_streaming` | 使用流式数据集 | False |
| `--max_seq_len` | 单个 branch 的最大长度 | 512 |
| `--branches_per_sample` | 固定每个样本的 branch 数量 | 8 |
| `--max_branches_per_sample` | 动态 branch 模式（0=固定模式） | 0 |

### 训练相关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 1 |
| `--batch_size` | 批次大小 | 16 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--grad_clip` | 梯度裁剪阈值 | 1.0 |
| `--dtype` | 训练精度（float32/float16/bfloat16） | bfloat16 |

### 其他

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out_dir` | 输出目录 | out |
| `--log_interval` | 日志打印间隔 | 50 |
| `--save_interval` | 模型保存间隔 | 500 |
| `--use_wandb` | 使用 Weights & Biases 记录 | False |

## 工作流程

### 1. 模型加载流程

```
HuggingFace 预训练模型
    ↓
应用 2D RoPE 修改
    ↓
添加 LoRA 层
    ↓
冻结非 LoRA 参数
    ↓
开始训练
```

### 2. 数据处理流程

```
Parallel 数据（main + branches）
    ↓
ParallelPretrainDataset 加载
    ↓
ParallelPretrainCollator 处理
    ↓
生成 pos2d（branch_id, time_id）
    ↓
构建 columnar_causal_mask
    ↓
送入模型训练
```

### 3. LoRA 权重保存

训练过程中会保存：
- 每隔 `save_interval` 步保存一次：`out/lora/{lora_name}_step{N}.pth`
- 训练结束保存最终版本：`out/lora/{lora_name}_final.pth`

只保存 LoRA 层的权重，不保存整个模型。

## 推理使用

训练完成后，可以这样加载模型进行推理：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_lora import load_lora
from parallel.columnar import patch_model_with_interleaved_2d_rope

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 应用 2D RoPE
from trainer.train_hf_parallel_lora import auto_pair_indices
pair_indices = auto_pair_indices(model, ratio=0.5)
patch_model_with_interleaved_2d_rope(model, pair_indices)

# 加载 LoRA 权重
from model.model_lora import apply_lora
apply_lora(model, rank=8)
load_lora(model, "out/lora/hf_parallel_lora_final.pth")

# 推理
model.eval()
# ... 进行推理
```

## 常见问题

### Q1: 显存不足怎么办？

1. 减小 `--batch_size`
2. 增加 `--accumulation_steps` 来保持有效 batch size
3. 减小 `--max_seq_len` 或 `--branches_per_sample`
4. 使用 `--dtype float16` 或 `bfloat16`

### Q2: 如何选择 LoRA rank？

- Rank 越大，参数越多，效果可能更好但训练更慢
- 建议：小模型（< 1B）用 rank=4-8，大模型用 rank=16-32

### Q3: rope_2d_ratio 应该设多少？

- 0.5 是一个平衡的选择（一半频率对用于 branch，一半用于 time）
- 如果 branch 数量较多，可以适当提高（如 0.6-0.7）
- 如果序列较长，可以适当降低（如 0.3-0.4）

### Q4: 如何从 SFT 数据转换为 Parallel 数据？

使用现有的转换脚本：

```bash
python scripts/convert_sft_to_pretrain.py \
    --input dataset/sft_data.jsonl \
    --output dataset/sft_data_parallel.jsonl
```

## 与其他训练脚本的对比

| 脚本 | 模型来源 | 数据格式 | LoRA | 用途 |
|------|---------|---------|------|------|
| `train_pretrain.py` | 从头训练 | Parallel | ❌ | 预训练 |
| `train_hf_lora.py` | HuggingFace | SFT | ✅ | 对话微调 |
| **`train_hf_parallel_lora.py`** | **HuggingFace** | **Parallel** | **✅** | **并行微调** |

## 技术细节

### 2D RoPE 实现

- 将部分频率对分配给 branch 维度，其余分配给 time 维度
- Branch 位置 = branch_id × stride（默认 128）
- Time 位置 = token 在 branch 内的序号
- 通过 `Interleaved2DRoPE` 模块实现交错编码

### Columnar Causal Mask

- 只允许同一 branch 内的 token 相互注意
- 遵循因果关系（未来 token 看不到过去）
- 不同 branch 之间完全隔离

### LoRA 实现

- 在所有 Linear 层添加低秩分解：`W + BA`
- 只训练 A 和 B 矩阵，原始权重冻结
- 典型的参数量减少 99% 以上

## 性能优化建议

1. **使用混合精度训练**：`--dtype bfloat16`
2. **启用 Flash Attention**（模型自动检测）
3. **合理设置 batch size 和梯度累积**
4. **使用分布式训练**：多 GPU 加速
5. **流式加载大数据集**：`--use_streaming`

## 示例配置

### 小模型（0.5B）快速测试

```bash
python trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "dataset/test_parallel.jsonl" \
    --lora_rank 4 \
    --batch_size 32 \
    --epochs 1 \
    --max_seq_len 256 \
    --branches_per_sample 4
```

### 中等模型（1.5B）生产训练

```bash
torchrun --nproc_per_node 4 trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data_path "dataset/full_parallel.jsonl" \
    --lora_rank 16 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --epochs 3 \
    --max_seq_len 512 \
    --branches_per_sample 8 \
    --use_wandb \
    --ddp
```

### 大模型（7B+）内存优化

```bash
torchrun --nproc_per_node 8 trainer/train_hf_parallel_lora.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --hf_dataset "your-dataset" \
    --use_streaming \
    --lora_rank 32 \
    --batch_size 2 \
    --accumulation_steps 16 \
    --dtype bfloat16 \
    --max_seq_len 512 \
    --branches_per_sample 8 \
    --ddp
```
