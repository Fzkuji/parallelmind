# HuggingFace + LoRA 推理统一指南

## 概述

本指南提供 ParallelMind 项目的推理脚本使用说明。我们有三个推理脚本，针对不同场景优化。

## 快速决策

**我该用哪个脚本？**

```
你想做什么？
│
├─ 测试单个问题 / 交互式对话
│  └─ 使用：inference_hf_lora.py
│
├─ 推理几个到几百个问题
│  ├─ 直接输入问题
│  │  └─ 使用：parallel_generate.py --prompts "问题1" "问题2" ...
│  ├─ 从文本文件读取
│  │  └─ 使用：parallel_generate.py --prompts_file questions.txt
│  └─ 从 JSONL 读取
│     └─ 使用：parallel_generate.py --data_path dataset/test.jsonl
│
└─ 大规模推理（10000+ 样本）
   └─ 使用：parallel_inference_hf_lora.py --data_path dataset/large.jsonl
```

---

## 推荐：`parallel_generate.py`（主要推理脚本）

### 适用场景

- ✅ 直接输入问题列表（命令行或文本文件）
- ✅ 中小规模推理（1-1000 个问题）
- ✅ 需要看到生成过程（streaming）
- ✅ 灵活的输入方式
- ✅ 快速测试和调试

### 使用方式 1：直接输入问题（最常用）

```bash
# 单 GPU
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --prompts "介绍一下人工智能" "讲解深度学习" "什么是强化学习"

# 多 GPU
torchrun --nproc_per_node 8 scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --prompts "介绍一下人工智能" "讲解深度学习" "什么是强化学习"
```

### 使用方式 2：从文本文件读取

创建 `questions.txt`：
```text
介绍一下人工智能
讲解深度学习
什么是强化学习
自然语言处理的应用
计算机视觉技术
```

运行推理：
```bash
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --prompts_file questions.txt
```

### 使用方式 3：从 JSONL 读取

对于 Parallel 格式数据：
```bash
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --data_path dataset/test.jsonl \
  --max_branches_per_sample 8 \
  --batch_by_samples \
  --batch_size 16
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hf_base_model` | HuggingFace 模型路径 | - |
| `--lora_path` | LoRA 权重路径 | - |
| `--lora_rank` | LoRA rank | 8 |
| `--rope_2d_ratio` | 2D RoPE 比例 | 0.5 |
| `--prompts` | 直接输入问题（空格分隔） | - |
| `--prompts_file` | 问题文本文件 | - |
| `--data_path` | JSONL 数据文件 | - |
| `--max_new_tokens` | 最大生成长度 | 512 |
| `--temperature` | 温度 | 0.7 |
| `--stream` | 流式输出 | True |
| `--no_patch_rope` | 禁用 2D RoPE | False |

---

## 单条推理：`inference_hf_lora.py`

### 适用场景

- ✅ 测试单个问题
- ✅ 交互式对话
- ✅ 快速验证模型效果

### 基本使用

```bash
# 单条推理
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --prompt "介绍一下人工智能"

# 交互式对话
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --interactive
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | HuggingFace 模型路径 | - |
| `--lora_path` | LoRA 权重路径（可选） | None |
| `--lora_rank` | LoRA rank | 8 |
| `--rope_2d_ratio` | 2D RoPE 比例 | 0.5 |
| `--prompt` | 输入问题 | - |
| `--interactive` | 交互模式 | False |
| `--max_new_tokens` | 最大生成长度 | 512 |
| `--no_patch_rope` | 禁用 2D RoPE | False |

详细文档：[docs/INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

---

## 大规模推理：`parallel_inference_hf_lora.py`（备选）

### 适用场景

- ✅ 超大规模推理（10000+ 样本）
- ✅ 需要最高吞吐量
- ✅ 已有 JSONL 数据集

### 限制

- ❌ 仅支持 JSONL 文件输入
- ❌ 不支持直接命令行输入问题
- ❌ 不支持实时输出

### 基本使用

```bash
# 单 GPU
python scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --data_path dataset/test_10k.jsonl \
  --out_path out/results_10k.jsonl \
  --batch_size 16 \
  --batch_by_samples

# 多 GPU（推荐）
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --data_path dataset/test_10k.jsonl \
  --out_path out/results_10k.jsonl \
  --batch_size 16 \
  --batch_by_samples \
  --max_branches_per_sample 12
```

详细文档：[docs/PARALLEL_INFERENCE_GUIDE.md](PARALLEL_INFERENCE_GUIDE.md)

---

## 脚本对比

| 特性 | parallel_generate.py | parallel_inference_hf_lora.py | inference_hf_lora.py |
|------|---------------------|------------------------------|---------------------|
| **推荐度** | ⭐⭐⭐⭐⭐ 主推 | ⭐⭐⭐ 备选 | ⭐⭐⭐⭐ 单用 |
| **输入方式** | 命令行/文本/JSONL | 仅 JSONL | 命令行/交互 |
| **多 GPU** | ✅ | ✅ | ❌ |
| **批量推理** | ✅ | ✅ | ❌ |
| **实时输出** | ✅ | ❌ | ✅ |
| **吞吐量** | 高 | 最高 | - |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 常见使用场景

### 场景 1：测试训练好的模型

```bash
# 单条测试
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompt "介绍一下人工智能"
```

### 场景 2：批量生成回复（10-100 个问题）

```bash
# 创建 questions.txt 包含你的问题
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompts_file questions.txt
```

### 场景 3：评估模型（1000+ 测试样本）

```bash
# 使用 JSONL 测试集
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --data_path dataset/test_1k.jsonl \
  --batch_size 16
```

### 场景 4：大规模生成（10000+ 样本）

```bash
# 多 GPU 高吞吐量
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --data_path dataset/test_10k.jsonl \
  --out_path out/results_10k.jsonl \
  --batch_size 8 \
  --batch_by_samples
```

---

## 性能优化建议

### 1. 选择合适的脚本

- **< 100 问题** → `parallel_generate.py`（直接输入或文本文件）
- **100-1000 问题** → `parallel_generate.py`（JSONL）
- **> 10000 问题** → `parallel_inference_hf_lora.py`

### 2. 批量大小调整

```bash
# 显存充足
--batch_size 32

# 显存不足
--batch_size 4
```

### 3. 多 GPU 加速

```bash
# 使用所有可用 GPU
torchrun --nproc_per_node $(nvidia-smi -L | wc -l) scripts/parallel_generate.py ...
```

### 4. 动态 branch 模式

```bash
# 更高效的显存利用
--max_branches_per_sample 16 \
--min_branches_per_sample 1 \
--batch_by_samples
```

---

## 常见问题

### Q1: 我只想输入几个问题，该用哪个？

**A**: 使用 `parallel_generate.py` + `--prompts`

```bash
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompts "问题1" "问题2" "问题3"
```

### Q2: 我有一个包含很多问题的文本文件，该用哪个？

**A**: 使用 `parallel_generate.py` + `--prompts_file`

```bash
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompts_file my_questions.txt
```

### Q3: 我有 10000+ 样本的 JSONL 数据集，该用哪个？

**A**: 使用 `parallel_inference_hf_lora.py`

```bash
torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --data_path dataset/large.jsonl \
  --out_path out/results.jsonl
```

### Q4: 可以不用 LoRA 吗？

**A**: 可以，不指定 `--lora_path` 即可使用纯基础模型。

### Q5: 如何禁用 2D RoPE？

**A**: 添加 `--no_patch_rope` 参数。

---

## 相关文档

- 📚 [单条推理详细指南](INFERENCE_GUIDE.md)
- 📚 [大规模推理指南](PARALLEL_INFERENCE_GUIDE.md)
- 📚 [Claude 与 GPT 共识](CLAUDE_GPT_CONSENSUS.md)
- 📚 [训练使用文档](TRAIN_HF_LORA_USAGE.md)
- 📚 [快速开始](QUICK_START_LORA.md)
- 📚 [pos2d 技术细节](POS2D_IMPLEMENTATION_SUMMARY.md)

---

## 总结

**默认推荐**：使用 `parallel_generate.py`

- ✅ 支持多种输入方式（命令行、文本文件、JSONL）
- ✅ 实时看到生成过程
- ✅ 支持多 GPU
- ✅ 最灵活易用

**适合 90% 的使用场景！**
