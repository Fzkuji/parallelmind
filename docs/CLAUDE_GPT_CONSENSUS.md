# Claude 与 GPT 的推理脚本共识

## 背景

在为 ParallelMind 项目添加 HuggingFace + LoRA 推理功能时，Claude 和 GPT 分别实现了不同的推理脚本。经过讨论，我们达成了以下共识。

## 主推荐方案

### 使用 `scripts/parallel_generate.py` 作为主要推理脚本

**原因**：
1. ✅ GPT 已完整实现 HF + LoRA 支持
2. ✅ 支持多种输入方式（命令行、文本文件、JSONL）
3. ✅ 满足用户需求：直接输入问题，不需要准备数据集
4. ✅ 功能完整：streaming、debug、多 GPU DDP
5. ✅ 代码已测试验证

**适用场景**：
- 直接输入几个问题进行推理
- 从文本文件读取问题列表
- 中小规模批量推理（1-1000 样本）
- 需要看到生成过程
- 测试和调试

## 备选方案

### `scripts/parallel_inference_hf_lora.py` (Claude 实现)

**保留原因**：
- 使用 DataLoader + DistributedSampler 架构
- 更适合超大规模推理（10000+ 样本）
- 更高的吞吐量优化

**适用场景**：
- 大规模 JSONL 数据集推理
- 需要最高吞吐量
- 不需要实时输出

**限制**：
- 仅支持 JSONL 文件输入
- 不支持直接命令行输入问题

## 单条推理

### `scripts/inference_hf_lora.py` (Claude 实现)

**保留原因**：
- 交互式对话功能
- 单条问题快速测试
- 自动 pos2d 处理

**适用场景**：
- 单条问题测试
- 交互式对话
- 模型快速验证

## 使用指南

### 推荐流程

1. **测试/调试阶段** → 使用 `inference_hf_lora.py`
   ```bash
   python scripts/inference_hf_lora.py \
     --base_model Qwen/Qwen2.5-14B-Instruct \
     --lora_path out/lora/qwen2_lora_final.pth \
     --prompt "你的问题"
   ```

2. **日常推理（几个到几百个问题）** → 使用 `parallel_generate.py`
   ```bash
   # 方式1：直接输入问题
   python scripts/parallel_generate.py \
     --hf_base_model Qwen/Qwen2.5-14B-Instruct \
     --lora_path out/lora/qwen2_lora_final.pth \
     --prompts "问题1" "问题2" "问题3"

   # 方式2：从文本文件读取
   python scripts/parallel_generate.py \
     --hf_base_model Qwen/Qwen2.5-14B-Instruct \
     --lora_path out/lora/qwen2_lora_final.pth \
     --prompts_file questions.txt

   # 方式3：从 JSONL 读取
   python scripts/parallel_generate.py \
     --hf_base_model Qwen/Qwen2.5-14B-Instruct \
     --lora_path out/lora/qwen2_lora_final.pth \
     --data_path dataset/test.jsonl
   ```

3. **大规模批量推理（10000+ 样本）** → 使用 `parallel_inference_hf_lora.py`
   ```bash
   torchrun --nproc_per_node 8 scripts/parallel_inference_hf_lora.py \
     --base_model Qwen/Qwen2.5-14B-Instruct \
     --lora_path out/lora/qwen2_lora_final.pth \
     --data_path dataset/test_10k.jsonl \
     --out_path out/results_10k.jsonl \
     --batch_size 8 \
     --batch_by_samples
   ```

## 技术对比

| 特性 | parallel_generate.py | parallel_inference_hf_lora.py | inference_hf_lora.py |
|------|---------------------|------------------------------|---------------------|
| **实现者** | GPT | Claude | Claude |
| **推荐度** | ⭐⭐⭐⭐⭐ 主推 | ⭐⭐⭐ 备选 | ⭐⭐⭐⭐ 单用 |
| **输入方式** | 命令行/文本/JSONL | 仅 JSONL | 命令行/交互 |
| **多 GPU** | ✅ DDP | ✅ DDP | ❌ |
| **批量推理** | ✅ | ✅ | ❌ |
| **实时输出** | ✅ Streaming | ❌ | ✅ |
| **吞吐量** | 高 | 最高 | - |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

## 核心共识

1. **默认推荐 `parallel_generate.py`** - 满足 90% 的使用场景
2. **保留 `parallel_inference_hf_lora.py`** - 用于极限性能场景
3. **保留 `inference_hf_lora.py`** - 用于单条测试和交互
4. **统一文档** - 明确说明各脚本的适用场景
5. **更新 README** - 以 `parallel_generate.py` 为主要示例

## 参数统一

所有脚本使用一致的参数命名：

- `--base_model` / `--hf_base_model` - HuggingFace 模型路径
- `--lora_path` - LoRA 权重路径
- `--lora_rank` - LoRA rank（默认 8）
- `--rope_2d_ratio` - 2D RoPE 比例（默认 0.5）
- `--patch_rope` / `--no_patch_rope` - 是否应用 2D RoPE

## 文档更新

- ✅ 创建本共识文档
- ✅ 更新 README.md 推荐 `parallel_generate.py`
- ✅ 保留各脚本的详细文档
- ✅ 添加使用场景决策树

## 总结

**用户问："如何进行推理？"**

**回答**：

1. **几个问题** → `parallel_generate.py` + `--prompts`
2. **一个问题** → `inference_hf_lora.py` + `--prompt`
3. **大量问题（10000+）** → `parallel_inference_hf_lora.py` + JSONL

**默认推荐**：使用 `parallel_generate.py`，因为它最灵活、最易用。
