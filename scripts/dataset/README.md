# Dataset Processing Scripts

## FineWeb-Edu 10BT 数据处理

### 功能特性

`process_fineweb_edu_10bt.py` 脚本提供以下功能：

1. **去重**
   - 使用 MD5 hash 检测并移除完全相同的文本
   - 内存高效的 hash set 实现

2. **质量过滤**
   - 移除乱码文本（非 ASCII 字符占比过高）
   - 移除包含过多重复句子的文本
   - 移除数字或特殊字符占比过高的文本
   - 移除过短的文本（少于50字符）

3. **智能切分**
   - 按句子边界切分（使用 `.!?` 等标点）
   - 使用 tokenizer 精确控制每段 token 数（默认512 tokens）
   - 避免切断句子
   - 对超长句子按逗号进一步切分

4. **格式化输出**
   - JSON Lines 格式（每行一个 JSON 对象）
   - 每段文本包含 `<|im_start|>` 和 `<|im_end|>` 标记
   - 与现有本地数据格式兼容

### 使用方法

#### 基本用法

```bash
# 处理全部 10BT 数据（默认设置）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2

# 测试处理（10万样本）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-samples 100000 \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2

# 使用离线模式（数据已缓存）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-samples 100000 \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2 \
    --offline
```

#### 参数说明

- `--dataset`: HuggingFace 数据集名称（默认：`HuggingFaceFW/fineweb-edu`）
- `--subset`: 数据集子集（默认：`sample-10BT`）
- `--split`: 数据集分割（默认：`train`）
- `--max-samples`: 最大处理样本数（默认：`None`，处理全部 10BT 数据）
- `--max-tokens-per-chunk`: 每个 chunk 最大 token 数（默认：512）
- `--tokenizer`: 用于计算 token 长度的 tokenizer（默认：`gpt2`）
- `--output-dir`: 输出目录（默认：`dataset/fineweb-edu-10BT`）
- `--offline`: 离线模式，使用已缓存的数据集

#### 自定义配置

```bash
# 使用更大的 chunk size (1024 tokens)
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-tokens-per-chunk 1024 \
    --tokenizer gpt2 \
    --max-samples 100000

# 使用不同的 tokenizer (例如 Qwen)
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --max-samples 50000

# 使用不同的子集
python scripts/dataset/process_fineweb_edu_10bt.py \
    --subset CC-MAIN-2024-10 \
    --tokenizer gpt2 \
    --max-samples 50000
```

### 输出格式

输出文件：`dataset/fineweb-edu-10BT/train.jsonl`

每行格式：
```json
{"text": "<|im_start|>This is a sample text with proper sentences. It demonstrates the format of the processed data.<|im_end|>"}
```

### 处理统计

脚本运行结束后会显示统计信息：

```
============================================================
Processing completed!
============================================================
Total samples processed: 100,000
Duplicates removed: 1,234
Garbage texts removed: 567
Repetitive texts removed: 890
Chunks created: 234,567
Output file: dataset/fineweb-edu-10BT/train.jsonl
============================================================
```

### 质量过滤规则

#### 乱码检测
- 非 ASCII 可打印字符占比 < 70%
- 数字占比 > 30%
- 特殊字符占比 > 20%
- 文本长度 < 50 字符

#### 重复句子检测
- 相同句子重复次数 > 30%
- 连续3个词的短语重复次数 > 10%

这些阈值可以在脚本中调整。

### Token 长度测试

使用 `test_token_length.py` 脚本可以测试不同 tokenizer 的单词/token 比例：

```bash
# 测试 gpt2 tokenizer
python scripts/dataset/test_token_length.py \
    --tokenizer gpt2 \
    --num-samples 1000

# 测试 Qwen tokenizer
python scripts/dataset/test_token_length.py \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --num-samples 1000
```

测试结果（GPT-2）：
- 平均 Token/Word 比例: **1.340**
- 382 单词 ≈ 512 tokens
- 400 单词 ≈ 536 tokens

### 注意事项

1. **内存使用**：去重功能使用内存中的 hash set，处理大规模数据时需要足够的 RAM
2. **处理速度**：约 1000-2000 samples/秒（取决于机器性能和 tokenizer）
3. **磁盘空间**：输出文件大小约为原始数据的 60-80%（经过去重和过滤）
4. **Token 精确控制**：使用 tokenizer 实时计算，保证每个 chunk 不超过指定 token 数

### 数据使用

处理完成后，可以使用以下方式训练：

```bash
# 使用本地数据训练
python trainer/train_pretrain.py \
    --data-path dataset/fineweb-edu-10BT/train.jsonl \
    --tokenizer gpt2 \
    --max-samples 200000
```

### 扩展

如需修改质量过滤规则，编辑 `process_fineweb_edu_10bt.py` 中的以下函数：
- `is_garbage_text()`: 修改乱码检测规则
- `has_repetitive_sentences()`: 修改重复检测阈值
- `create_chunks()`: 修改切分策略和 token 控制逻辑

如需使用不同的 tokenizer：
```bash
# 使用 Qwen tokenizer
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --max-tokens-per-chunk 512

# 使用 LLaMA tokenizer
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --max-tokens-per-chunk 512
```
