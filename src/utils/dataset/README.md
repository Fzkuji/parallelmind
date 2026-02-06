# Dataset Processing Scripts

## FineWeb-Edu 10BT 数据处理

### 功能特性

`process_fineweb_edu_10bt.py` 脚本提供以下功能：

1. **智能去重**
   - **精确去重**：使用 MD5 hash 检测并移除完全相同的文本
   - **近似去重**：使用 MinHash 算法检测并移除高度相似的文本
     - 基于 3-gram 特征和 Jaccard 相似度估算
     - 可配置相似度阈值（默认 0.85 = 85%相似即视为重复）
     - 可选择只使用精确去重或同时启用近似去重

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
# 处理全部 10BT 数据，使用96核并行处理（推荐）
# 默认启用近似去重（85%相似度阈值）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2 \
    --num-workers 96

# 测试处理（10万样本），使用精确去重
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-samples 100000 \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2 \
    --num-workers 96 \
    --dedup-threshold 1.0

# 使用离线模式（数据已缓存）+ 近似去重（90%相似度阈值）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-samples 100000 \
    --output-dir dataset/fineweb-edu-10BT \
    --tokenizer gpt2 \
    --num-workers 96 \
    --offline \
    --dedup-threshold 0.90
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
- `--num-workers`: 并行处理的进程数（默认：1，建议：96 for 128-core server）
- `--batch-size`: 批处理大小（默认：1000）
- `--dedup-threshold`: 近似去重的相似度阈值（默认：0.85）
  - 设置为 `1.0`：只使用精确去重（MD5 hash）
  - 设置为 `0.85`：相似度 ≥85% 的文本会被认为是重复
  - 设置为 `0.90`：更严格的去重（只移除 ≥90% 相似的文本）
- `--num-perm`: MinHash 签名的 hash 函数数量（默认：128）
  - 越大越精确，但计算和存储开销越大
  - 推荐值：64-256

#### 自定义配置

```bash
# 使用更大的 chunk size (1024 tokens) + 并行处理
python scripts/dataset/process_fineweb_edu_10bt.py \
    --max-tokens-per-chunk 1024 \
    --tokenizer gpt2 \
    --num-workers 96 \
    --max-samples 100000

# 使用不同的 tokenizer (例如 Qwen) + 并行处理
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --num-workers 96 \
    --max-samples 50000

# 使用不同的子集 + 并行处理
python scripts/dataset/process_fineweb_edu_10bt.py \
    --subset CC-MAIN-2024-10 \
    --tokenizer gpt2 \
    --num-workers 96 \
    --max-samples 50000

# 调整批处理大小（处理更多样本时）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer gpt2 \
    --num-workers 96 \
    --batch-size 2000

# 使用更严格的近似去重（90%相似度阈值）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer gpt2 \
    --num-workers 96 \
    --dedup-threshold 0.90 \
    --max-samples 100000

# 只使用精确去重（关闭近似去重）
python scripts/dataset/process_fineweb_edu_10bt.py \
    --tokenizer gpt2 \
    --num-workers 96 \
    --dedup-threshold 1.0 \
    --max-samples 100000
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
Exact duplicates removed: 1,234
Near-duplicates removed (>=85% similar): 3,456
Garbage texts removed: 567
Repetitive texts removed: 890
Chunks created: 234,567
Output file: dataset/fineweb-edu-10BT/train.jsonl
============================================================
```

注：如果使用 `--dedup-threshold 1.0`（只精确去重），则不会显示 "Near-duplicates removed" 行。

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

### 去重策略

脚本提供两种去重方式，可同时启用或单独使用：

#### 精确去重（MD5 Hash）
- 使用 MD5 hash 检测完全相同的文本
- 即使有一个字符不同，也会被认为是不同的文本
- 速度快，内存占用小
- 适合移除完全重复的内容

#### 近似去重（MinHash + Jaccard Similarity）
- 使用 MinHash 算法估算文本的 Jaccard 相似度
- 基于 3-gram 字符特征（适合多语言文本）
- 可配置相似度阈值（默认 0.85）
- 能够检测到稍有差异的近似重复文本
- 适合移除高度相似但不完全相同的内容（如：格式化不同、标点符号差异、轻微编辑等）

#### 性能影响
- **精确去重**：几乎无性能影响
- **近似去重**：每个文本增加约 10-20% 的计算时间
  - MinHash 签名计算：每个文本需要计算 128 个 hash 值（可配置）
  - 相似度比较：需要与已处理的文本进行比较（O(n) 复杂度）
  - 内存占用：每个文本额外存储 128 个整数（约 512 bytes）

#### 建议配置
- **快速处理**：`--dedup-threshold 1.0`（只精确去重）
- **标准去重**：`--dedup-threshold 0.85`（默认，移除 85% 以上相似的文本）
- **严格去重**：`--dedup-threshold 0.90`（只移除非常相似的文本）
- **最大去重**：`--dedup-threshold 0.75`（更激进，可能误删略有差异的文本）

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

### 并行处理性能

使用多进程并行处理可以显著提升速度：

| 配置 | 处理速度（samples/秒） | 相对提升 |
|------|----------------------|----------|
| 单进程 (`--num-workers 1`) | ~1,000-2,000 | 基线 |
| 96核并行 (`--num-workers 96`) | ~50,000-100,000 | 50-100x |

**推荐配置（128核服务器）**：
- `--num-workers 96`：使用96个核心并行处理
- `--batch-size 1000`：每批处理1000个样本（可根据内存调整）

### 注意事项

1. **内存使用**：
   - 去重功能使用内存中的 hash set，处理大规模数据时需要足够的 RAM
   - 并行处理会增加内存占用（每个进程加载tokenizer）
   - 建议：96核 + 1000 batch_size 约需 32-64GB RAM

2. **处理速度**：
   - 单进程：约 1000-2000 samples/秒
   - 96核并行：约 50,000-100,000 samples/秒
   - 速度取决于机器性能、tokenizer 和数据质量

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
