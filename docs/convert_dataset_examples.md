# 数据集转换脚本使用示例

## 快速开始

### 1. Pretrain数据准备

#### 纯文本文件
最简单的格式，每行一个样本：
```bash
# 输入文件: data.txt
# 每行一段文本
人工智能是计算机科学的一个分支。
深度学习是机器学习的一个子领域。
神经网络可以学习复杂的模式。

# 转换命令
python scripts/convert_dataset.py \
  --input data.txt \
  --output dataset/my_pretrain.jsonl \
  --mode pretrain

# 输出: dataset/my_pretrain.jsonl
# {"text": "人工智能是计算机科学的一个分支。"}
# {"text": "深度学习是机器学习的一个子领域。"}
# {"text": "神经网络可以学习复杂的模式。"}
```

#### CSV文件
```bash
# 输入文件: data.csv
# id,content,label
# 1,今天天气真好,positive
# 2,雨天令人沮丧,negative

# 转换命令（自动检测content列）
python scripts/convert_dataset.py \
  --input data.csv \
  --output dataset/my_pretrain.jsonl \
  --mode pretrain

# 或指定列名
python scripts/convert_dataset.py \
  --input data.csv \
  --output dataset/my_pretrain.jsonl \
  --mode pretrain \
  --text-column content
```

#### 已有的JSON/JSONL
```bash
# 自动提取text字段
python scripts/convert_dataset.py \
  --input existing.json \
  --output dataset/my_pretrain.jsonl \
  --mode pretrain
```

### 2. SFT数据准备

#### Alpaca格式（最常见）
```json
// alpaca.json
[
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。"
  },
  {
    "instruction": "翻译以下句子",
    "input": "Hello, how are you?",
    "output": "你好，你怎么样？"
  }
]
```

转换命令：
```bash
python scripts/convert_dataset.py \
  --input alpaca.json \
  --output dataset/my_sft.jsonl \
  --mode sft \
  --format alpaca
```

输出格式：
```json
{"conversations": [
  {"role": "user", "content": "解释什么是机器学习"},
  {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
]}
```

#### ShareGPT格式
```json
// sharegpt.json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么我可以帮助你的吗？"},
      {"from": "human", "value": "介绍一下你自己"},
      {"from": "gpt", "value": "我是一个AI助手..."}
    ]
  }
]
```

转换命令：
```bash
python scripts/convert_dataset.py \
  --input sharegpt.json \
  --output dataset/my_sft.jsonl \
  --mode sft \
  --format sharegpt
```

#### 简单prompt-response格式
```json
// simple.json
[
  {
    "prompt": "什么是Python?",
    "response": "Python是一种高级编程语言..."
  }
]
```

转换命令：
```bash
python scripts/convert_dataset.py \
  --input simple.json \
  --output dataset/my_sft.jsonl \
  --mode sft \
  --format simple
```

#### 自动检测格式
如果不确定格式，可以让脚本自动检测：
```bash
python scripts/convert_dataset.py \
  --input unknown_format.json \
  --output dataset/my_sft.jsonl \
  --mode sft \
  --format auto
```

### 3. 高级用法

#### 限制样本数量（用于测试）
```bash
python scripts/convert_dataset.py \
  --input large_dataset.json \
  --output dataset/test_small.jsonl \
  --mode pretrain \
  --max-samples 1000
```

#### 查看转换结果
转换完成后会自动显示前3个样本预览：
```
✓ 转换完成! 输出文件: dataset/my_pretrain.jsonl

前3个样本预览:

样本 1:
{"text": "人工智能是计算机科学的一个分支。"}

样本 2:
{"text": "深度学习是机器学习的一个子领域。"}
...
```

## 常见数据源转换示例

### Hugging Face Datasets

如果你从Hugging Face下载了数据集：

```python
# 先导出为JSON
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("your_dataset_name")

# 导出为JSON
dataset['train'].to_json("temp_data.json")
```

然后转换：
```bash
python scripts/convert_dataset.py \
  --input temp_data.json \
  --output dataset/hf_converted.jsonl \
  --mode sft \
  --format auto
```

### 自己爬取的数据

如果你有爬虫数据（如网页文本）：
```bash
# 假设每个文件是一段文本
cat crawled_data/*.txt > combined.txt

python scripts/convert_dataset.py \
  --input combined.txt \
  --output dataset/crawled_pretrain.jsonl \
  --mode pretrain
```

### Excel转换

先转换为CSV：
```python
import pandas as pd
df = pd.read_excel("data.xlsx")
df.to_csv("data.csv", index=False)
```

然后使用转换脚本：
```bash
python scripts/convert_dataset.py \
  --input data.csv \
  --output dataset/excel_converted.jsonl \
  --mode pretrain
```

## 训练使用

转换完成后，在训练时使用 `--data_path` 参数：

```bash
# Pretrain
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --data_path dataset/my_pretrain.jsonl \
  --pe rope \
  --epochs 1 \
  --batch_size 4 \
  --ddp

# SFT
torchrun --nproc_per_node 8 trainer/train_full_sft.py \
  --data_path dataset/my_sft.jsonl \
  --pe rope \
  --epochs 2 \
  --batch_size 4 \
  --ddp
```

## 注意事项

1. **编码问题**: 确保输入文件使用UTF-8编码
2. **数据质量**: 转换前最好清洗数据，去除空行、特殊字符等
3. **格式验证**: 转换后可以手动检查前几行，确保格式正确
4. **备份原始数据**: 转换前备份原始数据文件

## 支持的所有格式总结

### Pretrain模式
- ✅ 纯文本 (.txt) - 每行一个样本
- ✅ CSV文件 (.csv) - 自动检测或指定文本列
- ✅ JSON数组 - 自动提取text/content字段
- ✅ JSONL - 每行一个JSON对象

### SFT模式
- ✅ Standard格式 - OpenAI ChatML格式
- ✅ Alpaca格式 - instruction/input/output
- ✅ ShareGPT格式 - from/value字段
- ✅ OpenAI messages格式
- ✅ 简单prompt-response格式
- ✅ 自动检测 - 智能识别格式类型
