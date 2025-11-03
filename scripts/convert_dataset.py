#!/usr/bin/env python3
"""
通用数据集转换脚本
支持多种输入格式，自动转换为训练所需的JSONL格式

使用示例：
    # Pretrain - 纯文本文件（每行一个样本）
    python scripts/convert_dataset.py --input data.txt --output dataset/pretrain.jsonl --mode pretrain

    # Pretrain - CSV文件（指定文本列）
    python scripts/convert_dataset.py --input data.csv --output dataset/pretrain.jsonl --mode pretrain --text-column content

    # Hugging Face数据集（推荐！）
    python scripts/convert_dataset.py --hf-dataset HuggingFaceFW/fineweb-edu --hf-subset sample-10BT --hf-split train --output dataset/pretrain.jsonl --mode pretrain --max-samples 10000

    # SFT - ShareGPT格式
    python scripts/convert_dataset.py --input sharegpt.json --output dataset/sft.jsonl --mode sft --format sharegpt

    # SFT - Alpaca格式
    python scripts/convert_dataset.py --input alpaca.json --output dataset/sft.jsonl --mode sft --format alpaca

    # 自动检测格式
    python scripts/convert_dataset.py --input data.json --output dataset/output.jsonl --mode pretrain
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# 添加项目根目录到路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def load_huggingface_dataset(
    dataset_name: str,
    subset: str = None,
    split: str = "train",
    text_column: str = None,
    max_samples: int = None
) -> List[Dict[str, Any]]:
    """
    从Hugging Face加载数据集

    Args:
        dataset_name: HF数据集名称，如 "HuggingFaceFW/fineweb-edu"
        subset: 子集名称，如 "sample-10BT"
        split: 数据分割，如 "train", "test", "validation"
        text_column: 文本列名（None表示自动检测）
        max_samples: 最大样本数
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("请运行: pip install datasets")
        sys.exit(1)

    print(f"正在从 Hugging Face 加载数据集: {dataset_name}")
    if subset:
        print(f"  子集: {subset}")
    print(f"  分割: {split}")

    # 加载数据集
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True if max_samples else False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True if max_samples else False)
    except Exception as e:
        print(f"加载失败: {e}")
        print("\n提示:")
        print("1. 检查数据集名称是否正确")
        print("2. 某些数据集需要登录HF账号: huggingface-cli login")
        print("3. 查看数据集页面了解可用的subsets和splits")
        sys.exit(1)

    # 检查列名
    if hasattr(dataset, 'column_names'):
        available_columns = dataset.column_names
    elif hasattr(dataset, 'features'):
        available_columns = list(dataset.features.keys())
    else:
        # 对于streaming dataset，先取一个样本查看
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
        # 重新创建迭代器
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True if max_samples else False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True if max_samples else False)

    print(f"  可用列: {available_columns}")

    # 自动检测文本列
    if text_column is None:
        possible_text_columns = ['text', 'content', 'document', 'sentence', 'data', 'input', 'prompt']
        text_column = next((col for col in possible_text_columns if col in available_columns), available_columns[0])
        print(f"  自动检测文本列: {text_column}")

    # 提取样本
    samples = []
    count = 0

    for item in dataset:
        if max_samples and count >= max_samples:
            break

        # 提取文本
        text_content = item.get(text_column, "")
        if isinstance(text_content, str) and text_content.strip():
            samples.append(item)  # 保留原始字典，后续转换时再处理
            count += 1

            if count % 1000 == 0:
                print(f"  已加载 {count} 个样本...")

    print(f"✓ 从 Hugging Face 加载完成，共 {len(samples)} 个样本")
    return samples


def load_plain_text(file_path: str) -> List[Dict[str, str]]:
    """加载纯文本文件（每行一个样本）"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                samples.append({"text": text})
    return samples


def load_csv(file_path: str, text_column: str = None) -> List[Dict[str, str]]:
    """加载CSV文件"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # 自动检测文本列
        if text_column is None:
            possible_columns = ['text', 'content', 'document', 'sentence', 'data']
            text_column = next((col for col in possible_columns if col in headers), headers[0])
            print(f"自动检测到文本列: {text_column}")

        for row in reader:
            if text_column in row and row[text_column].strip():
                samples.append({"text": row[text_column].strip()})

    return samples


def load_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON或JSONL文件"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        # 尝试作为JSONL
        if '\n' in content:
            for line in content.split('\n'):
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # 如果JSONL失败，尝试作为JSON数组
        if not samples:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    samples = [data]
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")

    return samples


def convert_to_pretrain(samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """转换为Pretrain格式: {"text": "..."}"""
    converted = []
    for item in samples:
        text = None

        # 尝试不同的键名
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("document")

            # 如果是对话格式，拼接成文本
            if not text and "conversations" in item:
                parts = []
                for turn in item["conversations"]:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    parts.append(f"{role}: {content}")
                text = "\n".join(parts)
        elif isinstance(item, str):
            text = item

        if text and text.strip():
            converted.append({"text": text.strip()})

    return converted


def convert_to_sft_standard(samples: List[Dict[str, Any]], format_type: str = "auto") -> List[Dict]:
    """转换为SFT标准格式: {"conversations": [{"role": "...", "content": "..."}]}"""
    converted = []

    for item in samples:
        if not isinstance(item, dict):
            continue

        conversations = None

        # 检测格式类型
        if format_type == "auto":
            if "conversations" in item:
                format_type = "standard"
            elif "messages" in item:
                format_type = "openai"
            elif "instruction" in item or "input" in item or "output" in item:
                format_type = "alpaca"
            elif "prompt" in item and "response" in item:
                format_type = "simple"

        # 根据格式转换
        if format_type == "standard" and "conversations" in item:
            # 已经是标准格式
            conversations = item["conversations"]

        elif format_type == "openai" and "messages" in item:
            # OpenAI messages格式
            conversations = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in item["messages"]
            ]

        elif format_type == "sharegpt":
            # ShareGPT格式
            if "conversations" in item:
                convs = []
                for turn in item["conversations"]:
                    from_role = turn.get("from", "")
                    # 映射角色
                    if from_role == "human":
                        role = "user"
                    elif from_role == "gpt":
                        role = "assistant"
                    else:
                        role = from_role
                    convs.append({"role": role, "content": turn.get("value", "")})
                conversations = convs

        elif format_type == "alpaca":
            # Alpaca格式
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # 构造user问题
            if input_text:
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction

            conversations = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]

        elif format_type == "simple":
            # 简单的prompt-response格式
            conversations = [
                {"role": "user", "content": item.get("prompt", "")},
                {"role": "assistant", "content": item.get("response", "")}
            ]

        # 验证并添加
        if conversations and all(
            isinstance(turn, dict) and "role" in turn and "content" in turn
            for turn in conversations
        ):
            converted.append({"conversations": conversations})

    return converted


def main():
    parser = argparse.ArgumentParser(description="通用数据集转换工具")
    parser.add_argument("--input", "-i", help="输入文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSONL文件路径")
    parser.add_argument("--mode", required=True, choices=["pretrain", "sft"],
                        help="训练模式: pretrain(预训练) 或 sft(监督微调)")

    # Hugging Face 数据集参数
    parser.add_argument("--hf-dataset", help="Hugging Face数据集名称，如 'HuggingFaceFW/fineweb-edu'")
    parser.add_argument("--hf-subset", help="HF数据集子集名称，如 'sample-10BT'")
    parser.add_argument("--hf-split", default="train", help="HF数据集分割（默认: train）")

    # 其他参数
    parser.add_argument("--format", default="auto",
                        choices=["auto", "standard", "openai", "sharegpt", "alpaca", "simple"],
                        help="SFT数据格式（仅在mode=sft时使用）")
    parser.add_argument("--text-column", default=None,
                        help="文本列名（CSV文件或HF数据集，默认自动检测）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数量（用于测试）")

    args = parser.parse_args()

    # 验证输入参数
    if not args.hf_dataset and not args.input:
        parser.error("必须指定 --input 或 --hf-dataset 其中之一")

    output_path = Path(args.output)

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.hf_dataset:
        # 从Hugging Face加载
        samples = load_huggingface_dataset(
            dataset_name=args.hf_dataset,
            subset=args.hf_subset,
            split=args.hf_split,
            text_column=args.text_column,
            max_samples=args.max_samples
        )
    else:
        # 从本地文件加载
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            return

        print(f"正在加载数据: {input_path}")

        # 根据文件类型加载数据
        suffix = input_path.suffix.lower()

        if suffix == ".txt":
            samples = load_plain_text(str(input_path))
        elif suffix == ".csv":
            samples = load_csv(str(input_path), args.text_column)
        elif suffix in [".json", ".jsonl"]:
            samples = load_json_or_jsonl(str(input_path))
        else:
            print(f"警告: 未知文件类型 {suffix}，尝试作为文本文件处理")
            samples = load_plain_text(str(input_path))

    print(f"加载了 {len(samples)} 个原始样本")

    # 转换格式
    if args.mode == "pretrain":
        converted = convert_to_pretrain(samples)
    else:  # sft
        converted = convert_to_sft_standard(samples, args.format)

    # 限制样本数量（用于测试）
    if args.max_samples:
        converted = converted[:args.max_samples]

    print(f"转换后有 {len(converted)} 个有效样本")

    # 写入JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ 转换完成! 输出文件: {output_path}")

    # 显示示例
    if converted:
        print("\n前3个样本预览:")
        for i, item in enumerate(converted[:3]):
            print(f"\n样本 {i+1}:")
            print(json.dumps(item, ensure_ascii=False, indent=2)[:200] + "...")


if __name__ == "__main__":
    main()
