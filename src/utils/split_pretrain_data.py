"""
将预训练数据按照 <|im_start|>...<|im_end|> 切分成独立样本

原始格式：
{"text": "<|im_start|>对话1<|im_end|> <|im_start|>对话2<|im_end|> ..."}

输出格式：
{"text": "<|im_start|>对话1<|im_end|>"}
{"text": "<|im_start|>对话2<|im_end|>"}
...
"""

import json
import re
from pathlib import Path
from typing import List


def split_conversations(text: str) -> List[str]:
    """
    按照 <|im_start|>...<|im_end|> 切分对话
    """
    # 使用正则表达式匹配所有 <|im_start|>...<|im_end|> 片段
    pattern = r'<\|im_start\|>.*?<\|im_end\|>'
    matches = re.findall(pattern, text, re.DOTALL)

    # 清理每个片段（去除多余空格）
    conversations = []
    for match in matches:
        # 去除片段前后空格
        cleaned = match.strip()
        if cleaned:
            conversations.append(cleaned)

    return conversations


def process_file(input_path: Path, output_path: Path):
    """
    处理整个文件
    """
    total_lines = 0
    total_conversations = 0

    with input_path.open('r', encoding='utf-8') as f_in, \
         output_path.open('w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                text = data.get('text', '')

                if not text:
                    print(f"警告: 第 {line_num} 行没有 'text' 字段，跳过")
                    continue

                # 切分对话
                conversations = split_conversations(text)

                if not conversations:
                    print(f"警告: 第 {line_num} 行没有找到任何对话片段，跳过")
                    continue

                # 写入输出文件
                for conv in conversations:
                    output_data = {"text": conv}
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    total_conversations += 1

                total_lines += 1

                if line_num % 100 == 0:
                    print(f"已处理 {line_num} 行，切分出 {total_conversations} 个对话")

            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    return total_lines, total_conversations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="切分预训练数据")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/pretrain_hq.jsonl",
        help="输入文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/pretrain_hq_split.jsonl",
        help="输出文件路径"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"开始处理文件...")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print()

    total_lines, total_conversations = process_file(input_path, output_path)

    print()
    print("=" * 60)
    print(f"处理完成！")
    print(f"原始行数: {total_lines}")
    print(f"切分后样本数: {total_conversations}")
    print(f"平均每行包含: {total_conversations / total_lines:.1f} 个对话")
    print(f"输出文件: {output_path}")
    print("=" * 60)

    # 显示前几个样本
    print("\n前5个切分后的样本:")
    with output_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            data = json.loads(line)
            text = data['text']
            # 截断显示
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"{i+1}. {text}")


if __name__ == "__main__":
    main()
