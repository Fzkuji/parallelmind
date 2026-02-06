#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本用途：
将 SFT 格式的 JSONL 数据（形如 {"conversations": [{"role": "user"|"assistant", "content": "..."}, ...]} ）
转换为预训练格式的 JSONL：每行一个形如 {"text": "<|im_start|>...<|im_end|>"} 的样本。

特点与说明：
- 顺序保留：按 conversations 列表顺序拼接所有 turn 的 content。
- 包裹标记：默认在首尾添加 <|im_start|> 和 <|im_end|>；可通过参数自定义。
- 兼容性：若输入本身已有 {"text": "..."}，则直接套用首尾标记输出。
- 大文件友好：逐行流式处理，适合 10GB 级别数据。

示例：
  python scripts/convert_sft_to_pretrain.py \
    --input dataset/sft_512.jsonl \
    --output dataset/pretrain_512.jsonl

同理可处理 1024/2048：
  python scripts/convert_sft_to_pretrain.py --input dataset/sft_1024.jsonl --output dataset/pretrain_1024.jsonl
  python scripts/convert_sft_to_pretrain.py --input dataset/sft_2048.jsonl --output dataset/pretrain_2048.jsonl
"""

import os
import sys
import json
import argparse
from typing import List, Dict


def build_text_from_conversations(conversations: List[Dict], sep: str = " ") -> str:
    parts: List[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return sep.join(parts).strip()


def convert_file(input_path: str,
                 output_path: str,
                 start_token: str = "<|im_start|>",
                 end_token: str = "<|im_end|>",
                 sep: str = " ",
                 max_lines: int = 0,
                 log_every: int = 100000) -> None:
    total_in, total_out = 0, 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for total_in, line in enumerate(fin, 1):
            if max_lines and total_in > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = None

            # 优先从 conversations 构造
            if isinstance(obj, dict) and isinstance(obj.get("conversations"), list):
                text = build_text_from_conversations(obj["conversations"], sep=sep)
            # 兼容已是 {"text": "..."} 的情况
            if not text and isinstance(obj, dict) and isinstance(obj.get("text"), str):
                text = obj.get("text").strip()

            if not text:
                continue

            wrapped = f"{start_token}{text}{end_token}"
            fout.write(json.dumps({"text": wrapped}, ensure_ascii=False) + "\n")
            total_out += 1

            if log_every and total_in % log_every == 0:
                print(f"Processed {total_in:,} lines -> written {total_out:,} samples...")

    print(f"完成转换: 输入 {total_in:,} 行，输出 {total_out:,} 条样本 -> {output_path}")


def guess_output_path(input_path: str) -> str:
    # 常见映射：sft_512.jsonl -> pretrain_512.jsonl
    base = os.path.basename(input_path)
    if base.startswith("sft_"):
        base = base.replace("sft_", "pretrain_", 1)
    return os.path.join(os.path.dirname(input_path), base)


def main():
    parser = argparse.ArgumentParser(description="Convert SFT JSONL dataset to pretrain JSONL format")
    parser.add_argument("--input", type=str, default=os.path.join("dataset", "sft_512.jsonl"))
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path. Default: dataset/pretrain_*.jsonl")
    parser.add_argument("--start_token", type=str, default="<|im_start|>")
    parser.add_argument("--end_token", type=str, default="<|im_end|>")
    parser.add_argument("--sep", type=str, default=" ", help="Separator used when joining multi-turn contents")
    parser.add_argument("--max_lines", type=int, default=0, help="For quick test: only process first N lines (0=all)")
    parser.add_argument("--log_every", type=int, default=100000)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or guess_output_path(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    convert_file(
        input_path=input_path,
        output_path=output_path,
        start_token=args.start_token,
        end_token=args.end_token,
        sep=args.sep,
        max_lines=args.max_lines,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

