#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将对话类 SFT JSONL 转换为 parallel 预训练格式。

每个 assistant 回复会生成一个 branch：
<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>
并记录 answer_offset（在 token 级别，标记从哪个位置开始计算 loss）。
"""
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

USER_TOKEN = "<|im_start|>user"
ASSISTANT_TOKEN = "<|im_start|>assistant"
END_TOKEN = "<|im_end|>"


def _build_branch_text(user_text: str, assistant_text: str) -> Tuple[str, str, str]:
    user_block = f"{USER_TOKEN}\n{user_text.strip()}\n{END_TOKEN}\n"
    assistant_prefix = f"{ASSISTANT_TOKEN}\n"
    assistant_suffix = f"\n{END_TOKEN}"
    return user_block, assistant_prefix, assistant_suffix


def convert_sft_to_parallel_file(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    max_samples: int = 0,
    pad_min: int = 0,
    pad_max: int = 0,
    seed: int = 1337,
    log_every: int = 10000,
    quiet: bool = False,
) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    pad_token = tokenizer.pad_token or ""
    rng = random.Random(seed)

    total_in = 0
    total_out = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples and total_out >= max_samples:
                break
            total_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            conversations = obj.get("conversations", [])
            if not isinstance(conversations, list):
                continue

            idx = 0
            while idx < len(conversations):
                turn = conversations[idx]
                if turn.get("role") == "user":
                    user_text = str(turn.get("content", "")).strip()
                    if not user_text:
                        idx += 1
                        continue
                    if idx + 1 < len(conversations) and conversations[idx + 1].get("role") == "assistant":
                        assistant_text = str(conversations[idx + 1].get("content", "")).strip()
                        if not assistant_text:
                            idx += 2
                            continue

                        prefix_block, assistant_prefix, assistant_suffix = _build_branch_text(user_text, assistant_text)

                        pad_left = pad_token * rng.randint(pad_min, pad_max)
                        pad_right = pad_token * rng.randint(pad_min, pad_max)

                        full_text = f"{pad_left}{prefix_block}{assistant_prefix}{assistant_text}{assistant_suffix}{pad_right}"
                        prefix_text = f"{pad_left}{prefix_block}{assistant_prefix}"

                        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
                        answer_offset = len(prefix_ids)

                        fout.write(
                            json.dumps(
                                {
                                    "text": full_text,
                                    "answer_offset": answer_offset,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        total_out += 1

                        if (not quiet) and log_every and total_out % log_every == 0:
                            print(f"已转换 {total_out:,} 个分支 (处理 {total_in:,} 行)")

                        if max_samples and total_out >= max_samples:
                            break
                        idx += 2
                    else:
                        idx += 1
                else:
                    idx += 1

    if not quiet:
        print(f"✓ 完成转换: 读取 {total_in:,} 行, 输出 {total_out:,} 个branch -> {output_path}")
    return total_out


def main():
    parser = argparse.ArgumentParser(description="Convert conversation SFT JSONL to parallel branch JSONL")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--pad_min", type=int, default=0)
    parser.add_argument("--pad_max", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    convert_sft_to_parallel_file(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples,
        pad_min=args.pad_min,
        pad_max=args.pad_max,
        seed=args.seed,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
