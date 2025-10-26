#!/usr/bin/env python3
import argparse
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from parallel.columnar import build_columnar_causal_mask
from parallel_data.parallel_collator import ParallelPretrainCollator
from parallel_data.parallel_dataset import ParallelPretrainDataset


def format_matrix(tensor, max_rows=12, max_cols=32):
    tensor = tensor.to(torch.int32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    rows = []
    for row in tensor[:max_rows]:
        row_vals = " ".join(f"{val:2d}" for val in row[:max_cols])
        if row.numel() > max_cols:
            row_vals += " ..."
        rows.append(row_vals)
    if tensor.size(0) > max_rows:
        rows.append("...")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Inspect Stage-2 attention mask/collator output")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4, help="Samples per batch (Stage-2 uses batch_by_samples)")
    parser.add_argument("--batch_by_samples", action="store_true", default=True)
    parser.add_argument("--branches_per_sample", type=int, default=8)
    parser.add_argument("--max_branches_per_sample", type=int, default=32)
    parser.add_argument("--min_branches_per_sample", type=int, default=1)
    parser.add_argument("--max_total_tokens", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--min_visual_branches",
        type=int,
        default=2,
        help="Try to find a batch containing at least this many branches when using a dataset",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.data_path:
        data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(root, args.data_path)
        dataset = ParallelPretrainDataset(data_path)
        single_sample = False
        demo_features = None
    else:
        demo_features = [
            {"text": "<|im_start|>Branch-0 token a<|im_end|>"},
            {"text": "<|im_start|>Branch-1 token b<|im_end|>"},
        ]
        dataset = demo_features
        single_sample = True

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "model"))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        random_time_offset=True,
        interleave_branches=True,
    )

    if single_sample:
        collator.max_branches = collator.min_branches = len(demo_features)
        collator.target_samples = 1
        batch = collator(dataset)
    else:
        collator.target_samples = args.batch_size
        loader = DataLoader(
            dataset,
            batch_size=1 if args.batch_by_samples else args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            drop_last=False,
        )
        batch = None
        tries = 0
        for current in loader:
            tries += 1
            branch_counts = current["branch_counts"]
            if branch_counts.numel() > 0 and branch_counts.max().item() >= args.min_visual_branches:
                batch = current
                break
            if tries >= 32:
                batch = current
                break
        if batch is None:
            batch = current
        if batch["branch_counts"].max().item() < args.min_visual_branches:
            print(
                f"[warn] 未找到包含 {args.min_visual_branches} 个以上 branch 的样本，"
                f"当前 batch branch_counts={batch['branch_counts'].tolist()}，改用前 {args.min_visual_branches} 个样例演示。"
            )
            demo_cnt = min(args.min_visual_branches, len(dataset))
            demo_features = [dataset[i] for i in range(demo_cnt)]
            collator.max_branches = collator.min_branches = demo_cnt
            collator.target_samples = 1
            batch = collator(demo_features)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(args.device)

    column_mask = build_columnar_causal_mask(batch["time_ids"], batch["attention_mask"])

    print("=== Batch tensors ===")
    for name in ("input_ids", "attention_mask", "time_ids", "pos2d"):
        tensor = batch[name]
        print(f"{name}: shape={tuple(tensor.shape)}")
    if "branch_counts" in batch:
        print(f"branch_counts: {batch['branch_counts'].tolist()}")

    print("\n=== Attention mask (first sample) ===")
    print(format_matrix(batch["attention_mask"][0]))

    print("\n=== Column causal mask logits ===")
    full_mask = torch.where(column_mask == 0.0, torch.ones_like(column_mask), torch.zeros_like(column_mask))
    for sample_idx in range(full_mask.size(0)):
        print(f"[sample {sample_idx}]")
        print(format_matrix(full_mask[sample_idx, 0]))

    print("\n=== pos2d (branch,time) for first sample ===")
    pos2d = batch["pos2d"][0]
    valid_len = int(batch["attention_mask"][0].sum().item())
    pairs = [tuple(map(int, pair.tolist())) for pair in pos2d[:valid_len]]
    print(" ".join(f"({bx},{ty})" for bx, ty in pairs))

    print("\n=== Branch breakdown ===")
    branch_ids = sorted({bx for bx, _ in pairs})
    branch_map = {bx: idx for idx, bx in enumerate(branch_ids)}
    per_branch = {idx: [] for idx in branch_map.values()}
    for token_idx, (bx, ty) in enumerate(pairs):
        per_branch[branch_map[bx]].append((token_idx, ty))
    for b_idx, tokens in per_branch.items():
        snippet = " ".join(f"{tok_idx}:{time}" for tok_idx, time in tokens[:16])
        if len(tokens) > 16:
            snippet += " ..."
        print(f"branch {b_idx} -> {len(tokens)} tokens | {snippet}")


if __name__ == "__main__":
    main()
