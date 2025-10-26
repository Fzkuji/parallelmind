#!/usr/bin/env python3
"""
Quick script to evaluate pretrain loss on a small batch using the current model/checkpoint.
Usage:
  python scripts/check_pretrain_loss.py \
      --model_path out/pretrain_dynamic/pretrain_512.pth \
      --data_path dataset/pretrain_hq_split.jsonl \
      --batch_size 4 \
      --batch_by_samples \
      --max_branches_per_sample 32 \
      --min_branches_per_sample 1 \
      --max_total_tokens 0
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from parallel.columnar import build_columnar_causal_mask
from parallel_data.parallel_collator import ParallelPretrainCollator
from parallel_data.parallel_dataset import ParallelPretrainDataset


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    ckpt_path = args.model_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = MiniMindForCausalLM(config)
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Check pretrain loss on a sample batch")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrain_*.pth")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="batch_by_samples uses this many samples")
    parser.add_argument("--batch_by_samples", action="store_true", default=True)
    parser.add_argument("--branches_per_sample", type=int, default=8)
    parser.add_argument("--max_branches_per_sample", type=int, default=32)
    parser.add_argument("--min_branches_per_sample", type=int, default=1)
    parser.add_argument("--max_total_tokens", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model(args)

    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(ROOT, args.data_path)
    dataset = ParallelPretrainDataset(data_path)
    collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        random_time_offset=False,
        interleave_branches=True,
    )
    if args.batch_by_samples:
        collator.target_samples = args.batch_size

    effective_batch = args.batch_size * max(
        args.min_branches_per_sample, (args.min_branches_per_sample + args.max_branches_per_sample) // 2
    ) if args.batch_by_samples else args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        mask = build_columnar_causal_mask(batch["time_ids"], batch["attention_mask"]).to(args.device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=mask,
            position_ids=batch["position_ids"],
            pos2d=batch["pos2d"],
        )
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = batch["labels"][:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    print(f"Batch shape: {batch['input_ids'].shape}, loss={loss.item():.4f}")
    print(f"Branch counts: {batch['branch_counts'].tolist()}")


if __name__ == "__main__":
    main()
