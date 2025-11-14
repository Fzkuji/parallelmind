#!/usr/bin/env python3
"""
HuggingFace + LoRA + 2D RoPE 并行/批量推理脚本
支持多 GPU 分布式推理
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model_lora import apply_lora, load_lora
from parallel.columnar import (
    patch_model_with_interleaved_2d_rope,
    set_rope_pos2d,
    _find_rotary_holder,
)
from parallel_data.parallel_dataset import ParallelPretrainDataset
from parallel_data.parallel_collator import ParallelPretrainCollator


def Logger(msg, rank=None):
    """分布式训练日志"""
    if rank is None or rank == 0:
        print(msg)


def auto_pair_indices(model, ratio: float):
    """计算 2D RoPE 的频率对索引"""
    holder = _find_rotary_holder(model)
    inv_freq = getattr(holder.rotary_emb, "inv_freq", None)
    if inv_freq is None:
        raise RuntimeError("未找到 rotary_emb.inv_freq，无法应用 2D RoPE。")
    freq_count = inv_freq.numel()
    pair_count = max(1, min(freq_count, int(round(freq_count * ratio))))
    start = max(1, freq_count - pair_count + 1)
    return list(range(start, freq_count + 1))


def load_model_with_lora(args, rank=None):
    """
    加载 HuggingFace 模型并应用 LoRA + 2D RoPE

    Args:
        args: 命令行参数
        rank: DDP rank（None 表示单 GPU）

    Returns:
        model, tokenizer, patch_rope
    """
    Logger("=" * 80, rank)
    Logger(f"加载基础模型: {args.base_model}", rank)
    Logger("=" * 80, rank)

    # 设置数据类型
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    device = args.device if rank is None else f"cuda:{rank}"
    model.to(device)
    model.eval()

    # 应用 2D RoPE
    if args.patch_rope:
        Logger(f"应用 2D RoPE（ratio={args.rope_2d_ratio}）...", rank)
        pair_indices = auto_pair_indices(model, args.rope_2d_ratio)
        patch_model_with_interleaved_2d_rope(model, pair_indices)
        Logger(f"✓ 2D RoPE 已应用（{len(pair_indices)} 个频率对用于 branch 维度）", rank)
        model._uses_pos2d = True
    else:
        model._uses_pos2d = False

    # 应用 LoRA
    if args.lora_path:
        Logger(f"应用 LoRA（rank={args.lora_rank}）...", rank)
        apply_lora(model, rank=args.lora_rank)

        # 加载 LoRA 权重
        if os.path.exists(args.lora_path):
            Logger(f"加载 LoRA 权重: {args.lora_path}", rank)
            load_lora(model, args.lora_path)
            Logger("✓ LoRA 权重加载成功", rank)
        else:
            raise FileNotFoundError(f"LoRA 权重文件不存在: {args.lora_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    Logger("=" * 80, rank)
    Logger(f"模型加载完成！总参数量: {total_params / 1e6:.2f}M", rank)
    Logger("=" * 80, rank)

    return model, tokenizer, args.patch_rope


def batch_generate(
    model,
    tokenizer,
    dataloader,
    args,
    rank=None,
):
    """
    批量生成

    Args:
        model: 模型
        tokenizer: tokenizer
        dataloader: 数据加载器
        args: 参数
        rank: DDP rank
    """
    device = next(model.parameters()).device
    results = []

    Logger(f"\n开始批量生成（rank={rank or 0}）...", rank)
    Logger(f"总批次数: {len(dataloader)}", rank)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pos2d = batch["pos2d"].to(device) if "pos2d" in batch else None

            # 设置 pos2d（如果使用 2D RoPE）
            if args.patch_rope and pos2d is not None:
                set_rope_pos2d(model, pos2d)

            # 生成
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # 解码
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                input_len = attention_mask[i].sum().item()
                generated_tokens = outputs[i][input_len:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                results.append({
                    "input": tokenizer.decode(input_ids[i][:input_len], skip_special_tokens=True),
                    "output": response,
                })

            if (batch_idx + 1) % 10 == 0:
                Logger(f"已处理 {batch_idx + 1}/{len(dataloader)} 批次", rank)

    Logger(f"生成完成！共生成 {len(results)} 条结果", rank)
    return results


def save_results(results, output_path, rank=None):
    """保存结果到 JSONL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    Logger(f"结果已保存到: {output_path}", rank)


def init_distributed_mode(args):
    """初始化分布式训练"""
    if int(os.environ.get("RANK", -1)) != -1:
        # DDP 模式
        dist.init_process_group(backend="nccl")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.device)
        Logger(f"DDP 初始化完成: rank={args.rank}, world_size={args.world_size}", args.rank)
    else:
        # 单 GPU 模式
        args.rank = None
        args.world_size = 1
        args.local_rank = 0
        Logger("单 GPU 模式")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace + LoRA 并行推理")

    # 模型参数
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace 模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重路径（可选）")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")

    # 2D RoPE 参数
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5, help="2D RoPE ratio")
    parser.add_argument("--patch_rope", action="store_true", dest="patch_rope", help="应用 2D RoPE")
    parser.add_argument("--no_patch_rope", action="store_false", dest="patch_rope", help="不应用 2D RoPE")
    parser.set_defaults(patch_rope=True)

    # 数据参数
    parser.add_argument("--data_path", type=str, required=True, help="输入数据路径（JSONL）")
    parser.add_argument("--out_path", type=str, required=True, help="输出结果路径（JSONL）")
    parser.add_argument("--max_samples", type=int, default=None, help="限制处理的样本数量")

    # Parallel 数据参数
    parser.add_argument("--branches_per_sample", type=int, default=4, help="固定 branch 数量")
    parser.add_argument("--max_branches_per_sample", type=int, default=None, help="动态最大 branch 数")
    parser.add_argument("--min_branches_per_sample", type=int, default=1, help="动态最小 branch 数")
    parser.add_argument("--batch_by_samples", action="store_true", help="按样本数计算 batch")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_total_tokens", type=int, default=0, help="最大 token 数（0=动态）")
    parser.add_argument("--branch_stride", type=int, default=128, help="Branch 位置步长")
    parser.add_argument("--random_time_offset", action="store_true", default=False)
    parser.add_argument("--interleave_branches", action="store_true", default=True)

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling")

    # 系统参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")

    args = parser.parse_args()

    # 初始化分布式
    init_distributed_mode(args)

    # 加载模型
    model, tokenizer, patch_rope = load_model_with_lora(args, args.rank)

    # 加载数据集
    Logger(f"加载数据集: {args.data_path}", args.rank)
    dataset = ParallelPretrainDataset(
        data_path=args.data_path,
        max_samples=args.max_samples,
    )
    Logger(f"✓ 数据集加载完成：{len(dataset)} 个样本", args.rank)

    # 创建 sampler 和 collator
    if args.rank is not None:
        # DDP 模式
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        # 单 GPU 模式
        sampler = None

    collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        random_time_offset=args.random_time_offset,
        interleave_branches=args.interleave_branches,
        branch_stride=args.branch_stride,
    )

    # 计算有效 batch size
    if args.batch_by_samples:
        collator.target_samples = args.batch_size
        effective_batch_size = max(
            1, args.batch_size * (args.max_branches_per_sample or args.branches_per_sample)
        )
        Logger(f"batch_by_samples 模式：每个 batch 约 {args.batch_size} 个 sample -> {effective_batch_size} 条 branch", args.rank)
    else:
        effective_batch_size = args.batch_size
        Logger(f"batch_size: {effective_batch_size}", args.rank)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    # DDP 包装
    if args.rank is not None:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_for_generate = model.module
    else:
        model_for_generate = model

    # 批量生成
    results = batch_generate(
        model_for_generate,
        tokenizer,
        dataloader,
        args,
        args.rank,
    )

    # 保存结果
    if args.rank is None or args.rank == 0:
        # 只有 rank 0 保存结果
        if args.rank is not None:
            # DDP 模式：收集所有 rank 的结果
            Logger("收集所有 GPU 的结果...", args.rank)
            all_results = [None] * args.world_size
            dist.gather_object(results, all_results if args.rank == 0 else None, dst=0)

            if args.rank == 0:
                # 合并所有结果
                final_results = []
                for rank_results in all_results:
                    if rank_results:
                        final_results.extend(rank_results)
                save_results(final_results, args.out_path, args.rank)
        else:
            # 单 GPU 模式
            save_results(results, args.out_path)

    Logger("=" * 80, args.rank)
    Logger("推理完成！", args.rank)
    Logger("=" * 80, args.rank)

    # 清理 DDP
    if args.rank is not None:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
