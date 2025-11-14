import os
import sys
import argparse
import time
import math
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from model.model_lora import apply_lora, save_lora, load_lora
from parallel.columnar import (
    patch_model_with_interleaved_2d_rope,
    set_rope_pos2d,
    _find_rotary_holder,
    build_columnar_causal_mask,
)
from parallel_data.parallel_dataset import ParallelPretrainDataset
from parallel_data.parallel_collator import ParallelPretrainCollator
from scripts.convert_sft_to_parallel import convert_sft_to_parallel_file

warnings.filterwarnings("ignore")


def Logger(msg):
    if not ddp or dist.get_rank() == 0:
        print(msg)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def auto_pair_indices(model, ratio: float):
    holder = _find_rotary_holder(model)
    inv_freq = getattr(holder.rotary_emb, "inv_freq", None)
    if inv_freq is None:
        raise RuntimeError("未找到 rotary_emb.inv_freq，无法应用 2D RoPE。")
    freq_count = inv_freq.numel()
    pair_count = max(1, min(freq_count, int(round(freq_count * ratio))))
    start = max(1, freq_count - pair_count + 1)
    return list(range(start, freq_count + 1))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    start_time = time.time()

    if ddp and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        time_ids = batch["time_ids"].to(args.device)
        pos2d = batch["pos2d"].to(args.device)

        columnar_mask = build_columnar_causal_mask(time_ids, attention_mask).to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            # 设置 2D RoPE 位置
            if args.patch_rope:
                set_rope_pos2d(rope_target_model, pos2d)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=columnar_mask,
            )
            logits = outputs.logits

            # 计算损失
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            # 如果有 MoE 的 aux_loss
            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.8f} eta:{:.1f}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch / 60 - spend_time / 60,
                )
            )
            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
            lora_save_path = os.path.join(args.save_dir, "lora", f"{args.lora_name}_{args.model_tag}.pth")
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
            save_lora(model_to_save, lora_save_path)


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA finetune on HuggingFace model with Parallel data and 2D RoPE")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace模型名称或本地路径")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="若留空则与base_model一致")
    parser.add_argument("--out_dir", type=str, default=os.path.join(root_path, "out"))
    parser.add_argument("--data_path", type=str, default=os.path.join(root_path, "dataset", "pretrain_hq_split.jsonl"))
    parser.add_argument(
        "--data_mode",
        type=str,
        choices=["parallel", "parallel_sft"],
        default="parallel",
        help="parallel: data_path 已是并行格式; parallel_sft: 对话 SFT JSONL, 训练前自动转换",
    )
    parser.add_argument(
        "--parallel_cache_dir",
        type=str,
        default=None,
        help="parallel_sft 模式下用于存放转换后缓存文件的目录（默认 out/parallel_cache）",
    )
    parser.add_argument("--sft_pad_min", type=int, default=0, help="parallel_sft 随机左/右 padding 的最小token数")
    parser.add_argument("--sft_pad_max", type=int, default=0, help="parallel_sft 随机左/右 padding 的最大token数")
    parser.add_argument("--sft_seed", type=int, default=1337, help="parallel_sft 转换随机种子")
    parser.add_argument(
        "--sft_max_samples",
        type=int,
        default=0,
        help="parallel_sft 转换时最多处理的对话条数，0表示不限制",
    )
    parser.add_argument(
        "--rebuild_parallel_cache",
        action="store_true",
        help="parallel_sft 模式强制重新生成缓存文件",
    )
    parser.add_argument("--lora_name", type=str, default="hf_lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-HF-LoRA")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--patch_rope", action="store_true", dest="patch_rope")
    parser.add_argument("--no_patch_rope", action="store_false", dest="patch_rope")
    parser.set_defaults(patch_rope=True)
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5)
    parser.add_argument("--load_lora", type=str, default=None, help="可选：加载已有LoRA权重继续训练")
    parser.add_argument("--model_tag", type=str, default="hf", help="保存LoRA文件时使用的后缀")
    parser.add_argument("--max_total_tokens", type=int, default=0, help="Parallel collator pad length；设0则按实际长度")
    parser.add_argument("--max_parallel_samples", type=int, default=None, help="限制数据集样本数量")
    parser.add_argument("--branches_per_sample", type=int, default=8)
    parser.add_argument("--max_branches_per_sample", type=int, default=None)
    parser.add_argument("--min_branches_per_sample", type=int, default=1)
    parser.add_argument("--random_time_offset", action="store_true", default=False)
    parser.add_argument("--interleave_branches", action="store_true", default=True)
    parser.add_argument("--branch_stride", type=int, default=128)
    parser.add_argument("--batch_by_samples", action="store_true")
    parser.add_argument(
        "--align_to",
        type=str,
        choices=["left", "right"],
        default="left",
        help="列式布局对齐方式（left=默认，right=末尾对齐）",
    )

    args = parser.parse_args()
    args.dtype = "bfloat16"

    ddp = int(os.environ.get("RANK", -1)) != -1 if args.ddp else False
    ddp_local_rank, DEVICE = 0, args.device

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(1337 + rank)
        torch.cuda.manual_seed(1337 + rank)
    else:
        rank = 0
        args.device = torch.device(args.device)

    is_main_process = (not ddp) or rank == 0

    tokenizer_path = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    effective_data_path = args.data_path
    if args.data_mode == "parallel_sft":
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"SFT 数据文件不存在: {args.data_path}")

        cache_dir = Path(args.parallel_cache_dir) if args.parallel_cache_dir else Path(args.out_dir) / "parallel_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(args.data_path).stem or "parallel_sft"
        cache_name_parts = [base_name, "parallel", f"pad{args.sft_pad_min}-{args.sft_pad_max}", f"seed{args.sft_seed}"]
        if args.sft_max_samples:
            cache_name_parts.append(f"max{args.sft_max_samples}")
        converted_path = cache_dir / ("_".join(cache_name_parts) + ".jsonl")

        needs_rebuild = args.rebuild_parallel_cache or not converted_path.exists()
        if needs_rebuild and is_main_process:
            Logger(
                "parallel_sft 模式：开始将对话数据转换为并行格式 -> {}".format(converted_path)
            )
            convert_sft_to_parallel_file(
                input_path=args.data_path,
                output_path=str(converted_path),
                tokenizer_name=tokenizer_path,
                max_samples=args.sft_max_samples,
                pad_min=args.sft_pad_min,
                pad_max=args.sft_pad_max,
                seed=args.sft_seed,
                log_every=0,
                quiet=True,
            )
        if ddp:
            dist.barrier()

        if not needs_rebuild and is_main_process:
            Logger(f"parallel_sft 模式：使用已有缓存 {converted_path}")

        if not converted_path.exists():
            raise RuntimeError(f"并行缓存文件不存在: {converted_path}")

        effective_data_path = str(converted_path)

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    Logger(f"从 {args.base_model} 加载模型（dtype={args.dtype}）")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(args.device)
    model.model_tag = args.model_tag

    rope_target_model = model

    if args.patch_rope:
        pair_indices = auto_pair_indices(model, args.rope_2d_ratio)
        patch_model_with_interleaved_2d_rope(model, pair_indices)
        Logger(f"已应用 Interleaved 2D RoPE（ratio={args.rope_2d_ratio}, pairs={len(pair_indices)}）")

    apply_lora(model, rank=args.lora_rank)
    if args.load_lora and os.path.exists(args.load_lora):
        load_lora(model, args.load_lora)
        Logger(f"已加载 LoRA 权重: {args.load_lora}")

    for name, param in model.named_parameters():
        if not hasattr(param, "requires_grad"):
            continue
        param.requires_grad = bool("lora" in name)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    Logger(f"总参数量: {total_params / 1e6:.2f}M")
    Logger(f"可训练参数量（LoRA）: {trainable_params_count / 1e6:.2f}M ({trainable_params_count / total_params * 100:.2f}%)")

    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # 加载 Parallel 数据集
    Logger(f"加载 Parallel 数据集: {effective_data_path}")
    train_ds = ParallelPretrainDataset(
        data_path=effective_data_path,
        max_samples=args.max_parallel_samples,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None

    collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        random_time_offset=args.random_time_offset,
        interleave_branches=args.interleave_branches,
        branch_stride=args.branch_stride,
        align_to=args.align_to,
    )
    if args.batch_by_samples:
        collator.target_samples = args.batch_size
        effective_batch_size = max(
            1, args.batch_size * (args.max_branches_per_sample or args.branches_per_sample)
        )
        Logger(f"batch_by_samples 模式：每个 batch 约 {args.batch_size} 个 sample -> {effective_batch_size} 条 branch")
    else:
        effective_batch_size = args.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    iter_per_epoch = len(train_loader)
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cpu":
        ctx = nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
        # PyTorch 尚未对 bfloat16 提供 GradScaler 实现；仅在 float16 时启用缩放
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    wandb = None
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=f"HF-LoRA-{args.lora_name}")

    Logger("=" * 80)
    Logger("开始训练...")
    Logger("=" * 80)

    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

    if (not ddp) or dist.get_rank() == 0:
        final_path = os.path.join(args.save_dir, "lora", f"{args.lora_name}_{args.model_tag}_final.pth")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        save_lora(model_to_save, final_path)
        Logger("=" * 80)
        Logger(f"训练完成！LoRA 权重已保存至: {final_path}")
        Logger("=" * 80)
