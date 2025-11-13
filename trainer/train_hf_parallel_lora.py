import os
import sys
import argparse
import time
import math
import warnings
import json

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
from parallel_data.parallel_dataset import ParallelPretrainDataset, ParallelPretrainIterableDataset
from parallel_data.parallel_collator import ParallelPretrainCollator

warnings.filterwarnings("ignore")


def Logger(msg):
    if not ddp or dist.get_rank() == 0:
        print(msg)


def _append_jsonl(path: str, obj: dict):
    """Append a JSON object as one line to a JSONL file. Fail-silent in distributed context."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def auto_pair_indices(model, ratio: float):
    """自动计算 2D RoPE 的频率对索引"""
    holder = _find_rotary_holder(model)
    inv_freq = getattr(holder.rotary_emb, "inv_freq", None)
    if inv_freq is None:
        raise RuntimeError("未找到 rotary_emb.inv_freq，无法应用 2D RoPE。")
    freq_count = inv_freq.numel()
    pair_count = max(1, min(freq_count, int(round(freq_count * ratio))))
    start = max(1, freq_count - pair_count + 1)
    return list(range(start, freq_count + 1))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    start_time = time.time()
    total_loss = 0.0
    total_tokens = 0

    if ddp and hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(train_loader):
        # 移动数据到设备
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        pos2d = batch["pos2d"].to(args.device)

        # 构建 columnar causal mask
        columnar_mask = build_columnar_causal_mask(
            pos2d=pos2d,
            attention_mask=attention_mask,
            dtype=torch.float32,
            device=args.device,
        )

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

            # 计算损失（只对非 padding 位置）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = (shift_labels != tokenizer.pad_token_id)

            # 展平计算损失
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
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

        # 统计
        total_loss += loss.item() * args.accumulation_steps
        total_tokens += shift_mask.sum().item()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            avg_loss = total_loss / (step + 1)
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} avg_loss:{:.3f} lr:{:.8f} eta:{:.1f}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    avg_loss,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch / 60 - spend_time / 60,
                )
            )
            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "avg_loss": avg_loss,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch": epoch,
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
            lora_save_path = os.path.join(args.save_dir, "lora", f"{args.lora_name}_step{epoch * iter_per_epoch + step}.pth")
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
            save_lora(model_to_save, lora_save_path)
            Logger(f"LoRA checkpoint 已保存至: {lora_save_path}")


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA finetune on HuggingFace model with Parallel data")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace模型名称或本地路径")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="若留空则与base_model一致")
    parser.add_argument("--out_dir", type=str, default=os.path.join(root_path, "out"))
    parser.add_argument("--data_path", type=str, default=None, help="本地 JSONL 文件路径")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace 数据集名称")
    parser.add_argument("--hf_subset", type=str, default=None, help="HuggingFace 数据集子集")
    parser.add_argument("--hf_split", type=str, default="train", help="HuggingFace 数据集分片")
    parser.add_argument("--use_streaming", action="store_true", help="使用流式数据集")
    parser.add_argument("--lora_name", type=str, default="hf_parallel_lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=512, help="单个 branch 的最大长度")
    parser.add_argument("--branches_per_sample", type=int, default=8, help="固定每个样本的 branch 数量")
    parser.add_argument("--max_branches_per_sample", type=int, default=0, help="动态 branch 模式下的最大数量（0表示固定模式）")
    parser.add_argument("--max_total_tokens", type=int, default=0, help="每个样本的最大总 token 数（0表示使用 max_seq_len * branches）")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-HF-Parallel-LoRA")
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
    parser.add_argument("--stride", type=int, default=128, help="2D RoPE 中 branch 位置的步长")

    args = parser.parse_args()

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
        args.device = torch.device(args.device)

    # 加载 tokenizer
    tokenizer_path = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # 加载 HuggingFace 模型
    Logger(f"从 {args.base_model} 加载模型（dtype={args.dtype}）")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(args.device)

    rope_target_model = model

    # 应用 2D RoPE
    if args.patch_rope:
        pair_indices = auto_pair_indices(model, args.rope_2d_ratio)
        patch_model_with_interleaved_2d_rope(model, pair_indices)
        Logger(f"已应用 Interleaved 2D RoPE（ratio={args.rope_2d_ratio}, pairs={len(pair_indices)}）")

    # 应用 LoRA
    apply_lora(model, rank=args.lora_rank)
    if args.load_lora and os.path.exists(args.load_lora):
        load_lora(model, args.load_lora)
        Logger(f"已加载 LoRA 权重: {args.load_lora}")

    # 冻结非 LoRA 参数
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

    # 加载 Parallel 数据
    if args.use_streaming:
        Logger("使用流式数据集...")
        train_ds = ParallelPretrainIterableDataset(
            hf_dataset=args.hf_dataset,
            hf_subset=args.hf_subset,
            hf_split=args.hf_split,
        )
        train_sampler = None
    else:
        Logger("使用本地数据集...")
        train_ds = ParallelPretrainDataset(
            data_path=args.data_path,
            hf_dataset=args.hf_dataset,
            hf_subset=args.hf_subset,
            hf_split=args.hf_split,
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None

    # 创建 collator
    if args.max_branches_per_sample > 0:
        # 动态 branch 模式
        collator = ParallelPretrainCollator(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_branches_per_sample=args.max_branches_per_sample,
            max_total_tokens=args.max_total_tokens,
            stride=args.stride,
        )
        Logger(f"使用动态 branch 模式（max_branches={args.max_branches_per_sample}）")
    else:
        # 固定 branch 模式
        collator = ParallelPretrainCollator(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            branches_per_sample=args.branches_per_sample,
            max_total_tokens=args.max_total_tokens,
            stride=args.stride,
        )
        Logger(f"使用固定 branch 模式（branches={args.branches_per_sample}）")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None and not args.use_streaming),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    iter_per_epoch = len(train_loader) if not args.use_streaming else 10000  # 流式数据估算
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cpu":
        ctx = nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))

    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    wandb = None
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=f"HF-Parallel-LoRA-{args.lora_name}")

    Logger("=" * 80)
    Logger("开始训练...")
    Logger("=" * 80)

    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

    if (not ddp) or dist.get_rank() == 0:
        final_path = os.path.join(args.save_dir, "lora", f"{args.lora_name}_final.pth")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        save_lora(model_to_save, final_path)
        Logger(f"LoRA 权重已保存至: {final_path}")
        Logger("训练完成！")
