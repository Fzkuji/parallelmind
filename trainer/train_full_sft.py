import os
import sys

# 避免 tokenizers fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

__package__ = "trainer"
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from parallel.columnar import build_columnar_causal_mask
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from parallel_data.parallel_dataset import ParallelSFTDataset
from parallel_data.parallel_collator import ParallelSFTCollator

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    start_time = time.time()
    if ddp and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        column_mask = build_columnar_causal_mask(batch["time_ids"], batch["attention_mask"]).to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=column_mask,
                position_ids=batch["position_ids"],
                pos2d=batch["pos2d"],
            )
            # 使用collator构造的per-branch labels，无需shift
            logits = outputs.logits.contiguous()
            labels = batch["labels"].contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if hasattr(outputs, "aux_loss"):
                loss = loss + outputs.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_path, 'model'))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    lm_config.vocab_size = vocab_size
    max_positions = args.max_seq_len * args.branches_per_sample
    lm_config.max_position_embeddings = max(lm_config.max_position_embeddings, max_positions)
    lm_config.pad_token_id = tokenizer.pad_token_id
    lm_config.eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    model = MiniMindForCausalLM(lm_config)
    model.resize_token_embeddings(vocab_size)

    # 加载预训练权重
    if args.init_weight:
        ckp = args.init_weight
    else:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default=os.path.join(root_path, "out"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_attention_heads', default=8, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--branches_per_sample', type=int, default=4)
    parser.add_argument('--max_branches_per_sample', type=int, default=None,
                        help='Maximum branches per sample for dynamic mode (enables variable branch count)')
    parser.add_argument('--min_branches_per_sample', type=int, default=1,
                        help='Minimum branches per sample for dynamic mode')
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--pe', type=str, default='rope', choices=['rope', 'fpe'],
                        help='Position encoding: rope (RoPE 2D, branch_stride=128) or fpe (Fourier PE, branch_stride=1)')
    # Position Encoding 参数
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5,
                        help="RoPE 2D中用于branch维度的频率对比例 (0.0-1.0)。默认: 0.5")
    parser.add_argument("--fpe_theta", type=float, default=10000.0, help="Fourier PE基础频率（仅当--pe fpe时使用）")
    parser.add_argument("--fpe_max_positions", type=int, default=512, help="Fourier PE最大位置数")
    parser.add_argument("--fpe_learnable", action="store_true", help="使Fourier PE可学习（默认固定）")
    default_data_path = os.path.join(root_path, "dataset", "sft_512.jsonl")
    parser.add_argument("--data_path", type=str, default=default_data_path)

    # Hugging Face 数据集参数
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face数据集名称")
    parser.add_argument("--hf-subset", type=str, default=None, help="HF数据集子集名称")
    parser.add_argument("--hf-split", type=str, default="train", help="HF数据集分割（默认: train）")
    parser.add_argument("--max-samples", type=int, default=None, help="限制HF数据集样本数量")

    parser.add_argument("--init_weight", type=str, default=None,
                        help='Path to pretrained model checkpoint (e.g., out/pretrain_512.pth)')

    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        pe_type=args.pe,
        rope_2d_ratio=args.rope_2d_ratio,
        fpe_theta=args.fpe_theta,
        fpe_max_positions=args.fpe_max_positions,
        fpe_learnable=args.fpe_learnable,
    )
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(root_path, args.data_path)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(root_path, args.out_dir)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.branches_per_sample * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    if device_type == "cpu":
        ctx = nullcontext()
    else:
        amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)

    # 根据pe_type设置branch_stride
    # FPE使用1, RoPE 2D使用128
    branch_stride = 1 if args.pe == 'fpe' else 128

    # 根据是否提供了 HF 参数来初始化数据集
    if getattr(args, 'hf_dataset', None):
        # 从 Hugging Face 加载
        train_ds = ParallelSFTDataset(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            hf_dataset=args.hf_dataset,
            hf_subset=getattr(args, 'hf_subset', None),
            hf_split=getattr(args, 'hf_split', 'train'),
            max_samples=getattr(args, 'max_samples', None),
        )
    else:
        # 从本地文件加载
        train_ds = ParallelSFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    collator = ParallelSFTCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        pad_to=args.max_seq_len,
        branch_stride=branch_stride,
    )
    train_sampler = DistributedSampler(train_ds, drop_last=True) if ddp else None
    # 动态模式使用max_branches计算batch_size
    branches_for_batch = args.max_branches_per_sample if args.max_branches_per_sample else args.branches_per_sample
    effective_batch_size = args.batch_size * branches_for_batch
    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=collator,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
