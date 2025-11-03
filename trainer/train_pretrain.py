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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from parallel.columnar import build_columnar_causal_mask
from parallel_data.parallel_dataset import ParallelPretrainDataset
from parallel_data.parallel_collator import ParallelPretrainCollator
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    start_time = time.time()
    processed_dataset_samples = 0  # 累计消耗的数据集样本数（对应 jsonl 行数）

    if ddp and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        branch_counts = batch.pop("branch_counts")
        column_mask = build_columnar_causal_mask(batch["time_ids"], batch["attention_mask"]).to(args.device)

        # 更新计数
        batch_dataset_samples = int(branch_counts.sum().item())  # 这个 batch 使用的原始文本数
        processed_dataset_samples += batch_dataset_samples

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
            # 注意：不再需要shift操作，因为collator已经构造了正确的per-branch labels
            # logits[i]直接对应labels[i]（每个token预测同branch的下一个token）
            logits = outputs.logits.contiguous()
            labels = batch["labels"].contiguous()

            # 计算交叉熵损失
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
            # 计算实际的 batch 信息
            batch_seq_len = batch["input_ids"].shape[1]
            # 估算剩余时间（基于已处理的原始文本比例）
            progress_ratio = processed_dataset_samples / total_samples if total_samples > 0 else 0
            if progress_ratio > 0:
                estimated_total_time = spend_time / progress_ratio
                remaining_time = estimated_total_time - spend_time
            else:
                remaining_time = 0

            # 构建日志信息
            log_msg = 'Epoch:[{}/{}]({}/{}) step:{} loss:{:.3f} lr:{:.12f} batch:[{}x{}] epoch_Time:{}min:'.format(
                epoch + 1,
                args.epochs,
                processed_dataset_samples,
                total_samples,
                step,
                loss.item() * args.accumulation_steps,
                optimizer.param_groups[-1]['lr'],
                batch["input_ids"].shape[0],
                batch_seq_len,
                int(remaining_time // 60))

            Logger(log_msg)

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

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
    max_positions = args.max_total_tokens if args.max_total_tokens > 0 else args.max_seq_len * args.branches_per_sample
    lm_config.max_position_embeddings = max(lm_config.max_position_embeddings, max_positions)
    lm_config.pad_token_id = tokenizer.pad_token_id
    lm_config.eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    model = MiniMindForCausalLM(lm_config)
    model.resize_token_embeddings(vocab_size)
    model = model.to(args.device)

    if args.init_weight:
        init_path = args.init_weight
        if not os.path.isabs(init_path):
            init_path = os.path.join(root_path, init_path)
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"init_weight checkpoint not found: {init_path}")
        Logger(f"加载初始权重: {init_path}")
        state_dict = torch.load(init_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
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


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default=os.path.join(root_path, "out"))
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--branches_per_sample', type=int, default=8)
    parser.add_argument('--max_branches_per_sample', type=int, default=None, help='Maximum branches per sample for dynamic mode (1-32). If set, enables dynamic branches.')
    parser.add_argument('--min_branches_per_sample', type=int, default=1, help='Minimum branches per sample for dynamic mode')
    parser.add_argument('--random_time_offset', action='store_true', default=True, help='Enable random time offset for branches during training')
    parser.add_argument('--no_random_time_offset', action='store_false', dest='random_time_offset', help='Disable random time offset')
    parser.add_argument('--batch_by_samples', action='store_true', default=False, help='If True, batch_size refers to number of samples (not texts). Better for dynamic branches.')
    parser.add_argument('--use_moe', default=False, type=bool)
    default_data_path = os.path.join(root_path, "dataset", "pretrain_hq_split.jsonl")
    parser.add_argument("--data_path", type=str, default=default_data_path)

    # Hugging Face 数据集参数
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face数据集名称，如 'HuggingFaceFW/fineweb-edu'")
    parser.add_argument("--hf-subset", type=str, default=None, help="HF数据集子集名称，如 'sample-10BT'")
    parser.add_argument("--hf-split", type=str, default="train", help="HF数据集分割（默认: train）")
    parser.add_argument("--text-column", type=str, default=None, help="HF数据集文本列名（默认自动检测）")
    parser.add_argument("--max-samples", type=int, default=None, help="限制HF数据集样本数量")
    parser.add_argument("--chunk-length", type=int, default=None, help="HF数据集文本切分长度（tokens），用于将长文本切成多个片段，充分利用数据")

    parser.add_argument("--max_total_tokens", type=int, default=4096)
    parser.add_argument("--init_weight", type=str, default=None, help="Warm-start from an existing checkpoint (path to pretrain_*.pth)")
    parser.add_argument("--branch_slice_count", type=int, default=None, help="Divide dataset into N slices (e.g., branch dimension count)")
    parser.add_argument("--branch_slice_index", type=int, default=0, help="Select which slice to train on (0-based)")
    parser.add_argument("--branch_loop_all", action="store_true", help="Iterate through all branch slices sequentially in one run")
    # Position Encoding参数
    parser.add_argument("--pe", type=str, default='rope', choices=['rope', 'fpe'],
                        help="Position encoding type: 'rope' (RoPE 2D) or 'fpe' (Fourier PE + 1D RoPE)")
    parser.add_argument("--fpe_theta", type=float, default=10000.0, help="Fourier PE基础频率（仅当--pe fpe时使用）")
    parser.add_argument("--fpe_max_positions", type=int, default=512, help="Fourier PE最大位置数（branch数量一般很小，默认512）")
    parser.add_argument("--fpe_learnable", action="store_true", help="使Fourier PE可学习（默认固定）")
    args = parser.parse_args()

    # 声明全局变量
    global total_samples, effective_batch_size, iter_per_epoch

    if args.branch_slice_count is not None and args.branch_slice_count <= 0:
        raise ValueError("--branch_slice_count must be positive")

    if args.branch_slice_count is None:
        slice_indices = [None]
    else:
        if args.branch_loop_all:
            slice_indices = list(range(args.branch_slice_count))
        else:
            slice_indices = [args.branch_slice_index]
        for idx in slice_indices:
            if idx is not None and not (0 <= idx < args.branch_slice_count):
                raise ValueError(
                    f"--branch_slice_index must be within [0, {args.branch_slice_count - 1}], got {idx}"
                )

    # 计算 branches_multiplier（如果启用动态模式，使用 max_branches）
    branches_multiplier = args.max_branches_per_sample if args.max_branches_per_sample is not None else args.branches_per_sample

    # batch_by_samples 模式：batch_size 表示 samples 数量，需要乘以 branches 来获取文本数量
    # 默认模式：batch_size 已经表示文本数量（兼容旧行为）
    if args.batch_by_samples:
        if args.max_branches_per_sample is not None:
            avg_branches = (args.min_branches_per_sample + args.max_branches_per_sample) / 2
        else:
            avg_branches = args.branches_per_sample
        effective_batch_size = max(1, int(args.batch_size * avg_branches))
    else:
        effective_batch_size = args.batch_size * branches_multiplier

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        pe_type=args.pe,
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
    # 计算 tokens_per_iter
    if args.batch_by_samples:
        # batch_by_samples 模式：batch_size 表示 samples 数量
        tokens_per_iter = args.batch_size * (args.max_total_tokens if args.max_total_tokens > 0 else args.max_seq_len)
    else:
        # 默认模式：batch_size * branches 表示文本数量
        # 每个 sample 大约有 max_total_tokens 个 tokens
        # 但这只是估计值
        tokens_per_iter = args.batch_size * branches_multiplier * (args.max_total_tokens if args.max_total_tokens > 0 else args.max_seq_len)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
    # rope: 使用stride=128来拉大不同branch的位置距离
    # fpe: 直接用branch索引0,1,2,3...，stride=1
    branch_stride = 1 if args.pe == 'fpe' else 128

    collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=args.branches_per_sample,
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=args.max_branches_per_sample,
        min_branches_per_sample=args.min_branches_per_sample,
        random_time_offset=args.random_time_offset,
        interleave_branches=True,
        branch_stride=branch_stride,
    )
    if args.batch_by_samples:
        collator.target_samples = args.batch_size
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    total_slices = len(slice_indices)
    for slice_pos, slice_idx in enumerate(slice_indices):
        dataset_kwargs = {}
        slice_desc = "full"
        if args.branch_slice_count is not None and slice_idx is not None:
            dataset_kwargs = {"slice_count": args.branch_slice_count, "slice_index": slice_idx}
            slice_desc = f"{slice_idx + 1}/{args.branch_slice_count}"
            Logger(f"=== 开始训练 branch slice {slice_desc} ===")

        # 根据是否提供了 HF 参数来初始化数据集
        if getattr(args, 'hf_dataset', None):
            # 从 Hugging Face 加载
            train_ds = ParallelPretrainDataset(
                hf_dataset=args.hf_dataset,
                hf_subset=getattr(args, 'hf_subset', None),
                hf_split=getattr(args, 'hf_split', 'train'),
                text_column=getattr(args, 'text_column', None),
                max_samples=getattr(args, 'max_samples', None),
                chunk_length=getattr(args, 'chunk_length', None),
                tokenizer=tokenizer if getattr(args, 'chunk_length', None) else None,
                **dataset_kwargs
            )
        else:
            # 从本地文件加载
            train_ds = ParallelPretrainDataset(args.data_path, **dataset_kwargs)
        collator._buffer.clear()
        drop_last = False if args.batch_by_samples or args.max_branches_per_sample is not None else True
        train_sampler = DistributedSampler(train_ds, drop_last=drop_last) if ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=effective_batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=(train_sampler is None),  # 只有在没有sampler时才shuffle
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=collator,
        )

        total_samples = len(train_ds)
        iter_per_epoch = len(train_loader)

        if iter_per_epoch == 0:
            Logger("当前切片没有样本，跳过。")
            continue

        for epoch in range(args.epochs):
            train_epoch(epoch, wandb)

        if args.branch_slice_count is not None and slice_idx is not None:
            Logger(f"=== 完成 branch slice {slice_desc} ===")
