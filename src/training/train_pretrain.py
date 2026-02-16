import os
import sys

# 避免 tokenizers fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

__package__ = "trainer"
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import argparse
import json
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
import random
from contextlib import nullcontext
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from src.model.columnar import build_columnar_causal_mask, build_flex_columnar_mask
from src.data.parallel_dataset import ParallelPretrainDataset, ParallelPretrainIterableDataset
from src.data.parallel_collator import ParallelPretrainCollator
from src.model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')

# Will be set to True when --use_flex_attention is passed
_use_flex_attn = False


def _build_mask(batch, device):
    """Build attention mask: FlexAttention BlockMask or dense SDPA mask.

    When using FlexAttention, pads all batch tensors to the BlockMask's padded_len
    (multiple of 128) and returns the BlockMask.
    """
    time_ids = batch["time_ids"]
    attention_mask = batch["attention_mask"]
    if _use_flex_attn:
        block_mask, padded_len = build_flex_columnar_mask(time_ids, attention_mask)
        seq_len = time_ids.shape[1]
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            for key in ("input_ids", "labels"):
                pad_val = 0 if key == "input_ids" else -100
                batch[key] = torch.nn.functional.pad(batch[key], (0, pad_size), value=pad_val)
            batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, pad_size), value=0)
            batch["position_ids"] = torch.nn.functional.pad(batch["position_ids"], (0, pad_size), value=0)
            batch["time_ids"] = torch.nn.functional.pad(batch["time_ids"], (0, pad_size), value=-1)
            # pos2d: [B, S, 2]
            batch["pos2d"] = torch.nn.functional.pad(batch["pos2d"], (0, 0, 0, pad_size), value=0)
        return block_mask
    return build_columnar_causal_mask(time_ids, attention_mask).to(device)


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def _append_jsonl(path: str, obj: dict):
    """Append a JSON object as one line to a JSONL file. Fail-silent in distributed context."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def print_first_batch_sample(batch, tokenizer):
    """打印第一个batch的示例，帮助验证数据和labels的正确性"""
    Logger("=" * 80)
    Logger("第一个 Batch 数据示例:")
    Logger("=" * 80)

    # 显示batch维度信息
    batch_size = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]
    Logger(f"Batch shape: {batch_size} samples × {seq_len} tokens")
    Logger(f"Branch counts: {batch['branch_counts'].tolist()}")
    Logger("")

    # 显示第一个sample的详细信息
    sample_idx = 0
    Logger(f"Sample {sample_idx} 详细信息:")
    Logger(f"  该sample有 {batch['branch_counts'][sample_idx]} 个branches")

    # 找出该sample的所有branch
    pos2d = batch["pos2d"][sample_idx]  # [seq_len, 2]
    attention_mask = batch["attention_mask"][sample_idx]  # [seq_len]
    input_ids = batch["input_ids"][sample_idx]  # [seq_len]
    labels = batch["labels"][sample_idx]  # [seq_len]

    # 获取所有unique branch positions
    valid_mask = attention_mask == 1
    branch_positions = pos2d[valid_mask, 0].unique().tolist()

    Logger(f"  Branch positions: {branch_positions[:5]}{'...' if len(branch_positions) > 5 else ''}")
    Logger("")

    # 显示前2个branch的内容
    for branch_idx, branch_pos in enumerate(branch_positions[:2]):
        # 找到该branch的所有tokens
        branch_mask = (pos2d[:, 0] == branch_pos) & valid_mask
        branch_token_ids = input_ids[branch_mask]
        branch_labels = labels[branch_mask]

        # 解码文本
        text = tokenizer.decode(branch_token_ids[:50], skip_special_tokens=True)  # 只显示前50个token
        Logger(f"  Branch {branch_idx} (pos={branch_pos}):")
        Logger(f"    Tokens: {branch_token_ids[:10].tolist()}{'...' if len(branch_token_ids) > 10 else ''}")
        Logger(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        # 验证labels的正确性：显示前几个token和它们的label
        Logger(f"    Label验证 (input → label):")
        for i in range(min(5, len(branch_token_ids) - 1)):
            input_token = branch_token_ids[i].item()
            label_token = branch_labels[i].item()
            next_token = branch_token_ids[i + 1].item()

            # label应该等于下一个token
            status = "✓" if label_token == next_token else "✗"
            Logger(f"      {status} token[{i}]={input_token} → label={label_token} (expected={next_token})")
        Logger("")

    Logger("=" * 80)
    Logger("")


def validate(val_loader, epoch, global_step):
    """在验证集上评估模型（DDP：各rank均参与计算并做全局聚合）"""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    rank_loss = 0.0
    rank_batches = 0

    if (not ddp) or dist.get_rank() == 0:
        Logger("\n" + "="*80)
        Logger(f"开始验证 (Epoch {epoch + 1}, Step {global_step})...")

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            _ = batch.pop("branch_counts")
            column_mask = _build_mask(batch, args.device)

            with ctx:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=column_mask,
                    position_ids=batch["position_ids"],
                    pos2d=batch["pos2d"],
                )
                logits = outputs.logits.contiguous()
                labels = batch["labels"].contiguous()

                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                if hasattr(outputs, "aux_loss"):
                    loss = loss + outputs.aux_loss

            rank_loss += float(loss.item())
            rank_batches += 1

    # 全局聚合
    if ddp:
        import torch as _torch
        t = _torch.tensor([rank_loss, float(rank_batches)], device=args.device, dtype=_torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_batches = t[0].item(), int(t[1].item())
    else:
        total_loss, total_batches = rank_loss, rank_batches

    avg_val_loss = (total_loss / max(1, total_batches)) if total_batches > 0 else 0.0

    if (not ddp) or dist.get_rank() == 0:
        Logger(f"验证完成 - 平均loss: {avg_val_loss:.4f}")
        Logger("="*80 + "\n")
        # 记录到本地 JSONL 日志
        if getattr(args, 'val_log_path', None):
            _append_jsonl(args.val_log_path, {
                "time": datetime.utcnow().isoformat(),
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "val_loss": float(avg_val_loss),
            })

    model.train()

    # 记录到 wandb（仅rank0）
    if wandb is not None and ((not ddp) or dist.get_rank() == 0):
        wandb.log({
            "val_loss": avg_val_loss,
            "val_step": global_step,
        })

    return avg_val_loss


def train_epoch(epoch, wandb):
    global last_val_samples, last_val_tokens, trained_tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    start_time = time.time()
    processed_samples = 0  # 累计消耗的数据集样本数（对应 jsonl 行数 = branches 数量）
    # 在一次optimizer step窗口内累计统计（用于按优化步打印）
    opt_loss_accum = 0.0
    opt_micro_count = 0
    # 在一个epoch内累计的token（仅用于日志展示，真正的全局累计用全局变量trained_tokens）
    epoch_tokens_since_log = 0

    # 验证相关：跟踪上次验证时的样本数（用于基于样本数的验证间隔）
    if epoch == 0:
        last_val_samples = 0

    if ddp and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    is_rank0 = (not ddp) or dist.get_rank() == 0
    pbar = tqdm(enumerate(train_loader), total=iter_per_epoch,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                disable=not is_rank0, dynamic_ncols=True)
    for step, batch in pbar:
        # 在第一个epoch的第一个step，打印batch示例
        if epoch == 0 and step == 0:
            print_first_batch_sample(batch, tokenizer)

        batch = {k: v.to(args.device) for k, v in batch.items()}
        branch_counts = batch.pop("branch_counts")
        column_mask = _build_mask(batch, args.device)

        # 更新计数：1 sample = 1 branch = 1 个 JSONL 行
        batch_samples = int(branch_counts.sum().item())  # 这个 batch 消耗的样本数（JSONL行数）
        processed_samples += batch_samples

        # 计算学习率（streaming 模式下估算总步数）
        if iter_per_epoch is not None:
            current_step = epoch * iter_per_epoch + step
            total_steps = args.epochs * iter_per_epoch
        else:
            # IterableDataset: 估算步数（假设每个 epoch 大约相同步数）
            # 第一个 epoch 无法估算，使用简单的步数
            current_step = step
            # 估算总步数：如果 max_samples 提供了，可以粗略估算
            if args.max_samples:
                estimated_iter = args.max_samples // effective_batch_size
                total_steps = estimated_iter * args.epochs
            else:
                total_steps = 10000 * args.epochs  # 默认估算值

        lr = get_lr(current_step, total_steps, args.learning_rate)
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

            # 计算本micro-batch的有效token数（labels!=-100）
            batch_valid_tokens = int((labels != -100).sum().item())
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            if hasattr(outputs, "aux_loss"):
                loss = loss + outputs.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # 累计全局token计数（按DDP进程总和口径）。
        # 使用all_reduce聚合每个rank的有效token，确保各rank上的累计值一致。
        global_batch_tokens = batch_valid_tokens
        if ddp:
            import torch as _torch
            _t = _torch.tensor([batch_valid_tokens], device=args.device, dtype=_torch.long)
            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
            global_batch_tokens = int(_t.item())
        # 记录到全局与本epoch窗口
        epoch_tokens_since_log += global_batch_tokens
        # 使用全局变量记录训练至今累计token数
        trained_tokens += global_batch_tokens

        # 记录优化步窗口统计（用未缩放loss，便于与acc=1对齐）
        opt_loss_accum += loss.item() * args.accumulation_steps
        opt_micro_count += 1

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            # 更新进度条
            opt_step = (step + 1) // args.accumulation_steps
            if is_rank0:
                cur_loss = opt_loss_accum / max(1, opt_micro_count)
                pbar.set_postfix(loss=f"{cur_loss:.3f}",
                                 lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
                                 tokens=f"{trained_tokens:,}")

            # 仅在优化步边界按优化步频率打印日志
            if opt_step % max(1, args.log_interval) == 0:
                spend_time = time.time() - start_time
                # 计算实际的 batch 信息（当前微步）
                batch_seq_len = batch["input_ids"].shape[1]

                # 在 DDP 模式下，显示全局累计处理的样本数（所有GPU的总和）
                if ddp:
                    global_processed_samples = processed_samples * dist.get_world_size()
                else:
                    global_processed_samples = processed_samples

                # 估算剩余时间：优先使用样本进度（训练时长 / 进度）
                # 若 total_samples 可用，则用 processed_samples/total_samples 计算；否则退化到优化步口径
                epsilon = 1e-9
                if total_samples != float('inf') and total_samples > 0:
                    progress_ratio = min(1.0, max(0.0, global_processed_samples / max(total_samples, 1)))
                    if progress_ratio > epsilon:
                        estimated_total_time = spend_time / progress_ratio
                        remaining_time = max(0.0, estimated_total_time - spend_time)
                    else:
                        remaining_time = 0.0
                elif iter_per_epoch is not None:
                    import math as _math
                    opt_iters_per_epoch = max(1, _math.ceil(iter_per_epoch / max(1, args.accumulation_steps)))
                    progress_ratio = min(1.0, opt_step / max(1, args.epochs * opt_iters_per_epoch))
                    if progress_ratio > epsilon:
                        estimated_total_time = spend_time / progress_ratio
                        remaining_time = max(0.0, estimated_total_time - spend_time)
                    else:
                        remaining_time = 0.0
                else:
                    remaining_time = 0.0

                # 构建日志信息（按优化步口径）
                total_samples_str = str(int(total_samples)) if total_samples != float('inf') else 'streaming'
                avg_unscaled_loss = opt_loss_accum / max(1, opt_micro_count)
                # 以分钟显示，避免过早显示为0，使用四舍五入
                remaining_min = int((remaining_time / 60.0) + 0.5)
                log_msg = 'Epoch:[{}/{}]({}/{}) step:{} loss:{:.3f} lr:{:.12f} batch:[{}x{}] tokens_trained:{} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    global_processed_samples,
                    total_samples_str,
                    opt_step,
                    avg_unscaled_loss,
                    optimizer.param_groups[-1]['lr'],
                    batch["input_ids"].shape[0],
                    batch_seq_len,
                    trained_tokens,
                    remaining_min)

                # 不再打印到控制台（tqdm 进度条已覆盖），仅写文件日志
                if ((not ddp) or dist.get_rank() == 0) and getattr(args, 'train_log_path', None):
                    try:
                        world = dist.get_world_size() if ddp else 1
                    except Exception:
                        world = 1
                    _append_jsonl(args.train_log_path, {
                        "time": datetime.utcnow().isoformat(),
                        "epoch": int(epoch + 1),
                        "opt_step": int(opt_step),
                        "processed_samples": int(global_processed_samples),
                        "total_samples": (int(total_samples) if total_samples != float('inf') else None),
                        "batch_size": int(batch["input_ids"].shape[0]),
                        "seq_len": int(batch_seq_len),
                        "loss": float(avg_unscaled_loss),
                        "lr": float(optimizer.param_groups[-1]['lr']),
                        "tokens_trained": int(trained_tokens),
                        "remaining_min": int(remaining_min),
                        "world_size": int(world),
                    })

                if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                    # 与控制台一致的ETA估算（分钟）
                    estimated_epoch_time = remaining_min

                    wandb.log({
                        "loss": avg_unscaled_loss,
                        "lr": optimizer.param_groups[-1]['lr'],
                        "epoch_Time": estimated_epoch_time
                    })

                # 重置优化步窗口统计
                opt_loss_accum = 0.0
                opt_micro_count = 0

        # 验证
        if val_loader is not None:
            # 先本地计算，再在DDP下由rank0统一广播是否验证，避免不同rank条件不一致导致的死锁
            should_validate = False

            # 方式0：基于token数的验证间隔（最高优先级）
            if args.val_interval_tokens > 0:
                current_tokens = trained_tokens
                if (not ddp) or dist.get_rank() == 0:
                    if current_tokens - last_val_tokens >= args.val_interval_tokens:
                        should_validate = True
                        last_val_tokens = current_tokens

            # 方式1：基于优化步数的验证间隔
            if args.val_interval > 0 and not should_validate:
                opt_step = (step + 1) // args.accumulation_steps
                # 使用rank0做决策，避免步数/样本不一致
                if (not ddp) or dist.get_rank() == 0:
                    if opt_step % args.val_interval == 0:
                        should_validate = True

            # 方式2：基于样本数的验证间隔（优先级次之）
            if args.val_interval_samples > 0 and not should_validate:
                # 在 DDP 模式下，使用估算的全局样本数
                current_samples = processed_samples * (dist.get_world_size() if ddp else 1)
                if (not ddp) or dist.get_rank() == 0:
                    if current_samples - last_val_samples >= args.val_interval_samples:
                        should_validate = True
                        last_val_samples = current_samples

            # 在DDP下将rank0的should_validate广播给所有rank，确保一致进入/跳过验证
            if ddp:
                import torch as _torch
                flag = _torch.tensor([1 if should_validate else 0], device=args.device, dtype=_torch.int)
                dist.broadcast(flag, src=0)
                should_validate = bool(int(flag.item()))

            if should_validate:
                opt_step = (step + 1) // args.accumulation_steps
                global_step = epoch * (iter_per_epoch // args.accumulation_steps if iter_per_epoch else 0) + opt_step
                # 所有rank同时验证并做全局聚合，避免其它rank在barrier处等待导致NCCL watchdog超时
                validate(val_loader, epoch, global_step)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            _save_checkpoint(model, lm_config, args, final=False)


def _save_checkpoint(model, lm_config, args, final=False):
    """Save model checkpoint. final=True saves as the definitive model file."""
    model.eval()
    moe_path = '_moe' if lm_config.use_moe else ''
    if final:
        ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    else:
        ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}_ckpt.pth'

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state_dict = {k: v.half() for k, v in state_dict.items()}
    checkpoint = {
        'model_state_dict': state_dict,
        'tokenizer_path': lm_config.tokenizer_path,
        'vocab_size': lm_config.vocab_size,
    }
    torch.save(checkpoint, ckp)
    model.train()
    if final:
        # 训练完成后删除中间 checkpoint
        ckpt_path = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}_ckpt.pth'
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def _load_tokenizer(path):
    """Load tokenizer with fallback: AutoTokenizer -> PreTrainedTokenizerFast."""
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except (ValueError, OSError):
        return PreTrainedTokenizerFast.from_pretrained(path)


def init_model(lm_config):
    # 加载 tokenizer：优先使用命令行指定的，否则使用默认的 minimind tokenizer
    if hasattr(args, 'tokenizer') and args.tokenizer:
        tokenizer_path = args.tokenizer
    else:
        # 默认使用 minimind tokenizer
        tokenizer_path = os.path.join(root_path, 'model')
    tokenizer = _load_tokenizer(tokenizer_path)

    # 存储 tokenizer 路径，用于保存到 checkpoint
    lm_config.tokenizer_path = tokenizer_path

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
        checkpoint = torch.load(init_path, map_location=args.device)

        # 兼容新旧格式：新格式包含 model_state_dict，旧格式直接是 state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

    # 打印模型大小信息（总参数、可训练参数、近似权重占用内存）
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for b in model.buffers())
        # 估算内存：按当前参数dtype计算
        try:
            first_param = next(model.parameters())
            bytes_per_elem = first_param.element_size()
        except StopIteration:
            bytes_per_elem = 4  # 默认float32
        params_mem_mb = (total_params * bytes_per_elem) / (1024 ** 2)
        buffers_mem_mb = (total_buffers * bytes_per_elem) / (1024 ** 2)
        total_mem_mb = params_mem_mb + buffers_mem_mb

        Logger(f'LLM总参数量：{total_params / 1e6:.3f} 百万')
        Logger(f'LLM可训练总参数量：{trainable_params / 1e6:.3f} 百万')
        Logger(f'权重占用(含buffers)≈ {total_mem_mb:.1f} MB  (参数: {params_mem_mb:.1f} MB, buffer: {buffers_mem_mb:.1f} MB, 每元素{bytes_per_elem}字节)')
    except Exception:
        # 回退：至少打印可训练参数量
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
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_attention_heads', default=8, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--num_key_value_heads', default=None, type=int,
                        help='Number of key-value heads for GQA. Default: use MiniMindConfig default')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--branches_per_sample', type=int, default=8)
    parser.add_argument('--max_branches_per_sample', type=int, default=None, help='Maximum branches per sample for dynamic mode (1-32). If set, enables dynamic branches.')
    parser.add_argument('--min_branches_per_sample', type=int, default=1, help='Minimum branches per sample for dynamic mode')
    # Validation-specific branch controls
    parser.add_argument('--val_branches_per_sample', type=int, default=None, help='Validation: fixed branches per sample. Ignored if --val_max_branches_per_sample is set.')
    parser.add_argument('--val_max_branches_per_sample', type=int, default=None, help='Validation: max branches per sample (enables dynamic mode when set).')
    parser.add_argument('--val_min_branches_per_sample', type=int, default=None, help='Validation: min branches per sample (used in dynamic mode). Defaults to training min if unset.')
    parser.add_argument('--random_time_offset', action='store_true', default=True, help='Enable random time offset for branches during training')
    parser.add_argument('--no_random_time_offset', action='store_false', dest='random_time_offset', help='Disable random time offset')
    parser.add_argument('--batch_by_samples', action='store_true', default=False, help='If True, batch_size refers to number of samples (not texts). Better for dynamic branches.')
    parser.add_argument('--use_moe', default=False, type=bool)
    default_data_path = os.path.join(root_path, "dataset", "pretrain_hq_split.jsonl")
    parser.add_argument("--data_path", type=str, default=default_data_path)

    # Tokenizer 参数
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer 路径或HuggingFace模型名称。默认使用 model/ 目录下的 minimind tokenizer。"
                             "例如: --tokenizer gpt2 或 --tokenizer Qwen/Qwen2.5-0.5B-Instruct")

    # Hugging Face 数据集参数
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face数据集名称，如 'HuggingFaceFW/fineweb-edu'")
    parser.add_argument("--hf-subset", type=str, default=None, help="HF数据集子集名称，如 'sample-10BT'")
    parser.add_argument("--hf-split", type=str, default="train", help="HF数据集分割（默认: train）")
    parser.add_argument("--text-column", type=str, default=None, help="HF数据集文本列名（默认自动检测）")
    parser.add_argument("--max-samples", type=int, default=None, help="限制HF数据集样本数量")
    parser.add_argument("--chunk-length", type=int, default=None, help="HF数据集文本切分长度（tokens），用于将长文本切成多个片段，充分利用数据")
    parser.add_argument("--offline", action="store_true", help="离线模式：使用已缓存的数据集，不从网络验证和下载")

    parser.add_argument("--max_total_tokens", type=int, default=4096)
    parser.add_argument("--init_weight", type=str, default=None, help="Warm-start from an existing checkpoint (path to pretrain_*.pth)")
    parser.add_argument("--branch_slice_count", type=int, default=None, help="Divide dataset into N slices (e.g., branch dimension count)")
    parser.add_argument("--branch_slice_index", type=int, default=0, help="Select which slice to train on (0-based)")
    parser.add_argument("--branch_loop_all", action="store_true", help="Iterate through all branch slices sequentially in one run")
    # Position Encoding参数
    parser.add_argument("--pe", type=str, default='rope', choices=['rope', 'fpe'],
                        help="Position encoding type: 'rope' (RoPE 2D) or 'fpe' (Fourier PE + 1D RoPE)")
    parser.add_argument("--rope_2d_ratio", type=float, default=0.5,
                        help="RoPE 2D中用于branch维度的频率对比例 (0.0-1.0)。"
                             "例如: 0.5表示50%%的频率维度用于2D位置编码(branch+time)，50%%用于1D(time only)。"
                             "默认: 0.5")
    parser.add_argument("--fpe_theta", type=float, default=10000.0, help="Fourier PE基础频率（仅当--pe fpe时使用）")
    parser.add_argument("--fpe_max_positions", type=int, default=512, help="Fourier PE最大位置数（branch数量一般很小，默认512）")
    parser.add_argument("--fpe_learnable", action="store_true", help="使Fourier PE可学习（默认固定）")
    parser.add_argument("--use_flex_attention", action="store_true",
                        help="使用 FlexAttention 替代 SDPA（需 PyTorch 2.5+, CUDA）。"
                             "使用 BlockMask 避免 O(n^2) 显存占用，适合长序列多分支训练。")

    # 验证集参数
    parser.add_argument("--val_samples", type=int, default=0,
                        help="验证集样本数（从训练数据中随机抽取）。设置为0则不使用验证集。默认: 0")
    parser.add_argument("--val_interval", type=int, default=0,
                        help="验证间隔（优化步数）。每隔多少个优化步进行一次验证。设置为0则禁用基于步数的验证。默认: 0")
    parser.add_argument("--val_interval_samples", type=int, default=0,
                        help="验证间隔（样本数）。每训练多少个样本进行一次验证。"
                             "例如: --val_interval_samples 204800 表示每训练20.48万个样本验证一次。"
                             "此参数优先级高于--val_interval。设置为0则禁用。默认: 0")
    parser.add_argument("--val_interval_tokens", type=int, default=0,
                        help="验证间隔（token数）。每训练多少个有效token触发一次验证。"
                             "有效token指labels!=-100的token数。设置为0则禁用。")

    args = parser.parse_args()

    # 如果启用离线模式，立即设置环境变量（在加载数据集之前）
    if getattr(args, 'offline', False):
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

    # 声明全局变量
    global total_samples, effective_batch_size, iter_per_epoch, tokenizer, val_loader, last_val_samples, trained_tokens, last_val_tokens
    last_val_samples = 0   # 基于样本数的验证间隔
    trained_tokens = 0     # 全局累计训练过的有效token数量（labels!=-100）
    last_val_tokens = 0    # 上次验证时的累计token数

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

    config_kwargs = dict(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        pe_type=args.pe,
        rope_2d_ratio=args.rope_2d_ratio,
        fpe_theta=args.fpe_theta,
        fpe_max_positions=args.fpe_max_positions,
        fpe_learnable=args.fpe_learnable,
        use_flex_attention=getattr(args, 'use_flex_attention', False),
    )
    if args.num_key_value_heads is not None:
        config_kwargs['num_key_value_heads'] = args.num_key_value_heads
    lm_config = MiniMindConfig(**config_kwargs)

    # Set global flag for FlexAttention mask building
    _use_flex_attn = getattr(args, 'use_flex_attention', False)
    if _use_flex_attn:
        print("FlexAttention enabled: using BlockMask instead of dense O(n^2) mask")

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

    # 本地 JSONL 日志目录：out/logs/{YYYYmmdd-HHMMSS}-{hash8}
    try:
        cmd_str = " ".join(sys.argv)
        run_hash = hashlib.md5(cmd_str.encode("utf-8")).hexdigest()[:8]
        run_ts = time.strftime('%Y%m%d-%H%M%S')
        base_logs_dir = os.path.join(root_path, 'out', 'logs')
        args.log_run_dir = os.path.join(base_logs_dir, f"{run_ts}-{run_hash}")
        os.makedirs(args.log_run_dir, exist_ok=True)
        args.train_log_path = os.path.join(args.log_run_dir, 'train_loss.jsonl')
        args.val_log_path = os.path.join(args.log_run_dir, 'val_loss.jsonl')
        # 保存一次运行的命令行，方便区分不同运行
        with open(os.path.join(args.log_run_dir, 'run_cmd.txt'), 'w', encoding='utf-8') as f:
            f.write(cmd_str + "\n")
    except Exception:
        args.log_run_dir = None
        args.train_log_path = None
        args.val_log_path = None

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
            # 从 Hugging Face 加载 - 使用 IterableDataset 真正的 streaming
            train_ds = ParallelPretrainIterableDataset(
                hf_dataset=args.hf_dataset,
                hf_subset=getattr(args, 'hf_subset', None),
                hf_split=getattr(args, 'hf_split', 'train'),
                text_column=getattr(args, 'text_column', None),
                max_samples=getattr(args, 'max_samples', None),
                chunk_length=getattr(args, 'chunk_length', None),
                tokenizer=tokenizer if getattr(args, 'chunk_length', None) else None,
                offline=getattr(args, 'offline', False),
            )
            is_iterable = True
            val_ds = None  # IterableDataset 不支持简单的分割
        else:
            # 从本地文件加载 - 使用普通 Dataset
            # 重要：确保训练集与验证集互不挤占配额
            # 若指定了 max_samples(训练配额) 与 val_samples(验证配额)，
            # 则在构建完整数据集时需要读取 max_samples + val_samples 个样本，
            # 之后再切分，保证训练集有足量的 max_samples。
            _max_train = getattr(args, 'max_samples', None)
            _val_needed = getattr(args, 'val_samples', 0) or 0
            _max_for_full = (_max_train + _val_needed) if (_max_train is not None and _val_needed > 0) else _max_train

            full_ds = ParallelPretrainDataset(
                args.data_path,
                max_samples=_max_for_full,
                **dataset_kwargs
            )
            is_iterable = False

            # 分割训练集和验证集
            if args.val_samples > 0 and len(full_ds) > args.val_samples:
                # 随机分割
                total_size = len(full_ds)
                indices = list(range(total_size))
                random.shuffle(indices)

                val_indices = indices[:args.val_samples]
                train_indices = indices[args.val_samples:]

                # 如果限制了 full_ds 的 max_samples = max_train + val_samples，
                # 则 train_indices 的长度应接近 max_train；若数据真实总量不足，会更少。
                train_ds = Subset(full_ds, train_indices)
                val_ds = Subset(full_ds, val_indices)

                Logger(f"数据集分割: 训练集 {len(train_ds):,} 样本, 验证集 {len(val_ds):,} 样本")
                # 如果因为源数据不足，导致训练集小于期望的 max_samples，给出提示
                if _max_train is not None and len(train_ds) < _max_train:
                    Logger(f"提示: 可用训练样本数 {len(train_ds):,} 小于期望的 {int(_max_train):,}。请确认数据量是否足够 (max-samples + val_samples)。")
            else:
                train_ds = full_ds
                val_ds = None
                if args.val_samples > 0:
                    Logger(f"警告: 数据集大小 ({len(full_ds):,}) 小于等于验证集大小 ({args.val_samples:,})，不进行分割")

        collator._buffer.clear()
        drop_last = False if args.batch_by_samples or args.max_branches_per_sample is not None else True

        # IterableDataset 不支持 sampler（内部已处理 DDP 分片）
        if is_iterable:
            train_sampler = None
            shuffle = False
        else:
            train_sampler = DistributedSampler(train_ds, drop_last=drop_last) if ddp else None
            shuffle = (train_sampler is None)

        train_loader = DataLoader(
            train_ds,
            batch_size=effective_batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=collator,
        )

        # 创建验证集 DataLoader
        if val_ds is not None:
            # 为验证集创建一个新的 collator（避免与训练 collator 的 buffer 冲突）
            # 支持单独控制验证时的分支数量（固定或动态）
            val_branches = args.val_branches_per_sample if args.val_branches_per_sample is not None else args.branches_per_sample
            val_min_branches = args.val_min_branches_per_sample if args.val_min_branches_per_sample is not None else args.min_branches_per_sample
            # 若设置了 val_max，则启用动态；否则默认沿用训练侧的 max（保持旧行为）。
            val_max_branches = args.val_max_branches_per_sample if args.val_max_branches_per_sample is not None else args.max_branches_per_sample

            val_collator = ParallelPretrainCollator(
                tokenizer,
                branches_per_sample=val_branches,
                pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
                max_branches_per_sample=val_max_branches,
                min_branches_per_sample=val_min_branches,
                random_time_offset=args.random_time_offset,
                interleave_branches=True,
                branch_stride=branch_stride,
            )
            if args.batch_by_samples:
                # 验证目标样本数 = 训练的2倍
                val_collator.target_samples = max(1, args.batch_size * 2)

            # 记录验证分支配置
            if val_max_branches is not None:
                Logger(f"验证分支配置: 动态范围 [{val_min_branches}, {val_max_branches}]（min==max 将固定为该值）")
            else:
                Logger(f"验证分支配置: 固定 {val_branches} 个分支/样本")

            # 验证批大小 = 训练effective_batch_size的2倍
            _val_effective_batch_size = max(1, effective_batch_size * 2)
            val_loader = DataLoader(
                val_ds,
                batch_size=_val_effective_batch_size,
                pin_memory=True,
                drop_last=False,  # 验证集不丢弃最后一个batch
                shuffle=False,  # 验证集不需要shuffle
                num_workers=args.num_workers,
                collate_fn=val_collator,
            )
            Logger(f"验证集 DataLoader 创建完成，共 {len(val_loader)} 个 batch")
        else:
            val_loader = None

        # IterableDataset 没有长度，只能在训练时统计
        if is_iterable:
            total_samples = args.max_samples if args.max_samples else float('inf')
            iter_per_epoch = None  # 未知，训练时动态计算
            if ddp_local_rank == 0:
                Logger(f"开始训练（streaming 模式，数据集大小未知）")
        else:
            dataset_total = len(train_ds)
            total_samples = dataset_total  # 全局数据集总样本数（不是单个GPU的）
            iter_per_epoch = len(train_loader)
            if ddp_local_rank == 0:
                if ddp:
                    world_size = dist.get_world_size()
                    per_gpu_samples = dataset_total // world_size
                    Logger(f"数据集总样本数: {dataset_total:,}, 每个GPU分配: ~{per_gpu_samples:,}, 迭代次数: {iter_per_epoch}")
                    Logger(f"日志进度 (processed/total): processed是全局累计处理数(所有GPU总和), total是数据集总样本数")
                else:
                    Logger(f"数据集总样本数: {dataset_total:,}, 迭代次数: {iter_per_epoch}")
            if iter_per_epoch == 0:
                Logger("当前切片没有样本，跳过。")
                continue

        for epoch in range(args.epochs):
            train_epoch(epoch, wandb)

        # 训练完成，保存最终模型 (只有 rank 0)
        if not ddp or dist.get_rank() == 0:
            _save_checkpoint(model, lm_config, args, final=True)
            Logger("Training complete. Final model saved.")

        if args.branch_slice_count is not None and slice_idx is not None:
            Logger(f"=== 完成 branch slice {slice_desc} ===")
