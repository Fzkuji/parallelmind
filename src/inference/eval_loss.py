import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

# repo root on path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from src.model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from src.data.parallel_dataset import ParallelPretrainDataset
from src.data.parallel_collator import ParallelPretrainCollator
from src.model.columnar import build_columnar_causal_mask, build_flex_columnar_mask


class _ChunkedHFDataset(Dataset):
    """Map-style dataset: 预加载 HF 数据，token packing 成固定长度 chunks"""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def _get_cache_path(args, max_chunks):
    """生成 tokenized chunks 的缓存文件路径"""
    import hashlib
    key_parts = [
        args.hf_dataset or '',
        getattr(args, 'hf_subset', '') or '',
        str(getattr(args, 'chunk_length', 0)),
        str(max_chunks or 0),
        args.tokenizer or '',
    ]
    key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()[:12]
    cache_dir = os.path.join(root_path, '.cache', 'eval_chunks')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"chunks_{key}.pt")


def _load_hf_map_dataset(args, tokenizer, max_chunks=None):
    """从 HF 加载数据（非 streaming），tokenize + pack 成 map-style dataset

    优化:
    - 磁盘缓存: tokenized chunks 缓存到 .cache/eval_chunks/，后续直接读取
    - DDP: 只 rank 0 加载/tokenize，然后 broadcast 给其他 rank
    """
    from datasets import load_dataset
    import logging

    ddp = getattr(args, 'ddp', False)
    max_chunks = int(max_chunks) if max_chunks else None
    cache_path = _get_cache_path(args, max_chunks)

    # DDP: 只 rank 0 负责加载，其他 rank 等待
    is_main = (not ddp) or (dist.get_rank() == 0)

    chunks = None

    if is_main:
        # 尝试从磁盘缓存读取
        if os.path.exists(cache_path):
            Logger(f"从缓存加载 tokenized chunks: {cache_path}", rank0_only=True, ddp=ddp)
            chunks = torch.load(cache_path, map_location='cpu', weights_only=False)
            Logger(f"缓存加载完成: {len(chunks)} chunks", rank0_only=True, ddp=ddp)
        else:
            Logger("加载 HF 数据集（streaming → map-style）...", rank0_only=True, ddp=ddp)

            load_kwargs = dict(split=getattr(args, 'hf_split', 'train'), streaming=True)
            if getattr(args, 'hf_subset', None):
                load_kwargs['name'] = args.hf_subset
            raw_ds = load_dataset(args.hf_dataset, **load_kwargs)

            # 自动检测文本列
            first_item = next(iter(raw_ds))
            text_col = getattr(args, 'text_column', None)
            if text_col is None:
                cols = list(first_item.keys())
                for c in ['text', 'content', 'document', 'sentence', 'data']:
                    if c in cols:
                        text_col = c
                        break
                if text_col is None:
                    text_col = cols[0]

            chunk_length = getattr(args, 'chunk_length', None)

            # 重新创建迭代器（因为上面消耗了一个 item）
            raw_ds = load_dataset(args.hf_dataset, **load_kwargs)

            if chunk_length and tokenizer:
                # Token packing: 拼接文档，切成固定长度 chunks
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
                chunks = []
                token_buffer = []
                eos_id = tokenizer.eos_token_id

                for item in raw_ds:
                    text = item.get(text_col, "")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if token_buffer and eos_id is not None:
                        token_buffer.append(eos_id)
                    token_buffer.extend(tokenizer.encode(text.strip(), add_special_tokens=False))
                    while len(token_buffer) >= chunk_length:
                        chunks.append({"input_ids": token_buffer[:chunk_length]})
                        token_buffer = token_buffer[chunk_length:]
                        if max_chunks and len(chunks) >= max_chunks:
                            break
                    if max_chunks and len(chunks) >= max_chunks:
                        break
            else:
                # 不做 packing，直接用原始文本
                chunks = []
                for item in raw_ds:
                    text = item.get(text_col, "")
                    if isinstance(text, str) and text.strip():
                        chunks.append({"text": text.strip()})
                        if max_chunks and len(chunks) >= max_chunks:
                            break

            Logger(f"Token packing 完成: {len(chunks)} chunks, 缓存到 {cache_path}",
                   rank0_only=True, ddp=ddp)
            # 保存缓存
            torch.save(chunks, cache_path)

    # DDP: broadcast chunks 从 rank 0 到其他 rank
    if ddp:
        import pickle
        if is_main:
            data_bytes = pickle.dumps(chunks)
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device=args.device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=args.device)

        dist.broadcast(size_tensor, src=0)
        size = int(size_tensor.item())

        if is_main:
            data_tensor = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(args.device)
        else:
            data_tensor = torch.empty(size, dtype=torch.uint8, device=args.device)

        dist.broadcast(data_tensor, src=0)

        if not is_main:
            chunks = pickle.loads(data_tensor.cpu().numpy().tobytes())

        # 释放临时 tensor
        del data_tensor, size_tensor

    return _ChunkedHFDataset(chunks)


def setup_model_parallel(model, num_gpus=2):
    """使用 accelerate 的 dispatch_model 自动分配模型到多 GPU

    手动构建 device_map：embed/dropout/rotary/fourier 放 GPU 0，
    transformer layers 均匀分配，norm/lm_head 放最后一张卡。
    accelerate 自动处理跨 GPU 的 tensor 搬运（hook-based）。
    """
    from accelerate import dispatch_model

    inner = model.model
    n_layers = len(inner.layers)
    layers_per_gpu = (n_layers + num_gpus - 1) // num_gpus

    device_map = {
        "model.embed_tokens": 0,
        "model.dropout": 0,
        "model.rotary_emb": 0,
    }

    # fourier_pe 可能为 None（rope 模式），只有存在时才加入
    if inner.fourier_pe is not None:
        device_map["model.fourier_pe"] = 0

    # transformer layers 均匀分配
    for i in range(n_layers):
        gpu = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"model.layers.{i}"] = gpu

    # norm + lm_head 放最后一张卡
    last_gpu = min(num_gpus - 1, num_gpus - 1)
    device_map["model.norm"] = last_gpu
    device_map["lm_head"] = last_gpu

    model = dispatch_model(model, device_map=device_map)

    last_device = f"cuda:{last_gpu}"
    return model, last_device


def Logger(msg: str, *, rank0_only: bool = False, ddp: bool = False):
    if not ddp:
        print(msg)
    else:
        if (not rank0_only) or (dist.get_rank() == 0):
            print(msg)


def _is_ddp_env() -> bool:
    try:
        return int(os.environ.get("RANK", "-1")) != -1
    except Exception:
        return False


def init_distributed_if_needed(args):
    """Initialize torch.distributed if running under torchrun and set device."""
    ddp = _is_ddp_env()
    args.ddp = ddp
    if not ddp:
        return ddp

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    dist.init_process_group(backend="nccl")
    return ddp


def load_model_and_tokenizer(args):
    """Load model/tokenizer from a .pth checkpoint or a Transformers directory.

    - If args.model_path is a directory with config.json -> load via AutoModelForCausalLM.
    - Else treat as torch checkpoint (.pth). Requires minimal config args.
    """
    device = torch.device(args.device)

    # Transformers directory path
    if os.path.isdir(args.model_path) and os.path.exists(os.path.join(args.model_path, 'config.json')):
        Logger(f"从 Transformers 目录加载模型: {args.model_path}", rank0_only=True, ddp=args.ddp)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        model = model.to(device).eval()
        return model, tokenizer

    # Torch checkpoint (.pth)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型权重: {args.model_path}")

    Logger(f"从 checkpoint 加载模型: {args.model_path}", rank0_only=True, ddp=args.ddp)
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        tokenizer_path = checkpoint.get('tokenizer_path', args.tokenizer)
        vocab_size = checkpoint.get('vocab_size', None)
    else:
        state_dict = checkpoint
        tokenizer_path = args.tokenizer
        vocab_size = None

    # tokenizer loading (with fallback for directories without model config.json)
    tok_path = tokenizer_path or os.path.join(root_path, 'model')
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except (ValueError, OSError):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_path)

    # build config
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

    # adjust vocab size from tokenizer or checkpoint
    lm_config.vocab_size = len(tokenizer)
    if vocab_size is not None:
        lm_config.vocab_size = max(lm_config.vocab_size, int(vocab_size))

    model = MiniMindForCausalLM(lm_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        Logger(f"注意：有缺失权重未加载（{len(missing)} 项），例如: {missing[:5]}", rank0_only=True, ddp=args.ddp)
    if unexpected:
        Logger(f"注意：有未使用的权重（{len(unexpected)} 项），例如: {unexpected[:5]}", rank0_only=True, ddp=args.ddp)

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def _estimate_branches(args) -> int:
    if args.val_max_branches_per_sample is not None and args.val_min_branches_per_sample is not None:
        return max(1, int(round((args.val_min_branches_per_sample + args.val_max_branches_per_sample) / 2)))
    if args.val_branches_per_sample is not None:
        return max(1, int(args.val_branches_per_sample))
    return max(1, int(args.branches_per_sample))


def evaluate(model, tokenizer, args):
    # Build collator for evaluation with branch controls
    branch_stride = 1 if args.pe == 'fpe' else 128
    avg_branches = _estimate_branches(args)

    val_collator = ParallelPretrainCollator(
        tokenizer,
        branches_per_sample=(args.val_branches_per_sample or args.branches_per_sample),
        pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
        max_branches_per_sample=(args.val_max_branches_per_sample if args.val_max_branches_per_sample is not None else args.max_branches_per_sample),
        min_branches_per_sample=(args.val_min_branches_per_sample if args.val_min_branches_per_sample is not None else args.min_branches_per_sample),
        random_time_offset=args.random_time_offset,
        interleave_branches=True,
        branch_stride=branch_stride,
    )

    # 始终设置 target_samples，确保每个 batch 产生固定数量的样本
    val_collator.target_samples = max(1, args.batch_size)

    # Dataset — eval 始终用 map-style dataset，支持 DistributedSampler + 多 worker 并行加载
    desired_texts = args.eval_total_texts or args.eval_target_samples or getattr(args, 'eval_samples', 0) or getattr(args, 'max_samples', None)

    if getattr(args, 'hf_dataset', None):
        ds = _load_hf_map_dataset(args, tokenizer, max_chunks=desired_texts)
    else:
        ds = ParallelPretrainDataset(
            data_path=args.data_path,
            max_samples=None,
        )
        if desired_texts and len(ds) > int(desired_texts):
            import random as _random
            indices = list(range(len(ds)))
            _random.shuffle(indices)
            ds = Subset(ds, indices[:int(desired_texts)])

    # DataLoader (map-style: 支持 DistributedSampler + 多 worker)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False, drop_last=False) if args.ddp else None
    shuffle = sampler is None

    # batch_size = 目标样本数, 需要取出足够的文本来构建这些样本
    effective_batch_size = max(1, int(args.batch_size * avg_branches))

    loader = DataLoader(
        ds,
        batch_size=effective_batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=shuffle,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=val_collator,
    )

    device = torch.device(args.device)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    if args.dtype in ['float16', 'bfloat16'] and device.type == 'cuda':
        amp_dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    Logger("开始评估...", rank0_only=True, ddp=args.ddp)
    samples_done = 0  # 全局已评估的原始branch数量

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if getattr(args, 'use_flex_attention', False):
            block_mask, padded_len = build_flex_columnar_mask(batch["time_ids"], batch["attention_mask"])
            seq_len = batch["time_ids"].shape[1]
            if padded_len > seq_len:
                pad_size = padded_len - seq_len
                batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, pad_size), value=0)
                batch["labels"] = torch.nn.functional.pad(batch["labels"], (0, pad_size), value=-100)
                batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, pad_size), value=0)
                batch["position_ids"] = torch.nn.functional.pad(batch["position_ids"], (0, pad_size), value=0)
                batch["time_ids"] = torch.nn.functional.pad(batch["time_ids"], (0, pad_size), value=-1)
                batch["pos2d"] = torch.nn.functional.pad(batch["pos2d"], (0, 0, 0, pad_size), value=0)
            column_mask = block_mask
        else:
            column_mask = build_columnar_causal_mask(batch["time_ids"], batch["attention_mask"]).to(device)

        with autocast_ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=column_mask,
                position_ids=batch["position_ids"],
                pos2d=batch["pos2d"],
            )
            logits = outputs.logits.contiguous()
            labels = batch["labels"].contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if hasattr(outputs, 'aux_loss'):
                loss = loss + outputs.aux_loss

        # count valid tokens
        valid = (labels != -100).sum().item()
        total_tokens += valid
        total_loss += loss.item()
        total_batches += 1

        # 统计本批原始branch数量
        if "branch_counts" in batch:
            local_samples = int(batch["branch_counts"].sum().item())
        else:
            local_samples = 0
        if args.ddp:
            tmp = torch.tensor([local_samples], device=device, dtype=torch.long)
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            global_samples = int(tmp.item())
        else:
            global_samples = local_samples
        samples_done += global_samples

        if total_batches % max(1, args.log_interval) == 0:
            Logger(
                f"branches_processed {samples_done} | batch {total_batches}: loss={loss.item():.4f}, tokens={valid}",
                rank0_only=True,
                ddp=args.ddp,
            )

        if args.max_batches > 0 and total_batches >= args.max_batches:
            break

    # 在 DDP 下聚合所有 rank 的统计
    if args.ddp:
        stats = torch.tensor([total_loss, total_batches, total_tokens], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_batches = int(stats[1].item())
        total_tokens = int(stats[2].item())

    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    Logger(f"评估完成: 平均loss={avg_loss:.4f}, 近似ppl={ppl:.2f}, 批次数={total_batches}, 有效tokens={total_tokens}", rank0_only=True, ddp=args.ddp)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved MiniMind model loss on dataset (validation-like)")

    # model loading
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint (.pth) or Transformers model directory')
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer path/name (used when checkpoint lacks tokenizer info)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])

    # config (for .pth checkpoints)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--num_key_value_heads', type=int, default=None,
                        help='Number of key-value heads for GQA. Default: use MiniMindConfig default')
    parser.add_argument('--use_moe', type=bool, default=False)

    # position encoding
    parser.add_argument('--pe', type=str, default='rope', choices=['rope', 'fpe'])
    parser.add_argument('--rope_2d_ratio', type=float, default=0.5)
    parser.add_argument('--fpe_theta', type=float, default=10000.0)
    parser.add_argument('--fpe_max_positions', type=int, default=512)
    parser.add_argument('--fpe_learnable', action='store_true')
    parser.add_argument('--use_flex_attention', action='store_true',
                        help='Use FlexAttention BlockMask instead of dense mask')

    # dataset (local JSONL or HF)
    parser.add_argument('--data_path', type=str, default=os.path.join(root_path, 'dataset', 'pretrain_hq_split.jsonl'))
    parser.add_argument('--hf-dataset', type=str, default=None)
    parser.add_argument('--hf-subset', type=str, default=None)
    parser.add_argument('--hf-split', type=str, default='train')
    parser.add_argument('--text-column', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--chunk-length', type=int, default=None)
    parser.add_argument('--offline', action='store_true')

    # evaluation control
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_total_tokens', type=int, default=4096, help='Pad/truncate per sample sequence length; set 0 to disable pad_to')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--max_batches', type=int, default=0, help='Stop after N batches (0=all)')
    parser.add_argument('--eval_samples', type=int, default=0, help='DEPRECATED (raw texts). For map-style dataset: randomly subsample this many raw texts')
    parser.add_argument('--eval_target_samples', type=int, default=0, help='Target number of raw texts/branches to evaluate (global total across all GPUs)')
    parser.add_argument('--eval_total_texts', type=int, default=0, help='Directly limit number of raw texts to load from dataset (highest priority)')

    # branch controls for evaluation
    parser.add_argument('--branches_per_sample', type=int, default=8)
    parser.add_argument('--max_branches_per_sample', type=int, default=None)
    parser.add_argument('--min_branches_per_sample', type=int, default=1)
    parser.add_argument('--val_branches_per_sample', type=int, default=None)
    parser.add_argument('--val_max_branches_per_sample', type=int, default=None)
    parser.add_argument('--val_min_branches_per_sample', type=int, default=None)
    parser.add_argument('--random_time_offset', action='store_true', default=False)
    parser.add_argument('--no_random_time_offset', action='store_false', dest='random_time_offset')
    parser.add_argument('--batch_by_samples', action='store_true', help='Interpret batch_size as number of samples (collated groups); DataLoader batch size will scale by平均分支数')
    parser.add_argument('--model_parallel', type=int, default=0,
                        help='Split model across N GPUs (pipeline parallelism). 0=disabled, use DDP. >0: single process, model split across N GPUs.')

    args = parser.parse_args()

    # Model parallel 模式：单进程，模型分到多 GPU
    if args.model_parallel > 0:
        args.ddp = False
        args.device = 'cuda:0'
        # FlexAttention 的 BlockMask 无法跨 GPU，model_parallel 下强制用 dense mask
        if getattr(args, 'use_flex_attention', False):
            Logger("model_parallel 模式下禁用 FlexAttention，改用 dense mask")
            args.use_flex_attention = False
        model, tokenizer = load_model_and_tokenizer(args)
        model, last_device = setup_model_parallel(model, args.model_parallel)
        args.device = last_device  # loss 计算在最后一个 GPU
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        Logger(f"模型可训练参数量：{total_params / 1e6:.3f} 百万 (model_parallel={args.model_parallel})")
    else:
        # DDP 初始化（若通过 torchrun 启动会生效）
        init_distributed_if_needed(args)
        model, tokenizer = load_model_and_tokenizer(args)
        if (not args.ddp) or dist.get_rank() == 0:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            Logger(f"模型可训练参数量：{total_params / 1e6:.3f} 百万", rank0_only=True, ddp=args.ddp)

    evaluate(model, tokenizer, args)

    # 优雅退出 DDP
    if args.ddp:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    main()
