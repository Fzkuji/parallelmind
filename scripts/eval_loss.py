import os
import sys
import argparse
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

# repo root on path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from parallel_data.parallel_dataset import ParallelPretrainDataset, ParallelPretrainIterableDataset
from parallel_data.parallel_collator import ParallelPretrainCollator
from parallel.columnar import build_columnar_causal_mask


def Logger(msg: str):
    print(msg)


def load_model_and_tokenizer(args):
    """Load model/tokenizer from a .pth checkpoint or a Transformers directory.

    - If args.model_path is a directory with config.json -> load via AutoModelForCausalLM.
    - Else treat as torch checkpoint (.pth). Requires minimal config args.
    """
    device = torch.device(args.device)

    # Transformers directory path
    if os.path.isdir(args.model_path) and os.path.exists(os.path.join(args.model_path, 'config.json')):
        Logger(f"从 Transformers 目录加载模型: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        model = model.to(device).eval()
        return model, tokenizer

    # Torch checkpoint (.pth)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型权重: {args.model_path}")

    Logger(f"从 checkpoint 加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        tokenizer_path = checkpoint.get('tokenizer_path', args.tokenizer)
        vocab_size = checkpoint.get('vocab_size', None)
    else:
        state_dict = checkpoint
        tokenizer_path = args.tokenizer
        vocab_size = None

    # tokenizer loading
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_path, 'model'))

    # build config
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        pe_type=args.pe,
        rope_2d_ratio=args.rope_2d_ratio,
        fpe_theta=args.fpe_theta,
        fpe_max_positions=args.fpe_max_positions,
        fpe_learnable=args.fpe_learnable,
    )

    # adjust vocab size from tokenizer or checkpoint
    lm_config.vocab_size = len(tokenizer)
    if vocab_size is not None:
        lm_config.vocab_size = max(lm_config.vocab_size, int(vocab_size))

    model = MiniMindForCausalLM(lm_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        Logger(f"注意：有缺失权重未加载（{len(missing)} 项），例如: {missing[:5]}")
    if unexpected:
        Logger(f"注意：有未使用的权重（{len(unexpected)} 项），例如: {unexpected[:5]}")

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def evaluate(model, tokenizer, args):
    # Build collator for evaluation with branch controls
    branch_stride = 1 if args.pe == 'fpe' else 128
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

    # Dataset
    if getattr(args, 'hf_dataset', None):
        ds = ParallelPretrainIterableDataset(
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
    else:
        ds = ParallelPretrainDataset(
            data_path=args.data_path,
            max_samples=args.max_samples,
        )
        is_iterable = False
        if args.eval_samples > 0 and len(ds) > args.eval_samples:
            # Use a random subset for evaluation if requested
            import random as _random
            indices = list(range(len(ds)))
            _random.shuffle(indices)
            ds = Subset(ds, indices[: args.eval_samples])

    # DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
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

    Logger("开始评估...")
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
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

        if total_batches % max(1, args.log_interval) == 0:
            Logger(f"batch {total_batches}: loss={loss.item():.4f}, tokens={valid}")

        if args.max_batches > 0 and total_batches >= args.max_batches:
            break

    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    Logger(f"评估完成: 平均loss={avg_loss:.4f}, 近似ppl={ppl:.2f}, 批次数={total_batches}, 有效tokens={total_tokens}")
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
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=bool, default=False)

    # position encoding
    parser.add_argument('--pe', type=str, default='rope', choices=['rope', 'fpe'])
    parser.add_argument('--rope_2d_ratio', type=float, default=0.5)
    parser.add_argument('--fpe_theta', type=float, default=10000.0)
    parser.add_argument('--fpe_max_positions', type=int, default=512)
    parser.add_argument('--fpe_learnable', action='store_true')

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
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_total_tokens', type=int, default=4096, help='Pad/truncate per sample sequence length; set 0 to disable pad_to')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--max_batches', type=int, default=0, help='Stop after N batches (0=all)')
    parser.add_argument('--eval_samples', type=int, default=0, help='For local dataset: randomly subsample this many samples')

    # branch controls for evaluation
    parser.add_argument('--branches_per_sample', type=int, default=8)
    parser.add_argument('--max_branches_per_sample', type=int, default=None)
    parser.add_argument('--min_branches_per_sample', type=int, default=1)
    parser.add_argument('--val_branches_per_sample', type=int, default=None)
    parser.add_argument('--val_max_branches_per_sample', type=int, default=None)
    parser.add_argument('--val_min_branches_per_sample', type=int, default=None)
    parser.add_argument('--random_time_offset', action='store_true', default=False)
    parser.add_argument('--no_random_time_offset', action='store_false', dest='random_time_offset')

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"模型可训练参数量：{total_params / 1e6:.3f} 百万")

    evaluate(model, tokenizer, args)


if __name__ == '__main__':
    main()

