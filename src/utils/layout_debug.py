from typing import List, Optional, Sequence, Tuple

from src.model.columnar import BatchLayout
from transformers import PreTrainedTokenizer


def dump_branch_layout(
    layout: BatchLayout,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 256,
    sample_index: int = 0,
) -> None:
    """Print a human-readable view of branch/time positions for debugging."""
    if layout.input_ids.size(0) == 0:
        print("[layout] empty batch")
        return

    sample_index = max(0, min(sample_index, layout.input_ids.size(0) - 1))
    attn = layout.attention_mask[sample_index]
    seq_len = int(attn.sum().item())
    limit = min(seq_len, max_tokens)
    meta = layout.metadata[sample_index]

    branch_lookup = {pos: idx for idx, pos in enumerate(meta.branch_positions)}

    print(
        f"[layout] sample={sample_index} tokens={seq_len} branches={len(meta.branch_positions)} "
        f"limit={limit}"
    )
    if seq_len == 0:
        return

    for idx in range(limit):
        if attn[idx].item() <= 0:
            continue
        token_id = int(layout.input_ids[sample_index, idx].item())
        pos2d = layout.pos2d[sample_index, idx]
        branch_pos = int(pos2d[0].item())
        time_val = int(pos2d[1].item())
        branch_idx = branch_lookup.get(branch_pos, -1)
        token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
        print(
            f"  [t={time_val:4d}] idx={idx:4d} branch={branch_idx:2d} pos={branch_pos:4d} "
            f"id={token_id:5d} text='{token_text}'"
        )

    if seq_len > limit:
        print(f"  ... ({seq_len - limit} more tokens not shown)")


def dump_generated_sequences(
    branch_meta: Sequence[Sequence[Tuple[int, int, int, int]]],
    branch_tokens: Sequence[Sequence[int]],
    tokenizer: PreTrainedTokenizer,
) -> None:
    """Print generated token sequences with branch/time information."""
    print("\n[layout] Generated token sequences:")
    for idx, tokens in enumerate(branch_tokens):
        metas = branch_meta[idx] if idx < len(branch_meta) else []
        if not tokens:
            print(f"  Branch {idx}: (no tokens)")
            continue
        print(f"  Branch {idx}:")
        limit = len(tokens)
        for pos in range(limit):
            token_id = tokens[pos]
            token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
            meta = metas[pos] if pos < len(metas) else None
            if meta is None:
                print(f"    #{pos:02d} id={token_id:5d} text='{token_text}' (meta unavailable)")
                continue
            branch_pos, time_val, abs_idx, source_abs_idx = meta
            source_time = time_val - 1
            print(
                f"    #{pos:02d} id={token_id:5d} text='{token_text}' "
                f"@ branch_pos={branch_pos:4d} time={time_val:4d} abs_idx={abs_idx:5d} "
                f"(predicted from time={source_time:4d}, abs_idx={source_abs_idx:5d})"
            )
