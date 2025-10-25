import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


class Interleaved2DRoPE(torch.nn.Module):
    """Inject 2D coordinates into an existing rotary embedding module."""

    def __init__(self, base_rope_module: torch.nn.Module, pair_indices_1based: Sequence[int]) -> None:
        super().__init__()
        self.config = getattr(base_rope_module, "config", None)
        self.rope_init_fn = getattr(base_rope_module, "rope_init_fn", None)
        self.attention_scaling = getattr(base_rope_module, "attention_scaling", 1.0)
        self.register_buffer("inv_freq", base_rope_module.inv_freq, persistent=False)
        self.original_inv_freq = getattr(base_rope_module, "original_inv_freq", None)
        self.pair_indices = [i - 1 for i in pair_indices_1based]
        self.extra_pos2d: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.extra_pos2d is None:
            raise RuntimeError("extra_pos2d is not set. Call set_rope_pos2d first.")

        inv = self.inv_freq.to(dtype=torch.float32, device=x.device)
        pos1d = position_ids.to(dtype=torch.float32, device=x.device)
        freqs_1d = torch.einsum("f,bn->bnf", inv, pos1d.reshape(pos1d.size(0), -1)).reshape(
            *pos1d.shape, -1
        )

        pos2d = self.extra_pos2d.to(dtype=torch.float32, device=x.device)
        freqs_x = torch.einsum("f,bn->bnf", inv, pos2d[..., 0].reshape(pos1d.size(0), -1)).reshape(
            *pos1d.shape, -1
        )
        freqs_y = torch.einsum("f,bn->bnf", inv, pos2d[..., 1].reshape(pos1d.size(0), -1)).reshape(
            *pos1d.shape, -1
        )

        freqs_mix = freqs_1d.clone()
        for local_idx, freq_idx in enumerate(self.pair_indices):
            if 0 <= freq_idx < freqs_mix.size(-1):
                freqs_mix[..., freq_idx] = freqs_x[..., freq_idx] if (local_idx % 2 == 0) else freqs_y[..., freq_idx]

        emb = torch.cat((freqs_mix, freqs_mix), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _find_rotary_holder(module: torch.nn.Module) -> torch.nn.Module:
    queue: deque[torch.nn.Module] = deque([module])
    visited: set[int] = set()

    candidate_attrs = ("language_model", "model", "base_model", "module", "wrapped_model", "encoder")

    while queue:
        current = queue.popleft()
        ident = id(current)
        if ident in visited:
            continue
        visited.add(ident)

        if hasattr(current, "rotary_emb"):
            return current

        for attr in candidate_attrs:
            child = getattr(current, attr, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)

        get_base = getattr(current, "get_base_model", None)
        if callable(get_base):
            try:
                base = get_base()
            except Exception:
                base = None
            if isinstance(base, torch.nn.Module):
                queue.append(base)

    raise RuntimeError("Unable to locate rotary_emb inside the model.")


def patch_model_with_interleaved_2d_rope(model: torch.nn.Module, pair_indices_1based: Sequence[int]) -> Interleaved2DRoPE:
    holder = _find_rotary_holder(model)
    base_rope = holder.rotary_emb
    new_rope = Interleaved2DRoPE(base_rope, pair_indices_1based)
    holder.rotary_emb = new_rope
    return new_rope


def set_rope_pos2d(model: torch.nn.Module, pos2d: torch.Tensor) -> None:
    holder = _find_rotary_holder(model)
    rope = holder.rotary_emb
    if not isinstance(rope, Interleaved2DRoPE):
        raise RuntimeError("Model has not been patched with Interleaved2DRoPE.")
    rope.extra_pos2d = pos2d.to(next(model.parameters()).device)


@dataclass
class LayoutMetadata:
    branch_ids: List[int]
    branch_positions: List[int]
    branch_lengths: List[int]
    branch_start_y: List[int]
    branch_pos1d_end: List[int]
    background_branch_id: int
    answer_token_starts: List[int]


@dataclass
class BatchLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pos1d: torch.Tensor
    pos2d: torch.Tensor
    metadata: List[LayoutMetadata]
    pad_id: int
    time_ids: torch.Tensor


def pad_to_length(x: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    if x.size(-1) >= length:
        return x[..., :length]
    pad = x.new_full((*x.shape[:-1], length - x.size(-1)), pad_id)
    return torch.cat([x, pad], dim=-1)


DEFAULT_BRANCH_POSITION_STRIDE = 128


def build_flat_linear_layout(
    tokenizer: AutoTokenizer,
    samples: Sequence[Dict[str, Any]],
    device: torch.device,
    pad_to: Optional[int] = None,
    branch_stride: int = DEFAULT_BRANCH_POSITION_STRIDE,
    align_to: Literal["left", "right"] = "left",
    random_time_offset: bool = False,
) -> BatchLayout:
    if align_to not in {"left", "right"}:
        raise ValueError(f"Unsupported align_to value: {align_to}")
    tokenized: List[Dict[str, Any]] = []
    max_len = 0
    metadata: List[LayoutMetadata] = []

    for sample in samples:
        main_text = sample.get("main", "") or ""
        branches_text = sample.get("branches", [])

        if "input_ids" in sample:
            main_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        else:
            main_ids = tokenizer(
                main_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors="pt",
            ).input_ids[0]
        branch_ids_list: List[torch.Tensor] = []
        answer_offsets: List[int] = []

        for branch in branches_text:
            branch_data = branch if isinstance(branch, dict) else {"text": branch, "answer_offset": 0}
            if "input_ids" in branch_data:
                branch_ids = torch.tensor(branch_data["input_ids"], dtype=torch.long)
            else:
                branch_text = branch_data.get("text", branch_data.get("content", ""))
                branch_ids = tokenizer(
                    branch_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors="pt",
                ).input_ids[0]
            branch_ids_list.append(branch_ids)
            answer_offsets.append(int(branch_data.get("answer_offset", 0)))

        total_len = main_ids.numel() + sum(b.numel() for b in branch_ids_list)
        tokenized.append(
            {
                "main_ids": main_ids,
                "branches_ids": branch_ids_list,
                "main_len": main_ids.numel(),
                "answer_offsets": answer_offsets,
            }
        )
        max_len = max(max_len, total_len)

    global_T = pad_to if pad_to is not None else max_len
    pad_id = tokenizer.pad_token_id or (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    input_ids_batch: List[torch.Tensor] = []
    attn_batch: List[torch.Tensor] = []
    pos1d_batch: List[torch.Tensor] = []
    pos2d_batch: List[torch.Tensor] = []
    time_batch: List[torch.Tensor] = []

    for token_info in tokenized:
        main_tokens = token_info["main_ids"].to(device)
        branch_token_list = [ids.to(device) for ids in token_info["branches_ids"]]
        main_len = token_info["main_len"]
        answer_offsets = token_info["answer_offsets"]

        branch_sequences: List[torch.Tensor] = []
        if main_len > 0:
            branch_sequences.append(main_tokens)
        branch_sequences.extend(branch_token_list)

        branch_ids = list(range(len(branch_sequences)))
        branch_positions = [branch_id * branch_stride for branch_id in branch_ids]
        non_main_count = len(branch_sequences) - (1 if main_len > 0 else 0)
        max_branch_len = (
            max((seq.numel() for seq in branch_sequences[(1 if main_len > 0 else 0):]), default=0)
            if non_main_count > 0
            else 0
        )

        entries: List[Tuple[int, int, int, int]] = []
        branch_start_y: List[int] = []
        branch_pos1d_end = [-1 for _ in branch_sequences]
        answer_token_starts: List[int] = []

        # 为每个branch生成随机time offset（如果启用）
        import random
        branch_offsets = []
        if random_time_offset:
            for idx in range(len(branch_sequences)):
                if idx == 0 and main_len > 0:
                    # main branch不加offset
                    branch_offsets.append(0)
                else:
                    # 根据当前branch的长度动态决定offset范围
                    branch_len = branch_sequences[idx].numel()
                    if branch_len > 0:
                        # offset范围是 [0, branch_len]，确保不会超过文本长度
                        # 这样短文本offset小，长文本offset可以大
                        branch_offsets.append(random.randint(0, branch_len))
                    else:
                        branch_offsets.append(0)
        else:
            branch_offsets = [0] * len(branch_sequences)

        for idx, (branch_id, tokens) in enumerate(zip(branch_ids, branch_sequences)):
            seq_len = tokens.numel()
            if seq_len == 0:
                start_col = 0 if idx == 0 else (main_len if main_len > 0 else 0)
                branch_start_y.append(start_col)
                answer_token_starts.append(0)
                continue

            if idx == 0 and main_len > 0:
                times = torch.arange(seq_len, device=device)
            else:
                if align_to == "right" and non_main_count > 0:
                    base_offset = main_len if main_len > 0 else 0
                    shift = max(max_branch_len - seq_len, 0)
                    start_col = base_offset + shift
                else:
                    if non_main_count > 0:
                        if main_len > 0:
                            start_col = main_len
                        else:
                            start_col = 0
                    else:
                        start_col = 0 if main_len == 0 else main_len
                times = torch.arange(seq_len, device=device) + start_col

            # 应用branch offset
            times = times + branch_offsets[idx]
            branch_start_y.append(int(times[0].item()))

            if idx == 0 and main_len > 0:
                answer_token_starts.append(seq_len)
            else:
                offset_list_idx = idx - (1 if main_len > 0 else 0)
                raw_offset = answer_offsets[offset_list_idx] if 0 <= offset_list_idx < len(answer_offsets) else 0
                answer_token_starts.append(max(0, min(seq_len, raw_offset)))

            for order, token in enumerate(tokens):
                time_value = int(times[order].item())
                entries.append((time_value, branch_id, order, int(token.item())))

        entries.sort()
        token_count = len(entries)
        effective_len = min(token_count, global_T) if global_T is not None else token_count
        entries_eff = entries[:effective_len]

        sorted_ids = [entry[3] for entry in entries_eff]
        sorted_branch = [branch_positions[entry[1]] for entry in entries_eff]
        sorted_time = [entry[0] for entry in entries_eff]
        branch_lengths_eff = [0 for _ in branch_sequences]
        for pos_idx, (time_val, branch_id, _, _) in enumerate(entries_eff):
            branch_pos1d_end[branch_id] = pos_idx
            branch_lengths_eff[branch_id] += 1

        ids_tensor = (
            torch.tensor(sorted_ids, dtype=torch.long, device=device)
            if sorted_ids
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if global_T is not None:
            seq_padded = pad_to_length(ids_tensor[None, :], global_T, pad_id)
        else:
            seq_padded = ids_tensor[None, :]
        input_ids_batch.append(seq_padded)

        if global_T is not None:
            mask_row = torch.zeros(1, global_T, device=device, dtype=torch.long)
            mask_row[0, : effective_len] = 1
        else:
            mask_row = torch.ones(1, ids_tensor.size(0), device=device, dtype=torch.long)
        attn_batch.append(mask_row)

        if global_T is not None:
            pos1d_row = torch.arange(global_T, device=device)[None, :]
        else:
            pos1d_row = torch.arange(ids_tensor.size(0), device=device)[None, :]
        pos1d_batch.append(pos1d_row)

        if global_T is not None:
            pos2d_seq = torch.zeros(global_T, 2, device=device, dtype=torch.long)
            pos2d_seq[:effective_len, 0] = torch.tensor(sorted_branch, dtype=torch.long, device=device)
            pos2d_seq[:effective_len, 1] = torch.tensor(sorted_time, dtype=torch.long, device=device)
        else:
            pos2d_seq = torch.stack(
                [
                    torch.tensor(sorted_branch, dtype=torch.long, device=device),
                    torch.tensor(sorted_time, dtype=torch.long, device=device),
                ],
                dim=1,
            )
        pos2d_batch.append(pos2d_seq[None, :, :] if global_T is not None else pos2d_seq[None, :, :])

        if global_T is not None:
            time_row = torch.full((global_T,), -1, device=device, dtype=torch.long)
            time_row[:effective_len] = torch.tensor(sorted_time, dtype=torch.long, device=device)
        else:
            time_row = torch.tensor(sorted_time, dtype=torch.long, device=device)
        time_batch.append(time_row[None, :] if global_T is not None else time_row[None, :])

        metadata.append(
            LayoutMetadata(
                branch_ids=branch_ids,
                branch_positions=branch_positions,
                branch_lengths=branch_lengths_eff,
                branch_start_y=branch_start_y,
                branch_pos1d_end=branch_pos1d_end,
                background_branch_id=0 if main_len > 0 else -1,
                answer_token_starts=answer_token_starts,
            )
        )

    input_ids = torch.cat(input_ids_batch, dim=0)
    attention_mask = torch.cat(attn_batch, dim=0)
    pos1d = torch.cat(pos1d_batch, dim=0)
    pos2d = torch.cat(pos2d_batch, dim=0)
    time_ids = torch.cat(time_batch, dim=0)

    return BatchLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pos1d=pos1d,
        pos2d=pos2d,
        metadata=metadata,
        pad_id=pad_id,
        time_ids=time_ids,
    )


def build_columnar_causal_mask(time_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    device = time_ids.device
    batch, seq_len = time_ids.shape
    pad_bool = pad_mask.bool()
    time_clean = torch.where(pad_bool, time_ids, torch.full_like(time_ids, -1))

    time_i = time_clean.unsqueeze(2)
    time_j = time_clean.unsqueeze(1)

    allowed = time_j < time_i
    eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed | eye

    allowed = allowed & pad_bool.unsqueeze(2) & pad_bool.unsqueeze(1)

    mask = torch.full((batch, 1, seq_len, seq_len), fill_value=torch.finfo(torch.float32).min, device=device)
    mask = mask.masked_fill(allowed[:, None, :, :], 0.0)
    return mask


def build_incremental_causal_mask(time_list: List[int], device: torch.device) -> torch.Tensor:
    total_len = len(time_list)
    if total_len == 0:
        raise ValueError("time_list must contain at least one element")
    new_time = time_list[-1]
    mask = torch.full((1, 1, 1, total_len), fill_value=torch.finfo(torch.float32).min, device=device)
    for idx, t in enumerate(time_list):
        if t >= 0 and (t < new_time or idx == total_len - 1):
            mask[0, 0, 0, idx] = 0.0
    return mask


def slice_past(past_kv: Any, idx: int) -> Any:
    if past_kv is None:
        return None
    if isinstance(past_kv, tuple):
        legacy_layers = past_kv
        make_dynamic = False
    elif hasattr(past_kv, "to_legacy_cache"):
        legacy_layers = past_kv.to_legacy_cache()
        make_dynamic = True
    else:
        raise TypeError(f"Unsupported past_key_values type: {type(past_kv)}")

    sliced_layers = []
    for layer in legacy_layers:
        if layer is None:
            sliced_layers.append(None)
            continue
        if not (isinstance(layer, tuple) and len(layer) == 2):
            raise TypeError("Expected (key, value) tuple per layer.")
        key, value = layer
        sliced_layers.append((key[idx : idx + 1].contiguous(), value[idx : idx + 1].contiguous()))

    if make_dynamic:
        return DynamicCache.from_legacy_cache(tuple(sliced_layers))
    return tuple(sliced_layers)


def trim_past_seq(past_kv: Any, target_len: int) -> Any:
    if past_kv is None or target_len is None:
        return past_kv
    if target_len < 0:
        target_len = 0

    def _trim(legacy_layers: Tuple[Any, ...]) -> Tuple[Any, ...]:
        trimmed: List[Any] = []
        for layer in legacy_layers:
            if layer is None:
                trimmed.append(None)
                continue
            if not (isinstance(layer, tuple) and len(layer) == 2):
                trimmed.append(layer)
                continue
            key, value = layer
            if key is not None and key.shape[-2] > target_len:
                key = key[..., :target_len, :].contiguous()
            if value is not None and value.shape[-2] > target_len:
                value = value[..., :target_len, :].contiguous()
            trimmed.append((key, value))
        return tuple(trimmed)

    if isinstance(past_kv, tuple):
        return _trim(past_kv)
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
        trimmed = _trim(legacy)
        return DynamicCache.from_legacy_cache(trimmed)
    return past_kv
