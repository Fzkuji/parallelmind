from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.model.columnar import build_flat_linear_layout


class GroupedParallelCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pad_to: Optional[int] = None,
        random_time_offset: bool = False,
        interleave_branches: bool = True,
        branch_stride: int = 128,
        align_to: str = "left",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to = pad_to
        self.random_time_offset = random_time_offset
        self.interleave_branches = interleave_branches
        self.branch_stride = branch_stride
        self.align_to = align_to

    def _normalize_branch(self, branch: Any) -> Dict[str, Any]:
        if isinstance(branch, str):
            return {"text": branch, "answer_offset": 0}
        if isinstance(branch, dict):
            normalized = {
                "text": str(branch.get("text", "")),
                "answer_offset": int(branch.get("answer_offset", 0)),
            }
            if "input_ids" in branch:
                normalized["input_ids"] = branch["input_ids"]
            return normalized
        return {"text": str(branch), "answer_offset": 0}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples: List[Dict[str, Any]] = []
        for item in features:
            samples.append(
                {
                    "main": str(item.get("main", "")),
                    "branches": [self._normalize_branch(branch) for branch in item.get("branches", [])],
                }
            )

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.pad_to,
            branch_stride=self.branch_stride,
            random_time_offset=self.random_time_offset,
            interleave_branches=self.interleave_branches,
            align_to=self.align_to,
        )

        branch_counts = torch.tensor([len(sample["branches"]) for sample in samples], dtype=torch.long)
        labels = torch.full_like(layout.input_ids, fill_value=-100)

        for batch_idx, meta in enumerate(layout.metadata):
            for branch_idx, _ in enumerate(meta.branch_ids):
                branch_pos = meta.branch_positions[branch_idx]
                mask = (layout.pos2d[batch_idx, :, 0] == branch_pos) & (layout.attention_mask[batch_idx] == 1)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() <= 1:
                    continue

                answer_start = 0
                if branch_idx < len(meta.answer_token_starts):
                    answer_start = max(0, int(meta.answer_token_starts[branch_idx]))
                answer_start = min(answer_start, len(indices))

                for i in range(len(indices) - 1):
                    target_order = i + 1
                    if target_order < answer_start:
                        continue
                    src_pos = indices[i].item()
                    tgt_pos = indices[i + 1].item()
                    labels[batch_idx, src_pos] = layout.input_ids[batch_idx, tgt_pos].item()

        return {
            "input_ids": layout.input_ids,
            "attention_mask": layout.attention_mask,
            "position_ids": layout.pos1d,
            "pos2d": layout.pos2d,
            "time_ids": layout.time_ids,
            "labels": labels,
            "branch_counts": branch_counts,
        }
