from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from parallel.columnar import build_flat_linear_layout


class ParallelPretrainCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        branches_per_sample: int = 8,
        pad_to: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.branches_per_sample = max(1, branches_per_sample)
        self.pad_to = pad_to

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        texts = [str(f["text"]) for f in features]
        samples = []
        for idx in range(0, len(texts), self.branches_per_sample):
            chunk = texts[idx : idx + self.branches_per_sample]
            if len(chunk) < self.branches_per_sample:
                break
            branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
            samples.append({"main": "", "branches": branches})

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.pad_to,
        )

        labels = layout.input_ids.clone()
        labels[layout.attention_mask == 0] = -100

        return {
            "input_ids": layout.input_ids,
            "attention_mask": layout.attention_mask,
            "position_ids": layout.pos1d,
            "pos2d": layout.pos2d,
            "time_ids": layout.time_ids,
            "labels": labels,
        }


class ParallelSFTCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        branches_per_sample: int = 4,
        pad_to: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.branches_per_sample = max(1, branches_per_sample)
        self.pad_to = pad_to

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        samples = []
        for idx in range(0, len(features), self.branches_per_sample):
            chunk = features[idx : idx + self.branches_per_sample]
            if len(chunk) < self.branches_per_sample:
                break
            branches = []
            for item in chunk:
                branches.append(
                    {
                        "input_ids": item["input_ids"].tolist(),
                        "answer_offset": int(item["answer_offset"]),
                    }
                )
            samples.append({"main": "", "branches": branches})

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.pad_to,
        )

        labels = layout.input_ids.clone()
        labels[layout.attention_mask == 0] = -100

        for batch_idx, meta in enumerate(layout.metadata):
            for branch_idx, branch_id in enumerate(meta.branch_ids):
                branch_pos = meta.branch_positions[branch_idx]
                mask = (layout.pos2d[batch_idx, :, 0] == branch_pos) & (layout.attention_mask[batch_idx] == 1)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                answer_start = meta.answer_token_starts[branch_idx]
                answer_start = max(0, min(answer_start, indices.numel()))
                if answer_start > 0:
                    labels[batch_idx, indices[:answer_start]] = -100

        return {
            "input_ids": layout.input_ids,
            "attention_mask": layout.attention_mask,
            "position_ids": layout.pos1d,
            "pos2d": layout.pos2d,
            "time_ids": layout.time_ids,
            "labels": labels,
        }
