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
        max_branches_per_sample: Optional[int] = None,
        min_branches_per_sample: int = 1,
        random_time_offset: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.branches_per_sample = max(1, branches_per_sample)
        self.pad_to = pad_to
        # 如果设置了 max_branches，则使用动态模式
        self.max_branches = max_branches_per_sample if max_branches_per_sample is not None else branches_per_sample
        self.min_branches = max(1, min_branches_per_sample)
        self.dynamic_mode = max_branches_per_sample is not None
        self.random_time_offset = random_time_offset
        self.target_samples: Optional[int] = None  # 由训练脚本在 batch_by_samples 模式下设置

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        import random

        texts = [str(f["text"]) for f in features]
        samples = []

        if self.dynamic_mode:
            target_samples = self.target_samples
            if target_samples is not None and target_samples > 0:
                total_texts = len(texts)
                # 保证目标 sample 数不会超过可用文本数（每个 sample 至少需要 min_branches 个文本）
                max_possible_samples = max(1, total_texts // self.min_branches)
                target_samples = min(target_samples, max_possible_samples)

                remaining_texts = total_texts
                start_idx = 0
                for sample_idx in range(target_samples):
                    remaining_slots = target_samples - sample_idx
                    if remaining_texts <= 0:
                        break
                    # 计算当前 sample 可用的分配范围，确保剩余文本足够后续 sample 的最小需求
                    min_remaining = (remaining_slots - 1) * self.min_branches
                    max_remaining = (remaining_slots - 1) * self.max_branches

                    low = max(self.min_branches, remaining_texts - max_remaining)
                    high = min(self.max_branches, remaining_texts - min_remaining)
                    if remaining_slots == 1:
                        num_branches = remaining_texts
                    else:
                        if low > high:
                            low = max(self.min_branches, remaining_texts - min_remaining)
                            high = max(low, remaining_texts - min_remaining)
                        num_branches = random.randint(low, max(low, high))

                    chunk = texts[start_idx : start_idx + num_branches]
                    if chunk:
                        branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                        samples.append({"main": "", "branches": branches})
                        start_idx += num_branches
                    remaining_texts -= num_branches

                # 如果还有剩余文本，将其追加到最后一个 sample，避免浪费数据
                if remaining_texts > 0 and samples and start_idx < len(texts):
                    extra_chunk = texts[start_idx:]
                    samples[-1]["branches"].extend({"text": txt, "answer_offset": 0} for txt in extra_chunk)
            else:
                # 动态模式：每个 sample 的 branches 数量随机，samples 数量不固定
                idx = 0
                while idx < len(texts):
                    num_branches = random.randint(self.min_branches, self.max_branches)
                    num_branches = min(num_branches, len(texts) - idx)

                    if num_branches >= self.min_branches:
                        chunk = texts[idx : idx + num_branches]
                        branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                        samples.append({"main": "", "branches": branches})
                        idx += num_branches
                    elif num_branches > 0:
                        chunk = texts[idx : idx + num_branches]
                        branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                        samples.append({"main": "", "branches": branches})
                        break
                    else:
                        break
        else:
            # 固定模式：每个 sample 固定 branches 数量
            for idx in range(0, len(texts), self.branches_per_sample):
                chunk = texts[idx : idx + self.branches_per_sample]
                if len(chunk) >= self.branches_per_sample:
                    branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                    samples.append({"main": "", "branches": branches})
                elif len(chunk) > 0:
                    branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                    samples.append({"main": "", "branches": branches})

        if not samples and len(texts) > 0:
            branches = [{"text": txt, "answer_offset": 0} for txt in texts]
            samples.append({"main": "", "branches": branches})

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.pad_to,
            random_time_offset=self.random_time_offset,
        )

        branch_counts = torch.tensor(
            [len(sample["branches"]) for sample in samples],
            dtype=torch.long,
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
            "branch_counts": branch_counts,
        }


class ParallelSFTCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        branches_per_sample: int = 4,
        pad_to: Optional[int] = None,
        random_time_offset: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.branches_per_sample = max(1, branches_per_sample)
        self.pad_to = pad_to
        self.random_time_offset = random_time_offset

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
            random_time_offset=self.random_time_offset,
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
