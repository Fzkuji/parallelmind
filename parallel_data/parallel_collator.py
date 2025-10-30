from collections import deque
from typing import Deque, Dict, List, Optional

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
        interleave_branches: bool = False,
        branch_stride: int = 128,
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
        self._buffer: Deque[str] = deque()
        self.interleave_branches = interleave_branches
        self.branch_stride = branch_stride

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        import random

        texts = [str(f["text"]) for f in features]
        self._buffer.extend(texts)
        samples = []

        if self.dynamic_mode:
            target_samples = self.target_samples
            if target_samples is not None and target_samples > 0:
                for sample_idx in range(target_samples):
                    available = len(self._buffer)
                    remaining_slots = target_samples - sample_idx

                    # 如果没有足够的文本来构建一个 sample，则停止，让剩余文本留待下一次使用
                    if available < self.min_branches:
                        break

                    if remaining_slots > 1:
                        max_assignable = available - (remaining_slots - 1) * self.min_branches
                        if max_assignable < self.min_branches:
                            # 无法满足后续 sample 的最小需求，保留剩余文本到下次
                            break
                        max_assignable = min(max_assignable, self.max_branches)
                    else:
                        max_assignable = min(self.max_branches, available)

                    max_assignable = max(self.min_branches, max_assignable)
                    num_branches = random.randint(self.min_branches, max_assignable)
                    num_branches = min(num_branches, available)
                    chunk = [self._buffer.popleft() for _ in range(num_branches)]
                    if not chunk:
                        break

                    branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                    samples.append({"main": "", "branches": branches})
            else:
                # 动态模式：每个 sample 的 branches 数量随机，samples 数量不固定
                idx = 0
                buffered_texts = list(self._buffer)
                self._buffer.clear()
                while idx < len(buffered_texts):
                    num_branches = random.randint(self.min_branches, self.max_branches)
                    num_branches = min(num_branches, len(buffered_texts) - idx)

                    if num_branches >= self.min_branches:
                        chunk = buffered_texts[idx : idx + num_branches]
                        branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                        samples.append({"main": "", "branches": branches})
                        idx += num_branches
                    elif num_branches > 0:
                        chunk = buffered_texts[idx : idx + num_branches]
                        branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                        samples.append({"main": "", "branches": branches})
                        idx += num_branches
                    else:
                        break

                # 剩余未使用的文本重新放回缓冲区
                if idx < len(buffered_texts):
                    for txt in buffered_texts[idx:]:
                        self._buffer.append(txt)
        else:
            # 固定模式：每个 sample 固定 branches 数量
            current = list(self._buffer)
            self._buffer.clear()
            for idx in range(0, len(current), self.branches_per_sample):
                chunk = current[idx : idx + self.branches_per_sample]
                if len(chunk) >= self.branches_per_sample:
                    branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                    samples.append({"main": "", "branches": branches})
                elif len(chunk) > 0:
                    branches = [{"text": txt, "answer_offset": 0} for txt in chunk]
                    samples.append({"main": "", "branches": branches})

            # 固定模式下，剩余文本放回缓冲区
            unused = len(current) % self.branches_per_sample
            if unused:
                for txt in current[-unused:]:
                    self._buffer.append(txt)

        if not samples and len(texts) > 0:
            branches = [{"text": txt, "answer_offset": 0} for txt in list(self._buffer)]
            if not branches:
                branches = [{"text": txt, "answer_offset": 0} for txt in texts]
            self._buffer.clear()
            samples.append({"main": "", "branches": branches})

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.pad_to,
            branch_stride=self.branch_stride,
            random_time_offset=self.random_time_offset,
            interleave_branches=self.interleave_branches,
        )

        branch_counts = torch.tensor(
            [len(sample["branches"]) for sample in samples],
            dtype=torch.long,
        )

        # 构造正确的labels：每个token应该预测同一branch的下一个time的token
        # 而不是简单的shift=1（那样会预测到其他branch的token）
        labels = torch.full_like(layout.input_ids, fill_value=-100)

        for batch_idx, meta in enumerate(layout.metadata):
            # 对每个branch，找到它的所有token位置，并设置正确的label
            for branch_idx, branch_id in enumerate(meta.branch_ids):
                branch_pos = meta.branch_positions[branch_idx]

                # 找到该branch的所有有效token位置
                mask = (layout.pos2d[batch_idx, :, 0] == branch_pos) & (layout.attention_mask[batch_idx] == 1)
                indices = mask.nonzero(as_tuple=True)[0]

                if indices.numel() <= 1:
                    # 如果该branch只有0或1个token，没有下一个token可预测
                    continue

                # 对于该branch的每个token（除了最后一个），它的label是下一个token
                # indices[i]位置的token应该预测indices[i+1]位置的token
                for i in range(len(indices) - 1):
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


class ParallelSFTCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        branches_per_sample: int = 4,
        pad_to: Optional[int] = None,
        random_time_offset: bool = True,
        branch_stride: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.branches_per_sample = max(1, branches_per_sample)
        self.pad_to = pad_to
        self.random_time_offset = random_time_offset
        self.branch_stride = branch_stride

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
            branch_stride=self.branch_stride,
            random_time_offset=self.random_time_offset,
        )

        # 构造正确的labels：每个token应该预测同一branch的下一个time的token
        labels = torch.full_like(layout.input_ids, fill_value=-100)

        for batch_idx, meta in enumerate(layout.metadata):
            for branch_idx, branch_id in enumerate(meta.branch_ids):
                branch_pos = meta.branch_positions[branch_idx]
                mask = (layout.pos2d[batch_idx, :, 0] == branch_pos) & (layout.attention_mask[batch_idx] == 1)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() <= 1:
                    continue

                # 设置正确的next-token labels
                for i in range(len(indices) - 1):
                    src_pos = indices[i].item()
                    tgt_pos = indices[i + 1].item()
                    labels[batch_idx, src_pos] = layout.input_ids[batch_idx, tgt_pos].item()

                # 对于SFT，在answer_offset之前的token不参与loss计算
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
