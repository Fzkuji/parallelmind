import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ParallelPretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        slice_count: Optional[int] = None,
        slice_index: int = 0,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.offsets: List[int] = []
        self._file_handle = None
        self.slice_count = slice_count
        self.slice_index = slice_index

        # 只构建索引，不加载数据
        with open(data_path, "r", encoding="utf-8") as handle:
            offset = 0
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line.encode('utf-8'))

        total_samples = len(self.offsets)

        if self.slice_count is not None:
            if self.slice_count <= 0:
                raise ValueError(f"slice_count must be positive, got {self.slice_count}")
            if not (0 <= self.slice_index < self.slice_count):
                raise ValueError(
                    f"slice_index must be in [0, {self.slice_count - 1}], got {self.slice_index}"
                )

            original_offsets = self.offsets
            self.offsets = [
                offset_value
                for sample_idx, offset_value in enumerate(original_offsets)
                if sample_idx % self.slice_count == self.slice_index
            ]
            print(
                f"数据集初始化完成: {total_samples:,} 个样本 | 应用 branch 切分 {self.slice_index + 1}/{self.slice_count} 后剩余 {len(self.offsets):,} 个样本"
            )
        else:
            print(f"数据集初始化完成: {total_samples:,} 个样本")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, str]:
        # 懒加载：只在需要时读取
        # 保持文件句柄打开以提高性能
        if self._file_handle is None:
            self._file_handle = open(self.data_path, "r", encoding="utf-8")

        self._file_handle.seek(self.offsets[index])
        line = self._file_handle.readline()
        data = json.loads(line.strip())
        text = str(data.get("text", ""))
        return {"text": text}

    def __del__(self):
        # 清理文件句柄
        if self._file_handle is not None:
            self._file_handle.close()


class ParallelSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_tokens = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
        self.conversations: List[List[Dict[str, str]]] = []
        self.mapping: List[Tuple[int, int]] = []
        self._index_dataset(jsonl_path)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        conv_idx, turn_idx = self.mapping[index]
        history = self.conversations[conv_idx][: turn_idx + 1]
        prompt_text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        ).input_ids[: self.max_length]
        answer_offset = self._locate_answer_start(input_ids)
        if len(input_ids) < self.max_length:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_id] * (self.max_length - len(input_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "answer_offset": torch.tensor(answer_offset, dtype=torch.long),
        }

    def _index_dataset(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                convo = [{"role": turn["role"], "content": turn["content"]} for turn in data.get("conversations", [])]
                conv_idx = len(self.conversations)
                self.conversations.append(convo)
                for idx, turn in enumerate(convo):
                    if turn.get("role") == "assistant":
                        self.mapping.append((conv_idx, idx))

    def _locate_answer_start(self, input_ids: List[int]) -> int:
        bos = self.bos_tokens
        for idx in range(len(input_ids)):
            if input_ids[idx : idx + len(bos)] == bos:
                start = idx + len(bos)
                return start
        return 0
