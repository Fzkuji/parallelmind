import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ParallelPretrainDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.samples: List[str] = []
        with open(data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = str(data.get("text", ""))
                if text:
                    self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return {"text": self.samples[index]}


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
