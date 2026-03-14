import json
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


class GroupedParallelDataset(Dataset):
    """读取 grouped parallel JSONL。

    每行格式示例：
    {
      "main": "共享context文本",
      "branches": [
        {"text": "Question: ...\nAnswer: ...", "answer_offset": 42},
        ...
      ]
    }
    """

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.data_path = data_path
        self.samples: List[Dict[str, Any]] = []
        self._load(max_samples=max_samples)

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
            if "meta" in branch:
                normalized["meta"] = branch["meta"]
            return normalized
        return {"text": str(branch), "answer_offset": 0}

    def _load(self, max_samples: Optional[int] = None) -> None:
        with open(self.data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                branches = record.get("branches", [])
                if not branches:
                    continue
                sample = {
                    "main": str(record.get("main", "")),
                    "branches": [self._normalize_branch(branch) for branch in branches],
                }
                if "meta" in record:
                    sample["meta"] = record["meta"]
                self.samples.append(sample)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
