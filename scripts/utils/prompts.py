from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


def load_prompts(args) -> List[Any]:
    """Load prompts from CLI arguments or fallback defaults."""
    if args.prompts:
        return list(args.prompts)

    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            prompts: List[Any] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if args.mode == "pretrain":
                        prompts.append(record)
                    else:
                        prompts.append(str(record.get("prompt", record.get("question", ""))))
            return prompts
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return [
        "请介绍一下自己。",
        "推荐几本好书。",
        "未来的科技趋势是什么？",
        "如何理解大语言模型？",
    ]
