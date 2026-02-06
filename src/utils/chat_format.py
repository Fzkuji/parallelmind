from __future__ import annotations

from typing import Any, List, Sequence


def apply_chat_template(tokenizer, text: str, enable_chat: bool) -> List[int]:
    """Apply tokenizer chat template when needed."""
    if enable_chat:
        history = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = text
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def normalize_pretrain_branches(item: Any) -> List[str]:
    if isinstance(item, dict):
        if "branches" in item and isinstance(item["branches"], Sequence) and not isinstance(item["branches"], (str, bytes)):
            return [str(x) for x in item["branches"] if str(x).strip()]
        if "text" in item:
            text = str(item["text"])
            if "||" in text:
                return [seg.strip() for seg in text.split("||") if seg.strip()]
            return [text]
        if "prompt" in item:
            return [str(item["prompt"])]
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if str(x).strip()]
    text = str(item)
    if "||" in text:
        return [seg.strip() for seg in text.split("||") if seg.strip()]
    return [text]


def ensure_pretrain_template(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if not stripped.startswith("<|im_start|>"):
        stripped = "<|im_start|>" + stripped
    return stripped
