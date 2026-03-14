#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoTokenizer


def _normalize_context(raw_context: Any) -> List[Tuple[str, List[str]]]:
    if isinstance(raw_context, dict):
        titles = list(raw_context.get("title", []))
        sentences = list(raw_context.get("sentences", []))
        output = []
        for title, sents in zip(titles, sentences):
            s_list = [str(s).strip() for s in sents if str(s).strip()]
            output.append((str(title), s_list))
        return output

    output = []
    if isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sentences = [str(s).strip() for s in item[1] if str(s).strip()]
                output.append((title, sentences))
    return output


def _normalize_support_titles(raw_supporting_facts: Any) -> List[str]:
    if isinstance(raw_supporting_facts, dict):
        titles = raw_supporting_facts.get("title", [])
        return list(dict.fromkeys(str(title) for title in titles))

    titles: List[str] = []
    if isinstance(raw_supporting_facts, list):
        for item in raw_supporting_facts:
            if isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
    return list(dict.fromkeys(titles))


def _render_context(
    merged_context: Sequence[Tuple[str, List[str]]],
    max_sentences_per_title: int,
    max_context_chars: int,
) -> str:
    parts: List[str] = []
    total_chars = 0
    for title, sentences in merged_context:
        kept = [sent for sent in sentences[:max_sentences_per_title] if sent]
        if not kept:
            continue
        block = f"[{title}]\n" + " ".join(kept)
        if total_chars + len(block) > max_context_chars:
            remaining = max(0, max_context_chars - total_chars)
            if remaining > 64:
                parts.append(block[:remaining].rstrip())
            break
        parts.append(block)
        total_chars += len(block) + 2
    return "\n\n".join(parts)


def _build_main_text(shared_context_text: str) -> str:
    return (
        "You are given shared Wikipedia context. Each branch will answer a different but related question. "
        "Use the shared context to answer accurately and concisely.\n\n"
        f"Shared context:\n{shared_context_text}"
    )


def _build_branch_record(question: str, answer: str, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    prefix = f"Question: {question.strip()}\nAnswer:"
    text = f"{prefix} {answer.strip()}"
    answer_offset = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    return {
        "text": text,
        "answer_offset": answer_offset,
        "meta": {
            "question": question.strip(),
            "answer": answer.strip(),
        },
    }


def _sample_to_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    context = _normalize_context(sample.get("context", {}))
    support_titles = _normalize_support_titles(sample.get("supporting_facts", {}))
    context_titles = [title for title, _ in context]
    return {
        "id": sample.get("id", sample.get("_id", "")),
        "question": str(sample.get("question", "")).strip(),
        "answer": str(sample.get("answer", "")).strip(),
        "context": context,
        "support_titles": support_titles,
        "context_titles": context_titles,
    }


def load_hotpotqa(
    dataset_name: str,
    subset: str | None,
    split: str,
    max_samples: int | None,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(ds):
        records.append(_sample_to_meta(sample))
        if max_samples is not None and len(records) >= max_samples:
            break
        if (idx + 1) % 10000 == 0:
            print(f"loaded {idx + 1:,} raw samples...")
    return records


def build_groups(
    samples: List[Dict[str, Any]],
    group_size: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    rng = random.Random(seed)
    title_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        for title in set(sample["support_titles"]):
            title_to_indices[title].append(idx)

    order = list(range(len(samples)))
    rng.shuffle(order)
    used = set()
    groups: List[List[Dict[str, Any]]] = []

    for anchor_idx in order:
        if anchor_idx in used:
            continue
        anchor = samples[anchor_idx]
        candidate_scores: Dict[int, int] = defaultdict(int)

        for title in anchor["support_titles"]:
            for cand_idx in title_to_indices.get(title, []):
                if cand_idx == anchor_idx or cand_idx in used:
                    continue
                candidate_scores[cand_idx] += 2

        if len(candidate_scores) < group_size - 1:
            anchor_context_set = set(anchor["context_titles"])
            for cand_idx, cand in enumerate(samples):
                if cand_idx == anchor_idx or cand_idx in used:
                    continue
                overlap = len(anchor_context_set.intersection(cand["context_titles"]))
                if overlap > 0:
                    candidate_scores[cand_idx] += overlap

        ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        selected = [anchor_idx]
        for cand_idx, _score in ranked:
            if len(selected) >= group_size:
                break
            selected.append(cand_idx)

        if len(selected) < group_size:
            continue

        for idx in selected:
            used.add(idx)
        groups.append([samples[idx] for idx in selected])

    return groups


def build_group_record(
    group: Sequence[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_context_titles: int,
    max_sentences_per_title: int,
    max_context_chars: int,
) -> Dict[str, Any]:
    support_titles = []
    for sample in group:
        for title in sample["support_titles"]:
            if title not in support_titles:
                support_titles.append(title)

    title_to_sentences: Dict[str, List[str]] = {}
    for sample in group:
        for title, sentences in sample["context"]:
            if title not in title_to_sentences:
                title_to_sentences[title] = sentences

    ordered_titles = support_titles + [
        title for sample in group for title in sample["context_titles"] if title not in support_titles
    ]
    seen = set()
    merged_context: List[Tuple[str, List[str]]] = []
    for title in ordered_titles:
        if title in seen:
            continue
        seen.add(title)
        if title in title_to_sentences:
            merged_context.append((title, title_to_sentences[title]))
        if len(merged_context) >= max_context_titles:
            break

    shared_context_text = _render_context(
        merged_context,
        max_sentences_per_title=max_sentences_per_title,
        max_context_chars=max_context_chars,
    )

    branches = [
        _build_branch_record(sample["question"], sample["answer"], tokenizer)
        for sample in group
    ]

    return {
        "main": _build_main_text(shared_context_text),
        "branches": branches,
        "meta": {
            "ids": [sample["id"] for sample in group],
            "questions": [sample["question"] for sample in group],
            "shared_titles": [title for title, _ in merged_context],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA into grouped parallel JSONL")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--max_groups", type=int, default=3000)
    parser.add_argument("--group_size", type=int, default=3)
    parser.add_argument("--max_context_titles", type=int, default=8)
    parser.add_argument("--max_sentences_per_title", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading HotpotQA from {args.dataset} ({args.split})...")
    samples = load_hotpotqa(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples):,} normalized samples")

    groups = build_groups(samples, group_size=args.group_size, seed=args.seed)
    if args.max_groups:
        groups = groups[: args.max_groups]
    print(f"Built {len(groups):,} groups with size={args.group_size}")

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            record = build_group_record(
                group,
                tokenizer=tokenizer,
                max_context_titles=args.max_context_titles,
                max_sentences_per_title=args.max_sentences_per_title,
                max_context_chars=args.max_context_chars,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✓ Wrote {written:,} grouped parallel samples -> {output_path}")


if __name__ == "__main__":
    main()
