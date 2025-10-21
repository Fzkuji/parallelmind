import argparse
import json
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoTokenizer, TextStreamer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_prompts(source: str) -> List[str]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {source} not found")
    if path.suffix == ".jsonl":
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompts.append(str(record.get("prompt", record.get("question", ""))))
        return prompts
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def chunked(items: Sequence[str], chunk_size: int) -> List[List[str]]:
    return [list(items[idx : idx + chunk_size]) for idx in range(0, len(items), chunk_size)]


def init_chat_components(args):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        inference_rope_scaling=args.inference_rope_scaling,
    )
    model_path = Path(args.out_dir) / f"full_sft_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth"
    state_dict = torch.load(model_path, map_location=args.device)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(args.device), tokenizer


def apply_chat_template(tokenizer, history: List[dict], enable_thinking: bool = False) -> str:
    template_args = {
        "conversation": history,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        template_args["enable_thinking"] = True
    return tokenizer.apply_chat_template(**template_args)


def main():
    parser = argparse.ArgumentParser(description="Parallel prompt evaluation")
    parser.add_argument("--prompts-file", type=str, default="", help="文本或 JSONL，按行提供问题")
    parser.add_argument("--prompts", nargs="*", help="直接在命令行提供的问题列表")
    parser.add_argument("--branches-per-sample", type=int, default=4)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", default=False, action="store_true")
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.prompts:
        prompt_list = list(args.prompts)
    elif args.prompts_file:
        prompt_list = load_prompts(args.prompts_file)
    else:
        prompt_list = [
            "请介绍一下自己。",
            "你更擅长哪一个学科？",
            "详细介绍光速的物理概念。",
            "推荐一些杭州的特色美食。",
            "如何理解大语言模型？",
            "请将这句话翻译成英文：传统文化应该在现代社会传承。",
        ]
    if not prompt_list:
        raise ValueError("未提供任何 prompt。可以通过 --prompts 或 --prompts-file 指定。")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = init_chat_components(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_groups = chunked(prompt_list, args.branches_per_sample)
    for group_idx, group in enumerate(prompt_groups):
        print(f"\n=== 批次 {group_idx + 1}/{len(prompt_groups)} ===")
        for prompt in group:
            print(f"[问题] {prompt}")

        # 逐个问题生成回答
        for prompt in group:
            conversation = [{"role": "user", "content": prompt}]
            prompt_text = apply_chat_template(tokenizer, conversation)
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            ).to(args.device)

            print("[回答] ", end="")
            generated = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            response = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(response.strip())


if __name__ == "__main__":
    main()
