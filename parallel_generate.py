import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, TextStreamer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_prompts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {path} 不存在")
    if path.suffix == ".jsonl":
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompts.append(str(record.get("prompt", record.get("question", ""))))
        return prompts
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_prompt(tokenizer, text: str, enable_chat: bool) -> str:
    if enable_chat:
        conversation = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        inference_rope_scaling=args.inference_rope_scaling,
    )

    ckpt_path = Path(args.model_path) if args.model_path else Path(args.out_dir) / f"full_sft_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device).eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: List[str], args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    results = []
    for prompt in prompts:
        prompt_text = prepare_prompt(tokenizer, prompt, enable_chat=args.chat_template)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(args.device)
        with torch.no_grad():
            generated = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                attention_mask=inputs.get("attention_mask"),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer if args.stream else None,
            )
        output_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append((prompt, output_text.strip()))
    return results


def main():
    parser = argparse.ArgumentParser(description="MiniMind 批量推理脚本")
    parser.add_argument("--prompts", nargs="*", help="直接在命令行输入的 prompt")
    parser.add_argument("--prompts_file", type=str, default="", help="包含多条 prompt 的文本或 JSONL 文件")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="", help="可选，指定 .pth 权重路径")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--inference_rope_scaling", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--do_sample", action="store_true", help="使用采样 (默认关闭)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--chat_template", action="store_true", help="是否使用 ChatML 模板包装 prompt")
    parser.add_argument("--stream", action="store_true", help="实时流式输出")
    parser.add_argument("--batch_size", type=int, default=4, help="一次处理多少条 prompt")
    args = parser.parse_args()

    model, tokenizer = load_model(args)

    if args.prompts:
        prompt_list = list(args.prompts)
    elif args.prompts_file:
        prompt_list = load_prompts(Path(args.prompts_file))
    else:
        prompt_list = [
            "请介绍一下自己。",
            "推荐几本好书。",
            "未来的科技趋势是什么？",
            "如何理解大语言模型？",
        ]

    for idx in range(0, len(prompt_list), args.batch_size):
        sub_prompts = prompt_list[idx: idx + args.batch_size]
        outputs = generate_batch(model, tokenizer, sub_prompts, args)
        for prompt, response in outputs:
            print("\n=== Prompt ===")
            print(prompt)
            print("--- Response ---")
            print(response if response else "(空响应)")


if __name__ == "__main__":
    main()
