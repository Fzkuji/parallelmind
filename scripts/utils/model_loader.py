from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.inference_hf_lora import load_model_with_lora


def load_model(args):
    if args.hf_base_model:
        if not args.lora_path:
            raise ValueError("--lora_path 必须指定，用于加载 LoRA 权重")
        model, tokenizer, patch_rope = load_model_with_lora(
            base_model=args.hf_base_model,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            rope_2d_ratio=args.rope_2d_ratio,
            patch_rope=not args.no_patch_rope,
            device=args.device,
            dtype=args.hf_dtype,
        )
        model._uses_pos2d = patch_rope
        model._is_hf = True
        return model, tokenizer

    ckpt_path = Path(args.model_path) if args.model_path else _default_ckpt(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        tokenizer_path = checkpoint.get("tokenizer_path", "./model/")
    else:
        state_dict = checkpoint
        tokenizer_path = "./model/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(checkpoint, dict) and "vocab_size" in checkpoint:
        vocab_size = checkpoint["vocab_size"]
        print(f"✓ 从checkpoint加载 vocab_size: {vocab_size}")
    else:
        vocab_size = len(tokenizer)
        print(f"✓ 从tokenizer获取 vocab_size: {vocab_size}")

    has_fpe = any(k.startswith("model.fourier_pe.") for k in state_dict.keys())
    if getattr(args, "pe", None):
        pe_type = args.pe
        print(f"✓ 使用用户指定的位置编码: {pe_type}")
    else:
        pe_type = "fpe" if has_fpe else "rope"
        print(f"✓ 自动检测到位置编码类型: {pe_type}")

    fpe_max_positions = 512
    if pe_type == "fpe" and "model.fourier_pe.pe" in state_dict:
        fpe_max_positions = state_dict["model.fourier_pe.pe"].shape[0]
        print(f"✓ 自动检测到 fpe_max_positions: {fpe_max_positions}")

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        vocab_size=vocab_size,
        inference_rope_scaling=args.inference_rope_scaling,
        pe_type=pe_type,
        fpe_max_positions=fpe_max_positions,
    )

    model = MiniMindForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device).eval()
    model._uses_pos2d = (pe_type == "rope")
    model._is_hf = False
    return model, tokenizer


def _default_ckpt(args) -> Path:
    suffix = "_moe" if args.use_moe else ""
    prefix = "pretrain" if args.mode == "pretrain" else "full_sft"
    return Path(args.out_dir) / f"{prefix}_{args.hidden_size}{suffix}.pth"
