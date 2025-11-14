## Task Overview
We need to support "parallel-style" SFT fine-tuning for HuggingFace + LoRA models:
1. Convert standard conversation JSONL files (e.g. dataset/sft_512.jsonl) into the parallel pretrain format that `ParallelPretrainCollator` expects, by automatically slicing each conversation into branches, inserting optional padding/random offsets so that different branches can start at different time steps, and ensuring labels are only applied to assistant replies.
2. Update the training entry (trainer/train_hf_lora.py) to accept this converted dataset directly (no manual conversion step), controlled via a flag such as `--data_mode parallel_sft` that triggers the conversion pipeline on the fly.
3. Document the workflow in README so users know how to run the conversion + training pipeline, and describe how to run inference (both single-prompt and columnar multi-branch) with the resulting LoRA.

## Notes for Gemini
- Reuse existing conversion script logic if helpful (scripts/convert_sft_to_pretrain.py), but adapt it to handle arbitrary multi-turn conversations, generating branch texts + offsets.
- Keep the output compatible with ParallelPretrainCollator: each JSONL line should have `{"text": "..."}` (optionally `main`+`branches`) so that collator can build pos2d/time ids.
- The training script should have a clean flag, e.g. `--data_mode parallel_sft`, that: (a) runs the converter (either to temp file or in-memory) and (b) feeds the resulting dataset into the existing parallel data loader without requiring the user to manually run an extra script.
- README section needs to show how to run: conversion (if still needed), training command, and inference command (columnar multi-branch + single prompt).

Please implement the above and ping me when ready.
