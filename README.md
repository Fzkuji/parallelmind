<div align="center">

# ParallelMind

**Native Parallel Inference for Large Language Models via 2D RoPE**

</div>


## Introduction

ParallelMind is a project evolved from [MiniMind](https://github.com/jingyaogong/minimind), with the core goal of achieving **native parallel inference** for large language models by modifying RoPE (Rotary Position Embedding).

### Core Features

- **2D RoPE Parallel Inference**: Extends traditional 1D positional encoding to 2D, enabling multi-branch parallel generation
- **Flexible Training Methods**:
  - Train MiniMind models from scratch with 2D RoPE
  - Fine-tune pre-trained models with SFT
  - Fine-tune open-source models (e.g., Qwen, LLaMA) with modified RoPE logic using LoRA or full-parameter training
- **Complete Training Pipeline**: Supports pre-training, SFT, LoRA, DPO, and more

### 2D RoPE Principle

Traditional RoPE only encodes the position of tokens in a sequence (time dimension). 2D RoPE extends positional encoding to two dimensions:
- **Time Dimension**: Token position in the sequence (same as traditional RoPE)
- **Branch Dimension**: Which parallel branch the token belongs to

Control the ratio of frequency pairs allocated to the branch dimension via `rope_2d_ratio`:
- `0.25`: 25% for branch, 75% for time
- `0.5`: Default, balanced between branch and time (recommended)
- `0.75`: 75% for branch, 25% for time


## Project Structure

```
parallelmind/
├── model/                    # Model definitions
│   ├── model_minimind.py     # MiniMind model (with 2D RoPE support)
│   └── model_lora.py         # LoRA implementation
├── parallel/                 # Parallel inference core
│   └── columnar.py           # 2D RoPE implementation
├── parallel_data/            # Parallel data processing
│   ├── parallel_dataset.py   # Dataset
│   └── parallel_collator.py  # Data collator
├── trainer/                  # Training scripts
│   ├── train_pretrain.py     # Pre-training
│   ├── train_full_sft.py     # Full-parameter SFT
│   ├── train_hf_lora.py      # HF model LoRA fine-tuning
│   ├── train_hf_full.py      # HF model full-parameter fine-tuning
│   └── ...
├── scripts/                  # Utility scripts
│   ├── parallel_generate.py  # Parallel inference
│   ├── convert_dataset.py    # Dataset conversion
│   └── ...
├── dataset/                  # Dataset directory
└── out/                      # Output directory
```

# Quick Start

<details style="color:rgb(128,128,128)">
<summary>Hardware Configuration Reference</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

## Environment Setup

```bash
git clone https://github.com/your-repo/parallelmind.git
cd parallelmind
pip install -r requirements.txt
```

<details style="color:rgb(128,128,128)">
<summary>Note: Test if PyTorch can use CUDA</summary>

```python
import torch
print(torch.cuda.is_available())
```

If not available, download the appropriate whl file from [torch_stable](https://download.pytorch.org/whl/torch_stable.html).

</details>

## Dataset Download

Download required data files from the [dataset download link](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) and place them in `./dataset/`.

<details style="color:rgb(128,128,128)">
<summary>Dataset Notes</summary>

Recommended: Download `pretrain_hq.jsonl` + `sft_mini_512.jsonl` for quick reproduction.

</details>

### Using Custom Datasets

You can use the conversion script to convert your own dataset to the required JSONL format:

#### Option 1: Use Local Files

```bash
# Pretrain - Plain text file
python scripts/convert_dataset.py --input your_data.txt --output dataset/custom_pretrain.jsonl --mode pretrain

# SFT - Alpaca format JSON
python scripts/convert_dataset.py --input alpaca.json --output dataset/custom_sft.jsonl --mode sft --format alpaca

# SFT - ShareGPT format
python scripts/convert_dataset.py --input sharegpt.json --output dataset/custom_sft.jsonl --mode sft --format sharegpt
```

**Supported Input Formats**:
- **Pretrain**: Plain text (.txt), CSV (.csv), JSON/JSONL
- **SFT**:
  - Standard: `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
  - Alpaca: `{"instruction": "...", "input": "...", "output": "..."}`
  - ShareGPT: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
  - OpenAI: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

**Output Format**:
- **Pretrain**: `{"text": "content"}`
- **SFT**: `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

After conversion, use `--data_path` to specify your data file for training.

#### Option 2: Use Hugging Face Datasets

Convert any Hugging Face dataset with a single command:

```bash
# Example 1: FineWeb-Edu dataset (high-quality web text)
python scripts/convert_dataset.py \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --hf-subset sample-10BT \
  --hf-split train \
  --output dataset/fineweb_pretrain.jsonl \
  --mode pretrain \
  --max-samples 100000

# Example 2: Alpaca instruction dataset
python scripts/convert_dataset.py \
  --hf-dataset tatsu-lab/alpaca \
  --output dataset/alpaca_sft.jsonl \
  --mode sft \
  --format alpaca
```

**Parameters**:
- `--hf-dataset`: Hugging Face dataset name (required)
- `--hf-subset`: Subset name (if dataset has multiple subsets)
- `--hf-split`: Data split, default "train"
- `--text-column`: Text column name (auto-detected by default)
- `--max-samples`: Limit number of samples

**First-time use requires installing the datasets library**:
```bash
pip install datasets
```


# Training

All training scripts are in the `trainer/` directory.

## 1. Pre-training (with 2D RoPE)

Use 2D RoPE for branch position encoding.

**Option A: Train directly from Hugging Face (Recommended!)**:

```bash
# Example: Using FineWeb-Edu dataset
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --hf-subset sample-10BT \
  --max-samples 100000 \
  --chunk-length 512 \
  --pe rope \
  --rope_2d_ratio 0.5 \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --batch_by_samples \
  --max_branches_per_sample 16 \
  --min_branches_per_sample 1 \
  --max_total_tokens 0 \
  --out_dir out/rope_pretrain \
  --ddp
```

**HF Dataset Parameters**:
- `--hf-dataset`: Hugging Face dataset name (required)
- `--hf-subset`: Dataset subset name (optional)
- `--hf-split`: Dataset split, default "train"
- `--text-column`: Text column name (optional, auto-detected)
- `--max-samples`: Limit number of samples
- `--chunk-length`: **Important!** Text chunking length (tokens). For long-text datasets, set to 512 or 1024

**Option B: Use local JSONL files**:

```bash
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --pe rope \
  --rope_2d_ratio 0.5 \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --batch_by_samples \
  --max_branches_per_sample 16 \
  --min_branches_per_sample 1 \
  --val_max_branches_per_sample 4 \
  --val_min_branches_per_sample 4 \
  --max_total_tokens 0 \
  --data_path dataset/pretrain_512.jsonl \
  --max-samples 2048000 \
  --val_samples 500000 \
  --val_interval_tokens 100000000 \
  --out_dir out/rope_pretrain \
  --ddp
```

**Key Parameters**:
- `--rope_2d_ratio 0.5`: Ratio of RoPE dimensions for branch (50% for 2D, 50% for 1D time)
- `--max-samples`: Limit training samples
- `--val_samples`: Number of samples for validation set
- `--val_interval_tokens`: Validate every N effective tokens

**rope_2d_ratio Tuning Guide**:
- `0.25`: 25% for branch, 75% for time (when time dimension is more important)
- `0.5`: Default, balanced (recommended)
- `0.75`: 75% for branch, 25% for time (for stronger branch differentiation)

**Inference Test**:

```bash
python scripts/parallel_generate.py \
  --mode pretrain \
  --prompts "Why does the sun rise in the east" "Introduce large language models" \
  --branches_per_sample 2 \
  --model_path out/rope_pretrain/pretrain_512.pth \
  --streaming \
  --pe rope
```

**Loss Evaluation**:

```bash
torchrun --nproc_per_node 8 scripts/eval_loss.py \
  --model_path out/rope_pretrain/pretrain_512.pth \
  --data_path dataset/pretrain_hq_split.jsonl \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --pe rope \
  --rope_2d_ratio 0.5 \
  --eval_target_samples 2048000 \
  --batch_size 64 \
  --batch_by_samples \
  --val_max_branches_per_sample 4 \
  --val_min_branches_per_sample 4 \
  --max_total_tokens 0
```

> After pre-training, you get `pretrain_*.pth` as the output weights (where * is the model dimension, default 512)


## 2. Supervised Fine-Tuning (SFT)

Fine-tune the pre-trained model with multi-branch parallel training.

**Option A: Train directly from Hugging Face (Recommended!)**:

```bash
# Multi-GPU training (recommended)
torchrun --nproc_per_node 8 trainer/train_full_sft.py \
  --hf-dataset tatsu-lab/alpaca \
  --max-samples 10000 \
  --epochs 2 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 1 \
  --max_seq_len 512 \
  --pe rope \
  --init_weight out/pretrain_512.pth \
  --out_dir out \
  --ddp
```

**Option B: Use local JSONL files**:

```bash
# Multi-GPU training
torchrun --nproc_per_node 8 trainer/train_full_sft.py \
  --epochs 2 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 1 \
  --max_seq_len 512 \
  --pe rope \
  --data_path dataset/sft_512.jsonl \
  --init_weight out/pretrain_512.pth \
  --out_dir out \
  --ddp

# Single GPU training
python trainer/train_full_sft.py \
  --epochs 1 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --max_branches_per_sample 8 \
  --min_branches_per_sample 1 \
  --max_seq_len 512 \
  --pe rope \
  --data_path dataset/sft_512.jsonl \
  --init_weight out/pretrain_512.pth \
  --out_dir out
```

**Common Parameters**:
- `--max_branches_per_sample`: Maximum parallel branches per sample (enables dynamic branch mode)
- `--min_branches_per_sample`: Minimum parallel branches per sample (default 1)
- `--pe`: Position encoding method (rope=RoPE 2D recommended)
- `--init_weight`: Pre-trained model path
- `--data_path`: Local SFT data path (JSONL format)
- `--accumulation_steps`: Gradient accumulation steps. E.g., batch_size=4 + accumulation_steps=2 = effective batch_size of 8

> After SFT, you get `full_sft_*.pth` as the output weights

<details style="color:rgb(128,128,128)">
<summary>Training Notes</summary>

All training processes save parameters to `./out/***.pth` every 100 steps by default.

</details>



## 3. Parallel Inference

Test multi-branch parallel inference:

```bash
# Pretrain model - multi-branch parallel inference (without chat template)
python scripts/parallel_generate.py \
  --mode pretrain \
  --prompts "Why does the sun rise in the east" "Introduce large language models" \
  --branches_per_sample 2 \
  --model_path out/pretrain_512.pth \
  --streaming \
  --pe rope

# SFT model - multi-branch dialogue parallel inference (with chat template)
python scripts/parallel_generate.py \
  --prompts "Introduce yourself" "Explain large language models" "Recommend some books" \
  --branches_per_sample 3 \
  --model_path out/full_sft_512.pth \
  --chat_template \
  --streaming \
  --pe rope

# Using JSONL input
python scripts/parallel_generate.py \
  --mode pretrain \
  --branches_per_sample 4 \
  --prompts_file planner_inputs.jsonl \
  --model_path out/pretrain_512.pth
```

**Parameters**:
- `--mode pretrain`: Pretrain mode (without chat template, for completion)
- `--chat_template`: Enable chat template (for SFT model dialogue)
- `--streaming`: Real-time display of multi-branch generation progress
- `--pe rope`: Specify position encoding type
- JSONL format: `{"branches": ["branch-0","branch-1"]}` or `{"text": "branch-0||branch-1"}`


> [!TIP]
> All training scripts support multi-GPU acceleration. For N GPUs:

Multi-GPU training (DDP, supports multi-node clusters):

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details style="color:rgb(128,128,128)">
<summary>Other Notes</summary>

Enable wandb logging:

```bash
# Login first: wandb login
torchrun --nproc_per_node 8 train_xxx.py --use_wandb
python train_xxx.py --use_wandb
```

Use `--wandb_project` and `--wandb_run_name` to specify project and run names.

</details>


## 4. Fine-tuning HuggingFace Models with 2D RoPE

Fine-tune open-source models (like Qwen, LLaMA) with 2D RoPE for parallel inference.

### LoRA Fine-tuning

```bash
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --data_path dataset/sft_512.jsonl \
  --data_mode parallel_sft \
  --parallel_cache_dir out/parallel_cache \
  --batch_size 4 \
  --accumulation_steps 4 \
  --batch_by_samples \
  --max_branches_per_sample 4 \
  --min_branches_per_sample 1 \
  --branch_stride 512 \
  --align_to right \
  --rope_2d_ratio 0.5 \
  --lora_rank 8 \
  --epochs 1 \
  --save_interval 100 \
  --ddp
```

### Full-parameter Fine-tuning

```bash
torchrun --nproc_per_node 8 trainer/train_hf_full.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --data_path dataset/sft_512.jsonl \
  --data_mode parallel_sft \
  --parallel_cache_dir out/parallel_cache \
  --batch_size 2 \
  --accumulation_steps 8 \
  --batch_by_samples \
  --max_branches_per_sample 4 \
  --min_branches_per_sample 1 \
  --branch_stride 512 \
  --align_to right \
  --rope_2d_ratio 0.5 \
  --epochs 1 \
  --ddp
```

**Key Parameters**:
- `--base_model`: HuggingFace model name or local path
- `--patch_rope / --rope_2d_ratio`: Automatically replace model's RoPE with 2D RoPE
- `--lora_rank`: LoRA rank (4-8 for small models, 16-32 for large models)
- `--branch_stride`: Step size for different branches on the 2D RoPE X-axis
- `--align_to left/right`: Column layout alignment (left=default, right=align at the end)

### LoRA Inference

```bash
python scripts/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --branch_stride 256 \
  --branches_per_sample 3 \
  --out_path out/results.jsonl \
  --max_new_tokens 512 \
  --mode sft \
  --align_to right \
  --prompts "Introduce AI" "Explain deep learning" "NLP applications"
```


# Model Architecture

MiniMind-Dense uses Transformer Decoder-Only architecture (similar to [Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)):

* Pre-normalization with RMSNorm
* SwiGLU activation instead of ReLU
* Rotary Position Embedding (RoPE) instead of absolute position embedding

MiniMind-MoE uses MixFFN from [DeepSeek-V2/3](https://arxiv.org/pdf/2405.04434).

![structure](./images/LLM-structure.png)
![structure-moe](./images/LLM-structure-moe.png)

Model configurations can be found in [./model/model_minimind.py](./model/model_minimind.py).

| Model Name        | params | vocab_size | rope_theta | n_layers | d_model | kv_heads | q_heads | share+route |
|-------------------|--------|------------|------------|----------|---------|----------|---------|-------------|
| MiniMind2-Small   | 26M    | 6400       | 1e6        | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400       | 1e6        | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400       | 1e6        | 16       | 768     | 2        | 8       | -           |


# License

This repository is licensed under the [Apache-2.0 License](LICENSE).


*Based on [MiniMind](https://github.com/jingyaogong/minimind) project.*
