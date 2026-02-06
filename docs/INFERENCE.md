# 推理指南

## 快速选择

| 场景 | 推荐脚本 |
|------|----------|
| 单条测试 / 交互对话 | `src/inference/inference_hf_lora.py` |
| 批量推理（1-1000条） | `src/inference/parallel_generate.py` |
| 大规模推理（10000+） | `src/inference/parallel_inference_hf_lora.py` |

## 单条推理

```bash
python src/inference/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --prompt "你好"
```

**交互模式：**
```bash
python src/inference/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --mode chat
```

## 批量推理

```bash
# 直接输入问题
python src/inference/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompts "问题1" "问题2" "问题3"

# 从文件读取
python src/inference/parallel_generate.py \
  --hf_base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --prompts_file questions.txt
```

## 大规模推理（多 GPU）

```bash
torchrun --nproc_per_node 8 src/inference/parallel_inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_path out/lora/qwen2_lora_final.pth \
  --data_path dataset/test.jsonl \
  --out_path out/results.jsonl \
  --batch_size 8 \
  --batch_by_samples
```

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lora_rank` | LoRA rank（必须与训练一致） | 8 |
| `--rope_2d_ratio` | 2D RoPE 比例（必须与训练一致） | 0.5 |
| `--max_new_tokens` | 最大生成长度 | 512 |
| `--temperature` | 温度（越低越确定） | 0.7 |
| `--no_patch_rope` | 禁用 2D RoPE | - |

## 生成参数调优

| 场景 | temperature | top_p |
|------|-------------|-------|
| 事实问答 | 0.1-0.3 | 0.8 |
| 创意写作 | 0.7-0.9 | 0.95 |
| 代码生成 | 0.1-0.2 | 0.8 |

## 常见问题

**Q: `RuntimeError: extra_pos2d is not set`**

使用我们的推理脚本，pos2d 会自动处理。如果自己写代码，需要在生成前调用 `set_rope_pos2d()`。

**Q: `size mismatch` 错误**

确保 `--lora_rank` 与训练时一致。

**Q: 显存不足**

减小 `--batch_size` 或 `--max_new_tokens`，或使用 `--dtype bfloat16`。
