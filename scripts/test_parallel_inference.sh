#!/bin/bash
# 快速测试并行推理脚本

set -e

echo "=================================="
echo "并行推理快速测试"
echo "=================================="

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <base_model> <lora_path> <data_path> [num_gpus] [lora_rank]"
    echo ""
    echo "示例:"
    echo "  $0 Qwen/Qwen2-0.5B-Instruct out/lora/qwen2_lora_final.pth dataset/test.jsonl"
    echo "  $0 Qwen/Qwen2-0.5B-Instruct out/lora/qwen2_lora_final.pth dataset/test.jsonl 8 8"
    exit 1
fi

BASE_MODEL=$1
LORA_PATH=$2
DATA_PATH=$3
NUM_GPUS=${4:-1}
LORA_RANK=${5:-8}

echo "基础模型: $BASE_MODEL"
echo "LoRA 路径: $LORA_PATH"
echo "数据路径: $DATA_PATH"
echo "GPU 数量: $NUM_GPUS"
echo "LoRA Rank: $LORA_RANK"
echo ""

# 检查文件是否存在
if [ ! -f "$LORA_PATH" ]; then
    echo "错误: LoRA 文件不存在: $LORA_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p out/parallel_infer_test

# 输出文件路径
OUT_PATH="out/parallel_infer_test/results_$(date +%Y%m%d_%H%M%S).jsonl"

echo "开始推理..."
echo "输出将保存到: $OUT_PATH"
echo ""

# 根据 GPU 数量选择命令
if [ "$NUM_GPUS" -eq 1 ]; then
    # 单 GPU
    echo "=== 单 GPU 模式 ==="
    python scripts/parallel_inference_hf_lora.py \
      --base_model "$BASE_MODEL" \
      --lora_path "$LORA_PATH" \
      --lora_rank "$LORA_RANK" \
      --rope_2d_ratio 0.5 \
      --patch_rope \
      --data_path "$DATA_PATH" \
      --out_path "$OUT_PATH" \
      --batch_size 8 \
      --batch_by_samples \
      --max_branches_per_sample 8 \
      --min_branches_per_sample 1 \
      --max_new_tokens 256 \
      --max_samples 10
else
    # 多 GPU DDP
    echo "=== 多 GPU 模式（$NUM_GPUS 张卡）==="
    torchrun --nproc_per_node "$NUM_GPUS" scripts/parallel_inference_hf_lora.py \
      --base_model "$BASE_MODEL" \
      --lora_path "$LORA_PATH" \
      --lora_rank "$LORA_RANK" \
      --rope_2d_ratio 0.5 \
      --patch_rope \
      --data_path "$DATA_PATH" \
      --out_path "$OUT_PATH" \
      --batch_size 8 \
      --batch_by_samples \
      --max_branches_per_sample 8 \
      --min_branches_per_sample 1 \
      --max_new_tokens 256 \
      --max_samples 10
fi

echo ""
echo "=== 推理完成 ==="
echo ""
echo "查看结果:"
echo "  head -n 5 $OUT_PATH"
echo ""
echo "结果数量:"
wc -l "$OUT_PATH"
echo ""
echo "前3条结果:"
head -n 3 "$OUT_PATH" | python -m json.tool
