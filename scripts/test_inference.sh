#!/bin/bash
# 快速测试推理脚本

set -e

echo "=================================="
echo "LoRA 推理快速测试"
echo "=================================="

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <base_model> <lora_path> [lora_rank]"
    echo ""
    echo "示例:"
    echo "  $0 Qwen/Qwen2-0.5B-Instruct out/lora/qwen2_parallel_lora_hf_final.pth 8"
    exit 1
fi

BASE_MODEL=$1
LORA_PATH=$2
LORA_RANK=${3:-8}

echo "基础模型: $BASE_MODEL"
echo "LoRA 路径: $LORA_PATH"
echo "LoRA Rank: $LORA_RANK"
echo ""

# 检查 LoRA 文件是否存在
if [ ! -f "$LORA_PATH" ]; then
    echo "错误: LoRA 文件不存在: $LORA_PATH"
    exit 1
fi

echo "开始测试..."
echo ""

# 测试单次生成
echo "=== 测试 1: 单次生成 ==="
python scripts/inference_hf_lora.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --lora_rank "$LORA_RANK" \
  --mode generate \
  --prompt "你好，请介绍一下你自己" \
  --max_new_tokens 100

echo ""
echo "=== 测试完成 ==="
echo ""
echo "如需交互式对话，请运行："
echo "python scripts/inference_hf_lora.py \\"
echo "  --base_model $BASE_MODEL \\"
echo "  --lora_path $LORA_PATH \\"
echo "  --lora_rank $LORA_RANK \\"
echo "  --mode chat"
