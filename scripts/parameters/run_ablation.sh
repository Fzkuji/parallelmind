#!/bin/bash

# ============================================================================
# ParallelMind 消融实验
# ============================================================================
#
# 数据:    FineWeb-Edu (sample-10BT), Mistral tokenizer, token packing
# 消融变量: rope_2d_ratio × 训练分支配置 × 模型大小
#
# 模型:
#   - small: hidden=512,  heads=8,  layers=8,  kv_heads=2  (~26M params)
#   - large: hidden=1024, heads=16, layers=24, kv_heads=4  (~270M params)
#
# chunk_length=2048 (每个 branch 固定 2048 tokens)
#
# 实验矩阵:
#   rope_2d_ratio:  0, 0.25, 0.5, 0.75, 1.0           (5 个)
#   训练分支配置:   fixed1, 1-2, 1-4, 1-8, 1-16        (5 个)
#   模型:          small, large                          (2 个)
#   → 共 50 个训练 + 每个训练 7 个评估 = 350 个评估
#
# 训练方式:
#   每次前向 1 个 sample（随机 branch 数，无 padding）
#   累积 token ≥ 3/4 × 32768 = 24576 后 optimizer.step
#   每个 step 约 24576~32768 tokens/GPU
#
# 用法:
#   ./run_ablation.sh                          # 运行所有（跳过已完成）
#   ./run_ablation.sh --model small            # 只跑 small
#   ./run_ablation.sh --model large            # 只跑 large
#   ./run_ablation.sh --force                  # 强制重跑
#   ./run_ablation.sh --model large --force    # 强制重跑 large
#
# ============================================================================

# ============================================================================
# 信号处理
# ============================================================================
CHILD_PID=""

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received interrupt signal, cleaning up..."
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null
        sleep 2
        kill -9 "$CHILD_PID" 2>/dev/null
    fi
    pkill -P $$ 2>/dev/null
    pkill -9 -f "torchrun.*train_pretrain.py" 2>/dev/null
    pkill -9 -f "torchrun.*eval_loss.py" 2>/dev/null
    echo "Cleanup complete."
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 解析参数
# ============================================================================
FORCE_RERUN=false
RUN_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_RERUN=true; shift ;;
        --model) RUN_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--model small|large] [--force]"; exit 1 ;;
    esac
done

# ============================================================================
# GPU 配置
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
MASTER_PORT=29501

# ============================================================================
# 通用数据配置
# ============================================================================
HF_DATASET="HuggingFaceFW/fineweb-edu"
HF_SUBSET="sample-10BT"
CHUNK_LENGTH=2048
TOKENIZER="model_mistral_tok"
MAX_SEQ_LEN=2048

# ============================================================================
# 消融变量
# ============================================================================
ROPE_RATIOS=("0" "0.25" "0.5" "0.75" "1.0")
ROPE_STRS=("00" "025" "05" "075" "10")

EVAL_BRANCHES=(1 2 4 8 16 32 64)

# 训练分支配置: "min,max"
# 训练循环按 token 累积决定何时 optimizer.step（阈值 = 3/4 × MAX_TOTAL_TOKENS）
# 每次前向只处理 1 个 sample，无 batch、无 padding
#
# MAX_TOTAL_TOKENS = 每个 optimizer step 的 token 预算 (per GPU)
MAX_TOTAL_TOKENS=32768   # 16 × 2048

# 训练分支配置
TRAIN_CONFIGS=(
    "1,1"     # fixed1
    "1,2"     # 1-2
    "1,4"     # 1-4
    "1,8"     # 1-8
    "1,16"    # 1-16
)

# ============================================================================
# 工具函数
# ============================================================================
SCRIPT_START_TIME=$(date +%s)
GLOBAL_COMPLETED=0
GLOBAL_TOTAL=0

format_duration() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $(((seconds%3600)/60)) $((seconds%60))
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

record_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ERROR_FILE"
}

is_training_completed() {
    local OUT_DIR=$1 HIDDEN=$2
    [ -f "$OUT_DIR/pretrain_${HIDDEN}.pth" ]
}

is_eval_completed() {
    local EXP_KEY=$1
    [ -f "$COMPLETED_FILE" ] && grep -q "^${EXP_KEY}$" "$COMPLETED_FILE" 2>/dev/null
}

mark_eval_completed() {
    echo "$1" >> "$COMPLETED_FILE"
}

parse_eval_output() {
    local OUTPUT="$1"
    local LOSS=$(echo "$OUTPUT" | grep -oP '平均loss=\K[0-9.]+' | tail -1)
    local PPL=$(echo "$OUTPUT" | grep -oP '近似ppl=\K[0-9.]+|inf' | tail -1)
    echo "$LOSS,$PPL"
}

check_oom() {
    echo "$1" | grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory"
}

init_logs() {
    local DIR=$1
    mkdir -p "$DIR"
    LOG_FILE="${DIR}/train.log"
    ERROR_FILE="${DIR}/errors.txt"
    COMPLETED_FILE="${DIR}/completed.txt"
    CSV_FILE="${DIR}/results.csv"
    [ "$FORCE_RERUN" = true ] && > "$COMPLETED_FILE"
    if [ ! -f "$CSV_FILE" ] || [ "$FORCE_RERUN" = true ]; then
        echo "rope_ratio,train_branch,eval_branch,loss,ppl" > "$CSV_FILE"
    fi
}

# ============================================================================
# 训练函数
# ============================================================================
run_training() {
    local TRAIN_CMD=$1
    local OUT_DIR=$2
    local HIDDEN=$3
    local MAX_RETRIES=2
    local RETRY=0

    if [ "$FORCE_RERUN" = false ] && is_training_completed "$OUT_DIR" "$HIDDEN"; then
        log "[SKIP] Training exists: $OUT_DIR"
        return 0
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[TRAIN] Attempt $((RETRY+1))/$MAX_RETRIES → $OUT_DIR"

        set -o pipefail
        eval "$TRAIN_CMD" 2>&1 | tee /tmp/train_output_$$.txt
        local EC=$?
        set +o pipefail
        local OUTPUT=$(cat /tmp/train_output_$$.txt 2>/dev/null)
        rm -f /tmp/train_output_$$.txt
        echo "$OUTPUT" >> "$LOG_FILE"

        if [ $EC -eq 0 ]; then
            log "[TRAIN] Done: $OUT_DIR"
            return 0
        fi

        # 训练完成但 NCCL 退出时崩溃 — 模型已保存则视为成功
        if is_training_completed "$OUT_DIR" "$HIDDEN"; then
            log "[TRAIN] Model saved despite non-zero exit (NCCL cleanup issue): $OUT_DIR"
            return 0
        fi

        if check_oom "$OUTPUT"; then
            RETRY=$((RETRY + 1))
            log "[TRAIN] OOM → retry $((RETRY))/$MAX_RETRIES"
        else
            record_error "[TRAIN FAILED] $OUT_DIR (non-OOM)"
            return 1
        fi
    done
    record_error "[TRAIN FAILED] $OUT_DIR (max retries)"
    return 1
}

# ============================================================================
# 评估函数
# ============================================================================
run_evaluation() {
    local EVAL_CMD_BASE=$1
    local ROPE_RATIO=$2
    local ROPE_STR=$3
    local BRANCH_STR=$4
    local VAL_BRANCH=$5
    local EVAL_BATCH=$6
    local MAX_RETRIES=3
    local RETRY=0

    local EXP_KEY="r${ROPE_STR}-b${BRANCH_STR}-eval${VAL_BRANCH}"

    if [ "$FORCE_RERUN" = false ] && is_eval_completed "$EXP_KEY"; then
        local PREV=$(grep "^${ROPE_RATIO},${BRANCH_STR},${VAL_BRANCH}," "$CSV_FILE" 2>/dev/null | tail -1)
        [ -n "$PREV" ] && log "[SKIP] $EXP_KEY → $(echo $PREV | cut -d',' -f4)" || log "[SKIP] $EXP_KEY"
        return 0
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] $EXP_KEY batch=$EVAL_BATCH (attempt $((RETRY+1)))"
        local EVAL_OUTPUT=$(eval "$EVAL_CMD_BASE --batch_size $EVAL_BATCH" 2>&1)
        local EC=$?

        if [ $EC -eq 0 ]; then
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            local PARSED=$(parse_eval_output "$EVAL_OUTPUT")
            local LOSS=$(echo "$PARSED" | cut -d',' -f1)
            local PPL=$(echo "$PARSED" | cut -d',' -f2)
            if [ -n "$LOSS" ]; then
                echo "$ROPE_RATIO,$BRANCH_STR,$VAL_BRANCH,$LOSS,$PPL" >> "$CSV_FILE"
                log "[EVAL] $EXP_KEY → loss=$LOSS"
            fi
            mark_eval_completed "$EXP_KEY"
            return 0
        fi

        if check_oom "$EVAL_OUTPUT"; then
            RETRY=$((RETRY + 1))
            if [ $EVAL_BATCH -gt 1 ]; then
                EVAL_BATCH=$((EVAL_BATCH / 2))
            else
                log "[EVAL] OOM batch=1, skip $EXP_KEY"
                echo "$ROPE_RATIO,$BRANCH_STR,$VAL_BRANCH,OOM,OOM" >> "$CSV_FILE"
                mark_eval_completed "$EXP_KEY"
                return 0
            fi
        else
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            record_error "[EVAL FAILED] $EXP_KEY (non-OOM)"
            return 1
        fi
    done
    return 1
}

# ============================================================================
# 运行一组模型实验
# ============================================================================
run_model_experiments() {
    local MODEL_NAME=$1   # "small" or "large"
    local HIDDEN=$2
    local HEADS=$3
    local LAYERS=$4
    local KV_HEADS=$5
    local MAX_SAMPLES=$6

    local MODEL_TAG="${HIDDEN}-h${HEADS}-kv${KV_HEADS}-l${LAYERS}"
    local LOG_DIR="scripts/logs/ablation_${MODEL_TAG}"
    init_logs "$LOG_DIR"

    local STAGE_TOTAL=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))

    log ""
    log "================================================================"
    log "模型: ${MODEL_TAG} ($MODEL_NAME)"
    log "数据: $HF_DATASET ($HF_SUBSET), chunk=$CHUNK_LENGTH"
    log "样本: $MAX_SAMPLES (~$((MAX_SAMPLES * CHUNK_LENGTH / 1000000))M tokens)"
    log "token预算: $MAX_TOTAL_TOKENS/step/GPU (阈值 3/4)"
    log "训练: ${#TRAIN_CONFIGS[@]} configs × ${#ROPE_RATIOS[@]} ratios = $STAGE_TOTAL"
    log "评估: ${EVAL_BRANCHES[*]}"
    log "================================================================"

    local COUNT=0
    for i in "${!ROPE_RATIOS[@]}"; do
        local ROPE_RATIO="${ROPE_RATIOS[$i]}"
        local ROPE_STR="${ROPE_STRS[$i]}"

        for CONFIG in "${TRAIN_CONFIGS[@]}"; do
            local MIN_B=$(echo $CONFIG | cut -d',' -f1)
            local MAX_B=$(echo $CONFIG | cut -d',' -f2)
            COUNT=$((COUNT + 1))

            local BRANCH_STR
            if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

            local OUT_DIR="out/${MODEL_TAG}-r${ROPE_STR}-b${BRANCH_STR}"

            log ""
            log ">>> [${MODEL_TAG}] $COUNT/$STAGE_TOTAL | rope=$ROPE_RATIO, train=$BRANCH_STR"

            # Gradient checkpointing: 当最大 sample 长度 > 8192 时启用
            local MAX_SAMPLE_LEN=$((MAX_B * CHUNK_LENGTH))
            local GC_FLAG=""
            if [ "$MAX_SAMPLE_LEN" -gt 8192 ]; then
                GC_FLAG="--gradient_checkpointing"
                log "  gradient_checkpointing: ON (max_sample=$MAX_SAMPLE_LEN)"
            fi

            local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                --use_flex_attention $GC_FLAG \
                --pe rope --rope_2d_ratio $ROPE_RATIO \
                --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                --hf-dataset $HF_DATASET --hf-subset $HF_SUBSET --chunk-length $CHUNK_LENGTH --tokenizer $TOKENIZER --offline \
                --max_seq_len $MAX_SEQ_LEN --max_total_tokens $MAX_TOTAL_TOKENS \
                --epochs 1 \
                --max_branches_per_sample $MAX_B --min_branches_per_sample $MIN_B \
                --max-samples $MAX_SAMPLES \
                --out_dir $OUT_DIR --ddp"

            if ! run_training "$TRAIN_CMD" "$OUT_DIR" "$HIDDEN"; then
                continue
            fi

            local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
            [ ! -f "$MODEL_PATH" ] && continue

            # 评估 (eval 无 backward, 显存占用远小于训练)
            for VAL_B in "${EVAL_BRANCHES[@]}"; do
                local EVAL_MAX_TOTAL=$((VAL_B * CHUNK_LENGTH))
                local EVAL_BATCH
                if [ "$HIDDEN" -le 512 ]; then
                    EVAL_BATCH=16
                    [ "$VAL_B" -ge 8 ] && EVAL_BATCH=8
                    [ "$VAL_B" -ge 16 ] && EVAL_BATCH=4
                    [ "$VAL_B" -ge 32 ] && EVAL_BATCH=2
                    [ "$VAL_B" -ge 64 ] && EVAL_BATCH=1
                else
                    EVAL_BATCH=8
                    [ "$VAL_B" -ge 4 ] && EVAL_BATCH=4
                    [ "$VAL_B" -ge 8 ] && EVAL_BATCH=2
                    [ "$VAL_B" -ge 16 ] && EVAL_BATCH=1
                fi

                local EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/inference/eval_loss.py \
                    --model_path $MODEL_PATH --tokenizer $TOKENIZER \
                    --use_flex_attention \
                    --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                    --pe rope --rope_2d_ratio $ROPE_RATIO \
                    --hf-dataset $HF_DATASET --hf-subset $HF_SUBSET --chunk-length $CHUNK_LENGTH --offline \
                    --eval_target_samples 32768 \
                    --max_total_tokens $EVAL_MAX_TOTAL \
                    --val_max_branches_per_sample $VAL_B --val_min_branches_per_sample $VAL_B"

                run_evaluation "$EVAL_CMD" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_B" "$EVAL_BATCH"
            done

            GLOBAL_COMPLETED=$((GLOBAL_COMPLETED + 1))
            local NOW=$(date +%s)
            local ELAPSED=$((NOW - SCRIPT_START_TIME))
            if [ $GLOBAL_COMPLETED -gt 0 ]; then
                local REMAINING=$(( (GLOBAL_TOTAL - GLOBAL_COMPLETED) * ELAPSED / GLOBAL_COMPLETED ))
                log ">>> 全局: $GLOBAL_COMPLETED/$GLOBAL_TOTAL | 已用: $(format_duration $ELAPSED) | 剩余: ~$(format_duration $REMAINING)"
            fi
        done
    done
}

# ============================================================================
# 主流程
# ============================================================================

echo ""
echo "============================================================================"
echo "ParallelMind 消融实验 (FineWeb-Edu, chunk=$CHUNK_LENGTH)"
echo "Started at $(date)"
echo ""
echo "GPU: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Model filter: ${RUN_MODEL:-all}"
echo "Force: $FORCE_RERUN"
echo "============================================================================"

# --- 模型样本数 ---
SMALL_SAMPLES=512000     # ~1B tokens
LARGE_SAMPLES=256000     # ~524M tokens

# 计算全局总数（所有 config 共用 TRAIN_CONFIGS）
CONFIGS_PER_MODEL=${#TRAIN_CONFIGS[@]}
if [ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "small" ]; then
    GLOBAL_TOTAL=$((GLOBAL_TOTAL + ${#ROPE_RATIOS[@]} * CONFIGS_PER_MODEL))
fi
if [ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "large" ]; then
    GLOBAL_TOTAL=$((GLOBAL_TOTAL + ${#ROPE_RATIOS[@]} * CONFIGS_PER_MODEL))
fi

echo ""
echo "实验总数: $GLOBAL_TOTAL 个训练 + $((GLOBAL_TOTAL * ${#EVAL_BRANCHES[@]})) 个评估"
echo "token预算: $MAX_TOTAL_TOKENS/step/GPU"
echo "============================================================================"
echo ""

# 运行
if [ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "small" ]; then
    run_model_experiments "small" 512 8 8 2 $SMALL_SAMPLES
fi

if [ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "large" ]; then
    run_model_experiments "large" 1024 16 24 4 $LARGE_SAMPLES
fi

# ============================================================================
# 总结
# ============================================================================
TOTAL_TIME=$(($(date +%s) - SCRIPT_START_TIME))
echo ""
echo "============================================================================"
echo "实验完成！总耗时: $(format_duration $TOTAL_TIME)"
echo "完成: $GLOBAL_COMPLETED / $GLOBAL_TOTAL"
echo ""
echo "结果:"
[ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "small" ] && echo "  scripts/logs/ablation_512-h8-kv2-l8/results.csv"
[ -z "$RUN_MODEL" ] || [ "$RUN_MODEL" = "large" ] && echo "  scripts/logs/ablation_1024-h16-kv4-l24/results.csv"
echo "============================================================================"
