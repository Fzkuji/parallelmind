#!/bin/bash

# ============================================================================
# ParallelMind Packing 消融实验（chunk-length=2048, FineWeb-Edu）
# ============================================================================
#
# 目标：验证在长分支（2048 tokens/branch）下，time 维度的位置编码是否有用
#       即 rope_2d_ratio < 1.0 是否优于 1.0
#
# 模型配置：
#   hidden_size=1024, num_heads=16 (head_dim=64), num_layers=24, kv_heads=4
#   (~270M 参数，与之前 1024 模型一致)
#
# 数据：FineWeb-Edu sample-10BT (English, Mistral tokenizer)
#   Token packing: 每个 branch 固定 2048 tokens
#
# 显存限制：max_total_tokens=8192 (训练时最多 4 branches)
#
# 实验设计：
#   - rope_2d_ratio: 0, 0.25, 0.5, 0.75, 1.0
#   - 训练分支配置: fixed1, 1-2, 1-4
#   - 评估分支: 1, 2, 4, 8, 16 (eval 时动态调整 max_total_tokens)
#
# 总实验数: 5 ratios × 3 configs = 15 个训练 + 75 个评估
#
# 用法：
#   ./run_ablation_packing_2048.sh           # 正常运行（跳过已完成的）
#   ./run_ablation_packing_2048.sh --force   # 强制重新运行所有
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
        echo "Killing child process $CHILD_PID..."
        kill -TERM "$CHILD_PID" 2>/dev/null
        sleep 2
        kill -9 "$CHILD_PID" 2>/dev/null
    fi
    pkill -P $$ 2>/dev/null
    pkill -9 -f "torchrun.*train_pretrain.py" 2>/dev/null
    pkill -9 -f "torchrun.*eval_loss.py" 2>/dev/null
    echo "Cleanup complete. You can safely restart the script."
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 配置参数
# ============================================================================

# GPU 配置 (跳过 GPU 0)
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NUM_GPUS=7

# 模型配置
HIDDEN_SIZE=1024
NUM_HEADS=16
NUM_LAYERS=24
KV_HEADS=4

# 数据配置
HF_DATASET="HuggingFaceFW/fineweb-edu"
HF_SUBSET="sample-10BT"
CHUNK_LENGTH=2048
TOKENIZER="model_mistral_tok"

# 训练配置
MAX_TOTAL_TOKENS_TRAIN=8192    # 训练时总 token 上限
MAX_SAMPLES=256000              # ~256K packed chunks ≈ 524M tokens
MAX_SEQ_LEN=2048                # RoPE max_positions

# 评估配置
EVAL_SAMPLES=32768              # 评估用文本数量
EVAL_DATA="--hf-dataset ${HF_DATASET} --hf-subset ${HF_SUBSET} --chunk-length ${CHUNK_LENGTH} --tokenizer ${TOKENIZER} --offline"

# rope_2d_ratio 列表
ROPE_RATIOS=("0" "0.25" "0.5" "0.75" "1.0")
ROPE_STRS=("00" "025" "05" "075" "10")

# 训练分支配置: "min,max,batch_size,accum_steps"
TRAIN_CONFIGS=(
    "1,1,4,2"      # fixed1: batch=4, accum=2 (effective batch=8)
    "1,2,2,1"      # 1-2: batch=2, accum=1
    "1,4,1,1"      # 1-4: batch=1, accum=1
)

# 评估分支列表
EVAL_BRANCHES=(1 2 4 8 16)

# 解析参数
FORCE_RERUN=false
if [ "$1" == "--force" ]; then
    FORCE_RERUN=true
    echo "Force mode: will rerun all experiments"
fi

# ============================================================================
# 日志目录
# ============================================================================
LOG_DIR="scripts/logs/ablation_packing_2048"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train.log"
LOSS_FILE="${LOG_DIR}/loss_records.txt"
ERROR_FILE="${LOG_DIR}/errors.txt"
COMPLETED_FILE="${LOG_DIR}/completed.txt"
CSV_FILE="${LOG_DIR}/results.csv"

if [ "$FORCE_RERUN" = true ]; then
    > "$COMPLETED_FILE"
fi

if [ ! -f "$CSV_FILE" ] || [ "$FORCE_RERUN" = true ]; then
    echo "rope_ratio,train_branch,eval_branch,loss,ppl" > "$CSV_FILE"
fi

# ============================================================================
# 工具函数
# ============================================================================

SCRIPT_START_TIME=$(date +%s)
COMPLETED_EXPERIMENTS=0

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

show_progress() {
    local current=$1
    local total=$2
    local start_time=$3

    local now=$(date +%s)
    local elapsed=$((now - start_time))
    local elapsed_str=$(format_duration $elapsed)

    if [ $current -gt 0 ]; then
        local avg_time=$((elapsed / current))
        local remaining=$(((total - current) * avg_time))
        local remaining_str=$(format_duration $remaining)
        local eta=$(date -v+${remaining}S '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d "+${remaining} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        log ">>> 进度: $current/$total ($(( current * 100 / total ))%) | 已用时: $elapsed_str | 预计剩余: $remaining_str | ETA: $eta"
    else
        log ">>> 进度: $current/$total | 已用时: $elapsed_str"
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

record_loss() {
    echo "$1" | tee -a "$LOSS_FILE"
}

record_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ERROR_FILE"
}

is_training_completed() {
    local OUT_DIR=$1
    local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN_SIZE}.pth"
    [ -f "$MODEL_PATH" ]
}

is_eval_completed() {
    local EXP_KEY=$1
    if [ -f "$COMPLETED_FILE" ]; then
        grep -q "^${EXP_KEY}$" "$COMPLETED_FILE" 2>/dev/null
        return $?
    fi
    return 1
}

mark_eval_completed() {
    local EXP_KEY=$1
    echo "$EXP_KEY" >> "$COMPLETED_FILE"
}

parse_eval_output() {
    local OUTPUT="$1"
    local LOSS=$(echo "$OUTPUT" | grep -oP '平均loss=\K[0-9.]+' | tail -1)
    local PPL=$(echo "$OUTPUT" | grep -oP '近似ppl=\K[0-9.]+|inf' | tail -1)
    echo "$LOSS,$PPL"
}

record_csv() {
    local ROPE_RATIO=$1
    local BRANCH_STR=$2
    local EVAL_BRANCH=$3
    local LOSS=$4
    local PPL=$5
    echo "$ROPE_RATIO,$BRANCH_STR,$EVAL_BRANCH,$LOSS,$PPL" >> "$CSV_FILE"
}

check_oom() {
    local OUTPUT="$1"
    echo "$OUTPUT" | grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory"
}

# ============================================================================
# 训练函数
# ============================================================================
run_training() {
    local ROPE_RATIO=$1
    local MIN_BRANCH=$2
    local MAX_BRANCH=$3
    local BATCH_SIZE=$4
    local ACCUM_STEPS=$5
    local OUT_DIR=$6
    local MAX_RETRIES=3
    local RETRY=0

    if [ "$FORCE_RERUN" = false ] && is_training_completed "$OUT_DIR"; then
        log "[SKIP] Training already completed: $OUT_DIR"
        return 0
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log ""
        log "========================================================================"
        log "[TRAINING] Attempt $((RETRY + 1))/$MAX_RETRIES"
        log "  ROPE_2D_RATIO:    $ROPE_RATIO"
        log "  TRAIN_BRANCH:     $MIN_BRANCH-$MAX_BRANCH"
        log "  CHUNK_LENGTH:     $CHUNK_LENGTH"
        log "  MAX_TOTAL_TOKENS: $MAX_TOTAL_TOKENS_TRAIN"
        log "  BATCH_SIZE:       $BATCH_SIZE"
        log "  ACCUM_STEPS:      $ACCUM_STEPS"
        log "  OUT_DIR:          $OUT_DIR"
        log "========================================================================"

        TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port 29501 src/training/train_pretrain.py \
            --use_flex_attention \
            --pe rope \
            --rope_2d_ratio $ROPE_RATIO \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $KV_HEADS \
            --hf-dataset $HF_DATASET \
            --hf-subset $HF_SUBSET \
            --chunk-length $CHUNK_LENGTH \
            --tokenizer $TOKENIZER \
            --offline \
            --max_seq_len $MAX_SEQ_LEN \
            --max_total_tokens $MAX_TOTAL_TOKENS_TRAIN \
            --epochs 1 \
            --batch_size $BATCH_SIZE \
            --accumulation_steps $ACCUM_STEPS \
            --max_branches_per_sample $MAX_BRANCH \
            --min_branches_per_sample $MIN_BRANCH \
            --max-samples $MAX_SAMPLES \
            --out_dir $OUT_DIR \
            --ddp"

        log "[CMD] $TRAIN_CMD"

        set -o pipefail
        eval "$TRAIN_CMD" 2>&1 | tee /tmp/train_output_$$.txt
        TRAIN_EXIT=$?
        set +o pipefail
        TRAIN_OUTPUT=$(cat /tmp/train_output_$$.txt 2>/dev/null)
        rm -f /tmp/train_output_$$.txt

        cat <<< "$TRAIN_OUTPUT" >> "$LOG_FILE"

        if [ $TRAIN_EXIT -eq 0 ]; then
            log "[TRAINING] Completed: $OUT_DIR"
            return 0
        fi

        if check_oom "$TRAIN_OUTPUT"; then
            log "[TRAINING] OOM detected! Reducing batch size..."
            RETRY=$((RETRY + 1))
            if [ $BATCH_SIZE -gt 1 ]; then
                BATCH_SIZE=$((BATCH_SIZE / 2))
                ACCUM_STEPS=$((ACCUM_STEPS * 2))
            else
                ACCUM_STEPS=$((ACCUM_STEPS * 2))
            fi
            log "[TRAINING] Retry with BATCH_SIZE=$BATCH_SIZE, ACCUM_STEPS=$ACCUM_STEPS"
        else
            record_error "[TRAINING FAILED] $OUT_DIR (non-OOM error)"
            return 1
        fi
    done

    record_error "[TRAINING FAILED] $OUT_DIR (max retries reached)"
    return 1
}

# ============================================================================
# 评估函数
# ============================================================================
run_evaluation() {
    local MODEL_PATH=$1
    local ROPE_RATIO=$2
    local ROPE_STR=$3
    local BRANCH_STR=$4
    local VAL_BRANCH=$5
    local MAX_RETRIES=3
    local RETRY=0

    local EXP_KEY="r${ROPE_STR}-b${BRANCH_STR}-eval${VAL_BRANCH}"

    if [ "$FORCE_RERUN" = false ] && is_eval_completed "$EXP_KEY"; then
        local PREV_RESULT=$(grep "^${ROPE_RATIO},${BRANCH_STR},${VAL_BRANCH}," "$CSV_FILE" 2>/dev/null | tail -1)
        if [ -n "$PREV_RESULT" ]; then
            local PREV_LOSS=$(echo "$PREV_RESULT" | cut -d',' -f4)
            local PREV_PPL=$(echo "$PREV_RESULT" | cut -d',' -f5)
            log "[SKIP] Already done: rope=$ROPE_RATIO, train=$BRANCH_STR, eval=$VAL_BRANCH, loss=$PREV_LOSS, ppl=$PREV_PPL"
        else
            log "[SKIP] Evaluation already completed: $EXP_KEY"
        fi
        return 0
    fi

    # 动态计算 eval 时的 max_total_tokens = eval_branch × chunk_length
    local EVAL_MAX_TOTAL=$((VAL_BRANCH * CHUNK_LENGTH))
    local EVAL_BATCH=4

    # 根据 eval_branch 调整 batch_size
    if [ "$VAL_BRANCH" -ge 4 ]; then
        EVAL_BATCH=2
    fi
    if [ "$VAL_BRANCH" -ge 8 ]; then
        EVAL_BATCH=1
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] VAL_BRANCH=$VAL_BRANCH, MAX_TOTAL=$EVAL_MAX_TOTAL, BATCH=$EVAL_BATCH (attempt $((RETRY + 1)))"

        EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port 29501 src/inference/eval_loss.py \
            --model_path $MODEL_PATH \
            --tokenizer $TOKENIZER \
            --use_flex_attention \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $KV_HEADS \
            --pe rope \
            --rope_2d_ratio $ROPE_RATIO \
            --hf-dataset $HF_DATASET \
            --hf-subset $HF_SUBSET \
            --chunk-length $CHUNK_LENGTH \
            --offline \
            --eval_target_samples $EVAL_SAMPLES \
            --batch_size $EVAL_BATCH \
            --max_total_tokens $EVAL_MAX_TOTAL \
            --val_max_branches_per_sample $VAL_BRANCH \
            --val_min_branches_per_sample $VAL_BRANCH"

        EVAL_OUTPUT=$(eval "$EVAL_CMD" 2>&1)
        EVAL_EXIT=$?

        if [ $EVAL_EXIT -eq 0 ]; then
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            echo "$EVAL_OUTPUT" >> "$LOSS_FILE"

            local PARSED=$(parse_eval_output "$EVAL_OUTPUT")
            local LOSS=$(echo "$PARSED" | cut -d',' -f1)
            local PPL=$(echo "$PARSED" | cut -d',' -f2)
            if [ -n "$LOSS" ]; then
                record_csv "$ROPE_RATIO" "$BRANCH_STR" "$VAL_BRANCH" "$LOSS" "$PPL"
                log "[EVAL] Recorded: rope=$ROPE_RATIO, train=$BRANCH_STR, eval=$VAL_BRANCH, loss=$LOSS, ppl=$PPL"
            fi

            mark_eval_completed "$EXP_KEY"
            return 0
        fi

        if check_oom "$EVAL_OUTPUT"; then
            log "[EVAL] OOM detected! Reducing batch size..."
            RETRY=$((RETRY + 1))
            if [ $EVAL_BATCH -gt 1 ]; then
                EVAL_BATCH=$((EVAL_BATCH / 2))
            else
                # batch=1 还 OOM，跳过这个 eval branch
                log "[EVAL] OOM with batch=1, skipping eval_branch=$VAL_BRANCH"
                record_csv "$ROPE_RATIO" "$BRANCH_STR" "$VAL_BRANCH" "OOM" "OOM"
                mark_eval_completed "$EXP_KEY"
                return 0
            fi
        else
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            record_error "[EVAL FAILED] $EXP_KEY (non-OOM error)"
            return 1
        fi
    done

    record_error "[EVAL FAILED] $EXP_KEY (OOM with batch=1)"
    return 1
}

# ============================================================================
# 主流程
# ============================================================================

TOTAL_EXPERIMENTS=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))

echo ""
echo "============================================================================"
echo "ParallelMind Packing 消融实验 (chunk-length=$CHUNK_LENGTH)"
echo "Started at $(date)"
echo ""
echo "模型配置:"
echo "  hidden_size:    $HIDDEN_SIZE"
echo "  num_heads:      $NUM_HEADS (head_dim=$((HIDDEN_SIZE / NUM_HEADS)))"
echo "  num_layers:     $NUM_LAYERS"
echo "  kv_heads:       $KV_HEADS"
echo "  GPUs:           $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""
echo "数据配置:"
echo "  dataset:        $HF_DATASET ($HF_SUBSET)"
echo "  chunk_length:   $CHUNK_LENGTH tokens/branch"
echo "  tokenizer:      $TOKENIZER"
echo "  max_samples:    $MAX_SAMPLES (~$((MAX_SAMPLES * CHUNK_LENGTH / 1000000))M tokens)"
echo "  max_total:      $MAX_TOTAL_TOKENS_TRAIN (训练)"
echo ""
echo "实验配置:"
echo "  rope_2d_ratio:  ${ROPE_RATIOS[*]}"
echo "  训练分支配置:   fixed1, 1-2, 1-4"
echo "  评估分支:       ${EVAL_BRANCHES[*]}"
echo "  总实验数:       $TOTAL_EXPERIMENTS 个训练 + $((TOTAL_EXPERIMENTS * ${#EVAL_BRANCHES[@]})) 个评估"
echo "============================================================================"
echo "" | tee -a "$LOG_FILE"

record_loss ""
record_loss "# ========================================"
record_loss "# Session started at $(date)"
record_loss "# Packing ablation: chunk_length=$CHUNK_LENGTH"
record_loss "# ========================================"

EXPERIMENT_COUNT=0
COMPLETED_COUNT=0
FAILED_COUNT=0

for i in "${!ROPE_RATIOS[@]}"; do
    ROPE_RATIO="${ROPE_RATIOS[$i]}"
    ROPE_STR="${ROPE_STRS[$i]}"

    log ""
    log "########################################################################"
    log "# ROPE_2D_RATIO = $ROPE_RATIO"
    log "########################################################################"

    record_loss ""
    record_loss "========================================"
    record_loss "ROPE_2D_RATIO = $ROPE_RATIO"
    record_loss "========================================"

    for CONFIG in "${TRAIN_CONFIGS[@]}"; do
        MIN_BRANCH=$(echo $CONFIG | cut -d',' -f1)
        MAX_BRANCH=$(echo $CONFIG | cut -d',' -f2)
        BATCH_SIZE=$(echo $CONFIG | cut -d',' -f3)
        ACCUM_STEPS=$(echo $CONFIG | cut -d',' -f4)

        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

        if [ "$MIN_BRANCH" -eq "$MAX_BRANCH" ]; then
            BRANCH_STR="fixed${MAX_BRANCH}"
        else
            BRANCH_STR="${MIN_BRANCH}-${MAX_BRANCH}"
        fi

        OUT_DIR="out/pack2048-h${NUM_HEADS}-r${ROPE_STR}-b${BRANCH_STR}"

        log ""
        log ">>> Experiment $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS"
        log ">>> ROPE=$ROPE_RATIO, TRAIN=$BRANCH_STR"

        record_loss ""
        record_loss "--- EXPERIMENT $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS ---"
        record_loss "ROPE=$ROPE_RATIO, TRAIN=$BRANCH_STR"
        record_loss "OUT_DIR: $OUT_DIR"

        if ! run_training "$ROPE_RATIO" "$MIN_BRANCH" "$MAX_BRANCH" "$BATCH_SIZE" "$ACCUM_STEPS" "$OUT_DIR"; then
            record_loss "[FAILED] Training failed"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi

        MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN_SIZE}.pth"

        if [ ! -f "$MODEL_PATH" ]; then
            record_loss "[ERROR] Model file not found: $MODEL_PATH"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi

        record_loss "--- Evaluation Results ---"
        for VAL_BRANCH in "${EVAL_BRANCHES[@]}"; do
            record_loss "EVAL_BRANCH=$VAL_BRANCH:"
            run_evaluation "$MODEL_PATH" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_BRANCH"
        done

        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
        show_progress $COMPLETED_EXPERIMENTS $TOTAL_EXPERIMENTS $SCRIPT_START_TIME
        record_loss ""
    done
done

# ============================================================================
# 完成总结
# ============================================================================
TOTAL_TIME=$(( $(date +%s) - SCRIPT_START_TIME ))
log ""
log "============================================================================"
log "所有实验完成！"
log "  总耗时:     $(format_duration $TOTAL_TIME)"
log "  成功:       $COMPLETED_COUNT / $TOTAL_EXPERIMENTS"
log "  失败:       $FAILED_COUNT"
log "  结果文件:   $CSV_FILE"
log "============================================================================"

echo ""
echo "结果已保存到 $CSV_FILE"
echo "可用以下命令查看："
echo "  column -t -s',' $CSV_FILE"
