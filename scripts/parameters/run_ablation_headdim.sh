#!/bin/bash

# ============================================================================
# ParallelMind Head Dimension 消融实验（1024 hidden size）
# ============================================================================
#
# 固定 hidden_size=1024，变化 num_heads 来测试不同 head_dim 下
# rope_2d_ratio 的最优值是否变化
#
# Head 配置：
#   - num_heads=8  (head_dim=128, 64频率对, kv_heads=2)
#   - num_heads=32 (head_dim=32,  16频率对, kv_heads=8)
#   (num_heads=16 已在 run_ablation_branch_1024.sh 中完成)
#
# 实验设计：
#   - rope_2d_ratio: 0, 0.25, 0.5, 0.75, 1.0
#   - 训练分支配置: fixed1, 1-15
#   - 评估分支: 1, 2, 4, 8, 16, 24, 32, 48, 64
#
# 总实验数: 2 heads × 5 ratios × 2 configs = 20 个训练 + 180 个评估
#
# 用法：
#   ./run_ablation_headdim.sh           # 正常运行
#   ./run_ablation_headdim.sh --force   # 强制重新运行所有实验
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
    echo "Cleanup complete. You can safely restart the script."
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 配置参数
# ============================================================================

NUM_GPUS=8
HIDDEN_SIZE=1024
NUM_LAYERS=24

DATA_PATH="dataset/pretrain_512.jsonl"
MAX_SAMPLES=1024000
VAL_SAMPLES=200000
VAL_INTERVAL_TOKENS=0  # 不做中间验证

# Head 配置: "num_heads,kv_heads"
# num_heads=16 (head_dim=64) 已在 run_ablation_branch_1024.sh 中完成
HEAD_CONFIGS=(
    "8,2"      # head_dim=128, 64频率对, GQA 4:1
    "32,8"     # head_dim=32,  16频率对, GQA 4:1
)

# rope_2d_ratio 列表
ROPE_RATIOS=("0" "0.25" "0.5" "0.75" "1.0")
ROPE_STRS=("00" "025" "05" "075" "10")

# 只训练 fixed1 和 1-15
# "min,max,batch_size,accum_steps"
TRAIN_CONFIGS=(
    "1,15,1,1"     # avg=8, batch=1, accum=1
    "1,1,4,2"      # fixed=1, batch=4, accum=2
)

EVAL_BRANCHES=(1 2 4 8 16 24 32 48 64)

# 解析参数
FORCE_RERUN=false
if [ "$1" == "--force" ]; then
    FORCE_RERUN=true
    echo "Force mode: will rerun all experiments"
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
        local eta=$(date -d "+${remaining} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v+${remaining}S '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "N/A")
        log ">>> 进度: $current/$total ($(( current * 100 / total ))%) | 已用时: $elapsed_str | 预计剩余: $remaining_str | ETA: $eta"
    else
        log ">>> 进度: $current/$total | 已用时: $elapsed_str"
    fi
}

# LOG_FILE 等在外层按 head config 动态设置
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
    local CUR_NUM_HEADS=$7
    local CUR_KV_HEADS=$8
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
        log "  NUM_HEADS:        $CUR_NUM_HEADS (head_dim=$((HIDDEN_SIZE / CUR_NUM_HEADS)))"
        log "  KV_HEADS:         $CUR_KV_HEADS"
        log "  ROPE_2D_RATIO:    $ROPE_RATIO"
        log "  TRAIN_BRANCH:     $MIN_BRANCH-$MAX_BRANCH"
        log "  BATCH_SIZE:       $BATCH_SIZE"
        log "  ACCUM_STEPS:      $ACCUM_STEPS"
        log "  OUT_DIR:          $OUT_DIR"
        log "========================================================================"

        TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS src/training/train_pretrain.py \
            --pe rope \
            --rope_2d_ratio $ROPE_RATIO \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $CUR_NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $CUR_KV_HEADS \
            --epochs 1 \
            --batch_size $BATCH_SIZE \
            --accumulation_steps $ACCUM_STEPS \
            --batch_by_samples \
            --max_branches_per_sample $MAX_BRANCH \
            --min_branches_per_sample $MIN_BRANCH \
            --val_max_branches_per_sample 4 \
            --val_min_branches_per_sample 4 \
            --max_total_tokens 0 \
            --data_path $DATA_PATH \
            --max-samples $MAX_SAMPLES \
            --val_samples $VAL_SAMPLES \
            --val_interval_tokens $VAL_INTERVAL_TOKENS \
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
    local CUR_NUM_HEADS=$6
    local CUR_KV_HEADS=$7
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

    local EVAL_BATCH=16
    if [ "$VAL_BRANCH" -ge 8 ]; then
        EVAL_BATCH=8
    fi
    if [ "$VAL_BRANCH" -ge 32 ]; then
        EVAL_BATCH=4
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] VAL_BRANCH=$VAL_BRANCH, BATCH=$EVAL_BATCH (attempt $((RETRY + 1)))"

        EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS src/inference/eval_loss.py \
            --model_path $MODEL_PATH \
            --data_path dataset/pretrain_hq_split.jsonl \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $CUR_NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $CUR_KV_HEADS \
            --pe rope \
            --rope_2d_ratio $ROPE_RATIO \
            --eval_target_samples 32768 \
            --batch_size $EVAL_BATCH \
            --batch_by_samples \
            --val_max_branches_per_sample $VAL_BRANCH \
            --val_min_branches_per_sample $VAL_BRANCH \
            --max_total_tokens 0"

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
                break
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

TOTAL_EXPERIMENTS=$((${#HEAD_CONFIGS[@]} * ${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))

echo ""
echo "============================================================================"
echo "ParallelMind Head Dimension 消融实验 (hidden_size=$HIDDEN_SIZE)"
echo "Started at $(date)"
echo ""
echo "固定配置:"
echo "  hidden_size:    $HIDDEN_SIZE"
echo "  num_layers:     $NUM_LAYERS"
echo "  训练样本数:     $MAX_SAMPLES"
echo ""
echo "Head 配置:"
for HC in "${HEAD_CONFIGS[@]}"; do
    h=$(echo $HC | cut -d',' -f1)
    kv=$(echo $HC | cut -d',' -f2)
    echo "  - num_heads=$h, head_dim=$((HIDDEN_SIZE / h)), kv_heads=$kv, freq_pairs=$((HIDDEN_SIZE / h / 2))"
done
echo ""
echo "实验配置:"
echo "  rope_2d_ratio:  ${ROPE_RATIOS[*]}"
echo "  训练分支配置:   fixed1, 1-15"
echo "  评估分支:       ${EVAL_BRANCHES[*]}"
echo "  总实验数:       $TOTAL_EXPERIMENTS"
echo "============================================================================"

EXPERIMENT_COUNT=0
COMPLETED_COUNT=0
FAILED_COUNT=0

for HC in "${HEAD_CONFIGS[@]}"; do
    CUR_NUM_HEADS=$(echo $HC | cut -d',' -f1)
    CUR_KV_HEADS=$(echo $HC | cut -d',' -f2)
    CUR_HEAD_DIM=$((HIDDEN_SIZE / CUR_NUM_HEADS))
    CUR_FREQ_PAIRS=$((CUR_HEAD_DIM / 2))

    # 每个 head 配置独立的日志目录
    LOG_DIR="scripts/logs/ablation_1024_h${CUR_NUM_HEADS}_d${CUR_HEAD_DIM}"
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

    log ""
    log "============================================================================"
    log "Head Config: num_heads=$CUR_NUM_HEADS, head_dim=$CUR_HEAD_DIM, kv_heads=$CUR_KV_HEADS"
    log "  频率对数: $CUR_FREQ_PAIRS"
    log "  Log directory: $LOG_DIR"
    log "============================================================================"

    record_loss ""
    record_loss "# ========================================"
    record_loss "# Session started at $(date)"
    record_loss "# Model: ${HIDDEN_SIZE}-h${CUR_NUM_HEADS}-kv${CUR_KV_HEADS}-l${NUM_LAYERS} (head_dim=${CUR_HEAD_DIM})"
    record_loss "# ========================================"

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

            OUT_DIR="out/${HIDDEN_SIZE}-h${CUR_NUM_HEADS}-kv${CUR_KV_HEADS}-l${NUM_LAYERS}-r${ROPE_STR}-b${BRANCH_STR}"

            log ""
            log ">>> Experiment $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS"
            log ">>> HEADS=$CUR_NUM_HEADS (dim=$CUR_HEAD_DIM), ROPE=$ROPE_RATIO, TRAIN=$BRANCH_STR"

            record_loss ""
            record_loss "--- EXPERIMENT $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS ---"
            record_loss "HEADS=$CUR_NUM_HEADS, DIM=$CUR_HEAD_DIM, ROPE=$ROPE_RATIO, TRAIN=$BRANCH_STR"
            record_loss "OUT_DIR: $OUT_DIR"

            if ! run_training "$ROPE_RATIO" "$MIN_BRANCH" "$MAX_BRANCH" "$BATCH_SIZE" "$ACCUM_STEPS" "$OUT_DIR" "$CUR_NUM_HEADS" "$CUR_KV_HEADS"; then
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
                run_evaluation "$MODEL_PATH" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_BRANCH" "$CUR_NUM_HEADS" "$CUR_KV_HEADS"
            done

            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
            show_progress $COMPLETED_EXPERIMENTS $TOTAL_EXPERIMENTS $SCRIPT_START_TIME
            record_loss ""
        done
    done

    log ""
    log "============================================================================"
    log "Head Config h${CUR_NUM_HEADS}_d${CUR_HEAD_DIM} completed"
    log "============================================================================"
done

# ============================================================================
# 统计
# ============================================================================
echo ""
echo "=========================================="
echo "All experiments finished!"
echo ""
echo "Statistics:"
echo "  Total:     $TOTAL_EXPERIMENTS"
echo "  Completed: $COMPLETED_COUNT"
echo "  Failed:    $FAILED_COUNT"
echo ""
echo "Log directories:"
for HC in "${HEAD_CONFIGS[@]}"; do
    h=$(echo $HC | cut -d',' -f1)
    d=$((HIDDEN_SIZE / h))
    echo "  - scripts/logs/ablation_1024_h${h}_d${d}/results.csv"
done
echo ""
echo "To force rerun all: ./run_ablation_headdim.sh --force"
echo "=========================================="
