#!/bin/bash

# ============================================================================
# ParallelMind 1024 Hidden Size 消融实验（支持断点续训）
# ============================================================================
#
# 功能：
#   - 自动跳过已完成的实验（检测模型文件是否存在）
#   - 自动跳过已完成的评估（检测 loss 记录）
#   - 支持 OOM 重试（自动减半 batch size）
#   - 中断后可直接重新运行继续
#
# 用法：
#   ./run_ablation_branch_1024.sh           # 正常运行（跳过已完成的）
#   ./run_ablation_branch_1024.sh --force   # 强制重新运行所有实验
#
# 模型配置：
#   - hidden_size: 1024, heads: 16, layers: 24, kv_heads: 4 (~270M params)
#   - 500K samples (缩短版，验证 rope_2d_ratio 在大模型上的表现)
#
# 实验设计：
#   - rope_2d_ratio: 0, 0.25, 0.5, 0.75, 1.0
#   - 训练分支配置: 1-15, fixed1 (1-15 优先训练)
#   - 评估分支: 1, 2, 4, 8, 16, 24, 32, 48, 64
#
# 总实验数: 5 ratios × 2 configs = 10 个训练 + 90 个评估
#
# ============================================================================

# 不要 set -e，我们要自己处理错误
# set -e

# ============================================================================
# 信号处理 - 确保中断时清理子进程，避免 GPU 显存泄漏
# ============================================================================
CHILD_PID=""

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received interrupt signal, cleaning up..."

    # 杀死当前运行的子进程
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "Killing child process $CHILD_PID..."
        kill -TERM "$CHILD_PID" 2>/dev/null
        sleep 2
        kill -9 "$CHILD_PID" 2>/dev/null
    fi

    # 杀死所有相关的 torchrun/python 子进程
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
NUM_HEADS=16
NUM_LAYERS=24
NUM_KV_HEADS=4

# Data settings
DATA_PATH="dataset/pretrain_512.jsonl"
MAX_SAMPLES=500000
VAL_SAMPLES=200000
VAL_INTERVAL_TOKENS=25000000

# 日志目录（独立于 512 实验）
LOG_DIR="scripts/logs/ablation_1024"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train.log"
LOSS_FILE="${LOG_DIR}/loss_records.txt"
ERROR_FILE="${LOG_DIR}/errors.txt"
COMPLETED_FILE="${LOG_DIR}/completed.txt"  # 记录已完成的实验
CSV_FILE="${LOG_DIR}/results.csv"          # 结构化结果汇总

# 解析参数
FORCE_RERUN=false
if [ "$1" == "--force" ]; then
    FORCE_RERUN=true
    echo "Force mode: will rerun all experiments"
    # 清空已完成记录
    > "$COMPLETED_FILE"
fi

# 初始化 CSV 文件（如果不存在或强制重跑）
if [ ! -f "$CSV_FILE" ] || [ "$FORCE_RERUN" = true ]; then
    echo "rope_ratio,train_branch,eval_branch,loss,ppl" > "$CSV_FILE"
fi

# rope_2d_ratio 列表
ROPE_RATIOS=("0" "0.25" "0.5" "0.75" "1.0")
ROPE_STRS=("00" "025" "05" "075" "10")

# 训练分支配置: "min,max,batch_size,accum_steps"
# 目标：每个 step 总 token 数 ≈ 3.2w (8卡 × batch × accum × avg_branch × 512)
# 1-15 优先训练（主要实验条件），fixed1 作为 baseline
TRAIN_CONFIGS=(
    # 动态分支: 1-15 优先
    "1,15,1,1"     # avg=8,  batch=1, accum=1, tokens=8×1×1×8×512=32,768
    # 固定 1 分支 (baseline)
    "1,1,4,2"      # fixed=1, batch=4, accum=2, tokens=8×4×2×1×512=32,768
)

# 评估分支列表
EVAL_BRANCHES=(1 2 4 8 16 24 32 48 64)

# ============================================================================
# 工具函数
# ============================================================================

# 计时相关
SCRIPT_START_TIME=$(date +%s)
COMPLETED_EXPERIMENTS=0

# 格式化秒数为 HH:MM:SS
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# 计算并显示进度和预估时间
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

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

record_loss() {
    echo "$1" | tee -a "$LOSS_FILE"
}

record_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ERROR_FILE"
}

# 检查实验是否已完成（模型文件存在）
is_training_completed() {
    local OUT_DIR=$1
    local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN_SIZE}.pth"
    [ -f "$MODEL_PATH" ]
}

# 检查评估是否已完成
is_eval_completed() {
    local EXP_KEY=$1  # e.g., "r05-bfixed4-eval8"
    if [ -f "$COMPLETED_FILE" ]; then
        grep -q "^${EXP_KEY}$" "$COMPLETED_FILE" 2>/dev/null
        return $?
    fi
    return 1
}

# 标记评估完成
mark_eval_completed() {
    local EXP_KEY=$1
    echo "$EXP_KEY" >> "$COMPLETED_FILE"
}

# 从 eval_loss.py 输出中提取 loss 和 ppl
parse_eval_output() {
    local OUTPUT="$1"
    # 匹配: 评估完成: 平均loss=2.3456, 近似ppl=10.44, ...
    local LOSS=$(echo "$OUTPUT" | grep -oP '平均loss=\K[0-9.]+' | tail -1)
    local PPL=$(echo "$OUTPUT" | grep -oP '近似ppl=\K[0-9.]+|inf' | tail -1)
    echo "$LOSS,$PPL"
}

# 记录到 CSV
record_csv() {
    local ROPE_RATIO=$1
    local BRANCH_STR=$2
    local EVAL_BRANCH=$3
    local LOSS=$4
    local PPL=$5
    echo "$ROPE_RATIO,$BRANCH_STR,$EVAL_BRANCH,$LOSS,$PPL" >> "$CSV_FILE"
}

# 检查是否 OOM
check_oom() {
    local OUTPUT="$1"
    echo "$OUTPUT" | grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory"
}

# ============================================================================
# 训练函数（带 OOM 重试）
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

    # 检查是否已完成
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
        log "  BATCH_SIZE:       $BATCH_SIZE"
        log "  ACCUM_STEPS:      $ACCUM_STEPS"
        log "  OUT_DIR:          $OUT_DIR"
        log "========================================================================"

        TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS src/training/train_pretrain.py \
            --pe rope \
            --rope_2d_ratio $ROPE_RATIO \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $NUM_KV_HEADS \
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

        # 运行训练，捕获输出，记录 PID 以便中断时清理
        eval "$TRAIN_CMD" > /tmp/train_output_$$.txt 2>&1 &
        CHILD_PID=$!
        wait $CHILD_PID
        TRAIN_EXIT=$?
        CHILD_PID=""
        TRAIN_OUTPUT=$(cat /tmp/train_output_$$.txt)
        rm -f /tmp/train_output_$$.txt

        echo "$TRAIN_OUTPUT" >> "$LOG_FILE"

        if [ $TRAIN_EXIT -eq 0 ]; then
            log "[TRAINING] Completed: $OUT_DIR"
            return 0
        fi

        # 检查是否 OOM
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
            # 非 OOM 错误，直接失败
            record_error "[TRAINING FAILED] $OUT_DIR (non-OOM error)"
            return 1
        fi
    done

    record_error "[TRAINING FAILED] $OUT_DIR (max retries reached)"
    return 1
}

# ============================================================================
# 评估函数（带 OOM 重试）
# ============================================================================
run_evaluation() {
    local MODEL_PATH=$1
    local ROPE_RATIO=$2
    local ROPE_STR=$3
    local BRANCH_STR=$4
    local VAL_BRANCH=$5
    local MAX_RETRIES=3
    local RETRY=0

    # 生成唯一 key
    local EXP_KEY="r${ROPE_STR}-b${BRANCH_STR}-eval${VAL_BRANCH}"

    # 检查是否已完成
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

    # 根据分支数设置初始 batch size (更保守，因为 270M 模型)
    local EVAL_BATCH=2
    if [ "$VAL_BRANCH" -ge 16 ]; then
        EVAL_BATCH=1
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] VAL_BRANCH=$VAL_BRANCH, BATCH=$EVAL_BATCH (attempt $((RETRY + 1)))"

        EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS src/inference/eval_loss.py \
            --model_path $MODEL_PATH \
            --data_path dataset/pretrain_hq_split.jsonl \
            --hidden_size $HIDDEN_SIZE \
            --num_attention_heads $NUM_HEADS \
            --num_hidden_layers $NUM_LAYERS \
            --num_key_value_heads $NUM_KV_HEADS \
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

            # 解析并记录到 CSV
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

        # 检查是否 OOM
        if check_oom "$EVAL_OUTPUT"; then
            log "[EVAL] OOM detected! Reducing batch size..."
            RETRY=$((RETRY + 1))

            if [ $EVAL_BATCH -gt 1 ]; then
                EVAL_BATCH=$((EVAL_BATCH / 2))
            else
                # batch=1 还 OOM，放弃
                break
            fi
        else
            # 非 OOM 错误
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

log ""
log "============================================================================"
log "ParallelMind 1024 Hidden Size 消融实验"
log "Started/Resumed at $(date)"
log "Log directory: $LOG_DIR"
log "Force rerun: $FORCE_RERUN"
log ""
log "模型配置:"
log "  hidden_size:    $HIDDEN_SIZE"
log "  num_heads:      $NUM_HEADS"
log "  num_layers:     $NUM_LAYERS"
log "  num_kv_heads:   $NUM_KV_HEADS"
log "  参数量:         ~270M"
log ""
log "实验配置:"
log "  rope_2d_ratio:  ${ROPE_RATIOS[*]}"
log "  训练分支配置:   fixed1, 1-3, 1-7, 1-15"
log "  评估分支:       ${EVAL_BRANCHES[*]}"
log "  训练样本数:     $MAX_SAMPLES"
log "  验证样本数:     $VAL_SAMPLES"
log "  总实验数: ${#ROPE_RATIOS[@]} ratios × ${#TRAIN_CONFIGS[@]} configs = $((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]})) 个"
log "============================================================================"

record_loss ""
record_loss "# ========================================"
record_loss "# Session started at $(date)"
record_loss "# Model: ${HIDDEN_SIZE}-h${NUM_HEADS}-kv${NUM_KV_HEADS}-l${NUM_LAYERS}"
record_loss "# ========================================"

EXPERIMENT_COUNT=0
COMPLETED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
TOTAL_EXPERIMENTS=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))

# 遍历所有 rope_2d_ratio
for i in "${!ROPE_RATIOS[@]}"; do
    ROPE_RATIO="${ROPE_RATIOS[$i]}"
    ROPE_STR="${ROPE_STRS[$i]}"

    log ""
    log "############################################################################"
    log "# ROPE_2D_RATIO = $ROPE_RATIO"
    log "############################################################################"

    record_loss ""
    record_loss "========================================"
    record_loss "ROPE_2D_RATIO = $ROPE_RATIO"
    record_loss "========================================"

    # 遍历所有训练配置
    for CONFIG in "${TRAIN_CONFIGS[@]}"; do
        MIN_BRANCH=$(echo $CONFIG | cut -d',' -f1)
        MAX_BRANCH=$(echo $CONFIG | cut -d',' -f2)
        BATCH_SIZE=$(echo $CONFIG | cut -d',' -f3)
        ACCUM_STEPS=$(echo $CONFIG | cut -d',' -f4)

        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

        # 生成分支字符串
        if [ "$MIN_BRANCH" -eq "$MAX_BRANCH" ]; then
            BRANCH_STR="fixed${MAX_BRANCH}"
        else
            BRANCH_STR="${MIN_BRANCH}-${MAX_BRANCH}"
        fi

        OUT_DIR="out/${HIDDEN_SIZE}-h${NUM_HEADS}-kv${NUM_KV_HEADS}-l${NUM_LAYERS}-r${ROPE_STR}-b${BRANCH_STR}"

        log ""
        log ">>> Experiment $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS"
        log ">>> ROPE=$ROPE_RATIO, TRAIN_BRANCH=$BRANCH_STR"

        record_loss ""
        record_loss "--- EXPERIMENT $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS ---"
        record_loss "ROPE=$ROPE_RATIO, TRAIN_BRANCH=$BRANCH_STR"
        record_loss "OUT_DIR: $OUT_DIR"

        # 训练
        if ! run_training "$ROPE_RATIO" "$MIN_BRANCH" "$MAX_BRANCH" "$BATCH_SIZE" "$ACCUM_STEPS" "$OUT_DIR"; then
            record_loss "[FAILED] Training failed"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi

        # 评估
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
# 统计
# ============================================================================
log ""
log "============================================================================"
log "Experiments finished at $(date)"
log ""
log "Statistics:"
log "  Total:     $TOTAL_EXPERIMENTS"
log "  Completed: $COMPLETED_COUNT"
log "  Failed:    $FAILED_COUNT"
log ""
log "Log directory: $LOG_DIR"
log "============================================================================"

record_loss ""
record_loss "# ========================================"
record_loss "# Session completed at $(date)"
record_loss "# Total: $TOTAL_EXPERIMENTS, Completed: $COMPLETED_COUNT, Failed: $FAILED_COUNT"
record_loss "# ========================================"

echo ""
echo "=========================================="
echo "Experiments finished!"
echo ""
echo "Statistics:"
echo "  Total:     $TOTAL_EXPERIMENTS"
echo "  Completed: $COMPLETED_COUNT"
echo "  Failed:    $FAILED_COUNT"
echo ""
echo "Log directory: $LOG_DIR"
echo "  - train.log:        完整训练日志"
echo "  - loss_records.txt: Loss 记录（原始输出）"
echo "  - results.csv:      结构化结果汇总（用于绘图）"
echo "  - errors.txt:       错误记录"
echo "  - completed.txt:    已完成的评估记录"
echo ""
echo "To rerun failed experiments, just run this script again."
echo "To force rerun all, use: ./run_ablation_branch_1024.sh --force"
echo "=========================================="
