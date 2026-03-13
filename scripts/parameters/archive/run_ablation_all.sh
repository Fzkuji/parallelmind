#!/bin/bash

# ============================================================================
# ParallelMind 统一消融实验脚本
# ============================================================================
#
# 将所有消融实验合并为一个脚本，按阶段依次执行：
#
#   阶段 1: 512 模型 + 短分支 (batch_by_samples, pretrain_512.jsonl)
#   阶段 2: 1024 模型 + 短分支 (batch_by_samples, pretrain_512.jsonl)
#   阶段 3: 1024 模型 + head dim 消融 (8h/32h, batch_by_samples)
#   阶段 4: 1024 模型 + 长分支 packing (FineWeb-Edu, chunk_length=2048)
#
# 功能：
#   - 自动跳过已完成的训练（检测模型文件）和评估（检测 completed.txt）
#   - OOM 自动重试（减半 batch size）
#   - 中断后可直接重新运行继续
#   - 统一的 CSV 结果格式
#
# 用法：
#   ./run_ablation_all.sh                    # 运行所有阶段
#   ./run_ablation_all.sh --stage 4          # 只运行阶段 4
#   ./run_ablation_all.sh --force            # 强制重新运行所有
#   ./run_ablation_all.sh --stage 4 --force  # 强制重跑阶段 4
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
# 解析参数
# ============================================================================
FORCE_RERUN=false
RUN_STAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_RERUN=true; shift ;;
        --stage) RUN_STAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$FORCE_RERUN" = true ]; then
    echo "Force mode: will rerun all experiments"
fi
if [ -n "$RUN_STAGE" ]; then
    echo "Only running stage $RUN_STAGE"
fi

# ============================================================================
# GPU 配置（跳过 GPU 0）
# ============================================================================
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NUM_GPUS=7
MASTER_PORT=29501

# ============================================================================
# 通用常量
# ============================================================================
ROPE_RATIOS=("0" "0.25" "0.5" "0.75" "1.0")
ROPE_STRS=("00" "025" "05" "075" "10")

# ============================================================================
# 工具函数
# ============================================================================
SCRIPT_START_TIME=$(date +%s)
GLOBAL_COMPLETED=0
GLOBAL_TOTAL=0

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# log/record 函数使用全局 LOG_FILE 等变量（每个阶段会设置）
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
    local HIDDEN=$2
    local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
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
    echo "$1" >> "$COMPLETED_FILE"
}

parse_eval_output() {
    local OUTPUT="$1"
    local LOSS=$(echo "$OUTPUT" | grep -oP '平均loss=\K[0-9.]+' | tail -1)
    local PPL=$(echo "$OUTPUT" | grep -oP '近似ppl=\K[0-9.]+|inf' | tail -1)
    echo "$LOSS,$PPL"
}

record_csv() {
    echo "$1,$2,$3,$4,$5" >> "$CSV_FILE"
}

check_oom() {
    echo "$1" | grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory"
}

show_progress() {
    local current=$1
    local total=$2
    local now=$(date +%s)
    local elapsed=$((now - SCRIPT_START_TIME))
    local elapsed_str=$(format_duration $elapsed)
    if [ $current -gt 0 ]; then
        local avg_time=$((elapsed / current))
        local remaining=$(((total - current) * avg_time))
        local remaining_str=$(format_duration $remaining)
        log ">>> 全局进度: $current/$total | 已用时: $elapsed_str | 预计剩余: $remaining_str"
    fi
}

# ============================================================================
# 初始化阶段日志目录
# ============================================================================
init_stage_logs() {
    local STAGE_LOG_DIR=$1
    mkdir -p "$STAGE_LOG_DIR"
    LOG_FILE="${STAGE_LOG_DIR}/train.log"
    LOSS_FILE="${STAGE_LOG_DIR}/loss_records.txt"
    ERROR_FILE="${STAGE_LOG_DIR}/errors.txt"
    COMPLETED_FILE="${STAGE_LOG_DIR}/completed.txt"
    CSV_FILE="${STAGE_LOG_DIR}/results.csv"

    if [ "$FORCE_RERUN" = true ]; then
        > "$COMPLETED_FILE"
    fi
    if [ ! -f "$CSV_FILE" ] || [ "$FORCE_RERUN" = true ]; then
        echo "rope_ratio,train_branch,eval_branch,loss,ppl" > "$CSV_FILE"
    fi
}

# ============================================================================
# 通用训练函数
# ============================================================================
run_training() {
    local TRAIN_CMD=$1
    local OUT_DIR=$2
    local HIDDEN=$3
    local BATCH_SIZE=$4
    local ACCUM_STEPS=$5
    local MAX_RETRIES=3
    local RETRY=0

    if [ "$FORCE_RERUN" = false ] && is_training_completed "$OUT_DIR" "$HIDDEN"; then
        log "[SKIP] Training already completed: $OUT_DIR"
        return 0
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[TRAINING] Attempt $((RETRY + 1))/$MAX_RETRIES → $OUT_DIR"
        log "[CMD] $TRAIN_CMD"

        set -o pipefail
        eval "$TRAIN_CMD" 2>&1 | tee /tmp/train_output_$$.txt
        local EXIT_CODE=$?
        set +o pipefail
        local OUTPUT=$(cat /tmp/train_output_$$.txt 2>/dev/null)
        rm -f /tmp/train_output_$$.txt

        cat <<< "$OUTPUT" >> "$LOG_FILE"

        if [ $EXIT_CODE -eq 0 ]; then
            log "[TRAINING] Completed: $OUT_DIR"
            return 0
        fi

        if check_oom "$OUTPUT"; then
            RETRY=$((RETRY + 1))
            if [ $BATCH_SIZE -gt 1 ]; then
                BATCH_SIZE=$((BATCH_SIZE / 2))
                ACCUM_STEPS=$((ACCUM_STEPS * 2))
            else
                ACCUM_STEPS=$((ACCUM_STEPS * 2))
            fi
            log "[TRAINING] OOM! Retry with batch=$BATCH_SIZE, accum=$ACCUM_STEPS"
            # 重新构建命令 — 用 sed 替换 batch_size 和 accumulation_steps
            TRAIN_CMD=$(echo "$TRAIN_CMD" | sed "s/--batch_size [0-9]*/--batch_size $BATCH_SIZE/" | sed "s/--accumulation_steps [0-9]*/--accumulation_steps $ACCUM_STEPS/")
        else
            record_error "[TRAINING FAILED] $OUT_DIR (non-OOM error)"
            return 1
        fi
    done

    record_error "[TRAINING FAILED] $OUT_DIR (max retries)"
    return 1
}

# ============================================================================
# 通用评估函数
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
        if [ -n "$PREV" ]; then
            log "[SKIP] Already done: $EXP_KEY ($(echo $PREV | cut -d',' -f4))"
        else
            log "[SKIP] Already done: $EXP_KEY"
        fi
        return 0
    fi

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] $EXP_KEY, batch=$EVAL_BATCH (attempt $((RETRY + 1)))"

        local FULL_CMD="$EVAL_CMD_BASE --batch_size $EVAL_BATCH"
        local EVAL_OUTPUT=$(eval "$FULL_CMD" 2>&1)
        local EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            local PARSED=$(parse_eval_output "$EVAL_OUTPUT")
            local LOSS=$(echo "$PARSED" | cut -d',' -f1)
            local PPL=$(echo "$PARSED" | cut -d',' -f2)
            if [ -n "$LOSS" ]; then
                record_csv "$ROPE_RATIO" "$BRANCH_STR" "$VAL_BRANCH" "$LOSS" "$PPL"
                log "[EVAL] Recorded: $EXP_KEY → loss=$LOSS, ppl=$PPL"
            fi
            mark_eval_completed "$EXP_KEY"
            return 0
        fi

        if check_oom "$EVAL_OUTPUT"; then
            RETRY=$((RETRY + 1))
            if [ $EVAL_BATCH -gt 1 ]; then
                EVAL_BATCH=$((EVAL_BATCH / 2))
            else
                log "[EVAL] OOM with batch=1, skipping $EXP_KEY"
                record_csv "$ROPE_RATIO" "$BRANCH_STR" "$VAL_BRANCH" "OOM" "OOM"
                mark_eval_completed "$EXP_KEY"
                return 0
            fi
        else
            echo "$EVAL_OUTPUT" >> "$LOG_FILE"
            record_error "[EVAL FAILED] $EXP_KEY (non-OOM)"
            return 1
        fi
    done

    record_error "[EVAL FAILED] $EXP_KEY (OOM with batch=1)"
    return 1
}

# ============================================================================
# 计算 eval batch size
# ============================================================================
get_eval_batch_bysample() {
    local VAL_BRANCH=$1
    if [ "$VAL_BRANCH" -ge 24 ]; then echo 1
    elif [ "$VAL_BRANCH" -ge 16 ]; then echo 2
    else echo 4; fi
}

get_eval_batch_packing() {
    local VAL_BRANCH=$1
    if [ "$VAL_BRANCH" -ge 8 ]; then echo 1
    elif [ "$VAL_BRANCH" -ge 4 ]; then echo 2
    else echo 4; fi
}

# ############################################################################
#
#  阶段 1: 512 模型 + 短分支
#
# ############################################################################
run_stage_1() {
    local HIDDEN=512 HEADS=8 LAYERS=8 KV_HEADS=2
    local DATA_PATH="dataset/pretrain_512.jsonl"
    local MAX_SAMPLES=1024000
    local EVAL_BRANCHES=(1 2 4 8 16 24 32 48 64)

    # "min,max,batch,accum"
    local TRAIN_CONFIGS=(
        "1,1,8,1"      # fixed1
        "1,3,4,1"      # 1-3
        "1,7,2,1"      # 1-7
        "1,15,1,1"     # 1-15
    )

    local STAGE_TOTAL=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))
    init_stage_logs "scripts/logs/ablation"

    log ""
    log "################################################################"
    log "# 阶段 1: 512 模型 (h=$HIDDEN, ${HEADS}h, ${LAYERS}L)"
    log "# 训练配置: fixed1, 1-3, 1-7, 1-15"
    log "# 评估分支: ${EVAL_BRANCHES[*]}"
    log "# 实验数: $STAGE_TOTAL"
    log "################################################################"

    local COUNT=0
    for i in "${!ROPE_RATIOS[@]}"; do
        local ROPE_RATIO="${ROPE_RATIOS[$i]}"
        local ROPE_STR="${ROPE_STRS[$i]}"

        for CONFIG in "${TRAIN_CONFIGS[@]}"; do
            local MIN_B=$(echo $CONFIG | cut -d',' -f1)
            local MAX_B=$(echo $CONFIG | cut -d',' -f2)
            local BATCH=$(echo $CONFIG | cut -d',' -f3)
            local ACCUM=$(echo $CONFIG | cut -d',' -f4)
            COUNT=$((COUNT + 1))

            local BRANCH_STR
            if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

            local OUT_DIR="out/${HIDDEN}-h${HEADS}-r${ROPE_STR}-b${BRANCH_STR}"
            log ">>> Stage 1: $COUNT/$STAGE_TOTAL | rope=$ROPE_RATIO, train=$BRANCH_STR"

            local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                --pe rope --rope_2d_ratio $ROPE_RATIO \
                --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS \
                --epochs 1 --batch_size $BATCH --accumulation_steps $ACCUM \
                --batch_by_samples \
                --max_branches_per_sample $MAX_B --min_branches_per_sample $MIN_B \
                --val_max_branches_per_sample 4 --val_min_branches_per_sample 4 \
                --max_total_tokens 0 \
                --data_path $DATA_PATH --max-samples $MAX_SAMPLES \
                --out_dir $OUT_DIR --ddp"

            if ! run_training "$TRAIN_CMD" "$OUT_DIR" "$HIDDEN" "$BATCH" "$ACCUM"; then
                continue
            fi

            local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
            [ ! -f "$MODEL_PATH" ] && continue

            for VAL_B in "${EVAL_BRANCHES[@]}"; do
                local EVAL_BATCH=$(get_eval_batch_bysample $VAL_B)
                local EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/inference/eval_loss.py \
                    --model_path $MODEL_PATH \
                    --data_path dataset/pretrain_hq_split.jsonl \
                    --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS \
                    --pe rope --rope_2d_ratio $ROPE_RATIO \
                    --eval_target_samples 32768 \
                    --batch_by_samples \
                    --val_max_branches_per_sample $VAL_B --val_min_branches_per_sample $VAL_B \
                    --max_total_tokens 0"
                run_evaluation "$EVAL_CMD" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_B" "$EVAL_BATCH"
            done

            GLOBAL_COMPLETED=$((GLOBAL_COMPLETED + 1))
            show_progress $GLOBAL_COMPLETED $GLOBAL_TOTAL
        done
    done
}

# ############################################################################
#
#  阶段 2: 1024 模型 + 短分支
#
# ############################################################################
run_stage_2() {
    local HIDDEN=1024 HEADS=16 LAYERS=24 KV_HEADS=4
    local DATA_PATH="dataset/pretrain_512.jsonl"
    local MAX_SAMPLES=1024000
    local EVAL_BRANCHES=(1 2 4 8 16 24 32 48 64)

    local TRAIN_CONFIGS=(
        "1,15,1,1"     # 1-15
        "1,7,1,2"      # 1-7
        "1,3,2,2"      # 1-3
        "1,1,4,2"      # fixed1
    )

    local STAGE_TOTAL=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))
    init_stage_logs "scripts/logs/ablation_1024"

    log ""
    log "################################################################"
    log "# 阶段 2: 1024 模型 (h=$HIDDEN, ${HEADS}h, ${LAYERS}L, kv=$KV_HEADS)"
    log "# 训练配置: 1-15, 1-7, 1-3, fixed1"
    log "# 评估分支: ${EVAL_BRANCHES[*]}"
    log "# 实验数: $STAGE_TOTAL"
    log "################################################################"

    local COUNT=0
    for i in "${!ROPE_RATIOS[@]}"; do
        local ROPE_RATIO="${ROPE_RATIOS[$i]}"
        local ROPE_STR="${ROPE_STRS[$i]}"

        for CONFIG in "${TRAIN_CONFIGS[@]}"; do
            local MIN_B=$(echo $CONFIG | cut -d',' -f1)
            local MAX_B=$(echo $CONFIG | cut -d',' -f2)
            local BATCH=$(echo $CONFIG | cut -d',' -f3)
            local ACCUM=$(echo $CONFIG | cut -d',' -f4)
            COUNT=$((COUNT + 1))

            local BRANCH_STR
            if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

            local OUT_DIR="out/${HIDDEN}-h${HEADS}-kv${KV_HEADS}-l${LAYERS}-r${ROPE_STR}-b${BRANCH_STR}"
            log ">>> Stage 2: $COUNT/$STAGE_TOTAL | rope=$ROPE_RATIO, train=$BRANCH_STR"

            local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                --pe rope --rope_2d_ratio $ROPE_RATIO \
                --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                --epochs 1 --batch_size $BATCH --accumulation_steps $ACCUM \
                --batch_by_samples \
                --max_branches_per_sample $MAX_B --min_branches_per_sample $MIN_B \
                --val_max_branches_per_sample 4 --val_min_branches_per_sample 4 \
                --max_total_tokens 0 \
                --data_path $DATA_PATH --max-samples $MAX_SAMPLES \
                --out_dir $OUT_DIR --ddp"

            if ! run_training "$TRAIN_CMD" "$OUT_DIR" "$HIDDEN" "$BATCH" "$ACCUM"; then
                continue
            fi

            local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
            [ ! -f "$MODEL_PATH" ] && continue

            for VAL_B in "${EVAL_BRANCHES[@]}"; do
                local EVAL_BATCH=$(get_eval_batch_bysample $VAL_B)
                local EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/inference/eval_loss.py \
                    --model_path $MODEL_PATH \
                    --data_path dataset/pretrain_hq_split.jsonl \
                    --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                    --pe rope --rope_2d_ratio $ROPE_RATIO \
                    --eval_target_samples 32768 \
                    --batch_by_samples \
                    --val_max_branches_per_sample $VAL_B --val_min_branches_per_sample $VAL_B \
                    --max_total_tokens 0"
                run_evaluation "$EVAL_CMD" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_B" "$EVAL_BATCH"
            done

            GLOBAL_COMPLETED=$((GLOBAL_COMPLETED + 1))
            show_progress $GLOBAL_COMPLETED $GLOBAL_TOTAL
        done
    done
}

# ############################################################################
#
#  阶段 3: 1024 模型 + head dim 消融 (8h/32h)
#
# ############################################################################
run_stage_3() {
    local HIDDEN=1024 LAYERS=24
    local DATA_PATH="dataset/pretrain_512.jsonl"
    local MAX_SAMPLES=1024000
    local EVAL_BRANCHES=(1 2 4 8 16 24 32 48 64)

    # head 配置: "heads,kv_heads"
    local HEAD_CONFIGS=("8,2" "32,8")

    local TRAIN_CONFIGS=(
        "1,15,1,1"     # 1-15
        "1,1,4,2"      # fixed1
    )

    local STAGE_TOTAL=$((${#HEAD_CONFIGS[@]} * ${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))
    # 每个 head config 用独立 log 目录（与原始脚本兼容）

    log ""
    log "################################################################"
    log "# 阶段 3: Head Dim 消融 (h=1024, heads=8/32)"
    log "# 实验数: $STAGE_TOTAL"
    log "################################################################"

    for HC in "${HEAD_CONFIGS[@]}"; do
        local HEADS=$(echo $HC | cut -d',' -f1)
        local KV_HEADS=$(echo $HC | cut -d',' -f2)
        local HEAD_DIM=$((HIDDEN / HEADS))

        init_stage_logs "scripts/logs/ablation_1024_h${HEADS}_d${HEAD_DIM}"

        log "--- Head config: ${HEADS}h (dim=${HEAD_DIM}), kv=${KV_HEADS} ---"

        local COUNT=0
        for i in "${!ROPE_RATIOS[@]}"; do
            local ROPE_RATIO="${ROPE_RATIOS[$i]}"
            local ROPE_STR="${ROPE_STRS[$i]}"

            for CONFIG in "${TRAIN_CONFIGS[@]}"; do
                local MIN_B=$(echo $CONFIG | cut -d',' -f1)
                local MAX_B=$(echo $CONFIG | cut -d',' -f2)
                local BATCH=$(echo $CONFIG | cut -d',' -f3)
                local ACCUM=$(echo $CONFIG | cut -d',' -f4)
                COUNT=$((COUNT + 1))

                local BRANCH_STR
                if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

                local OUT_DIR="out/${HIDDEN}-h${HEADS}-kv${KV_HEADS}-l${LAYERS}-r${ROPE_STR}-b${BRANCH_STR}"
                log ">>> Stage 3 [${HEADS}h]: $COUNT | rope=$ROPE_RATIO, train=$BRANCH_STR"

                local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                    --pe rope --rope_2d_ratio $ROPE_RATIO \
                    --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                    --epochs 1 --batch_size $BATCH --accumulation_steps $ACCUM \
                    --batch_by_samples \
                    --max_branches_per_sample $MAX_B --min_branches_per_sample $MIN_B \
                    --val_max_branches_per_sample 4 --val_min_branches_per_sample 4 \
                    --max_total_tokens 0 \
                    --data_path $DATA_PATH --max-samples $MAX_SAMPLES \
                    --out_dir $OUT_DIR --ddp"

                if ! run_training "$TRAIN_CMD" "$OUT_DIR" "$HIDDEN" "$BATCH" "$ACCUM"; then
                    continue
                fi

                local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
                [ ! -f "$MODEL_PATH" ] && continue

                for VAL_B in "${EVAL_BRANCHES[@]}"; do
                    local EVAL_BATCH=$(get_eval_batch_bysample $VAL_B)
                    local EVAL_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/inference/eval_loss.py \
                        --model_path $MODEL_PATH \
                        --data_path dataset/pretrain_hq_split.jsonl \
                        --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                        --pe rope --rope_2d_ratio $ROPE_RATIO \
                        --eval_target_samples 32768 \
                        --batch_by_samples \
                        --val_max_branches_per_sample $VAL_B --val_min_branches_per_sample $VAL_B \
                        --max_total_tokens 0"
                    run_evaluation "$EVAL_CMD" "$ROPE_RATIO" "$ROPE_STR" "$BRANCH_STR" "$VAL_B" "$EVAL_BATCH"
                done

                GLOBAL_COMPLETED=$((GLOBAL_COMPLETED + 1))
                show_progress $GLOBAL_COMPLETED $GLOBAL_TOTAL
            done
        done
    done
}

# ############################################################################
#
#  阶段 4: 1024 模型 + 长分支 packing (FineWeb-Edu, chunk=2048)
#
# ############################################################################
run_stage_4() {
    local HIDDEN=1024 HEADS=16 LAYERS=24 KV_HEADS=4
    local HF_DATASET="HuggingFaceFW/fineweb-edu"
    local HF_SUBSET="sample-10BT"
    local CHUNK_LENGTH=2048
    local TOKENIZER="model_mistral_tok"
    local MAX_SAMPLES=256000
    local MAX_SEQ_LEN=2048
    local EVAL_BRANCHES=(1 2 4 8 16 32)

    # "min,max,batch,accum,max_total_tokens"
    local TRAIN_CONFIGS=(
        "1,1,4,4,2048"       # fixed1
        "1,2,4,2,4096"       # 1-2
        "1,4,2,2,8192"       # 1-4
        "1,8,1,2,16384"      # 1-8 (需要 GC)
        "1,16,1,1,32768"     # 1-16 (需要 GC)
    )

    local STAGE_TOTAL=$((${#ROPE_RATIOS[@]} * ${#TRAIN_CONFIGS[@]}))
    init_stage_logs "scripts/logs/ablation_packing_2048"

    log ""
    log "################################################################"
    log "# 阶段 4: 长分支 Packing (chunk=$CHUNK_LENGTH, FineWeb-Edu)"
    log "# 模型: h=$HIDDEN, ${HEADS}h, ${LAYERS}L, kv=$KV_HEADS"
    log "# 训练配置: fixed1, 1-2, 1-4, 1-8, 1-16"
    log "# 评估分支: ${EVAL_BRANCHES[*]}"
    log "# Gradient checkpointing: 序列>8192 时启用"
    log "# 实验数: $STAGE_TOTAL"
    log "################################################################"

    local COUNT=0
    for i in "${!ROPE_RATIOS[@]}"; do
        local ROPE_RATIO="${ROPE_RATIOS[$i]}"
        local ROPE_STR="${ROPE_STRS[$i]}"

        for CONFIG in "${TRAIN_CONFIGS[@]}"; do
            local MIN_B=$(echo $CONFIG | cut -d',' -f1)
            local MAX_B=$(echo $CONFIG | cut -d',' -f2)
            local BATCH=$(echo $CONFIG | cut -d',' -f3)
            local ACCUM=$(echo $CONFIG | cut -d',' -f4)
            local MAX_TOTAL=$(echo $CONFIG | cut -d',' -f5)
            COUNT=$((COUNT + 1))

            local BRANCH_STR
            if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

            local OUT_DIR="out/pack2048-h${HEADS}-r${ROPE_STR}-b${BRANCH_STR}"
            log ">>> Stage 4: $COUNT/$STAGE_TOTAL | rope=$ROPE_RATIO, train=$BRANCH_STR, max_total=$MAX_TOTAL"

            # gradient checkpointing 只在序列>8192时启用
            local GC_FLAG=""
            if [ "$MAX_TOTAL" -gt 8192 ]; then
                GC_FLAG="--gradient_checkpointing"
                log "  GRAD_CKPT: ON"
            fi

            local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                --use_flex_attention $GC_FLAG \
                --pe rope --rope_2d_ratio $ROPE_RATIO \
                --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                --hf-dataset $HF_DATASET --hf-subset $HF_SUBSET --chunk-length $CHUNK_LENGTH --tokenizer $TOKENIZER --offline \
                --max_seq_len $MAX_SEQ_LEN --max_total_tokens $MAX_TOTAL \
                --epochs 1 --batch_size $BATCH --accumulation_steps $ACCUM \
                --max_branches_per_sample $MAX_B --min_branches_per_sample $MIN_B \
                --max-samples $MAX_SAMPLES \
                --out_dir $OUT_DIR --ddp"

            if ! run_training "$TRAIN_CMD" "$OUT_DIR" "$HIDDEN" "$BATCH" "$ACCUM"; then
                continue
            fi

            local MODEL_PATH="$OUT_DIR/pretrain_${HIDDEN}.pth"
            [ ! -f "$MODEL_PATH" ] && continue

            for VAL_B in "${EVAL_BRANCHES[@]}"; do
                local EVAL_MAX_TOTAL=$((VAL_B * CHUNK_LENGTH))
                local EVAL_BATCH=$(get_eval_batch_packing $VAL_B)
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
            show_progress $GLOBAL_COMPLETED $GLOBAL_TOTAL
        done
    done
}

# ============================================================================
# 主流程
# ============================================================================

echo ""
echo "============================================================================"
echo "ParallelMind 统一消融实验"
echo "Started at $(date)"
echo ""
echo "GPU 配置: $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""
echo "实验阶段:"
echo "  1. 512 模型 + 短分支     (20 训练, 180 评估)"
echo "  2. 1024 模型 + 短分支    (20 训练, 180 评估)"
echo "  3. 1024 模型 + head dim  (20 训练, 180 评估)"
echo "  4. 1024 模型 + 2048 pack (25 训练, 150 评估)"
echo ""
echo "Force: $FORCE_RERUN | Stage filter: ${RUN_STAGE:-all}"
echo "============================================================================"

# 计算全局总数
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "1" ]; then GLOBAL_TOTAL=$((GLOBAL_TOTAL + 20)); fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "2" ]; then GLOBAL_TOTAL=$((GLOBAL_TOTAL + 20)); fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "3" ]; then GLOBAL_TOTAL=$((GLOBAL_TOTAL + 20)); fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "4" ]; then GLOBAL_TOTAL=$((GLOBAL_TOTAL + 25)); fi

# 运行选定的阶段
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "1" ]; then run_stage_1; fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "2" ]; then run_stage_2; fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "3" ]; then run_stage_3; fi
if [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "4" ]; then run_stage_4; fi

# 总结
TOTAL_TIME=$(($(date +%s) - SCRIPT_START_TIME))
echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "  总耗时: $(format_duration $TOTAL_TIME)"
echo "  完成:   $GLOBAL_COMPLETED / $GLOBAL_TOTAL"
echo ""
echo "结果文件:"
echo "  scripts/logs/ablation/results.csv              (512 模型)"
echo "  scripts/logs/ablation_1024/results.csv         (1024 模型)"
echo "  scripts/logs/ablation_1024_h8_d128/results.csv (head dim 8h)"
echo "  scripts/logs/ablation_1024_h32_d32/results.csv (head dim 32h)"
echo "  scripts/logs/ablation_packing_2048/results.csv (2048 packing)"
echo "============================================================================"
