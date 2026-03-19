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
#   每个 micro-batch 随机采样 branch 数 b，batch 内所有 sample 用相同 b
#   固定 batch_size × accumulation_steps，总 token/step = max_b × batch_size × chunk_length × accum
#   不同 branch 数训练的总 token 相同（b 小 → sample 多，b 大 → sample 少）
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

# 训练方式:
#   每个 micro-batch 随机采样一个 branch 数 b，batch 内所有 sample 都用 b 个 branch
#   DataLoader 每次取 max_b × batch_size 个文本，collator 按 b 分组
#   当 b 小时 sample 数多，b 大时 sample 数少，总 token 数恒定 = max_b × batch_size × chunk_length
#   固定 accumulation_steps，梯度累积后 optimizer.step
#
# MAX_TOTAL_TOKENS: 用于 pad_to（所有 sample pad 到相同长度以支持 batching）
MAX_TOTAL_TOKENS=32768   # 16 × 2048
ACCUMULATION_STEPS=2

# 训练分支配置: "min,max,batch_size"
TRAIN_CONFIGS=(
    "1,1,16"    # fixed1: 16 samples/micro-batch
    "1,2,8"     # 1-2: max 8 samples (when b=2), 16 samples (when b=1)
    "1,4,4"     # 1-4
    "1,8,2"     # 1-8
    "1,16,1"    # 1-16
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
    local OUTPUT="$1"
    local EXIT_CODE="${2:-0}"
    # 显式 OOM 消息
    echo "$OUTPUT" | grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory" && return 0
    # exitcode -9 = OOM killer (SIGKILL)
    [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq -9 ] && return 0
    # NCCL timeout / collective failure (通常由某个 rank OOM 引起)
    echo "$OUTPUT" | grep -qi "NCCL operations have failed or timed out" && return 0
    return 1
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
        eval "PYTHONUNBUFFERED=1 $TRAIN_CMD" 2>&1 | tee /tmp/train_output_$$.txt
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

        if check_oom "$OUTPUT" "$EC"; then
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
        local PREV_LOSS=$(echo "$PREV" | cut -d',' -f4)
        if [ -n "$PREV_LOSS" ] && [ "$PREV_LOSS" != "" ]; then
            log "[SKIP] $EXP_KEY → $PREV_LOSS"
            return 0
        else
            # completed 里有但 CSV 里没有 loss，说明之前没跑成功，需要重跑
            log "[REDO] $EXP_KEY → marked complete but no loss in CSV, re-running..."
            sed -i "/^${EXP_KEY}$/d" "$COMPLETED_FILE" 2>/dev/null
        fi
    fi

    local EVAL_TMP="/tmp/eval_output_$$.txt"

    while [ $RETRY -lt $MAX_RETRIES ]; do
        log "[EVAL] $EXP_KEY batch=$EVAL_BATCH (attempt $((RETRY+1))/$MAX_RETRIES)"
        eval "PYTHONUNBUFFERED=1 $EVAL_CMD_BASE --batch_size $EVAL_BATCH" > "$EVAL_TMP" 2>&1
        local EC=$?

        if [ $EC -eq 0 ]; then
            cat "$EVAL_TMP" >> "$LOG_FILE"
            local LOSS=$(grep -oP '平均loss=\K[0-9.]+' "$EVAL_TMP" | tail -1)
            local PPL=$(grep -oP '近似ppl=\K[0-9.]+|inf' "$EVAL_TMP" | tail -1)
            if [ -n "$LOSS" ]; then
                echo "$ROPE_RATIO,$BRANCH_STR,$VAL_BRANCH,$LOSS,$PPL" >> "$CSV_FILE"
                mark_eval_completed "$EXP_KEY"
                log "[DONE] $EXP_KEY → loss=$LOSS (attempt $((RETRY+1)), batch=$EVAL_BATCH)"
                rm -f "$EVAL_TMP"
                return 0
            else
                # exit 0 但没解析到 loss，不标记完成，重试
                RETRY=$((RETRY + 1))
                log "[WARN] $EXP_KEY → exit 0 but no loss parsed (attempt $RETRY/$MAX_RETRIES), retrying..."
                continue
            fi
        fi

        # 非 0 退出：判断是否 OOM（直接 grep 文件，避免大输出传参问题）
        if check_oom "$(grep -m1 -i 'out of memory\|OutOfMemoryError\|CUDA out of memory\|NCCL operations have failed' "$EVAL_TMP" 2>/dev/null)" "$EC"; then
            RETRY=$((RETRY + 1))
            if [ $EVAL_BATCH -gt 1 ]; then
                EVAL_BATCH=1
                log "[OOM]  $EXP_KEY → retry $RETRY/$MAX_RETRIES, batch reduced to 1"
            else
                log "[OOM]  $EXP_KEY → batch=1 still OOM, skipping"
                echo "$ROPE_RATIO,$BRANCH_STR,$VAL_BRANCH,OOM,OOM" >> "$CSV_FILE"
                mark_eval_completed "$EXP_KEY"
                rm -f "$EVAL_TMP"
                return 0
            fi
        else
            cat "$EVAL_TMP" >> "$LOG_FILE"
            # 未知错误也重试，不直接放弃
            RETRY=$((RETRY + 1))
            log "[FAIL] $EXP_KEY → non-OOM error (attempt $RETRY/$MAX_RETRIES), retrying..."
        fi
    done
    rm -f "$EVAL_TMP"
    record_error "[FAIL] $EXP_KEY → max retries ($MAX_RETRIES) exhausted"
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
    log "accumulation_steps: $ACCUMULATION_STEPS"
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
            local BATCH_SIZE=$(echo $CONFIG | cut -d',' -f3)
            COUNT=$((COUNT + 1))

            local BRANCH_STR
            if [ "$MIN_B" -eq "$MAX_B" ]; then BRANCH_STR="fixed${MAX_B}"; else BRANCH_STR="${MIN_B}-${MAX_B}"; fi

            local OUT_DIR="out/${MODEL_TAG}-r${ROPE_STR}-b${BRANCH_STR}"

            # 每个 micro-batch 的总 token 数 = max_b × batch_size × chunk_length
            local MICRO_TOKENS=$((MAX_B * BATCH_SIZE * CHUNK_LENGTH))

            log ""
            log ">>> [${MODEL_TAG}] $COUNT/$STAGE_TOTAL | rope=$ROPE_RATIO, train=$BRANCH_STR, batch=$BATCH_SIZE, micro_tokens=$MICRO_TOKENS"

            # Gradient checkpointing 自适应：训练循环在 seq_len > 8192 时自动启用
            local TRAIN_CMD="PYTHONPATH=. torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT src/training/train_pretrain.py \
                --use_flex_attention \
                --pe rope --rope_2d_ratio $ROPE_RATIO \
                --hidden_size $HIDDEN --num_attention_heads $HEADS --num_hidden_layers $LAYERS --num_key_value_heads $KV_HEADS \
                --hf-dataset $HF_DATASET --hf-subset $HF_SUBSET --chunk-length $CHUNK_LENGTH --tokenizer $TOKENIZER --offline \
                --max_seq_len $MAX_SEQ_LEN --max_total_tokens $MAX_TOTAL_TOKENS \
                --batch_size $BATCH_SIZE --accumulation_steps $ACCUMULATION_STEPS \
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
                    EVAL_BATCH=8
                    [ "$VAL_B" -ge 8 ] && EVAL_BATCH=4
                    [ "$VAL_B" -ge 16 ] && EVAL_BATCH=2
                    [ "$VAL_B" -ge 32 ] && EVAL_BATCH=1
                else
                    EVAL_BATCH=4
                    [ "$VAL_B" -ge 4 ] && EVAL_BATCH=2
                    [ "$VAL_B" -ge 8 ] && EVAL_BATCH=1
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
SMALL_SAMPLES=256000     # ~524M tokens
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
echo "accumulation_steps: $ACCUMULATION_STEPS"
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
