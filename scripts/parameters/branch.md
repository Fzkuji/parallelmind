(mind) [zichuanfu@ip-10-206-0-17 parallelmind]$ ./experiments/scripts/run_ablation_branch.sh 
[2026-02-05 23:25:06] 
[2026-02-05 23:25:06] ============================================================================
[2026-02-05 23:25:06] ParallelMind 完整消融实验
[2026-02-05 23:25:06] Started/Resumed at Thu Feb  5 23:25:06 CST 2026
[2026-02-05 23:25:06] Log directory: experiments/logs/ablation
[2026-02-05 23:25:06] Force rerun: false
[2026-02-05 23:25:06] 
[2026-02-05 23:25:06] 实验配置:
[2026-02-05 23:25:06]   rope_2d_ratio: 0 0.25 0.5 0.75 1.0
[2026-02-05 23:25:06]   固定分支训练: 1, 2, 4, 8, 16
[2026-02-05 23:25:06]   动态分支训练: 1-3, 1-7, 1-15, 1-31
[2026-02-05 23:25:06]   评估分支: 1 2 4 8 16 24 32
[2026-02-05 23:25:06]   总实验数: 5 ratios × 4 configs = 20 个
[2026-02-05 23:25:06] ============================================================================

# ========================================
# Session started at Thu Feb  5 23:25:06 CST 2026
# ========================================
[2026-02-05 23:25:06] 
[2026-02-05 23:25:06] ############################################################################
[2026-02-05 23:25:06] # ROPE_2D_RATIO = 0
[2026-02-05 23:25:06] ############################################################################

========================================
ROPE_2D_RATIO = 0
========================================
[2026-02-05 23:25:06] 
[2026-02-05 23:25:06] >>> Experiment 1 / 20
[2026-02-05 23:25:06] >>> ROPE=0, TRAIN_BRANCH=fixed1

--- EXPERIMENT 1 / 20 ---
ROPE=0, TRAIN_BRANCH=fixed1
OUT_DIR: out/512-h8-r00-bfixed1
[2026-02-05 23:25:06] 
[2026-02-05 23:25:06] ========================================================================
[2026-02-05 23:25:06] [TRAINING] Attempt 1/3
[2026-02-05 23:25:06]   ROPE_2D_RATIO:    0
[2026-02-05 23:25:06]   TRAIN_BRANCH:     1-1
[2026-02-05 23:25:06]   BATCH_SIZE:       8
[2026-02-05 23:25:06]   ACCUM_STEPS:      1
[2026-02-05 23:25:06]   OUT_DIR:          out/512-h8-r00-bfixed1
[2026-02-05 23:25:06] ========================================================================
[2026-02-05 23:25:06] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 8             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 1             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r00-bfixed1             --ddp
[2026-02-06 01:49:30] [TRAINING] Completed: out/512-h8-r00-bfixed1
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 01:49:30] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 01:51:02] [EVAL] Recorded: rope=0, train=fixed1, eval=1, loss=2.4375, ppl=11.44
EVAL_BRANCH=2:
[2026-02-06 01:51:02] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 01:52:03] [EVAL] Recorded: rope=0, train=fixed1, eval=2, loss=3.6508, ppl=38.50
EVAL_BRANCH=4:
[2026-02-06 01:52:03] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 01:53:02] [EVAL] Recorded: rope=0, train=fixed1, eval=4, loss=4.7192, ppl=112.08
EVAL_BRANCH=8:
[2026-02-06 01:53:02] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 01:53:59] [EVAL] Recorded: rope=0, train=fixed1, eval=8, loss=5.3685, ppl=214.55
EVAL_BRANCH=16:
[2026-02-06 01:53:59] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 01:54:59] [EVAL] Recorded: rope=0, train=fixed1, eval=16, loss=5.8068, ppl=332.54
EVAL_BRANCH=24:
[2026-02-06 01:54:59] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 01:55:59] [EVAL] Recorded: rope=0, train=fixed1, eval=24, loss=5.9046, ppl=366.73
EVAL_BRANCH=32:
[2026-02-06 01:55:59] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 01:56:59] [EVAL] Recorded: rope=0, train=fixed1, eval=32, loss=5.9810, ppl=395.84
[2026-02-06 01:56:59] >>> 进度: 1/20 (5%) | 已用时: 02:31:53 | 预计剩余: 48:05:47 | ETA: 2026-02-08 02:02:46

[2026-02-06 01:56:59] 
[2026-02-06 01:56:59] >>> Experiment 2 / 20
[2026-02-06 01:56:59] >>> ROPE=0, TRAIN_BRANCH=1-3

--- EXPERIMENT 2 / 20 ---
ROPE=0, TRAIN_BRANCH=1-3
OUT_DIR: out/512-h8-r00-b1-3
[2026-02-06 01:56:59] 
[2026-02-06 01:56:59] ========================================================================
[2026-02-06 01:56:59] [TRAINING] Attempt 1/3
[2026-02-06 01:56:59]   ROPE_2D_RATIO:    0
[2026-02-06 01:56:59]   TRAIN_BRANCH:     1-3
[2026-02-06 01:56:59]   BATCH_SIZE:       4
[2026-02-06 01:56:59]   ACCUM_STEPS:      1
[2026-02-06 01:56:59]   OUT_DIR:          out/512-h8-r00-b1-3
[2026-02-06 01:56:59] ========================================================================
[2026-02-06 01:56:59] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 4             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 3             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r00-b1-3             --ddp
[2026-02-06 04:27:17] [TRAINING] Completed: out/512-h8-r00-b1-3
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 04:27:17] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 04:28:51] [EVAL] Recorded: rope=0, train=1-3, eval=1, loss=2.5004, ppl=12.19
EVAL_BRANCH=2:
[2026-02-06 04:28:51] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 04:29:54] [EVAL] Recorded: rope=0, train=1-3, eval=2, loss=2.5324, ppl=12.58
EVAL_BRANCH=4:
[2026-02-06 04:29:54] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 04:30:52] [EVAL] Recorded: rope=0, train=1-3, eval=4, loss=3.2524, ppl=25.85
EVAL_BRANCH=8:
[2026-02-06 04:30:52] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 04:31:51] [EVAL] Recorded: rope=0, train=1-3, eval=8, loss=4.7508, ppl=115.68
EVAL_BRANCH=16:
[2026-02-06 04:31:51] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 04:32:49] [EVAL] Recorded: rope=0, train=1-3, eval=16, loss=5.5363, ppl=253.73
EVAL_BRANCH=24:
[2026-02-06 04:32:49] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 04:33:48] [EVAL] Recorded: rope=0, train=1-3, eval=24, loss=5.8327, ppl=341.29
EVAL_BRANCH=32:
[2026-02-06 04:33:48] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 04:34:48] [EVAL] Recorded: rope=0, train=1-3, eval=32, loss=6.0255, ppl=413.84
[2026-02-06 04:34:48] >>> 进度: 2/20 (10%) | 已用时: 05:09:42 | 预计剩余: 46:27:18 | ETA: 2026-02-08 03:02:06

[2026-02-06 04:34:48] 
[2026-02-06 04:34:48] >>> Experiment 3 / 20
[2026-02-06 04:34:48] >>> ROPE=0, TRAIN_BRANCH=1-7

--- EXPERIMENT 3 / 20 ---
ROPE=0, TRAIN_BRANCH=1-7
OUT_DIR: out/512-h8-r00-b1-7
[2026-02-06 04:34:48] 
[2026-02-06 04:34:48] ========================================================================
[2026-02-06 04:34:48] [TRAINING] Attempt 1/3
[2026-02-06 04:34:48]   ROPE_2D_RATIO:    0
[2026-02-06 04:34:48]   TRAIN_BRANCH:     1-7
[2026-02-06 04:34:48]   BATCH_SIZE:       2
[2026-02-06 04:34:48]   ACCUM_STEPS:      1
[2026-02-06 04:34:48]   OUT_DIR:          out/512-h8-r00-b1-7
[2026-02-06 04:34:48] ========================================================================
[2026-02-06 04:34:48] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 2             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 7             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r00-b1-7             --ddp
[2026-02-06 07:11:24] [TRAINING] Completed: out/512-h8-r00-b1-7
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 07:11:24] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 07:12:56] [EVAL] Recorded: rope=0, train=1-7, eval=1, loss=2.6127, ppl=13.64
EVAL_BRANCH=2:
[2026-02-06 07:12:56] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 07:13:58] [EVAL] Recorded: rope=0, train=1-7, eval=2, loss=2.6290, ppl=13.86
EVAL_BRANCH=4:
[2026-02-06 07:13:58] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 07:14:56] [EVAL] Recorded: rope=0, train=1-7, eval=4, loss=2.8750, ppl=17.73
EVAL_BRANCH=8:
[2026-02-06 07:14:56] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 07:15:55] [EVAL] Recorded: rope=0, train=1-7, eval=8, loss=3.7542, ppl=42.70
EVAL_BRANCH=16:
[2026-02-06 07:15:55] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 07:16:54] [EVAL] Recorded: rope=0, train=1-7, eval=16, loss=4.7802, ppl=119.13
EVAL_BRANCH=24:
[2026-02-06 07:16:54] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 07:17:52] [EVAL] Recorded: rope=0, train=1-7, eval=24, loss=5.4270, ppl=227.46
EVAL_BRANCH=32:
[2026-02-06 07:17:52] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 07:18:51] [EVAL] Recorded: rope=0, train=1-7, eval=32, loss=5.8463, ppl=345.95
[2026-02-06 07:18:51] >>> 进度: 3/20 (15%) | 已用时: 07:53:45 | 预计剩余: 44:44:35 | ETA: 2026-02-08 04:03:26

[2026-02-06 07:18:51] 
[2026-02-06 07:18:51] >>> Experiment 4 / 20
[2026-02-06 07:18:51] >>> ROPE=0, TRAIN_BRANCH=1-15

--- EXPERIMENT 4 / 20 ---
ROPE=0, TRAIN_BRANCH=1-15
OUT_DIR: out/512-h8-r00-b1-15
[2026-02-06 07:18:51] 
[2026-02-06 07:18:51] ========================================================================
[2026-02-06 07:18:51] [TRAINING] Attempt 1/3
[2026-02-06 07:18:51]   ROPE_2D_RATIO:    0
[2026-02-06 07:18:51]   TRAIN_BRANCH:     1-15
[2026-02-06 07:18:51]   BATCH_SIZE:       1
[2026-02-06 07:18:51]   ACCUM_STEPS:      1
[2026-02-06 07:18:51]   OUT_DIR:          out/512-h8-r00-b1-15
[2026-02-06 07:18:51] ========================================================================
[2026-02-06 07:18:51] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 1             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 15             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r00-b1-15             --ddp
[2026-02-06 09:09:18] [TRAINING] Completed: out/512-h8-r00-b1-15
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 09:09:18] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 09:10:50] [EVAL] Recorded: rope=0, train=1-15, eval=1, loss=2.7636, ppl=15.86
EVAL_BRANCH=2:
[2026-02-06 09:10:50] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 09:11:53] [EVAL] Recorded: rope=0, train=1-15, eval=2, loss=2.7755, ppl=16.05
EVAL_BRANCH=4:
[2026-02-06 09:11:53] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 09:12:53] [EVAL] Recorded: rope=0, train=1-15, eval=4, loss=2.9982, ppl=20.05
EVAL_BRANCH=8:
[2026-02-06 09:12:53] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 09:13:51] [EVAL] Recorded: rope=0, train=1-15, eval=8, loss=3.7896, ppl=44.24
EVAL_BRANCH=16:
[2026-02-06 09:13:51] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 09:14:50] [EVAL] Recorded: rope=0, train=1-15, eval=16, loss=4.8697, ppl=130.28
EVAL_BRANCH=24:
[2026-02-06 09:14:50] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 09:15:49] [EVAL] Recorded: rope=0, train=1-15, eval=24, loss=5.2942, ppl=199.17
EVAL_BRANCH=32:
[2026-02-06 09:15:49] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 09:16:48] [EVAL] Recorded: rope=0, train=1-15, eval=32, loss=5.5628, ppl=260.55
[2026-02-06 09:16:48] >>> 进度: 4/20 (20%) | 已用时: 09:51:42 | 预计剩余: 39:26:40 | ETA: 2026-02-08 00:43:28

[2026-02-06 09:16:48] 
[2026-02-06 09:16:48] ############################################################################
[2026-02-06 09:16:48] # ROPE_2D_RATIO = 0.25
[2026-02-06 09:16:48] ############################################################################

========================================
ROPE_2D_RATIO = 0.25
========================================
[2026-02-06 09:16:48] 
[2026-02-06 09:16:48] >>> Experiment 5 / 20
[2026-02-06 09:16:48] >>> ROPE=0.25, TRAIN_BRANCH=fixed1

--- EXPERIMENT 5 / 20 ---
ROPE=0.25, TRAIN_BRANCH=fixed1
OUT_DIR: out/512-h8-r025-bfixed1
[2026-02-06 09:16:48] 
[2026-02-06 09:16:48] ========================================================================
[2026-02-06 09:16:48] [TRAINING] Attempt 1/3
[2026-02-06 09:16:48]   ROPE_2D_RATIO:    0.25
[2026-02-06 09:16:48]   TRAIN_BRANCH:     1-1
[2026-02-06 09:16:48]   BATCH_SIZE:       8
[2026-02-06 09:16:48]   ACCUM_STEPS:      1
[2026-02-06 09:16:48]   OUT_DIR:          out/512-h8-r025-bfixed1
[2026-02-06 09:16:48] ========================================================================
[2026-02-06 09:16:48] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.25             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 8             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 1             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r025-bfixed1             --ddp
[2026-02-06 11:42:20] [TRAINING] Completed: out/512-h8-r025-bfixed1
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 11:42:20] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 11:43:55] [EVAL] Recorded: rope=0.25, train=fixed1, eval=1, loss=2.4390, ppl=11.46
EVAL_BRANCH=2:
[2026-02-06 11:43:55] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 11:44:57] [EVAL] Recorded: rope=0.25, train=fixed1, eval=2, loss=3.6415, ppl=38.15
EVAL_BRANCH=4:
[2026-02-06 11:44:57] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 11:45:59] [EVAL] Recorded: rope=0.25, train=fixed1, eval=4, loss=4.7215, ppl=112.33
EVAL_BRANCH=8:
[2026-02-06 11:45:59] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 11:46:57] [EVAL] Recorded: rope=0.25, train=fixed1, eval=8, loss=5.3212, ppl=204.64
EVAL_BRANCH=16:
[2026-02-06 11:46:57] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 11:47:56] [EVAL] Recorded: rope=0.25, train=fixed1, eval=16, loss=5.7546, ppl=315.64
EVAL_BRANCH=24:
[2026-02-06 11:47:56] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 11:48:54] [EVAL] Recorded: rope=0.25, train=fixed1, eval=24, loss=5.8635, ppl=351.97
EVAL_BRANCH=32:
[2026-02-06 11:48:54] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 11:49:52] [EVAL] Recorded: rope=0.25, train=fixed1, eval=32, loss=5.9413, ppl=380.42
[2026-02-06 11:49:52] >>> 进度: 5/20 (25%) | 已用时: 12:24:46 | 预计剩余: 37:14:15 | ETA: 2026-02-08 01:04:07

[2026-02-06 11:49:52] 
[2026-02-06 11:49:52] >>> Experiment 6 / 20
[2026-02-06 11:49:52] >>> ROPE=0.25, TRAIN_BRANCH=1-3

--- EXPERIMENT 6 / 20 ---
ROPE=0.25, TRAIN_BRANCH=1-3
OUT_DIR: out/512-h8-r025-b1-3
[2026-02-06 11:49:52] 
[2026-02-06 11:49:52] ========================================================================
[2026-02-06 11:49:52] [TRAINING] Attempt 1/3
[2026-02-06 11:49:52]   ROPE_2D_RATIO:    0.25
[2026-02-06 11:49:52]   TRAIN_BRANCH:     1-3
[2026-02-06 11:49:52]   BATCH_SIZE:       4
[2026-02-06 11:49:52]   ACCUM_STEPS:      1
[2026-02-06 11:49:52]   OUT_DIR:          out/512-h8-r025-b1-3
[2026-02-06 11:49:52] ========================================================================
[2026-02-06 11:49:52] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.25             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 4             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 3             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r025-b1-3             --ddp
[2026-02-06 14:19:27] [TRAINING] Completed: out/512-h8-r025-b1-3
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 14:19:27] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 14:20:59] [EVAL] Recorded: rope=0.25, train=1-3, eval=1, loss=2.5038, ppl=12.23
EVAL_BRANCH=2:
[2026-02-06 14:20:59] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 14:22:03] [EVAL] Recorded: rope=0.25, train=1-3, eval=2, loss=2.5147, ppl=12.36
EVAL_BRANCH=4:
[2026-02-06 14:22:03] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 14:23:01] [EVAL] Recorded: rope=0.25, train=1-3, eval=4, loss=3.0474, ppl=21.06
EVAL_BRANCH=8:
[2026-02-06 14:23:01] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 14:23:59] [EVAL] Recorded: rope=0.25, train=1-3, eval=8, loss=4.5993, ppl=99.41
EVAL_BRANCH=16:
[2026-02-06 14:23:59] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 14:24:58] [EVAL] Recorded: rope=0.25, train=1-3, eval=16, loss=5.7190, ppl=304.62
EVAL_BRANCH=24:
[2026-02-06 14:24:58] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 14:25:58] [EVAL] Recorded: rope=0.25, train=1-3, eval=24, loss=6.0214, ppl=412.14
EVAL_BRANCH=32:
[2026-02-06 14:25:58] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 14:26:58] [EVAL] Recorded: rope=0.25, train=1-3, eval=32, loss=6.1156, ppl=452.86
[2026-02-06 14:26:58] >>> 进度: 6/20 (30%) | 已用时: 15:01:52 | 预计剩余: 35:04:12 | ETA: 2026-02-08 01:31:10

[2026-02-06 14:26:58] 
[2026-02-06 14:26:58] >>> Experiment 7 / 20
[2026-02-06 14:26:58] >>> ROPE=0.25, TRAIN_BRANCH=1-7

--- EXPERIMENT 7 / 20 ---
ROPE=0.25, TRAIN_BRANCH=1-7
OUT_DIR: out/512-h8-r025-b1-7
[2026-02-06 14:26:58] 
[2026-02-06 14:26:58] ========================================================================
[2026-02-06 14:26:58] [TRAINING] Attempt 1/3
[2026-02-06 14:26:58]   ROPE_2D_RATIO:    0.25
[2026-02-06 14:26:58]   TRAIN_BRANCH:     1-7
[2026-02-06 14:26:58]   BATCH_SIZE:       2
[2026-02-06 14:26:58]   ACCUM_STEPS:      1
[2026-02-06 14:26:58]   OUT_DIR:          out/512-h8-r025-b1-7
[2026-02-06 14:26:58] ========================================================================
[2026-02-06 14:26:58] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.25             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 2             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 7             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r025-b1-7             --ddp
[2026-02-06 17:05:24] [TRAINING] Completed: out/512-h8-r025-b1-7
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 17:05:24] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 17:06:57] [EVAL] Recorded: rope=0.25, train=1-7, eval=1, loss=2.5683, ppl=13.04
EVAL_BRANCH=2:
[2026-02-06 17:06:57] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 17:08:00] [EVAL] Recorded: rope=0.25, train=1-7, eval=2, loss=2.5904, ppl=13.33
EVAL_BRANCH=4:
[2026-02-06 17:08:00] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 17:09:01] [EVAL] Recorded: rope=0.25, train=1-7, eval=4, loss=2.6614, ppl=14.32
EVAL_BRANCH=8:
[2026-02-06 17:09:01] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 17:09:59] [EVAL] Recorded: rope=0.25, train=1-7, eval=8, loss=2.9755, ppl=19.60
EVAL_BRANCH=16:
[2026-02-06 17:09:59] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 17:11:00] [EVAL] Recorded: rope=0.25, train=1-7, eval=16, loss=5.1213, ppl=167.55
EVAL_BRANCH=24:
[2026-02-06 17:11:00] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 17:11:59] [EVAL] Recorded: rope=0.25, train=1-7, eval=24, loss=5.9501, ppl=383.78
EVAL_BRANCH=32:
[2026-02-06 17:11:59] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 17:13:00] [EVAL] Recorded: rope=0.25, train=1-7, eval=32, loss=6.3222, ppl=556.81
[2026-02-06 17:13:00] >>> 进度: 7/20 (35%) | 已用时: 17:47:54 | 预计剩余: 33:03:09 | ETA: 2026-02-08 02:16:09

[2026-02-06 17:13:00] 
[2026-02-06 17:13:00] >>> Experiment 8 / 20
[2026-02-06 17:13:00] >>> ROPE=0.25, TRAIN_BRANCH=1-15

--- EXPERIMENT 8 / 20 ---
ROPE=0.25, TRAIN_BRANCH=1-15
OUT_DIR: out/512-h8-r025-b1-15
[2026-02-06 17:13:00] 
[2026-02-06 17:13:00] ========================================================================
[2026-02-06 17:13:00] [TRAINING] Attempt 1/3
[2026-02-06 17:13:00]   ROPE_2D_RATIO:    0.25
[2026-02-06 17:13:00]   TRAIN_BRANCH:     1-15
[2026-02-06 17:13:00]   BATCH_SIZE:       1
[2026-02-06 17:13:00]   ACCUM_STEPS:      1
[2026-02-06 17:13:00]   OUT_DIR:          out/512-h8-r025-b1-15
[2026-02-06 17:13:00] ========================================================================
[2026-02-06 17:13:00] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.25             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 1             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 15             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r025-b1-15             --ddp
[2026-02-06 19:03:38] [TRAINING] Completed: out/512-h8-r025-b1-15
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 19:03:38] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 19:05:10] [EVAL] Recorded: rope=0.25, train=1-15, eval=1, loss=2.7059, ppl=14.97
EVAL_BRANCH=2:
[2026-02-06 19:05:10] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 19:06:13] [EVAL] Recorded: rope=0.25, train=1-15, eval=2, loss=2.6979, ppl=14.85
EVAL_BRANCH=4:
[2026-02-06 19:06:13] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 19:07:12] [EVAL] Recorded: rope=0.25, train=1-15, eval=4, loss=2.7274, ppl=15.29
EVAL_BRANCH=8:
[2026-02-06 19:07:12] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 19:08:11] [EVAL] Recorded: rope=0.25, train=1-15, eval=8, loss=2.8714, ppl=17.66
EVAL_BRANCH=16:
[2026-02-06 19:08:11] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 19:09:09] [EVAL] Recorded: rope=0.25, train=1-15, eval=16, loss=3.5872, ppl=36.13
EVAL_BRANCH=24:
[2026-02-06 19:09:09] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 19:10:10] [EVAL] Recorded: rope=0.25, train=1-15, eval=24, loss=4.5737, ppl=96.90
EVAL_BRANCH=32:
[2026-02-06 19:10:10] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 19:11:08] [EVAL] Recorded: rope=0.25, train=1-15, eval=32, loss=4.9964, ppl=147.88
[2026-02-06 19:11:08] >>> 进度: 8/20 (40%) | 已用时: 19:46:02 | 预计剩余: 29:39:00 | ETA: 2026-02-08 00:50:08

[2026-02-06 19:11:08] 
[2026-02-06 19:11:08] ############################################################################
[2026-02-06 19:11:08] # ROPE_2D_RATIO = 0.5
[2026-02-06 19:11:08] ############################################################################

========================================
ROPE_2D_RATIO = 0.5
========================================
[2026-02-06 19:11:08] 
[2026-02-06 19:11:08] >>> Experiment 9 / 20
[2026-02-06 19:11:08] >>> ROPE=0.5, TRAIN_BRANCH=fixed1

--- EXPERIMENT 9 / 20 ---
ROPE=0.5, TRAIN_BRANCH=fixed1
OUT_DIR: out/512-h8-r05-bfixed1
[2026-02-06 19:11:08] 
[2026-02-06 19:11:08] ========================================================================
[2026-02-06 19:11:08] [TRAINING] Attempt 1/3
[2026-02-06 19:11:08]   ROPE_2D_RATIO:    0.5
[2026-02-06 19:11:08]   TRAIN_BRANCH:     1-1
[2026-02-06 19:11:08]   BATCH_SIZE:       8
[2026-02-06 19:11:08]   ACCUM_STEPS:      1
[2026-02-06 19:11:08]   OUT_DIR:          out/512-h8-r05-bfixed1
[2026-02-06 19:11:08] ========================================================================
[2026-02-06 19:11:08] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.5             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 8             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 1             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r05-bfixed1             --ddp
[2026-02-06 21:38:55] [TRAINING] Completed: out/512-h8-r05-bfixed1
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-06 21:38:55] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-06 21:40:26] [EVAL] Recorded: rope=0.5, train=fixed1, eval=1, loss=2.4413, ppl=11.49
EVAL_BRANCH=2:
[2026-02-06 21:40:26] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-06 21:41:29] [EVAL] Recorded: rope=0.5, train=fixed1, eval=2, loss=3.6995, ppl=40.43
EVAL_BRANCH=4:
[2026-02-06 21:41:29] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-06 21:42:29] [EVAL] Recorded: rope=0.5, train=fixed1, eval=4, loss=4.7415, ppl=114.60
EVAL_BRANCH=8:
[2026-02-06 21:42:29] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-06 21:43:28] [EVAL] Recorded: rope=0.5, train=fixed1, eval=8, loss=5.3302, ppl=206.48
EVAL_BRANCH=16:
[2026-02-06 21:43:28] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-06 21:44:26] [EVAL] Recorded: rope=0.5, train=fixed1, eval=16, loss=5.7298, ppl=307.90
EVAL_BRANCH=24:
[2026-02-06 21:44:26] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-06 21:45:26] [EVAL] Recorded: rope=0.5, train=fixed1, eval=24, loss=5.8416, ppl=344.34
EVAL_BRANCH=32:
[2026-02-06 21:45:26] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-06 21:46:24] [EVAL] Recorded: rope=0.5, train=fixed1, eval=32, loss=5.9084, ppl=368.12
[2026-02-06 21:46:24] >>> 进度: 9/20 (45%) | 已用时: 22:21:18 | 预计剩余: 27:19:22 | ETA: 2026-02-08 01:05:46

[2026-02-06 21:46:24] 
[2026-02-06 21:46:24] >>> Experiment 10 / 20
[2026-02-06 21:46:24] >>> ROPE=0.5, TRAIN_BRANCH=1-3

--- EXPERIMENT 10 / 20 ---
ROPE=0.5, TRAIN_BRANCH=1-3
OUT_DIR: out/512-h8-r05-b1-3
[2026-02-06 21:46:24] 
[2026-02-06 21:46:24] ========================================================================
[2026-02-06 21:46:24] [TRAINING] Attempt 1/3
[2026-02-06 21:46:24]   ROPE_2D_RATIO:    0.5
[2026-02-06 21:46:24]   TRAIN_BRANCH:     1-3
[2026-02-06 21:46:24]   BATCH_SIZE:       4
[2026-02-06 21:46:24]   ACCUM_STEPS:      1
[2026-02-06 21:46:24]   OUT_DIR:          out/512-h8-r05-b1-3
[2026-02-06 21:46:24] ========================================================================
[2026-02-06 21:46:24] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.5             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 4             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 3             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r05-b1-3             --ddp
[2026-02-07 00:15:46] [TRAINING] Completed: out/512-h8-r05-b1-3
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-07 00:15:46] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-07 00:17:19] [EVAL] Recorded: rope=0.5, train=1-3, eval=1, loss=2.4736, ppl=11.87
EVAL_BRANCH=2:
[2026-02-07 00:17:19] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-07 00:18:21] [EVAL] Recorded: rope=0.5, train=1-3, eval=2, loss=2.4721, ppl=11.85
EVAL_BRANCH=4:
[2026-02-07 00:18:21] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-07 00:19:21] [EVAL] Recorded: rope=0.5, train=1-3, eval=4, loss=2.7301, ppl=15.33
EVAL_BRANCH=8:
[2026-02-07 00:19:21] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-07 00:20:19] [EVAL] Recorded: rope=0.5, train=1-3, eval=8, loss=3.8857, ppl=48.70
EVAL_BRANCH=16:
[2026-02-07 00:20:19] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-07 00:21:19] [EVAL] Recorded: rope=0.5, train=1-3, eval=16, loss=5.1408, ppl=170.85
EVAL_BRANCH=24:
[2026-02-07 00:21:19] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-07 00:22:17] [EVAL] Recorded: rope=0.5, train=1-3, eval=24, loss=5.4474, ppl=232.16
EVAL_BRANCH=32:
[2026-02-07 00:22:17] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-07 00:23:17] [EVAL] Recorded: rope=0.5, train=1-3, eval=32, loss=5.5439, ppl=255.67
[2026-02-07 00:23:17] >>> 进度: 10/20 (50%) | 已用时: 24:58:11 | 预计剩余: 24:58:10 | ETA: 2026-02-08 01:21:27

[2026-02-07 00:23:17] 
[2026-02-07 00:23:17] >>> Experiment 11 / 20
[2026-02-07 00:23:17] >>> ROPE=0.5, TRAIN_BRANCH=1-7

--- EXPERIMENT 11 / 20 ---
ROPE=0.5, TRAIN_BRANCH=1-7
OUT_DIR: out/512-h8-r05-b1-7
[2026-02-07 00:23:17] 
[2026-02-07 00:23:17] ========================================================================
[2026-02-07 00:23:17] [TRAINING] Attempt 1/3
[2026-02-07 00:23:17]   ROPE_2D_RATIO:    0.5
[2026-02-07 00:23:17]   TRAIN_BRANCH:     1-7
[2026-02-07 00:23:17]   BATCH_SIZE:       2
[2026-02-07 00:23:17]   ACCUM_STEPS:      1
[2026-02-07 00:23:17]   OUT_DIR:          out/512-h8-r05-b1-7
[2026-02-07 00:23:17] ========================================================================
[2026-02-07 00:23:17] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.5             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 2             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 7             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r05-b1-7             --ddp
[2026-02-07 03:01:34] [TRAINING] Completed: out/512-h8-r05-b1-7
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-07 03:01:34] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-07 03:03:07] [EVAL] Recorded: rope=0.5, train=1-7, eval=1, loss=2.5551, ppl=12.87
EVAL_BRANCH=2:
[2026-02-07 03:03:07] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-07 03:04:09] [EVAL] Recorded: rope=0.5, train=1-7, eval=2, loss=2.5601, ppl=12.94
EVAL_BRANCH=4:
[2026-02-07 03:04:09] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-07 03:05:10] [EVAL] Recorded: rope=0.5, train=1-7, eval=4, loss=2.5958, ppl=13.41
EVAL_BRANCH=8:
[2026-02-07 03:05:10] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-07 03:06:09] [EVAL] Recorded: rope=0.5, train=1-7, eval=8, loss=2.8091, ppl=16.59
EVAL_BRANCH=16:
[2026-02-07 03:06:09] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-07 03:07:09] [EVAL] Recorded: rope=0.5, train=1-7, eval=16, loss=3.6047, ppl=36.77
EVAL_BRANCH=24:
[2026-02-07 03:07:09] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-07 03:08:08] [EVAL] Recorded: rope=0.5, train=1-7, eval=24, loss=4.6056, ppl=100.05
EVAL_BRANCH=32:
[2026-02-07 03:08:08] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-07 03:09:09] [EVAL] Recorded: rope=0.5, train=1-7, eval=32, loss=5.3546, ppl=211.58
[2026-02-07 03:09:09] >>> 进度: 11/20 (55%) | 已用时: 27:44:03 | 预计剩余: 22:41:24 | ETA: 2026-02-08 01:50:33

[2026-02-07 03:09:09] 
[2026-02-07 03:09:09] >>> Experiment 12 / 20
[2026-02-07 03:09:09] >>> ROPE=0.5, TRAIN_BRANCH=1-15

--- EXPERIMENT 12 / 20 ---
ROPE=0.5, TRAIN_BRANCH=1-15
OUT_DIR: out/512-h8-r05-b1-15
[2026-02-07 03:09:09] 
[2026-02-07 03:09:09] ========================================================================
[2026-02-07 03:09:09] [TRAINING] Attempt 1/3
[2026-02-07 03:09:09]   ROPE_2D_RATIO:    0.5
[2026-02-07 03:09:09]   TRAIN_BRANCH:     1-15
[2026-02-07 03:09:09]   BATCH_SIZE:       1
[2026-02-07 03:09:09]   ACCUM_STEPS:      1
[2026-02-07 03:09:09]   OUT_DIR:          out/512-h8-r05-b1-15
[2026-02-07 03:09:09] ========================================================================
[2026-02-07 03:09:09] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.5             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 1             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 15             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r05-b1-15             --ddp
[2026-02-07 04:59:27] [TRAINING] Completed: out/512-h8-r05-b1-15
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-07 04:59:27] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-07 05:01:00] [EVAL] Recorded: rope=0.5, train=1-15, eval=1, loss=2.7044, ppl=14.94
EVAL_BRANCH=2:
[2026-02-07 05:01:00] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-07 05:02:03] [EVAL] Recorded: rope=0.5, train=1-15, eval=2, loss=2.7002, ppl=14.88
EVAL_BRANCH=4:
[2026-02-07 05:02:03] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-07 05:03:02] [EVAL] Recorded: rope=0.5, train=1-15, eval=4, loss=2.7217, ppl=15.21
EVAL_BRANCH=8:
[2026-02-07 05:03:02] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-07 05:04:00] [EVAL] Recorded: rope=0.5, train=1-15, eval=8, loss=2.8221, ppl=16.81
EVAL_BRANCH=16:
[2026-02-07 05:04:00] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-07 05:04:58] [EVAL] Recorded: rope=0.5, train=1-15, eval=16, loss=3.0306, ppl=20.71
EVAL_BRANCH=24:
[2026-02-07 05:04:58] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-07 05:05:56] [EVAL] Recorded: rope=0.5, train=1-15, eval=24, loss=3.2347, ppl=25.40
EVAL_BRANCH=32:
[2026-02-07 05:05:56] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-07 05:06:56] [EVAL] Recorded: rope=0.5, train=1-15, eval=32, loss=3.4602, ppl=31.82
[2026-02-07 05:06:56] >>> 进度: 12/20 (60%) | 已用时: 29:41:50 | 预计剩余: 19:47:52 | ETA: 2026-02-08 00:54:48

[2026-02-07 05:06:56] 
[2026-02-07 05:06:56] ############################################################################
[2026-02-07 05:06:56] # ROPE_2D_RATIO = 0.75
[2026-02-07 05:06:56] ############################################################################

========================================
ROPE_2D_RATIO = 0.75
========================================
[2026-02-07 05:06:56] 
[2026-02-07 05:06:56] >>> Experiment 13 / 20
[2026-02-07 05:06:56] >>> ROPE=0.75, TRAIN_BRANCH=fixed1

--- EXPERIMENT 13 / 20 ---
ROPE=0.75, TRAIN_BRANCH=fixed1
OUT_DIR: out/512-h8-r075-bfixed1
[2026-02-07 05:06:56] 
[2026-02-07 05:06:56] ========================================================================
[2026-02-07 05:06:56] [TRAINING] Attempt 1/3
[2026-02-07 05:06:56]   ROPE_2D_RATIO:    0.75
[2026-02-07 05:06:56]   TRAIN_BRANCH:     1-1
[2026-02-07 05:06:56]   BATCH_SIZE:       8
[2026-02-07 05:06:56]   ACCUM_STEPS:      1
[2026-02-07 05:06:56]   OUT_DIR:          out/512-h8-r075-bfixed1
[2026-02-07 05:06:56] ========================================================================
[2026-02-07 05:06:56] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.75             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 8             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 1             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r075-bfixed1             --ddp
[2026-02-07 07:33:05] [TRAINING] Completed: out/512-h8-r075-bfixed1
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-07 07:33:05] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-07 07:34:38] [EVAL] Recorded: rope=0.75, train=fixed1, eval=1, loss=2.4461, ppl=11.54
EVAL_BRANCH=2:
[2026-02-07 07:34:38] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-07 07:35:40] [EVAL] Recorded: rope=0.75, train=fixed1, eval=2, loss=3.6530, ppl=38.59
EVAL_BRANCH=4:
[2026-02-07 07:35:40] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-07 07:36:40] [EVAL] Recorded: rope=0.75, train=fixed1, eval=4, loss=4.8042, ppl=122.02
EVAL_BRANCH=8:
[2026-02-07 07:36:40] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-07 07:37:38] [EVAL] Recorded: rope=0.75, train=fixed1, eval=8, loss=5.4034, ppl=222.17
EVAL_BRANCH=16:
[2026-02-07 07:37:38] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-07 07:38:39] [EVAL] Recorded: rope=0.75, train=fixed1, eval=16, loss=5.7346, ppl=309.38
EVAL_BRANCH=24:
[2026-02-07 07:38:39] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-07 07:39:39] [EVAL] Recorded: rope=0.75, train=fixed1, eval=24, loss=5.8634, ppl=351.91
EVAL_BRANCH=32:
[2026-02-07 07:39:39] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-07 07:40:38] [EVAL] Recorded: rope=0.75, train=fixed1, eval=32, loss=5.9604, ppl=387.77
[2026-02-07 07:40:38] >>> 进度: 13/20 (65%) | 已用时: 32:15:32 | 预计剩余: 17:22:11 | ETA: 2026-02-08 01:02:49

[2026-02-07 07:40:38] 
[2026-02-07 07:40:38] >>> Experiment 14 / 20
[2026-02-07 07:40:38] >>> ROPE=0.75, TRAIN_BRANCH=1-3

--- EXPERIMENT 14 / 20 ---
ROPE=0.75, TRAIN_BRANCH=1-3
OUT_DIR: out/512-h8-r075-b1-3
[2026-02-07 07:40:38] 
[2026-02-07 07:40:38] ========================================================================
[2026-02-07 07:40:38] [TRAINING] Attempt 1/3
[2026-02-07 07:40:38]   ROPE_2D_RATIO:    0.75
[2026-02-07 07:40:38]   TRAIN_BRANCH:     1-3
[2026-02-07 07:40:38]   BATCH_SIZE:       4
[2026-02-07 07:40:38]   ACCUM_STEPS:      1
[2026-02-07 07:40:38]   OUT_DIR:          out/512-h8-r075-b1-3
[2026-02-07 07:40:38] ========================================================================
[2026-02-07 07:40:38] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.75             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 4             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 3             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r075-b1-3             --ddp
[2026-02-07 10:11:58] [TRAINING] Completed: out/512-h8-r075-b1-3
--- Evaluation Results ---
EVAL_BRANCH=1:
[2026-02-07 10:11:58] [EVAL] VAL_BRANCH=1, BATCH=4 (attempt 1)
[2026-02-07 10:13:31] [EVAL] Recorded: rope=0.75, train=1-3, eval=1, loss=2.4734, ppl=11.86
EVAL_BRANCH=2:
[2026-02-07 10:13:31] [EVAL] VAL_BRANCH=2, BATCH=4 (attempt 1)
[2026-02-07 10:14:34] [EVAL] Recorded: rope=0.75, train=1-3, eval=2, loss=2.4757, ppl=11.89
EVAL_BRANCH=4:
[2026-02-07 10:14:34] [EVAL] VAL_BRANCH=4, BATCH=4 (attempt 1)
[2026-02-07 10:15:34] [EVAL] Recorded: rope=0.75, train=1-3, eval=4, loss=2.6787, ppl=14.57
EVAL_BRANCH=8:
[2026-02-07 10:15:34] [EVAL] VAL_BRANCH=8, BATCH=4 (attempt 1)
[2026-02-07 10:16:33] [EVAL] Recorded: rope=0.75, train=1-3, eval=8, loss=3.2375, ppl=25.47
EVAL_BRANCH=16:
[2026-02-07 10:16:33] [EVAL] VAL_BRANCH=16, BATCH=2 (attempt 1)
[2026-02-07 10:17:31] [EVAL] Recorded: rope=0.75, train=1-3, eval=16, loss=4.4666, ppl=87.06
EVAL_BRANCH=24:
[2026-02-07 10:17:31] [EVAL] VAL_BRANCH=24, BATCH=1 (attempt 1)
[2026-02-07 10:18:28] [EVAL] Recorded: rope=0.75, train=1-3, eval=24, loss=4.9195, ppl=136.94
EVAL_BRANCH=32:
[2026-02-07 10:18:28] [EVAL] VAL_BRANCH=32, BATCH=1 (attempt 1)
[2026-02-07 10:19:27] [EVAL] Recorded: rope=0.75, train=1-3, eval=32, loss=5.1229, ppl=167.82
[2026-02-07 10:19:27] >>> 进度: 14/20 (70%) | 已用时: 34:54:21 | 预计剩余: 14:57:30 | ETA: 2026-02-08 01:16:57

[2026-02-07 10:19:27] 
[2026-02-07 10:19:27] >>> Experiment 15 / 20
[2026-02-07 10:19:27] >>> ROPE=0.75, TRAIN_BRANCH=1-7

--- EXPERIMENT 15 / 20 ---
ROPE=0.75, TRAIN_BRANCH=1-7
OUT_DIR: out/512-h8-r075-b1-7
[2026-02-07 10:19:27] 
[2026-02-07 10:19:27] ========================================================================
[2026-02-07 10:19:27] [TRAINING] Attempt 1/3
[2026-02-07 10:19:27]   ROPE_2D_RATIO:    0.75
[2026-02-07 10:19:27]   TRAIN_BRANCH:     1-7
[2026-02-07 10:19:27]   BATCH_SIZE:       2
[2026-02-07 10:19:27]   ACCUM_STEPS:      1
[2026-02-07 10:19:27]   OUT_DIR:          out/512-h8-r075-b1-7
[2026-02-07 10:19:27] ========================================================================
[2026-02-07 10:19:27] [CMD] torchrun --nproc_per_node 8 trainer/train_pretrain.py             --pe rope             --rope_2d_ratio 0.75             --hidden_size 512             --num_attention_heads 8             --num_hidden_layers 8             --epochs 1             --batch_size 2             --accumulation_steps 1             --batch_by_samples             --max_branches_per_sample 7             --min_branches_per_sample 1             --val_max_branches_per_sample 4             --val_min_branches_per_sample 4             --max_total_tokens 0             --data_path dataset/pretrain_512.jsonl             --max-samples 2048000             --val_samples 500000             --val_interval_tokens 100000000             --out_dir out/512-h8-r075-b1-7             --ddp
