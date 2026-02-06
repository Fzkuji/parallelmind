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
