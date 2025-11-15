# 预训练 vs HF LoRA微调 - 训练配置对比

## 1. Collator 配置对比

### train_pretrain.py (从头预训练)
```python
collator = ParallelPretrainCollator(
    tokenizer,
    branches_per_sample=args.branches_per_sample,
    pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
    max_branches_per_sample=args.max_branches_per_sample,
    min_branches_per_sample=args.min_branches_per_sample,
    random_time_offset=args.random_time_offset,
    interleave_branches=True,  # ⚠️ 硬编码为True
    branch_stride=branch_stride,
    # ⚠️ 没有align_to参数，使用ParallelPretrainCollator的默认值
)
```

### train_hf_lora.py (HF + LoRA微调)
```python
collator = ParallelPretrainCollator(
    tokenizer,
    branches_per_sample=args.branches_per_sample,
    pad_to=args.max_total_tokens if args.max_total_tokens > 0 else None,
    max_branches_per_sample=args.max_branches_per_sample,
    min_branches_per_sample=args.min_branches_per_sample,
    random_time_offset=args.random_time_offset,
    interleave_branches=args.interleave_branches,  # ✓ 可配置（默认True）
    branch_stride=args.branch_stride,
    align_to=args.align_to,  # ✓ 可配置（默认left）
)
```

## 2. Attention Mask 构建

### 两者完全相同
```python
columnar_mask = build_columnar_causal_mask(time_ids, attention_mask).to(args.device)
```

**关键特性**：
- 仅基于time_ids构建因果mask
- **没有分支隔离**（同一time的不同branch可以互相看到）
- 可见性规则：token i 可以看到所有 time_j < time_i 的token

## 3. Loss 计算

### 两者完全相同
```python
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
```

## 4. 模型Forward

### train_pretrain.py (MiniMind模型)
```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=column_mask,
    position_ids=batch["position_ids"],
    pos2d=batch["pos2d"],
)
```

### train_hf_lora.py (HF模型)
```python
# 通过set_rope_pos2d设置2D位置
if args.patch_rope:
    set_rope_pos2d(rope_target_model, pos2d)

outputs = model(
    input_ids=input_ids,
    attention_mask=columnar_mask,
    # ⚠️ 不传递position_ids和pos2d（HF模型不支持）
)
```

## 5. Labels 构建 (ParallelPretrainCollator)

### 完全相同的逻辑
```python
# 每个token预测同一branch的下一个time的token
for batch_idx, meta in enumerate(layout.metadata):
    for branch_idx, _ in enumerate(meta.branch_ids):
        branch_pos = meta.branch_positions[branch_idx]
        mask = (layout.pos2d[batch_idx, :, 0] == branch_pos) & (layout.attention_mask[batch_idx] == 1)
        indices = mask.nonzero(as_tuple=True)[0]

        for i in range(len(indices) - 1):
            src_pos = indices[i].item()
            tgt_pos = indices[i + 1].item()
            labels[batch_idx, src_pos] = layout.input_ids[batch_idx, tgt_pos].item()
```

## 6. 关键参数默认值对比

| 参数 | train_pretrain.py | train_hf_lora.py | 说明 |
|------|-------------------|------------------|------|
| `interleave_branches` | True (硬编码) | True (默认) | ✓ 一致 |
| `align_to` | 未设置（使用collator默认） | "left" (默认) | ⚠️ 需要检查collator默认值 |
| `random_time_offset` | False (默认) | False (默认) | ✓ 一致 |
| `branch_stride` | 128 | 128 | ✓ 一致 |

## 7. ParallelPretrainCollator 默认参数

需要检查 `parallel_data/parallel_collator.py` 中 `ParallelPretrainCollator.__init__` 的默认参数，特别是：
- `align_to` 的默认值
- `interleave_branches` 的默认值

## 结论

**核心一致性**：
1. ✓ Loss计算完全相同
2. ✓ Mask构建完全相同（都不隔离分支）
3. ✓ Labels构建逻辑完全相同
4. ✓ interleave_branches默认都是True

**需要注意的差异**：
1. ⚠️ HF模型不接收position_ids/pos2d参数，通过set_rope_pos2d设置
2. ⚠️ align_to参数在train_pretrain.py中未显式设置，需确认是否使用collator默认值

**验证点**：
- 需要确认ParallelPretrainCollator的align_to默认值是"left"
- 如果是，则两者在数据处理上完全一致
