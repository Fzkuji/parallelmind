"""
验证修复后的labels是否正确
"""
import torch
from transformers import AutoTokenizer
from parallel_data.parallel_collator import ParallelPretrainCollator

tokenizer = AutoTokenizer.from_pretrained("./model/")

# 创建测试数据：2个branches，故意让长度不同
features = [
    {"text": "春天的花朵很美丽，让人心情愉悦"},  # 较长
    {"text": "夏天很热"},  # 较短
]

collator = ParallelPretrainCollator(
    tokenizer=tokenizer,
    branches_per_sample=2,
    pad_to=None,
    interleave_branches=True,  # 交错布局
    branch_stride=128,
)

batch = collator(features)

print("=" * 80)
print("测试 Columnar Parallel 的 Labels 修复")
print("=" * 80)

input_ids = batch["input_ids"][0]
labels = batch["labels"][0]
pos2d = batch["pos2d"][0]
time_ids = batch["time_ids"][0]
attn_mask = batch["attention_mask"][0]

# 打印每个位置的信息
print("\n位置 | Branch | Time | Input Token → Label Token | 状态")
print("-" * 80)

seq_len = attn_mask.sum().item()
for i in range(seq_len):
    input_id = input_ids[i].item()
    label_id = labels[i].item()
    branch_pos = pos2d[i, 0].item()
    time_pos = pos2d[i, 1].item()

    input_token = tokenizer.decode([input_id]) if input_id != -100 else "<PAD>"
    label_token = tokenizer.decode([label_id]) if label_id != -100 else "<IGNORE>"

    # 检查label是否来自同一个branch
    status = "✅"
    if label_id != -100:
        # 找到label对应的位置
        for j in range(seq_len):
            if input_ids[j].item() == label_id and pos2d[j, 0].item() == branch_pos and pos2d[j, 1].item() == time_pos + 1:
                status = "✅ 正确"
                break
        else:
            # 检查是否预测了错误的branch
            for j in range(seq_len):
                if input_ids[j].item() == label_id:
                    other_branch = pos2d[j, 0].item()
                    if other_branch != branch_pos:
                        status = f"❌ 错误! label来自branch {other_branch}"
                        break

    print(f"{i:4d} | B{branch_pos:3d} | t{time_pos:2d} | '{input_token}' → '{label_token}' | {status}")

print("\n" + "=" * 80)
print("验证结果:")
print("=" * 80)

# 统计label的正确性
valid_labels = labels[labels != -100]
print(f"总token数: {seq_len}")
print(f"有效label数: {valid_labels.numel()}")
print(f"忽略的label数: {(labels == -100).sum().item()}")

# 检查每个branch的label
print("\n每个branch的详细验证:")
for branch_idx in [0, 128]:
    mask = (pos2d[:, 0] == branch_idx) & (attn_mask == 1)
    indices = mask.nonzero(as_tuple=True)[0]

    if len(indices) == 0:
        continue

    print(f"\n{'='*60}")
    print(f"Branch {branch_idx} (共 {len(indices)} 个tokens)")
    print(f"{'='*60}")

    tokens = []
    for idx in indices:
        token_id = input_ids[idx].item()
        token = tokenizer.decode([token_id])
        tokens.append(token)
    print(f"完整序列: {''.join(tokens)}")

    # 详细验证每个token的label
    print(f"\n位置映射 (flat_pos → time):")
    all_correct = True
    for i, idx in enumerate(indices):
        flat_pos = idx.item()
        time_pos = pos2d[idx, 1].item()
        input_id = input_ids[idx].item()
        label_id = labels[idx].item()

        input_token = tokenizer.decode([input_id])

        if i < len(indices) - 1:
            # 不是最后一个token，应该有label
            next_idx = indices[i + 1].item()
            expected_label_id = input_ids[next_idx].item()
            next_time = pos2d[indices[i + 1], 1].item()

            if label_id == expected_label_id:
                label_token = tokenizer.decode([label_id])
                print(f"  t{time_pos} (pos {flat_pos:2d}): '{input_token}' → '{label_token}' (t{next_time}) ✅")
            else:
                expected_token = tokenizer.decode([expected_label_id])
                actual_token = tokenizer.decode([label_id]) if label_id != -100 else "<IGNORE>"
                print(f"  t{time_pos} (pos {flat_pos:2d}): '{input_token}' → '{actual_token}' ❌ 应为 '{expected_token}'")
                all_correct = False
        else:
            # 最后一个token，应该没有label (=-100)
            if label_id == -100:
                print(f"  t{time_pos} (pos {flat_pos:2d}): '{input_token}' → <IGNORE> ✅ (最后一个token)")
            else:
                actual_token = tokenizer.decode([label_id])
                print(f"  t{time_pos} (pos {flat_pos:2d}): '{input_token}' → '{actual_token}' ❌ 最后token应为-100")
                all_correct = False

    if all_correct:
        print(f"\n✅ Branch {branch_idx}: 所有labels都正确！")
    else:
        print(f"\n❌ Branch {branch_idx}: 发现错误的labels！")

print("\n" + "=" * 80)
print("✅ 如果所有label都标记为'✅'，说明修复成功！")
print("=" * 80)
