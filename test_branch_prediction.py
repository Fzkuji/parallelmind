"""
测试脚本：验证当前训练中的分支预测机制
"""
import torch
from transformers import AutoTokenizer
from parallel_data.parallel_collator import ParallelPretrainCollator


def test_branch_prediction():
    """
    测试分支预测机制

    假设有3个分支，每个分支有3个token：
    - 分支0: [100, 101, 102]
    - 分支1: [200, 201, 202]
    - 分支2: [300, 301, 302]

    在interleave模式下，排列应该是：
    位置0: 100 (分支0, order=0)
    位置1: 200 (分支1, order=0)
    位置2: 300 (分支2, order=0)
    位置3: 101 (分支0, order=1)
    位置4: 201 (分支1, order=1)
    位置5: 301 (分支2, order=1)
    位置6: 102 (分支0, order=2)
    位置7: 202 (分支1, order=2)
    位置8: 302 (分支2, order=2)

    我们要检查：
    - 位置0的token(100)预测的目标是什么？
      - 跨分支预测：应该是位置1的token(200)
      - 分支内预测：应该是位置3的token(101)
    """

    # 使用一个简单的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建collator，使用interleave模式
    collator = ParallelPretrainCollator(
        tokenizer=tokenizer,
        branches_per_sample=3,
        interleave_branches=True,
        branch_stride=128,
        random_time_offset=False,  # 关闭随机offset以便测试
        max_branches_per_sample=3,
        min_branches_per_sample=3,
    )

    # 创建测试数据：3个简单的文本
    features = [
        {"text": "A B C"},      # 分支0
        {"text": "D E F"},      # 分支1
        {"text": "G H I"},      # 分支2
    ]

    # 处理数据
    batch = collator(features)

    print("=" * 80)
    print("测试分支预测机制")
    print("=" * 80)

    # 提取第一个样本的数据
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    pos2d = batch["pos2d"][0]
    time_ids = batch["time_ids"][0]
    attention_mask = batch["attention_mask"][0]

    # 打印布局
    print("\n序列布局：")
    print("-" * 80)
    print(f"{'Position':<10} {'Token ID':<10} {'Label':<10} {'Branch':<10} {'Time':<10} {'Mask':<10}")
    print("-" * 80)

    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            continue
        token_id = input_ids[i].item()
        label_id = labels[i].item()
        branch_pos = pos2d[i, 0].item()
        time_val = time_ids[i].item()
        mask_val = attention_mask[i].item()

        # 解码token（如果可能）
        try:
            token_text = tokenizer.decode([token_id])
        except:
            token_text = f"<{token_id}>"

        label_text = str(label_id) if label_id != -100 else "IGNORE"

        print(f"{i:<10} {token_id:<10} {label_text:<10} {branch_pos:<10} {time_val:<10} {mask_val:<10} | {token_text}")

    print("\n" + "=" * 80)
    print("分析预测机制：")
    print("=" * 80)

    # 找到每个分支的token位置
    branches = {}
    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            continue
        branch_pos = pos2d[i, 0].item()
        if branch_pos not in branches:
            branches[branch_pos] = []
        branches[branch_pos].append(i)

    print(f"\n找到 {len(branches)} 个分支")

    for branch_pos in sorted(branches.keys()):
        positions = branches[branch_pos]
        print(f"\n分支 {branch_pos} (stride位置):")
        print(f"  包含的序列位置: {positions}")
        print(f"  Token IDs: {[input_ids[p].item() for p in positions]}")

        # 检查预测目标
        print(f"  预测关系:")
        for i, pos in enumerate(positions[:-1]):
            token_id = input_ids[pos].item()
            label_id = labels[pos].item()
            next_pos = positions[i + 1]
            next_token_id = input_ids[next_pos].item()

            if label_id == next_token_id:
                print(f"    位置{pos}(token={token_id}) -> 预测位置{next_pos}(token={next_token_id}) ✓ 分支内预测")
            elif label_id != -100:
                print(f"    位置{pos}(token={token_id}) -> 预测token={label_id} (不是下一个位置{next_pos}的{next_token_id})")

    # 检查是否是跨分支预测
    print("\n" + "=" * 80)
    print("检查是否实现了跨分支预测：")
    print("=" * 80)

    all_positions = []
    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            continue
        all_positions.append(i)

    cross_branch_predictions = 0
    intra_branch_predictions = 0

    for i in range(len(all_positions) - 1):
        pos = all_positions[i]
        next_pos = all_positions[i + 1]

        token_id = input_ids[pos].item()
        label_id = labels[pos].item()
        next_token_id = input_ids[next_pos].item()

        branch_pos = pos2d[pos, 0].item()
        next_branch_pos = pos2d[next_pos, 0].item()

        if label_id == -100:
            continue

        # 跨分支预测：预测的是下一个位置的token（不同分支）
        if label_id == next_token_id and branch_pos != next_branch_pos:
            cross_branch_predictions += 1
            print(f"  位置{pos}(分支{branch_pos}) -> 位置{next_pos}(分支{next_branch_pos}): 跨分支预测 ✓")
        # 分支内预测：预测的是同一分支内的下一个token
        elif branch_pos == next_branch_pos:
            # 检查是否预测的是同一分支的下一个token
            same_branch_positions = branches[branch_pos]
            pos_idx = same_branch_positions.index(pos)
            if pos_idx + 1 < len(same_branch_positions):
                expected_next = same_branch_positions[pos_idx + 1]
                expected_token = input_ids[expected_next].item()
                if label_id == expected_token:
                    intra_branch_predictions += 1

    print(f"\n总结:")
    print(f"  跨分支预测数量: {cross_branch_predictions}")
    print(f"  分支内预测数量: {intra_branch_predictions}")

    if cross_branch_predictions > 0:
        print(f"\n✓ 实现了跨分支预测！")
    elif intra_branch_predictions > 0:
        print(f"\n✗ 当前实现的是分支内预测，不是跨分支预测！")
    else:
        print(f"\n? 无法确定预测机制")


if __name__ == "__main__":
    test_branch_prediction()
