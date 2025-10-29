"""
验证prefill阶段的attention mask是否正确隔离了不同branch
"""
import torch
from transformers import AutoTokenizer
from parallel.columnar import build_flat_linear_layout, build_columnar_causal_mask

def verify_mask_isolation():
    tokenizer = AutoTokenizer.from_pretrained("./model/")

    # 构造两个不同的问题
    samples = [{
        "main": "",
        "branches": [
            {"input_ids": tokenizer("<|im_start|>现在给我生成一首关于秋天", add_special_tokens=False).input_ids, "answer_offset": 9},
            {"input_ids": tokenizer("<|im_start|>请介绍下大语言模型的基本原理", add_special_tokens=False).input_ids, "answer_offset": 9}
        ]
    }]

    layout = build_flat_linear_layout(
        tokenizer,
        samples,
        device="cpu",
        pad_to=None,
        align_to="left",
        interleave_branches=True,  # pretrain模式
    )

    # 构建mask
    mask = build_columnar_causal_mask(layout.time_ids, layout.attention_mask)

    # 分析mask结构
    print("=" * 80)
    print("Mask Isolation Verification")
    print("=" * 80)

    seq_len = int(layout.attention_mask.sum().item())
    print(f"\nTotal sequence length: {seq_len}")

    # 找到两个branch的最后位置
    metadata = layout.metadata[0]
    branch_pos1d_end = metadata.branch_pos1d_end

    print(f"\nBranch 0 last position: {branch_pos1d_end[0]}")
    print(f"Branch 1 last position: {branch_pos1d_end[1]}")

    # 检查每个branch最后位置的token能看到什么
    for branch_idx in range(2):
        pos = branch_pos1d_end[branch_idx]
        token_id = layout.input_ids[0, pos].item()
        token_text = tokenizer.decode([token_id])
        token_time = layout.time_ids[0, pos].item()
        token_pos2d = layout.pos2d[0, pos].tolist()

        print(f"\n{'='*80}")
        print(f"Branch {branch_idx} - Last Token Analysis")
        print(f"{'='*80}")
        print(f"Position: {pos}")
        print(f"Token: '{token_text}' (id={token_id})")
        print(f"Time: {token_time}")
        print(f"Pos2D: {token_pos2d}")

        # 检查这个token能看到哪些位置
        attention_row = mask[0, 0, pos, :seq_len]  # [seq_len]
        visible_positions = (attention_row > -1e8).nonzero(as_tuple=True)[0].tolist()

        print(f"\nCan see {len(visible_positions)} positions: {visible_positions}")

        # 分析能看到的token中，有多少来自另一个branch
        branch_positions = metadata.branch_positions
        own_branch_pos = branch_positions[branch_idx]
        other_branch_pos = branch_positions[1 - branch_idx]

        own_count = 0
        other_count = 0

        for vis_pos in visible_positions:
            vis_pos2d = layout.pos2d[0, vis_pos, 0].item()
            vis_time = layout.time_ids[0, vis_pos].item()

            if vis_pos2d == own_branch_pos:
                own_count += 1
            elif vis_pos2d == other_branch_pos:
                other_count += 1
                # 这是问题！不应该看到另一个branch的token
                vis_token_id = layout.input_ids[0, vis_pos].item()
                vis_token_text = tokenizer.decode([vis_token_id])
                print(f"  !!! WARNING: Can see other branch token at pos={vis_pos}")
                print(f"      Token: '{vis_token_text}' (id={vis_token_id})")
                print(f"      Pos2D: [{vis_pos2d}, {vis_time}]")

        print(f"\nSummary:")
        print(f"  Own branch tokens: {own_count}")
        print(f"  Other branch tokens: {other_count}")

        if other_count > 0:
            print(f"  ❌ PROBLEM: Branch {branch_idx} can see {other_count} tokens from the other branch!")
        else:
            print(f"  ✓ OK: Branch {branch_idx} is properly isolated")

    # 检查同一time的token是否互相隔离
    print(f"\n{'='*80}")
    print("Same-Time Token Isolation Check")
    print(f"{'='*80}")

    max_time = layout.time_ids[0, :seq_len].max().item()
    print(f"Max time: {max_time}")

    for t in range(max_time + 1):
        # 找到所有time=t的token
        time_mask = (layout.time_ids[0, :seq_len] == t)
        positions_at_time_t = time_mask.nonzero(as_tuple=True)[0].tolist()

        if len(positions_at_time_t) <= 1:
            continue

        print(f"\nTime {t}: {len(positions_at_time_t)} tokens at positions {positions_at_time_t}")

        # 检查这些token是否能互相看到
        for i, pos_i in enumerate(positions_at_time_t):
            for j, pos_j in enumerate(positions_at_time_t):
                if i == j:
                    continue
                can_see = mask[0, 0, pos_i, pos_j].item() > -1e8
                if can_see:
                    token_i = tokenizer.decode([layout.input_ids[0, pos_i].item()])
                    token_j = tokenizer.decode([layout.input_ids[0, pos_j].item()])
                    print(f"  ❌ PROBLEM: Token at {pos_i} ('{token_i}') can see token at {pos_j} ('{token_j}') in same time!")

if __name__ == "__main__":
    verify_mask_isolation()
