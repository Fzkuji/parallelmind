"""Visualize all ParallelMind ablation experiment results.

Covers:
  - 512 model branch ablation (5 rope ratios × 4 train configs)
  - 1024 model head-dim ablation (h8/h16/h32 × 5 rope ratios)
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

OUT_DIR = Path("Experiments/Figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# DATA: 512 model (hidden_size=512, 8 heads, 8 layers, ~26M params)
# ══════════════════════════════════════════════════════════════════
EVAL_BRANCHES_512 = [1, 2, 4, 8, 16, 24, 32]

# data_512[rope][train_config][eval_branch] = loss
data_512: dict = {
    0: {
        "fixed1": {1: 2.4375, 2: 3.6508, 4: 4.7192, 8: 5.3685, 16: 5.8068, 24: 5.9046, 32: 5.9810},
        "1-3":    {1: 2.5004, 2: 2.5324, 4: 3.2524, 8: 4.7508, 16: 5.5363, 24: 5.8327, 32: 6.0255},
        "1-7":    {1: 2.6127, 2: 2.6290, 4: 2.8750, 8: 3.7542, 16: 4.7802, 24: 5.4270, 32: 5.8463},
        "1-15":   {1: 2.7636, 2: 2.7755, 4: 2.9982, 8: 3.7896, 16: 4.8697, 24: 5.2942, 32: 5.5628},
    },
    0.25: {
        "fixed1": {1: 2.4390, 2: 3.6415, 4: 4.7215, 8: 5.3212, 16: 5.7546, 24: 5.8635, 32: 5.9413},
        "1-3":    {1: 2.5038, 2: 2.5147, 4: 3.0474, 8: 4.5993, 16: 5.7190, 24: 6.0214, 32: 6.1156},
        "1-7":    {1: 2.5683, 2: 2.5904, 4: 2.6614, 8: 2.9755, 16: 5.1213, 24: 5.9501, 32: 6.3222},
        "1-15":   {1: 2.7059, 2: 2.6979, 4: 2.7274, 8: 2.8714, 16: 3.5872, 24: 4.5737, 32: 4.9964},
    },
    0.5: {
        "fixed1": {1: 2.4413, 2: 3.6995, 4: 4.7415, 8: 5.3302, 16: 5.7298, 24: 5.8416, 32: 5.9084},
        "1-3":    {1: 2.4736, 2: 2.4721, 4: 2.7301, 8: 3.8857, 16: 5.1408, 24: 5.4474, 32: 5.5439},
        "1-7":    {1: 2.5551, 2: 2.5601, 4: 2.5958, 8: 2.8091, 16: 3.6047, 24: 4.6056, 32: 5.3546},
        "1-15":   {1: 2.7044, 2: 2.7002, 4: 2.7217, 8: 2.8221, 16: 3.0306, 24: 3.2347, 32: 3.4602},
    },
    0.75: {
        "fixed1": {1: 2.4461, 2: 3.6530, 4: 4.8042, 8: 5.4034, 16: 5.7346, 24: 5.8634, 32: 5.9604},
        "1-3":    {1: 2.4734, 2: 2.4757, 4: 2.6787, 8: 3.2375, 16: 4.4666, 24: 4.9195, 32: 5.1229},
        "1-7":    {1: 2.5108, 2: 2.4953, 4: 2.4971, 8: 2.5405, 16: 3.0241, 24: 3.4677, 32: 3.8966},
        "1-15":   {1: 2.6223, 2: 2.6070, 4: 2.5998, 8: 2.6006, 16: 2.6497, 24: 2.7585, 32: 2.9389},
    },
    1.0: {
        "fixed1": {1: 2.4609, 2: 3.7923, 4: 4.5955, 8: 5.1308, 16: 5.5021, 24: 5.6867, 32: 5.8063},
        "1-3":    {1: 2.4799, 2: 2.4695, 4: 2.5423, 8: 2.6774, 16: 3.2281, 24: 3.7215, 32: 4.0977},
        "1-7":    {1: 2.5092, 2: 2.5035, 4: 2.4977, 8: 2.4992, 16: 2.5389, 24: 2.6115, 32: 2.7805},
        "1-15":   {1: 2.5997, 2: 2.5815, 4: 2.5851, 8: 2.5806, 16: 2.5887, 24: 2.6096, 32: 2.6340},
    },
}

# ══════════════════════════════════════════════════════════════════
# DATA: 1024 model head-dim ablation (~270M params, 24 layers)
# Training config: 1-15, eval branches: 1,2,4,8,16,24,32,48,64
# ══════════════════════════════════════════════════════════════════
EVAL_BRANCHES_1024 = [1, 2, 4, 8, 16, 24, 32, 48, 64]

# Full scaling curves (train 1-15)
scaling_1024: dict = {
    # h8 (head_dim=128, 64 freq pairs)
    ("h8", 1.0):  {1: 3.052, 2: 2.97, 4: 2.93, 8: 2.922, 16: 2.917, 24: 2.93, 32: 2.93, 48: 2.94, 64: 2.956},
    ("h8", 0.75): {1: 3.035, 2: 2.94, 4: 2.92, 8: 2.904, 16: 2.920, 24: 3.18, 32: 3.33, 48: 3.60, 64: 3.731},
    ("h8", 0.5):  {1: 2.912, 8: 3.006, 16: 3.184, 64: 4.323},
    ("h8", 0.25): {1: 3.704, 8: 4.342, 16: 4.571, 64: 5.114},
    ("h8", 0):    {1: 3.621, 8: 3.876, 16: 4.925, 64: 5.762},
    # h16 (head_dim=64, 32 freq pairs) — reference baseline
    ("h16", 1.0):  {1: 2.77, 2: 2.76, 4: 2.75, 8: 2.75, 16: 2.75, 24: 2.76, 32: 2.77, 48: 2.79, 64: 2.82},
    ("h16", 0.75): {1: 2.73, 2: 2.71, 4: 2.70, 8: 2.72, 16: 2.74, 24: 2.87, 32: 3.03, 48: 3.20, 64: 3.31},
    ("h16", 0.5):  {1: 2.8506, 64: 4.6163},
    ("h16", 0.25): {1: 3.3023, 64: 5.2793},
    ("h16", 0):    {1: 3.2204, 64: 5.6485},
    # h32 (head_dim=32, 16 freq pairs)
    ("h32", 1.0):  {1: 2.672, 2: 2.66, 4: 2.66, 8: 2.652, 16: 2.661, 24: 2.77, 32: 2.84, 48: 2.91, 64: 2.992},
    ("h32", 0.75): {1: 2.719, 8: 2.735, 16: 2.829, 64: 3.520},
    ("h32", 0.5):  {1: 2.918, 8: 3.024, 16: 3.201, 64: 4.883},
    ("h32", 0.25): {1: 3.146, 8: 3.970, 16: 4.511, 64: 5.324},
    ("h32", 0):    {1: 3.233, 8: 5.282, 16: 5.596, 64: 5.779},
}

# 1024 model fixed1 baseline (single-branch training)
fixed1_1024: dict = {
    ("h8", 0):    {1: 2.4194, 64: 5.884},
    ("h8", 0.75): {1: 2.3948, 64: 5.720},
    ("h8", 1.0):  {1: 2.4294, 64: 5.554},
    ("h16", 0):   {1: 2.4285},
    ("h16", 0.5): {1: 2.3817},
    ("h16", 1.0): {1: 2.4802},
    ("h32", 0):   {1: 2.4036, 64: 5.966},
    ("h32", 0.75):{1: 2.3769, 64: 5.885},
    ("h32", 1.0): {1: 2.4351, 64: 5.809},
}

# ── Color / marker palettes ───────────────────────────────────────
RATIO_COLORS = {0: "#d62728", 0.25: "#ff7f0e", 0.5: "#2ca02c", 0.75: "#1f77b4", 1.0: "#9467bd"}
RATIO_MARKERS = {0: "o", 0.25: "s", 0.5: "^", 0.75: "D", 1.0: "P"}
TC_COLORS = {"fixed1": "#d62728", "1-3": "#ff7f0e", "1-7": "#2ca02c", "1-15": "#1f77b4"}
TC_MARKERS = {"fixed1": "o", "1-3": "s", "1-7": "^", "1-15": "D"}
HEAD_COLORS = {"h8": "#d62728", "h16": "#2ca02c", "h32": "#1f77b4"}
HEAD_MARKERS = {"h8": "o", "h16": "s", "h32": "^"}

RATIOS = [0, 0.25, 0.5, 0.75, 1.0]
TRAIN_CONFIGS = ["fixed1", "1-3", "1-7", "1-15"]


# ══════════════════════════════════════════════════════════════════
# Figure 1: 512 Model — Loss vs Eval Branches (by Train Config)
# ══════════════════════════════════════════════════════════════════
def fig1_512_by_train():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("512 Model: Loss vs Eval Branches (by Training Config)", fontsize=15, y=1.02)
    for ax, tc in zip(axes, TRAIN_CONFIGS):
        for r in RATIOS:
            losses = [data_512[r][tc][eb] for eb in EVAL_BRANCHES_512]
            ax.plot(EVAL_BRANCHES_512, losses, marker=RATIO_MARKERS[r], color=RATIO_COLORS[r],
                    label=f"r={r}", linewidth=2, markersize=6)
        ax.set_title(f"Train: {tc}")
        ax.set_xlabel("Eval Branches")
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_BRANCHES_512)
        ax.set_xticklabels([str(b) for b in EVAL_BRANCHES_512], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(2.0, 6.5)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(title="rope_2d_ratio", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_512_by_train.png")
    fig.savefig(OUT_DIR / "fig1_512_by_train.pdf")
    print("Saved fig1_512_by_train")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 2: 512 Model — Loss vs Eval Branches (by Rope Ratio)
# ══════════════════════════════════════════════════════════════════
def fig2_512_by_rope():
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
    fig.suptitle("512 Model: Loss vs Eval Branches (by rope_2d_ratio)", fontsize=15, y=1.02)
    for ax, r in zip(axes, RATIOS):
        for tc in TRAIN_CONFIGS:
            losses = [data_512[r][tc][eb] for eb in EVAL_BRANCHES_512]
            ax.plot(EVAL_BRANCHES_512, losses, marker=TC_MARKERS[tc], color=TC_COLORS[tc],
                    label=tc, linewidth=2, markersize=6)
        ax.set_title(f"rope_2d_ratio = {r}")
        ax.set_xlabel("Eval Branches")
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_BRANCHES_512)
        ax.set_xticklabels([str(b) for b in EVAL_BRANCHES_512], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(2.0, 6.5)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(title="Train Config", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_512_by_rope.png")
    fig.savefig(OUT_DIR / "fig2_512_by_rope.pdf")
    print("Saved fig2_512_by_rope")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 3: 512 Model — Heatmap (rope_ratio × eval_branches)
# ══════════════════════════════════════════════════════════════════
def fig3_512_heatmap():
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("512 Model: Loss Heatmap (rope_ratio vs eval_branches)", fontsize=15, y=1.02)
    for ax_idx, tc in enumerate(TRAIN_CONFIGS):
        mat = np.zeros((len(RATIOS), len(EVAL_BRANCHES_512)))
        for i, r in enumerate(RATIOS):
            for j, eb in enumerate(EVAL_BRANCHES_512):
                mat[i, j] = data_512[r][tc][eb]
        im = ax_idx
        im = axes[ax_idx].imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=2.3, vmax=6.0)
        axes[ax_idx].set_xticks(range(len(EVAL_BRANCHES_512)))
        axes[ax_idx].set_xticklabels([str(b) for b in EVAL_BRANCHES_512])
        axes[ax_idx].set_yticks(range(len(RATIOS)))
        axes[ax_idx].set_yticklabels([str(r) for r in RATIOS])
        axes[ax_idx].set_xlabel("Eval Branches")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("rope_2d_ratio")
        axes[ax_idx].set_title(f"Train: {tc}")
        for i in range(len(RATIOS)):
            for j in range(len(EVAL_BRANCHES_512)):
                axes[ax_idx].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                  color="white" if mat[i, j] > 4.5 else "black")
    plt.colorbar(im, ax=axes[-1], shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_512_heatmap.png")
    fig.savefig(OUT_DIR / "fig3_512_heatmap.pdf")
    print("Saved fig3_512_heatmap")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 4: 1024 Model — Head-Dim Scaling Curves (rope=1.0)
# ══════════════════════════════════════════════════════════════════
def fig4_1024_scaling():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    configs = [
        ("h8", 1.0, "h8 (d=128), r=1.0"),
        ("h16", 1.0, "h16 (d=64), r=1.0"),
        ("h32", 1.0, "h32 (d=32), r=1.0"),
        ("h16", 0.75, "h16 (d=64), r=0.75"),
        ("h8", 0.75, "h8 (d=128), r=0.75"),
    ]
    linestyles = ["-", "-", "-", "--", "--"]
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "s", "o"]

    for (head, rope, label), ls, c, m in zip(configs, linestyles, colors, markers):
        d = scaling_1024[(head, rope)]
        ebs = sorted(d.keys())
        losses = [d[eb] for eb in ebs]
        ax.plot(ebs, losses, marker=m, color=c, label=label, linewidth=2.5, markersize=7, linestyle=ls)

    ax.set_xlabel("Eval Branches")
    ax.set_ylabel("Loss")
    ax.set_title("1024 Model (~270M): Scaling Curves by Head Dimension\n(Training config: 1-15 branches)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(EVAL_BRANCHES_1024)
    ax.set_xticklabels([str(b) for b in EVAL_BRANCHES_1024])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(2.4, 4.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_1024_scaling.png")
    fig.savefig(OUT_DIR / "fig4_1024_scaling.pdf")
    print("Saved fig4_1024_scaling")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5: 1024 Model — rope_ratio impact per head config (train 1-15)
# ══════════════════════════════════════════════════════════════════
def fig5_1024_rope_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    fig.suptitle("1024 Model: rope_2d_ratio Impact by Head Config (Train 1-15)", fontsize=15, y=1.02)
    heads = ["h8", "h16", "h32"]
    head_labels = {"h8": "h8 (d=128, 64 freq pairs)", "h16": "h16 (d=64, 32 freq pairs)", "h32": "h32 (d=32, 16 freq pairs)"}

    for ax, head in zip(axes, heads):
        for r in RATIOS:
            key = (head, r)
            if key not in scaling_1024:
                continue
            d = scaling_1024[key]
            ebs = sorted(d.keys())
            losses = [d[eb] for eb in ebs]
            ax.plot(ebs, losses, marker=RATIO_MARKERS[r], color=RATIO_COLORS[r],
                    label=f"r={r}", linewidth=2, markersize=6)
        ax.set_title(head_labels[head])
        ax.set_xlabel("Eval Branches")
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_BRANCHES_1024)
        ax.set_xticklabels([str(b) for b in EVAL_BRANCHES_1024], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(2.4, 6.0)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(title="rope_2d_ratio", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_1024_rope.png")
    fig.savefig(OUT_DIR / "fig5_1024_rope.pdf")
    print("Saved fig5_1024_rope")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 6: Cross-Scale — Best configs 512 vs 1024
# ══════════════════════════════════════════════════════════════════
def fig6_cross_scale():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 512 best: rope=1.0, train=1-15
    losses_512 = [data_512[1.0]["1-15"][eb] for eb in EVAL_BRANCHES_512]
    ax.plot(EVAL_BRANCHES_512, losses_512, marker="o", color="#d62728",
            label="512 model (~26M), r=1.0, train 1-15", linewidth=2.5, markersize=7)

    # 512 second best: rope=0.75, train=1-15
    losses_512_075 = [data_512[0.75]["1-15"][eb] for eb in EVAL_BRANCHES_512]
    ax.plot(EVAL_BRANCHES_512, losses_512_075, marker="o", color="#d62728",
            label="512 model (~26M), r=0.75, train 1-15", linewidth=2, markersize=6, linestyle="--")

    # 1024 best: h16, rope=1.0, train=1-15
    d_1024 = scaling_1024[("h16", 1.0)]
    ebs_1024 = sorted(d_1024.keys())
    losses_1024 = [d_1024[eb] for eb in ebs_1024]
    ax.plot(ebs_1024, losses_1024, marker="s", color="#2ca02c",
            label="1024 model (~270M), h16, r=1.0, train 1-15", linewidth=2.5, markersize=7)

    # 1024 h32: rope=1.0, train=1-15
    d_h32 = scaling_1024[("h32", 1.0)]
    ebs_h32 = sorted(d_h32.keys())
    losses_h32 = [d_h32[eb] for eb in ebs_h32]
    ax.plot(ebs_h32, losses_h32, marker="^", color="#1f77b4",
            label="1024 model (~270M), h32, r=1.0, train 1-15", linewidth=2.5, markersize=7)

    # Baselines: fixed1 single branch (standard transformer)
    ax.axhline(y=2.44, color="gray", linestyle=":", linewidth=1.5, label="512 fixed1 baseline (~2.44)")
    ax.axhline(y=2.40, color="gray", linestyle="-.", linewidth=1.5, label="1024 fixed1 baseline (~2.40)")

    ax.set_xlabel("Eval Branches")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-Scale Comparison: Best Parallel Decoding Configs")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 24, 32, 48, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24", "32", "48", "64"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(2.2, 3.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_cross_scale.png")
    fig.savefig(OUT_DIR / "fig6_cross_scale.pdf")
    print("Saved fig6_cross_scale")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 7: 512 Model — Key Comparison (Baseline vs Best Parallel)
# ══════════════════════════════════════════════════════════════════
def fig7_512_key():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    configs = [
        (0, "fixed1", "Standard Transformer\n(rope=0, fixed1)", "#d62728", "o", "--"),
        (1.0, "fixed1", "rope=1.0, fixed1", "#ff7f0e", "s", "--"),
        (0.5, "1-15", "rope=0.5, train 1-15", "#2ca02c", "^", "-"),
        (0.75, "1-15", "rope=0.75, train 1-15", "#17becf", "D", "-"),
        (1.0, "1-15", "rope=1.0, train 1-15", "#9467bd", "P", "-"),
        (1.0, "1-7", "rope=1.0, train 1-7", "#8c564b", "X", "-"),
    ]
    for r, tc, label, color, marker, ls in configs:
        losses = [data_512[r][tc][eb] for eb in EVAL_BRANCHES_512]
        ax.plot(EVAL_BRANCHES_512, losses, marker=marker, color=color,
                label=label, linewidth=2.5, markersize=8, linestyle=ls)
    ax.set_xlabel("Eval Branches")
    ax.set_ylabel("Loss")
    ax.set_title("512 Model (~26M): Standard Transformer vs Best Parallel Configs")
    ax.set_xscale("log", base=2)
    ax.set_xticks(EVAL_BRANCHES_512)
    ax.set_xticklabels([str(b) for b in EVAL_BRANCHES_512])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(2.0, 6.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_512_key.png")
    fig.savefig(OUT_DIR / "fig7_512_key.pdf")
    print("Saved fig7_512_key")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 8: Loss Degradation Rate — Δloss from branch=1 to max branch
# ══════════════════════════════════════════════════════════════════
def fig8_degradation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 512 model: delta from b=1 to b=32
    ax = axes[0]
    ax.set_title("512 Model: Loss Increase (b=1 → b=32)")
    bar_data = []
    for r in RATIOS:
        for tc in ["1-7", "1-15"]:
            delta = data_512[r][tc][32] - data_512[r][tc][1]
            bar_data.append((f"r={r}\n{tc}", delta, RATIO_COLORS[r]))
    labels, deltas, colors = zip(*bar_data)
    bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Δ Loss (b=32 − b=1)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{d:.2f}", ha="center", va="bottom", fontsize=7)

    # 1024 model: delta from b=1 to b=64
    ax = axes[1]
    ax.set_title("1024 Model: Loss Increase (b=1 → b=64)")
    bar_data_1024 = []
    for head in ["h8", "h16", "h32"]:
        for r in [0.75, 1.0]:
            key = (head, r)
            d = scaling_1024.get(key, {})
            if 1 in d and 64 in d:
                delta = d[64] - d[1]
                bar_data_1024.append((f"{head}\nr={r}", delta, HEAD_COLORS[head]))
    if bar_data_1024:
        labels, deltas, colors = zip(*bar_data_1024)
        bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Δ Loss (b=64 − b=1)")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{d:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_degradation.png")
    fig.savefig(OUT_DIR / "fig8_degradation.pdf")
    print("Saved fig8_degradation")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_512_by_train()
    fig2_512_by_rope()
    fig3_512_heatmap()
    fig4_1024_scaling()
    fig5_1024_rope_comparison()
    fig6_cross_scale()
    fig7_512_key()
    fig8_degradation()
    print(f"\nAll figures saved to {OUT_DIR}/")
