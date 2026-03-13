"""Scaling trend analysis for 2D RoPE ratio vs branch length and branch count."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

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
# Data
# ══════════════════════════════════════════════════════════════════

RATIOS = [0, 0.25, 0.5, 0.75, 1.0]

# 512 model, train 1-15, eval branches 1..32
data_512 = {
    0:    {1: 2.7636, 2: 2.7755, 4: 2.9982, 8: 3.7896, 16: 4.8697, 24: 5.2942, 32: 5.5628},
    0.25: {1: 2.7059, 2: 2.6979, 4: 2.7274, 8: 2.8714, 16: 3.5872, 24: 4.5737, 32: 4.9964},
    0.5:  {1: 2.7044, 2: 2.7002, 4: 2.7217, 8: 2.8221, 16: 3.0306, 24: 3.2347, 32: 3.4602},
    0.75: {1: 2.6223, 2: 2.6070, 4: 2.5998, 8: 2.6006, 16: 2.6497, 24: 2.7585, 32: 2.9389},
    1.0:  {1: 2.5997, 2: 2.5815, 4: 2.5851, 8: 2.5806, 16: 2.5887, 24: 2.6096, 32: 2.6340},
}

# 1024 model h16, train 1-15, eval branches 1..64
data_1024 = {
    0:    {1: 3.2204, 64: 5.6485},
    0.25: {1: 3.3023, 64: 5.2793},
    0.5:  {1: 2.8506, 64: 4.6163},
    0.75: {1: 2.73, 2: 2.71, 4: 2.70, 8: 2.72, 16: 2.74, 24: 2.87, 32: 3.03, 48: 3.20, 64: 3.31},
    1.0:  {1: 2.77, 2: 2.76, 4: 2.75, 8: 2.75, 16: 2.75, 24: 2.76, 32: 2.77, 48: 2.79, 64: 2.82},
}

# 2048 packing, train 1-4, eval branches 1..4
data_2048 = {
    0:    {1: 3.3316, 2: 3.3221, 4: 3.3220},
    0.25: {1: 3.4309, 2: 3.6512, 4: 3.6512},
    0.5:  {1: 3.3256, 2: 3.3123, 4: 3.3122},
    0.75: {1: 3.2704, 2: 3.2532, 4: 3.2531},
    1.0:  {1: 3.3038, 2: 3.2881, 4: 3.2880},
}


# ══════════════════════════════════════════════════════════════════
# Figure 1: Optimal ratio vs branch length
# ══════════════════════════════════════════════════════════════════
def fig1_optimal_ratio_vs_length():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Optimal ratio vs branch length ---
    ax = axes[0]

    # 512 model: best ratio at max eval branches (32)
    # Find best ratio for each eval branch
    best_512_32 = min(RATIOS, key=lambda r: data_512[r][32])  # 1.0
    best_512_16 = min(RATIOS, key=lambda r: data_512[r][16])  # 1.0
    best_512_8 = min(RATIOS, key=lambda r: data_512[r][8])    # 1.0

    # 1024 model: best ratio at 64 branches
    best_1024_64 = min([0.75, 1.0], key=lambda r: data_1024[r][64])  # 1.0

    # 2048 packing: best ratio at 4 branches
    best_2048_4 = min(RATIOS, key=lambda r: data_2048[r][4])  # 0.75

    # Data points: (branch_length, optimal_ratio, label)
    points = [
        (300, 1.0, "512 model\n(~300 tok, b=32)"),
        (300, 1.0, "1024 model\n(~300 tok, b=64)"),
        (2048, 0.75, "1024 model\n(2048 tok, b=4)"),
    ]

    lengths = [300, 300, 2048]
    opt_ratios = [1.0, 1.0, 0.75]

    ax.scatter([300], [1.0], s=150, c="#2ca02c", zorder=5, marker="o")
    ax.scatter([2048], [0.75], s=150, c="#d62728", zorder=5, marker="s")

    ax.annotate("512 & 1024 model\n(~300 tok/branch)", (300, 1.0),
                textcoords="offset points", xytext=(60, -30), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("1024 model + packing\n(2048 tok/branch)", (2048, 0.75),
                textcoords="offset points", xytext=(-60, 30), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))

    # Log extrapolation
    # ratio = 1.0 - k * ln(length / L0)
    # At L=300, ratio=1.0 → baseline
    # At L=2048, ratio=0.75 → 0.75 = 1.0 - k * ln(2048/300) → k = 0.25/ln(6.83) = 0.130
    L0 = 300
    k = 0.25 / np.log(2048 / L0)
    lengths_ext = np.logspace(np.log10(100), np.log10(16384), 100)
    ratios_ext = np.clip(1.0 - k * np.log(lengths_ext / L0), 0, 1)

    ax.plot(lengths_ext, ratios_ext, "--", color="gray", linewidth=1.5, alpha=0.7,
            label=f"Extrapolation: r = 1.0 - {k:.3f}·ln(L/{L0})")

    # Mark predictions
    for pred_len in [512, 1024, 4096, 8192]:
        pred_ratio = max(0, 1.0 - k * np.log(pred_len / L0))
        ax.plot(pred_len, pred_ratio, "x", color="gray", markersize=8, alpha=0.5)
        ax.annotate(f"L={pred_len}\nr≈{pred_ratio:.2f}", (pred_len, pred_ratio),
                    textcoords="offset points", xytext=(10, 5), fontsize=7, color="gray")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Branch Length (tokens)")
    ax.set_ylabel("Optimal rope_2d_ratio")
    ax.set_title("Optimal ratio shifts with branch length")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(80, 20000)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # --- Right: Loss at best ratio vs branch length ---
    ax = axes[1]

    # For each ratio, show loss at eval=max_branches, normalized to ratio=0 baseline
    # 512 model (eval=32): relative loss = loss[ratio] / loss[ratio=0]
    rel_512 = {r: data_512[r][32] / data_512[0][32] for r in RATIOS}
    # 2048 packing (eval=4)
    rel_2048 = {r: data_2048[r][4] / data_2048[0][4] for r in RATIOS}

    x = np.arange(len(RATIOS))
    width = 0.35

    bars1 = ax.bar(x - width/2, [rel_512[r] for r in RATIOS], width,
                   label="~300 tok (eval=32)", color="#2ca02c", alpha=0.8)
    bars2 = ax.bar(x + width/2, [rel_2048[r] for r in RATIOS], width,
                   label="2048 tok (eval=4)", color="#d62728", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATIOS])
    ax.set_xlabel("rope_2d_ratio")
    ax.set_ylabel("Relative Loss (normalized to ratio=0)")
    ax.set_title("Ratio impact: short vs long branches")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scaling_optimal_ratio.png")
    fig.savefig(OUT_DIR / "scaling_optimal_ratio.pdf")
    print("Saved scaling_optimal_ratio")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 2: Branch degradation rate as function of ratio
# ══════════════════════════════════════════════════════════════════
def fig2_degradation_curve():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: 512 model degradation curves ---
    ax = axes[0]
    colors = {0: "#d62728", 0.25: "#ff7f0e", 0.5: "#2ca02c", 0.75: "#1f77b4", 1.0: "#9467bd"}
    markers = {0: "o", 0.25: "s", 0.5: "^", 0.75: "D", 1.0: "P"}

    ebs_512 = [1, 2, 4, 8, 16, 24, 32]
    for r in RATIOS:
        # Normalize: delta from b=1
        base = data_512[r][1]
        deltas = [data_512[r][eb] - base for eb in ebs_512]
        ax.plot(ebs_512, deltas, marker=markers[r], color=colors[r],
                label=f"r={r}", linewidth=2, markersize=6)

    ax.set_xlabel("Eval Branches")
    ax.set_ylabel("ΔLoss (relative to b=1)")
    ax.set_title("512 Model (~300 tok/branch): Degradation")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ebs_512)
    ax.set_xticklabels([str(b) for b in ebs_512])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.1, 3.2)

    # --- Right: Degradation rate vs ratio (slope) ---
    ax = axes[1]

    # For 512 model: compute degradation slope (b=1 to b=32)
    degradation_512 = [(data_512[r][32] - data_512[r][1]) for r in RATIOS]
    # For 2048 packing: compute degradation (b=1 to b=4)
    degradation_2048 = [(data_2048[r][4] - data_2048[r][1]) for r in RATIOS]

    ax.plot(RATIOS, degradation_512, "o-", color="#2ca02c", linewidth=2.5, markersize=8,
            label="~300 tok (b=1→32)")
    ax.plot(RATIOS, degradation_2048, "s-", color="#d62728", linewidth=2.5, markersize=8,
            label="2048 tok (b=1→4)")

    # Add value labels
    for r, d in zip(RATIOS, degradation_512):
        ax.annotate(f"{d:.2f}", (r, d), textcoords="offset points", xytext=(8, 5), fontsize=8, color="#2ca02c")
    for r, d in zip(RATIOS, degradation_2048):
        ax.annotate(f"{d:.3f}", (r, d), textcoords="offset points", xytext=(8, -12), fontsize=8, color="#d62728")

    ax.set_xlabel("rope_2d_ratio")
    ax.set_ylabel("ΔLoss (b=1 → max_b)")
    ax.set_title("Branch Degradation vs Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scaling_degradation.png")
    fig.savefig(OUT_DIR / "scaling_degradation.pdf")
    print("Saved scaling_degradation")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 3: "Capacity" analysis — max branches before X% degradation
# ══════════════════════════════════════════════════════════════════
def fig3_capacity():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # For 512 model: at each ratio, find the max branch count where
    # loss is within threshold of b=1 loss
    thresholds = [0.05, 0.1, 0.2, 0.5]
    th_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    ebs_512 = [1, 2, 4, 8, 16, 24, 32]

    for th, tc in zip(thresholds, th_colors):
        capacities = []
        for r in RATIOS:
            base = data_512[r][1]
            max_b = 1
            for eb in ebs_512:
                if data_512[r][eb] - base <= th:
                    max_b = eb
            capacities.append(max_b)
        ax.plot(RATIOS, capacities, "o-", color=tc, linewidth=2, markersize=8,
                label=f"ΔLoss ≤ {th}")

    ax.set_xlabel("rope_2d_ratio")
    ax.set_ylabel("Max Branches within threshold")
    ax.set_title("512 Model: Branch Capacity by Ratio\n(max branches before loss degrades by threshold)")
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.grid(True, alpha=0.3)
    ax.legend(title="Degradation threshold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scaling_capacity.png")
    fig.savefig(OUT_DIR / "scaling_capacity.pdf")
    print("Saved scaling_capacity")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 4: Combined story — 3 experiments on one plot
# ══════════════════════════════════════════════════════════════════
def fig4_combined():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss vs ratio for multi-branch eval, one line per experiment
    # 512 model, train 1-15, eval=32
    losses_512_b32 = [data_512[r][32] for r in RATIOS]
    ax.plot(RATIOS, losses_512_b32, "o-", color="#2ca02c", linewidth=2.5, markersize=8,
            label="512 model (~300 tok, eval=32 branches)")

    # 1024 model h16, train 1-15, eval=64
    losses_1024_b64 = []
    for r in RATIOS:
        if 64 in data_1024[r]:
            losses_1024_b64.append(data_1024[r][64])
        else:
            losses_1024_b64.append(np.nan)
    ax.plot(RATIOS, losses_1024_b64, "s-", color="#1f77b4", linewidth=2.5, markersize=8,
            label="1024 model (~300 tok, eval=64 branches)")

    # 2048 packing, train 1-4, eval=4
    losses_2048_b4 = [data_2048[r][4] for r in RATIOS]
    ax.plot(RATIOS, losses_2048_b4, "^-", color="#d62728", linewidth=2.5, markersize=8,
            label="1024 model (2048 tok, eval=4 branches)")

    # Mark optimal for each
    best_512 = RATIOS[np.argmin(losses_512_b32)]
    best_2048 = RATIOS[np.argmin(losses_2048_b4)]
    ax.axvline(x=best_512, color="#2ca02c", linestyle=":", alpha=0.5)
    ax.axvline(x=best_2048, color="#d62728", linestyle=":", alpha=0.5)

    ax.annotate(f"Optimal: {best_512}", (best_512, min(losses_512_b32)),
                textcoords="offset points", xytext=(-50, -20), fontsize=9, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax.annotate(f"Optimal: {best_2048}", (best_2048, min(losses_2048_b4)),
                textcoords="offset points", xytext=(15, 15), fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax.set_xlabel("rope_2d_ratio")
    ax.set_ylabel("Loss")
    ax.set_title("Optimal rope_2d_ratio shifts with branch length\n(higher ratio = more branch PE, less time PE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scaling_combined.png")
    fig.savefig(OUT_DIR / "scaling_combined.pdf")
    print("Saved scaling_combined")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5: Frequency pair allocation analysis
# ══════════════════════════════════════════════════════════════════
def fig5_frequency_allocation():
    fig, ax = plt.subplots(figsize=(10, 6))

    # For h16 (head_dim=64, 32 freq pairs):
    # At ratio r, branch gets r*32 pairs, time gets (1-r)*32 pairs
    total_pairs = 32  # h16, head_dim=64

    # Plot: x = branch_freq_pairs, y = time_freq_pairs, color = loss
    # Each point is one experiment config

    configs = []
    # 512 model, eval=32
    for r in RATIOS:
        bp = r * total_pairs
        tp = (1 - r) * total_pairs
        loss = data_512[r][32]
        configs.append((bp, tp, loss, "512 (~300 tok, b=32)"))

    # 2048 packing, eval=4
    for r in RATIOS:
        bp = r * total_pairs
        tp = (1 - r) * total_pairs
        loss = data_2048[r][4]
        configs.append((bp, tp, loss, "2048 (2048 tok, b=4)"))

    # Separate by experiment
    for label, color, marker in [
        ("512 (~300 tok, b=32)", "#2ca02c", "o"),
        ("2048 (2048 tok, b=4)", "#d62728", "s"),
    ]:
        pts = [(bp, tp, loss) for bp, tp, loss, l in configs if l == label]
        bps = [p[0] for p in pts]
        tps = [p[1] for p in pts]
        losses = [p[2] for p in pts]

        sc = ax.scatter(bps, tps, c=losses, cmap="RdYlGn_r", s=200, marker=marker,
                       edgecolors="black", linewidth=1, zorder=5,
                       vmin=2.5, vmax=5.5)

        # Label each point with ratio and loss
        for bp, tp, loss in pts:
            ratio = bp / total_pairs
            ax.annotate(f"r={ratio:.2f}\n{loss:.2f}",
                       (bp, tp), textcoords="offset points",
                       xytext=(12, 0), fontsize=7)

    ax.plot([0, total_pairs], [total_pairs, 0], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Branch Frequency Pairs")
    ax.set_ylabel("Time Frequency Pairs")
    ax.set_title(f"Frequency Pair Allocation (total={total_pairs}, head_dim=64)\nColor = Loss (green=low, red=high)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Loss", shrink=0.8)

    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='512 (~300 tok, b=32)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='2048 (2048 tok, b=4)'),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scaling_frequency_allocation.png")
    fig.savefig(OUT_DIR / "scaling_frequency_allocation.pdf")
    print("Saved scaling_frequency_allocation")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "=" * 70)
    print("SCALING TREND SUMMARY")
    print("=" * 70)

    print("\n1. Optimal ratio vs branch length:")
    print(f"   ~300 tokens/branch  → optimal ratio = 1.0")
    print(f"   2048 tokens/branch  → optimal ratio = 0.75")

    # Log extrapolation
    L0 = 300
    k = 0.25 / np.log(2048 / L0)
    print(f"\n   Extrapolation: ratio = 1.0 - {k:.4f} · ln(L/{L0})")
    for L in [512, 1024, 2048, 4096, 8192]:
        r = max(0, 1.0 - k * np.log(L / L0))
        print(f"   L={L:5d} → predicted ratio ≈ {r:.2f}")

    print("\n2. Branch degradation (512 model, b=1→32):")
    for r in RATIOS:
        delta = data_512[r][32] - data_512[r][1]
        print(f"   ratio={r:.2f}: ΔLoss = {delta:+.2f}")

    print("\n3. Branch degradation (2048 packing, b=1→4):")
    for r in RATIOS:
        delta = data_2048[r][4] - data_2048[r][1]
        print(f"   ratio={r:.2f}: ΔLoss = {delta:+.4f}")

    print("\n4. Key insight:")
    print("   - Short branches: all PE budget to branch separation (ratio→1.0)")
    print("   - Long branches: need some PE for time position (ratio→0.75)")
    print("   - Prediction: 4096+ tokens → ratio≈0.5-0.6 may be optimal")
    print("=" * 70)


if __name__ == "__main__":
    fig1_optimal_ratio_vs_length()
    fig2_degradation_curve()
    fig3_capacity()
    fig4_combined()
    fig5_frequency_allocation()
    print_summary()
    print(f"\nAll figures saved to {OUT_DIR}/")
