"""Generate the pipeline/architecture diagram and the per-episode results chart."""

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PERCEPTION = "#2E6DA4"
CONTROL = "#4B8B3B"
TEXT = "#FFFFFF"


def diagram():
    stages = [
        ("Conveyor\nFeed", CONTROL),
        ("RGB-D\nCamera", PERCEPTION),
        ("Object\nDetection", PERCEPTION),
        ("Color\nClassification", PERCEPTION),
        ("3D\nLocalization", PERCEPTION),
        ("IK Motion\nPlanning", CONTROL),
        ("Weld\nGrasp", CONTROL),
        ("Place\nin Bin", CONTROL),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    bw, bh = 2.2, 1.15
    gap = 0.75
    positions = {}

    # Top row left-to-right (stages 0..3)
    y_top = 4.2
    for i in range(4):
        x = 0.4 + i * (bw + gap)
        positions[i] = (x, y_top)
    # Bottom row right-to-left (stages 4..7) so the flow snakes
    y_bot = 1.4
    for k, i in enumerate(range(4, 8)):
        x = 0.4 + (3 - k) * (bw + gap)
        positions[i] = (x, y_bot)

    for i, (label, color) in enumerate(stages):
        x, y = positions[i]
        box = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.5, edgecolor="white", facecolor=color, zorder=2)
        ax.add_patch(box)
        ax.text(x + bw / 2, y + bh / 2, label, ha="center", va="center",
                color=TEXT, fontsize=10.5, fontweight="bold", zorder=3)

    def arrow(a, b, rad=0.0):
        xa, ya = positions[a]; xb, yb = positions[b]
        # connect nearest edges
        if abs(ya - yb) < 0.1 and xb > xa:      # left-to-right (top)
            start = (xa + bw, ya + bh / 2); end = (xb, yb + bh / 2)
        elif abs(ya - yb) < 0.1 and xb < xa:    # right-to-left (bottom)
            start = (xa, ya + bh / 2); end = (xb + bw, yb + bh / 2)
        else:                                   # vertical connector
            start = (xa + bw / 2, ya); end = (xb + bw / 2, yb + bh)
        ap = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                             linewidth=2, color="#333333",
                             connectionstyle=f"arc3,rad={rad}", zorder=1)
        ax.add_patch(ap)

    for i in range(3):          # top row
        arrow(i, i + 1)
    arrow(3, 4)                 # top-right down to bottom-right
    for i in range(4, 7):       # bottom row (right to left)
        arrow(i, i + 1)

    # Feedback loop: Place in Bin -> Conveyor Feed (next item)
    xp, yp = positions[7]; xc, yc = positions[0]
    fb = FancyArrowPatch((xp, yp + bh / 2), (xc, yc + bh / 2),
                         arrowstyle="-|>", mutation_scale=18, linewidth=2,
                         color="#B0662A", linestyle=(0, (4, 3)),
                         connectionstyle="arc3,rad=-0.35", zorder=1)
    ax.add_patch(fb)
    ax.text(0.15, (y_top + y_bot) / 2 + bh / 2, "next item", rotation=90,
            ha="center", va="center", color="#B0662A", fontsize=9.5, fontweight="bold")

    # Legend
    ax.add_patch(FancyBboxPatch((7.7, 0.15), 0.35, 0.35, boxstyle="round,pad=0.02",
                                facecolor=PERCEPTION, edgecolor="white"))
    ax.text(8.15, 0.32, "Perception", va="center", fontsize=9.5)
    ax.add_patch(FancyBboxPatch((9.8, 0.15), 0.35, 0.35, boxstyle="round,pad=0.02",
                                facecolor=CONTROL, edgecolor="white"))
    ax.text(10.25, 0.32, "Control", va="center", fontsize=9.5)

    ax.set_title("Pick-and-Place Pipeline Architecture", fontsize=14, fontweight="bold",
                 color="#1F3A5F", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / "fig_pipeline.png")


def results_chart(episodes=6, objects=6):
    from run_pipeline import Pipeline

    pipe = Pipeline(headless=True, num_objects=objects)
    placement, sort_acc = [], []
    for ep in range(episodes):
        r = pipe.run_episode(ep)
        placement.append(r["success_rate"] * 100.0)
        sort_acc.append(r["sort_accuracy"] * 100.0)
    pipe.shutdown()

    x = np.arange(1, episodes + 1)
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - width / 2, placement, width, label="Physical placement success",
                color="#2E6DA4", edgecolor="white")
    b2 = ax.bar(x + width / 2, sort_acc, width, label="Sort accuracy",
                color="#4B8B3B", edgecolor="white")

    avg_p = float(np.mean(placement))
    ax.axhline(avg_p, color="#2E6DA4", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.text(episodes + 0.35, avg_p, f"avg {avg_p:.0f}%", color="#2E6DA4",
            va="center", fontsize=9, fontweight="bold")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Percent (%)")
    ax.set_title(f"Per-Episode Performance ({objects} products per episode)",
                 fontsize=13, fontweight="bold", color="#1F3A5F")
    ax.set_xticks(x)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower center", ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_results.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / "fig_results.png")
    print(f"AVG placement={avg_p:.1f}%  AVG sort={np.mean(sort_acc):.1f}%")


if __name__ == "__main__":
    diagram()
    results_chart()
