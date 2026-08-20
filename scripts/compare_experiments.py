"""
Compare two evaluation experiments side by side (Phases 1 and 4).

Loads two result files written by evaluate.py and prints a side-by-side table
of vision / localization / robotics / end-to-end metrics, then saves a grouped
bar chart. Use it for baseline-vs-improved and ground-truth-vs-YOLO comparisons.

Usage:
  python scripts/compare_experiments.py --a baseline_segmentation --b yolo_learned
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path("results/experiments")
FIG = Path("docs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    p = EXP / (name if name.endswith(".json") else f"{name}.json")
    with open(p) as f:
        return json.load(f)["summary"]


def row(label, a, b, fmt="{:.1%}"):
    return f"  {label:24s} {fmt.format(a):>12s} {fmt.format(b):>12s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first experiment name")
    ap.add_argument("--b", required=True, help="second experiment name")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    na = A.get("meta", {}).get("backend", args.a)
    nb = B.get("meta", {}).get("backend", args.b)

    print("=" * 52)
    print(f"  COMPARISON: {args.a}  vs  {args.b}")
    print("=" * 52)
    print(f"  {'metric':24s} {na:>12s} {nb:>12s}")
    print("-" * 52)
    print(row("detection_rate", A["vision"]["detection_rate"], B["vision"]["detection_rate"]))
    print(row("class_accuracy", A["vision"]["class_accuracy"], B["vision"]["class_accuracy"]))
    print(row("inference_ms", A["vision"]["avg_inference_ms"], B["vision"]["avg_inference_ms"], "{:.1f}"))
    print(row("fps", A["vision"]["fps"], B["vision"]["fps"], "{:.1f}"))
    print(row("loc_mae_cm", A["localization"]["mae_3d_cm"], B["localization"]["mae_3d_cm"], "{:.2f}"))
    print(row("pick_success", A["robotics"]["pick_success"], B["robotics"]["pick_success"]))
    print(row("place_success", A["robotics"]["place_success"], B["robotics"]["place_success"]))
    print(row("end_to_end", A["end_to_end"]["success_rate"], B["end_to_end"]["success_rate"]))
    print("=" * 52)

    # Grouped bar chart of the key rates
    labels = ["detection", "class acc", "pick", "place", "end-to-end"]
    av = [A["vision"]["detection_rate"], A["vision"]["class_accuracy"],
          A["robotics"]["pick_success"], A["robotics"]["place_success"],
          A["end_to_end"]["success_rate"]]
    bv = [B["vision"]["detection_rate"], B["vision"]["class_accuracy"],
          B["robotics"]["pick_success"], B["robotics"]["place_success"],
          B["end_to_end"]["success_rate"]]
    av = [v * 100 for v in av]; bv = [v * 100 for v in bv]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, av, w, label=na, color="#2E6DA4", edgecolor="white")
    ax.bar(x + w / 2, bv, w, label=nb, color="#B0662A", edgecolor="white")
    for i, (a, b) in enumerate(zip(av, bv)):
        ax.text(i - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("Percent (%)"); ax.set_ylim(0, 112)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"Perception backend comparison: {na} vs {nb}",
                 fontsize=13, fontweight="bold", color="#1F3A5F")
    ax.legend(frameon=False, ncol=2, loc="lower center")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG / "fig_perception_comparison.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
