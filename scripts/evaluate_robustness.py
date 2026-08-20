"""
Domain-randomization robustness evaluation (Phase 7).

Evaluates the trained YOLO detector on two distributions:
  - in-distribution  : datasets/factory       (same randomization as training)
  - out-of-distribution : datasets/factory_ood (stronger randomization: bigger
    camera/lighting/appearance/background jitter and more noise)

The model is trained ONLY on the in-distribution training split, so the OOD
val split measures generalization beyond the exact scenes it saw.

Prepare the OOD set first (harder distribution):
  python scripts/generate_dataset.py --config-key dataset_ood

Then:
  python scripts/evaluate_robustness.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/experiments"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("docs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def val_on(model, data_yaml, imgsz):
    m = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False, device="cpu")
    b = m.box
    return {
        "precision": float(b.mp), "recall": float(b.mr),
        "mAP50": float(b.map50), "mAP50_95": float(b.map),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/factory_yolov8n.pt")
    ap.add_argument("--id-data", default="datasets/factory/data.yaml")
    ap.add_argument("--ood-data", default="datasets/factory_ood/data.yaml")
    ap.add_argument("--imgsz", type=int, default=416)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"ERROR: weights {args.weights} not found. Train first.")
        sys.exit(1)
    if not Path(args.ood_data).exists():
        print(f"ERROR: OOD dataset {args.ood_data} not found.\n"
              f"Run: python scripts/generate_dataset.py --config-key dataset_ood")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    idm = val_on(model, args.id_data, args.imgsz)
    oodm = val_on(model, args.ood_data, args.imgsz)

    report = {"in_distribution": idm, "out_of_distribution": oodm,
              "weights": args.weights}
    (OUT / "robustness.json").write_text(json.dumps(report, indent=2))

    print("=" * 58)
    print("  ROBUSTNESS: in-distribution vs out-of-distribution")
    print("=" * 58)
    print(f"  {'metric':14s} {'in-dist':>12s} {'OOD':>12s} {'drop':>10s}")
    for k in ("precision", "recall", "mAP50", "mAP50_95"):
        d = idm[k] - oodm[k]
        print(f"  {k:14s} {idm[k]:>12.3f} {oodm[k]:>12.3f} {d:>+10.3f}")
    print("=" * 58)

    labels = ["precision", "recall", "mAP50", "mAP50-95"]
    idv = [idm["precision"], idm["recall"], idm["mAP50"], idm["mAP50_95"]]
    oov = [oodm["precision"], oodm["recall"], oodm["mAP50"], oodm["mAP50_95"]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w / 2, idv, w, label="in-distribution", color="#2E6DA4", edgecolor="white")
    ax.bar(x + w / 2, oov, w, label="out-of-distribution", color="#B0662A", edgecolor="white")
    ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("score"); ax.legend(frameon=False)
    ax.set_title("Detector robustness under stronger domain randomization",
                 fontsize=12, fontweight="bold", color="#1F3A5F")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "fig_robustness.png", dpi=160); plt.close(fig)
    print(f"  Saved: {OUT / 'robustness.json'}  and  {FIG / 'fig_robustness.png'}")


if __name__ == "__main__":
    main()
