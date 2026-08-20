"""
Finalize a trained YOLO run: validate the best checkpoint, measure CPU inference
latency, export weights to models/, copy training plots, and write real metrics.

Used when training is stopped at a converged checkpoint (best.pt is saved every
validation epoch by Ultralytics). All metrics are measured, not fabricated.

Usage:
  python scripts/finalize_yolo.py
"""

import sys
import json
import time
import shutil
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
OUT = project_root / "results" / "experiments"; OUT.mkdir(parents=True, exist_ok=True)
FIG = project_root / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def find_best():
    cands = list(project_root.glob("runs/**/weights/best.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main():
    with open("config/training.yaml") as f:
        cfg = yaml.safe_load(f)["training"]
    data_path = cfg["data"]
    imgsz = cfg["imgsz"]

    best = find_best()
    if best is None:
        print("ERROR: no best.pt found under runs/. Train first.")
        sys.exit(1)
    print(f"Using checkpoint: {best}")

    from ultralytics import YOLO
    model = YOLO(str(best))

    metrics = model.val(data=data_path, imgsz=imgsz, device="cpu", verbose=False)
    box = metrics.box
    val_metrics = {
        "precision": float(box.mp), "recall": float(box.mr),
        "mAP50": float(box.map50), "mAP50_95": float(box.map),
        "per_class_mAP50": {},
    }
    try:
        for i, ap50 in enumerate(box.ap50):
            val_metrics["per_class_mAP50"][model.names[i]] = float(ap50)
    except Exception:
        pass

    export_to = project_root / cfg["export_to"]
    export_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, export_to)
    print(f"Exported -> {export_to}")

    # CPU latency over up to 50 val images
    val_dir = Path(data_path).parent / "images" / "val"
    val_imgs = sorted(val_dir.glob("*.png"))[:50]
    lat = []
    for p in val_imgs:
        t = time.time()
        model.predict(str(p), imgsz=imgsz, device="cpu", verbose=False)
        lat.append((time.time() - t) * 1000)
    avg_ms = float(np.mean(lat)) if lat else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    save_dir = best.parent.parent
    for src_name, dst_name in [("results.png", "fig_yolo_training.png"),
                               ("PR_curve.png", "fig_yolo_pr_curve.png"),
                               ("confusion_matrix.png", "fig_yolo_confusion.png"),
                               ("BoxPR_curve.png", "fig_yolo_pr_curve.png")]:
        src = save_dir / src_name
        if src.exists():
            shutil.copy(src, FIG / dst_name)

    result = {
        "config": {"base_model": cfg["base_model"], "imgsz": imgsz,
                   "device": "cpu", "data": data_path,
                   "note": "validated from converged best.pt checkpoint"},
        "val_metrics": val_metrics,
        "inference": {"avg_latency_ms": avg_ms, "fps": fps, "n_images": len(lat)},
        "weights": str(export_to),
        "runs_dir": str(save_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "yolo_train.json").write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 56)
    print("  YOLO (converged checkpoint) VALIDATION")
    print("=" * 56)
    print(f"  Precision:   {val_metrics['precision']:.3f}")
    print(f"  Recall:      {val_metrics['recall']:.3f}")
    print(f"  mAP@50:      {val_metrics['mAP50']:.3f}")
    print(f"  mAP@50-95:   {val_metrics['mAP50_95']:.3f}")
    print(f"  Inference:   {avg_ms:.1f} ms ({fps:.1f} FPS) CPU")
    print("  Per-class mAP50: " +
          ", ".join(f"{k}={v:.2f}" for k, v in val_metrics["per_class_mAP50"].items()))
    print("=" * 56)
    print(f"  Saved: {OUT / 'yolo_train.json'}")


if __name__ == "__main__":
    main()
