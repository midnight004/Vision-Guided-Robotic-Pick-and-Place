"""
Train / validate / export the learned YOLO detector (Phase 3).

Fine-tunes a lightweight YOLOv8n on the synthetic MuJoCo dataset (CPU by
default), validates it, measures inference latency, exports the best weights to
models/, and writes real metrics (precision, recall, mAP50, mAP50-95, latency,
FPS) to results/experiments/yolo_train.json. No metrics are fabricated.

Usage:
  python scripts/generate_dataset.py            # first, build the dataset
  python scripts/train_yolo.py                  # then train
  python scripts/train_yolo.py --epochs 30      # override epochs
"""

import sys
import argparse
import json
import time
import shutil
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
OUT_DIR = project_root / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/training.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["training"]

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    batch = args.batch if args.batch is not None else cfg["batch"]

    data_path = Path(cfg["data"])
    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}. Run generate_dataset.py first.")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"Training {cfg['base_model']} on {data_path}  "
          f"(epochs={epochs}, imgsz={imgsz}, batch={batch}, device={cfg['device']})")

    model = YOLO(cfg["base_model"])
    t0 = time.time()
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=cfg["device"],
        workers=cfg["workers"],
        patience=cfg["patience"],
        seed=cfg["seed"],
        project=cfg["project"],
        name=cfg["name"],
        deterministic=True,
        verbose=True,
        plots=True,
    )
    train_time = time.time() - t0

    # Validation metrics on the val split
    metrics = model.val(data=str(data_path), imgsz=imgsz, device=cfg["device"], verbose=False)
    box = metrics.box
    val_metrics = {
        "precision": float(np.mean(box.p)) if hasattr(box, "p") and len(box.p) else float(box.mp),
        "recall": float(np.mean(box.r)) if hasattr(box, "r") and len(box.r) else float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class_mAP50": {},
    }
    try:
        names = model.names
        for i, ap50 in enumerate(box.ap50):
            val_metrics["per_class_mAP50"][names[i]] = float(ap50)
    except Exception:
        pass

    # Export best weights
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    export_to = project_root / cfg["export_to"]
    export_to.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, export_to)
        print(f"Exported best weights -> {export_to}")

    # Measure inference latency on val images (CPU)
    val_imgs = sorted((data_path.parent / "images" / "val").glob("*.png"))[:50]
    latencies = []
    infer_model = YOLO(str(export_to)) if export_to.exists() else model
    for p in val_imgs:
        t = time.time()
        infer_model.predict(str(p), imgsz=imgsz, device=cfg["device"], verbose=False)
        latencies.append((time.time() - t) * 1000)
    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    result = {
        "config": {"base_model": cfg["base_model"], "epochs": epochs, "imgsz": imgsz,
                   "batch": batch, "device": cfg["device"], "data": str(data_path)},
        "train_time_s": train_time,
        "val_metrics": val_metrics,
        "inference": {"avg_latency_ms": avg_ms, "fps": fps, "n_images": len(latencies)},
        "weights": str(export_to),
        "runs_dir": str(save_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = OUT_DIR / "yolo_train.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    # Copy key plots for the report
    figdir = project_root / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [("results.png", "fig_yolo_training.png"),
                               ("PR_curve.png", "fig_yolo_pr_curve.png"),
                               ("confusion_matrix.png", "fig_yolo_confusion.png")]:
        src = save_dir / src_name
        if src.exists():
            shutil.copy(src, figdir / dst_name)

    print("\n" + "=" * 60)
    print("  YOLO TRAINING RESULTS")
    print("=" * 60)
    print(f"  Precision:   {val_metrics['precision']:.3f}")
    print(f"  Recall:      {val_metrics['recall']:.3f}")
    print(f"  mAP@50:      {val_metrics['mAP50']:.3f}")
    print(f"  mAP@50-95:   {val_metrics['mAP50_95']:.3f}")
    print(f"  Inference:   {avg_ms:.1f} ms ({fps:.1f} FPS) on CPU")
    print(f"  Train time:  {train_time/60:.1f} min")
    print(f"  Saved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
