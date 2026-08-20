"""
Evaluation harness for the pick-and-place pipeline.

Runs the existing Pipeline (unchanged control logic) with a selectable perception
backend and a fixed seed, records per-item vision / localization / task / failure
data, and writes both machine-readable JSON and a human-readable summary.

Used for:
  - Phase 1: baseline capture and version comparison
  - Phase 4: ground-truth segmentation vs learned YOLO
  - Phase 7: domain-randomization robustness
  - Phase 8: large-scale evaluation with failure categorization

Examples:
  python scripts/evaluate.py --backend segmentation --episodes 100 --seed 42 --name baseline_seg
  python scripts/evaluate.py --backend yolo --episodes 100 --seed 42 --name yolo_seg
"""

import sys
import argparse
import json
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np

from run_pipeline import Pipeline
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")

OUT_DIR = project_root / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class EvalRecorder:
    """Collects per-item records emitted by the instrumented Pipeline."""

    def __init__(self):
        self.records = []

    def record(self, r: dict) -> None:
        self.records.append(r)


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else 0.0


def aggregate(records, task_summary):
    total = len(records)
    detected = [r for r in records if r["detected"]]
    n_det = len(detected)

    # VISION
    det_rate = n_det / max(1, total)
    class_correct = sum(1 for r in detected if r["class_correct"])
    class_acc = class_correct / max(1, n_det)
    inf_ms = _safe_mean([r.get("inference_ms") for r in records])
    fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0

    # LOCALIZATION (only where detected)
    loc_errors = [r["loc_error_m"] for r in detected if r.get("loc_error_m") is not None]
    loc = {
        "mae_3d_cm": float(np.mean(loc_errors) * 100) if loc_errors else 0.0,
        "rmse_3d_cm": float(np.sqrt(np.mean(np.square(loc_errors))) * 100) if loc_errors else 0.0,
        "median_3d_cm": float(np.median(loc_errors) * 100) if loc_errors else 0.0,
        "max_3d_cm": float(np.max(loc_errors) * 100) if loc_errors else 0.0,
        "samples": len(loc_errors),
    }

    # ROBOTICS
    picks = sum(1 for r in detected if r["pick_success"])
    lands = sum(1 for r in detected if r["landed"])
    pick_rate = picks / max(1, n_det)
    place_rate = lands / max(1, picks)

    # END-TO-END: detected AND landed AND routed to the correct bin
    e2e = sum(1 for r in records if r.get("detected") and r.get("landed") and r.get("routed_right"))
    e2e_rate = e2e / max(1, total)

    # FAILURE CATEGORIZATION
    failures = {}
    for r in records:
        f = r.get("failure")
        if f:
            failures[f] = failures.get(f, 0) + 1

    return {
        "counts": {"items": total, "detected": n_det},
        "vision": {
            "detection_rate": det_rate,
            "class_accuracy": class_acc,
            "avg_inference_ms": inf_ms,
            "fps": fps,
        },
        "localization": loc,
        "robotics": {
            "pick_success": pick_rate,
            "place_success": place_rate,
            "avg_cycle_time_s": task_summary["task"]["avg_cycle_time_s"],
            "objects_per_minute": task_summary["task"]["objects_per_minute"],
        },
        "end_to_end": {"success_rate": e2e_rate, "successes": e2e},
        "failures": failures,
    }


def print_summary(name, agg):
    print("\n" + "=" * 66)
    print(f"  EVALUATION SUMMARY: {name}")
    print("=" * 66)
    c = agg["counts"]
    print(f"  Items: {c['items']}   Detected: {c['detected']}")
    v = agg["vision"]
    print("\n  VISION")
    print(f"    Detection rate:   {v['detection_rate']:.1%}")
    print(f"    Class accuracy:   {v['class_accuracy']:.1%}")
    print(f"    Inference:        {v['avg_inference_ms']:.1f} ms ({v['fps']:.1f} FPS)")
    l = agg["localization"]
    print("\n  LOCALIZATION (detected items)")
    print(f"    MAE 3D:  {l['mae_3d_cm']:.2f} cm   RMSE: {l['rmse_3d_cm']:.2f} cm   "
          f"median: {l['median_3d_cm']:.2f} cm   max: {l['max_3d_cm']:.2f} cm   n={l['samples']}")
    r = agg["robotics"]
    print("\n  ROBOTICS")
    print(f"    Pick success:     {r['pick_success']:.1%}")
    print(f"    Place success:    {r['place_success']:.1%}")
    print(f"    Cycle time:       {r['avg_cycle_time_s']:.2f} s   ({r['objects_per_minute']:.1f} obj/min)")
    e = agg["end_to_end"]
    print("\n  END-TO-END")
    print(f"    Success rate:     {e['success_rate']:.1%}  ({e['successes']}/{c['items']})")
    if agg["failures"]:
        print("\n  FAILURES")
        for k, n in sorted(agg["failures"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:18s} {n}")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(description="Pipeline evaluation harness")
    ap.add_argument("--backend", default="segmentation",
                    choices=["segmentation", "yolo", "color"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="Experiment name (output file stem)")
    args = ap.parse_args()

    name = args.name or f"{args.backend}_ep{args.episodes}_seed{args.seed}"
    np.random.seed(args.seed)

    recorder = EvalRecorder()
    pipe = Pipeline(headless=True, num_objects=min(args.objects, 14),
                    detector_mode=args.backend, recorder=recorder)

    t0 = time.time()
    pipe.run(num_episodes=args.episodes)
    elapsed = time.time() - t0
    task_summary = pipe.metrics.get_summary()
    pipe.shutdown()

    agg = aggregate(recorder.records, task_summary)
    agg["meta"] = {
        "name": name, "backend": args.backend, "episodes": args.episodes,
        "objects_per_episode": args.objects, "seed": args.seed,
        "wall_time_s": elapsed, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_json = OUT_DIR / f"{name}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": agg, "records": recorder.records}, f, indent=2, default=str)

    print_summary(name, agg)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
