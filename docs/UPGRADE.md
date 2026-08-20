# Vision Upgrade: Learned Perception, Calibration, and Large-Scale Evaluation

This document describes the upgrade that adds a genuine learned computer-vision
pipeline alongside the original segmentation reference system, plus explicit
camera calibration, tracking analysis, robustness testing, large-scale
evaluation, and an optional visual-servoing mode. The proven pick-and-place
controller and the segmentation backend are preserved unchanged.

See `docs/BASELINE.md` for the frozen baseline and subsystem map.

## What was added

| Area | New / changed |
|------|---------------|
| Deterministic runs | `--seed` flag on `run_pipeline.py` |
| Evaluation harness | `scripts/evaluate.py` — vision + localization + robotics + end-to-end metrics, failure categorization, backend-selectable, machine + human readable output |
| Synthetic data | `scripts/generate_dataset.py`, `config/dataset.yaml` — domain-randomized RGB + YOLO labels from MuJoCo |
| Learned detector | `scripts/train_yolo.py`, `config/training.yaml`; YOLO backend in `src/detection/detector.py` predicts the color taxonomy directly |
| Comparison | `scripts/compare_experiments.py` — side-by-side table + chart |
| Calibration | `scripts/validate_localization.py` — explicit intrinsics/extrinsics + X/Y/Z/3D error |
| Tracking | `scripts/evaluate_tracking.py` — measured ID consistency on the moving conveyor object |
| Robustness | `scripts/evaluate_robustness.py`, `config/dataset.yaml [dataset_ood]` — in-distribution vs harder distribution |
| Visual servoing | optional `relocalize_fn` in `pick_place.execute`, `--visual-servo` flag (off by default) |
| Tests | `tests/test_pipeline_units.py` |

## End-to-end workflow

```powershell
# 1. Baseline (reference segmentation perception), 100 episodes, deterministic
python scripts/evaluate.py --backend segmentation --episodes 100 --seed 42 --name baseline_segmentation

# 2. Generate the synthetic dataset (deterministic given the seed in config/dataset.yaml)
python scripts/generate_dataset.py --preview

# 3. Train the learned YOLO detector on synthetic data (CPU)
python scripts/train_yolo.py --epochs 40
#    -> exports models/factory_yolov8n.pt, writes results/experiments/yolo_train.json,
#       copies training curves/PR/confusion plots to docs/figures/

# 4. Evaluate the learned backend end-to-end (same control stack)
python scripts/evaluate.py --backend yolo --episodes 100 --seed 42 --name yolo_learned

# 5. Compare ground-truth segmentation vs learned YOLO
python scripts/compare_experiments.py --a baseline_segmentation --b yolo_learned

# 6. Calibration / localization accuracy
python scripts/validate_localization.py

# 7. Tracking (moving conveyor object)
python scripts/evaluate_tracking.py

# 8. Robustness: train distribution vs harder distribution
python scripts/generate_dataset.py --config-key dataset_ood
python scripts/evaluate_robustness.py

# 9. Large-scale benchmark (1000+ manipulation attempts)
python scripts/evaluate.py --backend segmentation --episodes 170 --seed 1 --name seg_1000
python scripts/evaluate.py --backend yolo         --episodes 170 --seed 1 --name yolo_1000
```

## Switching perception backends

- At the demo level: `python scripts/run_pipeline.py --backend yolo --episodes 10`
- In config: set `detection.mode` in `config/detection.yaml` to `segmentation`, `yolo`, or `color`.
- The learned weights path is `detection.model.custom_weights` (default `models/factory_yolov8n.pt`).

## How the learned detector fits the pipeline

The YOLO model is trained on the same color taxonomy the pipeline consumes
(`red/blue/green/yellow/unknown`), so its predicted class is used directly as
the sorting category. Everything downstream (localization, IK, grasp, place,
landing check) is identical to the segmentation path, which is what makes the
ground-truth-vs-learned comparison a clean, controlled experiment.

## Camera geometry (explicit)

- Resolution 640x480; intrinsics from vertical FOV: `fx = fy = 432.97`,
  principal point `cx = 320, cy = 240`.
- Extrinsics: overhead camera at world `(0.5, -0.05, 1.2)`, looking down.
- Transform chain: `pixel + depth -> camera frame -> world frame` (via camera
  extrinsics) `-> robot target` (world coordinates are the robot's frame here,
  since the Panda base is at the world origin).

## Results

Populated from actual runs (no fabricated numbers). See:
- `results/experiments/baseline_segmentation.json`
- `results/experiments/yolo_train.json`
- `results/experiments/yolo_learned.json`
- `results/experiments/localization_validation.json`
- `results/experiments/tracking_eval.json`
- `results/experiments/robustness.json`

A consolidated results table is in `docs/RESULTS.md`.

## Simulation-only scope

This is a simulation project. There is no physical robot and no real camera, so
there is no sim-to-real transfer claim. Depth and segmentation are simulator
ground truth; the learned detector consumes rendered RGB with domain
randomization (lighting, camera jitter, appearance, background, sensor noise,
occlusion) to approximate real-camera variability within the simulator.
