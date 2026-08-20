# Results

All numbers below come from actual runs on this machine (CPU inference; the AMD
GPU has no CUDA). Machine-readable copies live in `results/experiments/`.
No metrics are fabricated.

## Baseline vs learned perception (end-to-end, 100 episodes / 600 items, seed 42)

The control stack (IK, weld grasp, state machine, landing check) is identical
for both backends; only the perception module differs, so the differences below
are attributable to perception alone.

| Metric | Segmentation (reference) | Learned YOLO |
|--------|--------------------------|--------------|
| Detection rate | 100.0% | 97.8% |
| Class accuracy | 98.8% | 92.2% |
| Inference (CPU) | 25.3 ms (39.5 FPS) | 18.6 ms (53.6 FPS) |
| Localization MAE (3D) | 3.00 cm | 3.05 cm |
| Pick success | 98.8% | 98.3% |
| Place success | 98.7% | 98.3% |
| End-to-end (landed in correct bin) | 96.3% (578/600) | 87.0% (522/600) |

Failure breakdown:
- Segmentation: placement 8, motion/grasp 7, classification 7
- YOLO: classification 45, detection 13, motion/grasp 10, placement 10

Interpretation: the learned detector is ~1.35x faster but trades accuracy. Its
main weakness is an orange->yellow confusion (orange spheres/boxes should route
to trash but are sometimes called yellow) plus occasional missed orange boxes.
Because grasping and placement are unchanged (~98% both backends), the ~9-point
end-to-end gap is almost entirely a perception (classification) effect. This is
the headline finding: perception quality, not control, drives the difference.

See `docs/figures/fig_perception_comparison.png`.

## Learned detector training (YOLOv8n, synthetic data, CPU)

Fine-tuned from `yolov8n.pt` on 800 synthetic MuJoCo images (200 val) with
domain randomization. Trained ~18 epochs to convergence (best checkpoint).
Taxonomy: red / blue / green / yellow / unknown.

| Metric | Value |
|--------|-------|
| Precision | 0.939 |
| Recall | 0.904 |
| mAP@50 | 0.936 |
| mAP@50-95 | 0.691 |
| Inference latency (CPU) | 19.9 ms (50.2 FPS) |
| Per-class mAP@50 | red 0.96, blue 0.94, green 0.95, yellow 0.90, unknown 0.92 |

PR curve and confusion matrix: `docs/figures/fig_yolo_pr_curve.png`,
`fig_yolo_confusion.png`. Raw metrics: `results/experiments/yolo_train.json`.

## 3D localization accuracy (Phase 5, n=74 probe placements)

Objects placed at known world positions across the pick zone; estimate compared
to the simulator ground truth.

| Axis | Mean | Median | Max |
|------|------|--------|-----|
| X | 0.47 cm | 0.29 cm | 2.23 cm |
| Y | 0.56 cm | 0.47 cm | 1.35 cm |
| Z | 3.87 cm | 3.50 cm | 6.73 cm |
| 3D (Euclidean) | 3.99 cm | 3.52 cm | 6.91 cm |

The 3D error is dominated by a **systematic +3.87 cm Z bias**: the depth
back-projection estimates the object's top surface, which sits above the body
center by roughly half the object height. Lateral (X/Y) error is sub-centimeter,
which is why top-down grasps succeed at ~99%.

Camera geometry: 640x480, fx = fy = 432.97, cx = 320, cy = 240; overhead camera
at world (0.5, -0.05, 1.2). See `docs/figures/fig_localization.png`.

## Tracking (Phase 6, moving conveyor object)

Tracking is off during the sorting pipeline's perception step because each
product is settled and static then. Measured on the scenario where it matters
(a product moving down the belt):

| Metric | Value |
|--------|-------|
| Trials | 8 |
| Single-ID rate | 100.0% |
| Total ID switches | 0 |
| Mean frames detected / tracked | 40.0 / 39.9 |
| Detection without confirmed track | 1 (min-hits warm-up) |

## Robustness (Phase 7, in-distribution vs out-of-distribution)

The detector is trained only on the in-distribution training split; the OOD val
split applies stronger camera/lighting/appearance/background jitter and more
noise.

| Metric | In-distribution | Out-of-distribution | Drop |
|--------|-----------------|---------------------|------|
| Precision | 0.939 | 0.880 | -0.059 |
| Recall | 0.904 | 0.832 | -0.072 |
| mAP@50 | 0.936 | 0.904 | -0.033 |
| mAP@50-95 | 0.691 | 0.633 | -0.058 |

The small mAP@50 drop (0.033) under much stronger randomization indicates the
detector generalizes well beyond its exact training scenes.
See `docs/figures/fig_robustness.png`.

## Large-scale benchmark (Phase 8, 1000+ attempts)

Segmentation backend, 170 episodes (1020 items), seed 1
(`results/experiments/seg_1000.json`):

| Metric | Value |
|--------|-------|
| Detection rate | 100.0% |
| Class accuracy | 99.0% |
| Pick success | 98.7% |
| Place success | 98.5% |
| End-to-end (correct bin) | 96.3% (982/1020) |
| Localization MAE (3D) | 2.99 cm |
| Failure categories | placement 15, motion/grasp 13, classification 10 |

The reference pipeline holds its ~96% end-to-end performance over 1020
manipulation attempts, confirming the baseline is stable at scale and was not
regressed by the upgrade. The 100-episode results above are the quick
benchmark. The YOLO backend can be run at the same scale with `--backend yolo`.

## Reproduction

See `docs/UPGRADE.md` for exact commands. All randomized components accept a
seed for determinism. Unit tests: `python tests/test_pipeline_units.py`.
