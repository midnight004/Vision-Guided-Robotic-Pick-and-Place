# Baseline Architecture and Reproduction

This document freezes the working baseline before the learned-vision upgrade,
records how to reproduce it, and separates the system into four subsystems so
future changes can be compared cleanly.

## Subsystem map

The project is deliberately split so each layer is independently testable and
swappable.

| Subsystem | Modules | Role |
|-----------|---------|------|
| **Ground-truth (reference) perception** | `src/detection/detector.py` (`mode: segmentation`) | Uses MuJoCo segmentation masks for pixel-perfect object regions, then classifies each object's color from its RGB pixels (HSV hue voting). This is the reliable reference perception. |
| **Learned perception** | `src/detection/detector.py` (`mode: yolo`), `scripts/generate_dataset.py`, `scripts/train_yolo.py` | A YOLOv8n detector trained on synthetic MuJoCo images. Predicts the same color taxonomy (`red/blue/green/yellow/unknown`) so it is a drop-in perception backend. |
| **3D perception / geometry** | `src/camera/camera_interface.py`, `src/camera/depth_processor.py`, `src/localization/localizer.py` | Camera intrinsics from FOV, extrinsics from the simulator, depth back-projection, camera→world transform, pick-zone filtering. |
| **Robotics / control** | `src/robot_control/{arm_controller,gripper_controller,pick_place}.py` | Damped least-squares IK, contact-triggered weld grasp, and the finite-state pick-and-place controller. **Unchanged by the upgrade.** |
| **Evaluation** | `src/evaluation/metrics.py`, `scripts/evaluate.py`, `scripts/compare_experiments.py` | Task/vision/localization metrics, failure categorization, and backend comparison. |

## Data flow

```
overhead_cam (RGB-D)
  -> ObjectDetector           (segmentation masks OR learned YOLO -> color class)
  -> ObjectLocalizer          (depth back-projection -> camera -> world; pick-zone filter)
  -> target selection         (detection nearest to the staging point)
  -> ArmController (DLS IK)    (approach / descend)
  -> GripperController         (close -> contact-triggered weld equality constraint)
  -> PickPlaceExecutor         (APPROACH->DESCEND->GRASP->LIFT->TRANSPORT->LOWER->RELEASE->RETREAT)
  -> physical landing check    (object's true position must be inside the target bin)
```

The perception backend (segmentation vs YOLO) is the only part that changes
between experiments; everything downstream is identical, which is what makes
the ground-truth-vs-learned comparison meaningful.

## Reproducing the baseline

The pipeline is deterministic given a seed.

```powershell
# Quick demo (viewer)
.\.venv\Scripts\python.exe scripts/run_pipeline.py --episodes 10 --objects 6 --seed 42

# 100-episode baseline with full vision/robotics/end-to-end metrics
.\.venv\Scripts\python.exe scripts/evaluate.py --backend segmentation --episodes 100 --seed 42 --name baseline_segmentation
```

Results are written to `results/experiments/<name>.json` (machine-readable) and
printed as a human-readable summary. A 20-episode seeded run reproduces:

- Pick success ~99%, Place success 100%, Complete ~99%
- End-to-end (physical landing in the correct bin) ~98%
- Localization MAE ~2.9 cm (segmentation backend)

These match the stated baseline (100-episode average ~96.6%). Cycle time is
lower in headless mode because there are no viewer sleeps.

## Comparing versions

```powershell
python scripts/evaluate.py --backend segmentation --episodes 100 --seed 42 --name baseline_segmentation
python scripts/evaluate.py --backend yolo         --episodes 100 --seed 42 --name yolo_learned
python scripts/compare_experiments.py --a baseline_segmentation --b yolo_learned
```

`compare_experiments.py` prints a side-by-side table and saves a grouped bar
chart. Because the control stack is identical, differences reflect perception
quality alone.

## What must not regress

- The contact-triggered weld grasp and the pick-and-place state machine are the
  proven core; they are not modified by the upgrade.
- The segmentation backend remains the default and the reference.
- Task performance must stay at the ~98% baseline; the learned backend is
  additive and selectable via `detection.mode` (config) or `--backend` (eval).
