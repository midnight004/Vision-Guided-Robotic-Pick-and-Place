# Vision-Guided Robotic Pick-and-Place

**Computer Vision, 3D Object Localization and Robotic Manipulation in Simulation**

An autonomous factory sorting cell: a conveyor feeds a stream of random products into the cell, an overhead RGB-D camera perceives each item, the system localizes it in 3D and classifies its color, and a Franka Emika Panda arm sorts it into the matching color bin using contact-triggered physics grasping. Products of unrecognized colors are routed to a reject/trash bin.

---

## Pipeline

```
Conveyor feed → Camera (RGB-D) → Object Detection → 3D Localization (depth → world)
    → Color classification (known bin vs. reject) → IK Motion Planning
    → Contact-triggered weld grasp → Pick & Place into Bins
```

Workflow per item: **feed next product to staging → park & scan → detect + classify → localize in 3D → pick → sort into matching (or trash) bin → repeat for the whole queue**.

---

## Run It

```bash
# Activate environment (Python 3.12)
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux

# Full pipeline with 3D viewer
python scripts/run_pipeline.py --episodes 10 --objects 6

# Headless evaluation
python scripts/run_pipeline.py --headless --episodes 10 --objects 6
```

---

## Key Features

- **Official Franka Emika Panda** from MuJoCo Menagerie (photorealistic meshes, calibrated inertias, gravity-compensated actuators)
- **Contact-triggered grasping** — the gripper closes on the nearest object and a weld equality constraint locks it at its grasped pose (no teleportation/attach hacks); releasing removes the constraint and the item drops into the bin
- **Conveyor feed** — products are spawned at the belt entrance and driven to a staging gate, one at a time, mimicking a real sorting line
- **Randomized product stream** — each episode draws a random mix of known-color and unknown-color products in random order and orientation
- **Reliable perception** — segmentation-based detection with color classification from RGB pixels; the arm parks aside during scanning to avoid occluding the camera
- **3D localization** — pixel + depth → camera frame → world frame using the pinhole model and camera extrinsics
- **Damped least-squares IK** with a downward-orientation constraint for top-down grasps
- **Color sorting task** — 14 product types across 5 bins (red, blue, green, yellow, and a trash bin for unrecognized colors)
- **Honest evaluation** — success is measured by the item's actual final position inside the target bin, not by motion completion
- **Factory scene** — work table, conveyor belt, five sorting bins, industrial lighting

---

## Verified Results

Measured over 6 episodes of 6 randomized products each (physical landing verified per item):

| Metric | Value |
|--------|-------|
| Physical placement success | ~94% (item lands inside target bin) |
| Sort accuracy | 100% (correct bin for every placed item) |
| Cycle time | ~0.8 s per item |
| Failed items | automatically cleared so they don't disrupt the next pick |

---

## Architecture

```
src/
├── simulation/       # MuJoCo environment (Menagerie Franka, physics, RGB-D + segmentation rendering)
├── camera/           # RGB-D camera interface, depth processing
├── detection/        # Segmentation / color / YOLO detection backends
├── tracking/         # Multi-object tracking
├── localization/     # Depth → 3D world coordinate transformation
├── robot_control/    # IK arm controller, gripper, pick-and-place state machine
├── task_logic/       # Color-based sorting rules
├── evaluation/       # Metrics collection
└── utils/            # Logging, visualization

assets/
├── scene.xml         # Factory scene (table, conveyor, bins, products, camera)
└── franka/           # MuJoCo Menagerie Franka Emika Panda model
```

---

## Technologies

| Category | Technology |
|----------|-----------|
| Simulation | MuJoCo 3.11 |
| Robot | Franka Emika Panda (7-DOF) — MuJoCo Menagerie |
| Vision | OpenCV, segmentation rendering, depth back-projection |
| Deep Learning | PyTorch / YOLOv8 (for custom-trained detection) |
| Language | Python 3.12 |
| Math | NumPy, SciPy |

---

## Setup

```bash
python3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Sorting Task

| Product | Color Bin |
|---------|-----------|
| Red box / Red can | Red |
| Blue box / Blue capsule | Blue |
| Green cylinder / Green box | Green |
| Yellow sphere / Yellow bottle | Yellow |
| Unrecognized colors (purple, orange, white, black) | Trash / reject |

---

## Limitations

- Simulation only — no physical robot or real camera
- No sim-to-real transfer validation
- Detection uses simulator segmentation (ground-truth masks); a custom-trained YOLO backend is included for learned detection

## Future Work

- Custom YOLOv8 trained on synthetic data with domain randomization
- Visual servoing for closed-loop final approach
- Grasp quality estimation (antipodal analysis)
- Realistic depth sensor noise modeling
- Moving conveyor belt with dynamic picking

---

## Attribution

The Franka Emika Panda model in `assets/franka/` is from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
by Google DeepMind, used under its BSD-3-Clause license (see `assets/franka/LICENSE`).

## License

MIT (project code)
