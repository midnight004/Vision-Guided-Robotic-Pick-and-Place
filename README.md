# Vision-Guided Robotic Pick-and-Place

**Computer Vision, 3D Object Localization and Robotic Manipulation in Simulation**

An autonomous factory sorting cell: an overhead RGB-D camera perceives products on a work table, the system localizes each item in 3D, and a Franka Emika Panda arm sorts them into color-coded bins using real physics-based grasping.

---

## Pipeline

```
Camera (RGB-D) → Object Detection → 3D Localization (depth → world)
    → Task Planning (color sorting) → IK Motion Planning
    → Real Physics Grasping → Pick & Place into Bins
```

Workflow per cycle: **park & scan → detect all products → pick closest → sort into matching bin → repeat until table is clear**.

---

## Run It

```bash
# Activate environment (Python 3.12)
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux

# Full pipeline with 3D viewer
python scripts/run_pipeline.py --episodes 10 --objects 4

# Headless evaluation
python scripts/run_pipeline.py --headless --episodes 10 --objects 4
```

---

## Key Features

- **Official Franka Emika Panda** from MuJoCo Menagerie (photorealistic meshes, calibrated inertias, gravity-compensated actuators)
- **Real physics grasping** — friction-based holding with the Panda's fingertip pads (no teleportation/attach hacks)
- **Reliable perception** — segmentation-based detection produces pixel-accurate bounding boxes; the arm parks aside during scanning to avoid occluding the camera
- **3D localization** — pixel + depth → camera frame → world frame using the pinhole model and camera extrinsics
- **Damped least-squares IK** with a downward-orientation constraint for top-down grasps
- **Color sorting task** — 8 product types mapped to 4 color bins (red, blue, green, yellow)
- **Factory scene** — work table, conveyor belt, sorting bins, industrial lighting

---

## Verified Results

| Metric | Value |
|--------|-------|
| Objects sorted per episode | 4 / 4 |
| Task success rate | 100% |
| Placement accuracy | < 1 cm from bin center |
| Cycle time | ~2.5 s per 4-object episode |

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
