# Vision-Guided Robotic Pick-and-Place

### Computer Vision, 3D Object Localization and Robotic Manipulation in Simulation

**Author:** Rabah Bouguezel
**Repository:** https://github.com/midnight004/Vision-Guided-Robotic-Pick-and-Place
**Simulator:** MuJoCo 3.11 · **Language:** Python 3.12

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements and Technology Stack](#2-system-requirements-and-technology-stack)
3. [Installation and Setup](#3-installation-and-setup)
4. [Running the System](#4-running-the-system)
5. [Repository Layout](#5-repository-layout)
6. [The Simulation Scene](#6-the-simulation-scene)
7. [Perception: Detection and Color Classification](#7-perception-detection-and-color-classification)
8. [3D Localization](#8-3d-localization)
9. [Robot Control](#9-robot-control)
10. [Grasping Mechanism](#10-grasping-mechanism)
11. [Conveyor Feed](#11-conveyor-feed)
12. [Sorting Logic and Bins](#12-sorting-logic-and-bins)
13. [Evaluation and Metrics](#13-evaluation-and-metrics)
14. [Custom Labels](#14-custom-labels)
15. [Engineering Decisions and Notable Fixes](#15-engineering-decisions-and-notable-fixes)
16. [Verified Results](#16-verified-results)
17. [Limitations and Future Work](#17-limitations-and-future-work)
18. [Configuration Reference](#18-configuration-reference)
19. [Attribution and License](#19-attribution-and-license)

---

## 1. Project Overview

This project is an autonomous factory sorting cell simulated in MuJoCo. A conveyor belt feeds a stream of random products into the cell one at a time. An overhead RGB-D camera perceives each item, the system localizes it in 3D and classifies its color, and a Franka Emika Panda arm picks the item and places it into the matching color bin. Products whose color is not one of the four known categories are routed to a reject (trash) bin.

The cell exercises a complete robotics-and-vision pipeline end to end:

```
Conveyor feed -> RGB-D camera -> Object detection -> 3D localization
   -> Color classification (known bin vs. reject) -> Inverse kinematics
   -> Contact-triggered physics grasp -> Pick and place into bins
```

Each episode draws a randomized mix of known-color and unknown-color products, in random order and orientation, so no two runs are identical.

---

## 2. System Requirements and Technology Stack

### Environment
- Operating system: Windows (developed on win32, PowerShell)
- Python: 3.12 (required for the installed MuJoCo build)
- Virtual environment: `.venv`
- GPU: AMD (the deep-learning detection path is optional; the active perception path is CPU-light)

### Core dependencies

| Purpose | Package |
|---------|---------|
| Physics / simulation | `mujoco` (3.11.0) |
| Robot model | Franka Emika Panda from MuJoCo Menagerie |
| Computer vision | `opencv-python`, `opencv-contrib-python` |
| Numerics | `numpy`, `scipy` |
| Label texture generation | `Pillow` |
| Configuration | `PyYAML` |
| Deep learning (optional path) | `torch`, `torchvision`, `ultralytics` (YOLOv8) |
| Tracking (optional path) | `filterpy` |
| Plotting / reporting | `matplotlib`, `seaborn`, `pandas` |

The full list is pinned in `requirements.txt`.

---

## 3. Installation and Setup

```powershell
# Create and activate the Python 3.12 virtual environment
python3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The Franka Panda model (meshes, inertias, actuators) lives under `assets/franka/` and is loaded directly by the scene, so no extra download step is required.

---

## 4. Running the System

```powershell
# Full pipeline with the interactive 3D viewer
.\.venv\Scripts\python.exe scripts/run_pipeline.py --episodes 10 --objects 6

# Headless evaluation (no viewer, metrics only)
.\.venv\Scripts\python.exe scripts/run_pipeline.py --headless --episodes 6 --objects 6
```

Command-line options:

| Flag | Meaning |
|------|---------|
| `--episodes N` | Number of sorting episodes to run |
| `--objects N` | Number of products fed per episode |
| `--headless` | Run without the 3D viewer (faster, for evaluation) |

Regenerating the label textures (for example to change the nameplate text):

```powershell
.\.venv\Scripts\python.exe scripts/make_labels.py
```

---

## 5. Repository Layout

```
VISION ROBOT/
|-- assets/
|   |-- scene.xml            Factory scene: table, conveyor, 5 bins, 14 products, cameras, welds
|   |-- franka/              Franka Emika Panda model (MuJoCo Menagerie)
|   +-- textures/            Generated label images (label_trash.png, label_rabah.png)
|-- config/                  Per-module YAML configuration
|-- scripts/
|   |-- run_pipeline.py      Main entry point (episode loop, evaluation)
|   +-- make_labels.py       Generates the TRASH / RABAH label textures
|-- src/
|   |-- simulation/          MuJoCo environment, scene rules
|   |-- camera/              RGB-D camera interface, depth processing
|   |-- detection/           Detection backends + color classifier + visualizer
|   |-- localization/        Depth -> 3D world coordinate transformation
|   |-- tracking/            Multi-object tracker
|   |-- robot_control/       Arm controller (IK), gripper, pick-and-place state machine
|   |-- task_logic/          Color -> bin sorting rules
|   |-- evaluation/          Metrics collection
|   +-- utils/               Logging, visualization
|-- requirements.txt
|-- pyproject.toml
+-- README.md
```

### Key source modules

| Module | Responsibility |
|--------|----------------|
| `src/simulation/environment.py` | Loads the model, steps physics, renders RGB-D and segmentation, runs the conveyor feed, activates/releases grasp welds, parks objects |
| `src/simulation/scene_builder.py` | Maps a detected color category to a destination bin position |
| `src/camera/camera_interface.py` | Captures RGB and depth frames, exposes the intrinsic matrix |
| `src/detection/detector.py` | Segmentation / color / YOLO detection backends and the HSV color classifier |
| `src/localization/localizer.py` | Back-projects masked pixels + depth into world coordinates; filters to the pick zone |
| `src/robot_control/arm_controller.py` | Damped least-squares IK, Cartesian moves, home and scan poses |
| `src/robot_control/gripper_controller.py` | Opens/closes the gripper and drives the contact-triggered weld grasp |
| `src/robot_control/pick_place.py` | The pick-and-place state machine and motion heights |
| `scripts/run_pipeline.py` | Builds the episode queue, feeds items, orchestrates perception + motion, verifies landings |

---

## 6. The Simulation Scene

The scene is defined in `assets/scene.xml`.

### Global physics options
- Timestep: 0.002 s
- Integrator: `implicitfast`
- Contact model: `cone="pyramidal"`, `impratio="1"` (chosen for stability with small round objects)
- Gravity: -9.81 m/s^2

### Fixed elements
- Work table centered at (0.5, 0, 0.28)
- Conveyor belt with side rails feeding along +Y toward the cell
- Overhead perception camera at roughly (0.5, -0.05, 1.2), 58 degrees vertical FOV, rendering 640x480
- A side camera for viewing
- Diffuse-only industrial lighting (specular disabled) to keep object colors accurate for the color classifier

### Sorting bins (5)
Corner bins are 0.08 half-size; the trash bin is 0.07 half-size; all walls are 0.05 tall.

| Bin | World position (x, y, z) |
|-----|--------------------------|
| Red | (0.30, -0.30, 0.30) |
| Blue | (0.30, 0.30, 0.30) |
| Green | (0.62, -0.30, 0.30) |
| Yellow | (0.62, 0.30, 0.30) |
| Trash | (0.46, 0.31, 0.30) |

### Products (14)
Parked off-screen when not in play and fed one at a time.

**Known colors (8) -> color bins:** red_box, red_can, blue_box, blue_capsule, green_cylinder, green_box, yellow_sphere, yellow_bottle

**Unknown colors (6) -> trash bin:** purple_box, orange_sphere, white_cylinder, black_box, purple_cylinder, orange_box

Each product is a free body with per-object friction, mass ~0.04-0.05 kg. The two spheres use `condim="6"` with rolling friction so they do not roll away (see Section 15).

### Grasp welds
An `<equality>` block defines 14 weld constraints, one per product, connecting the robot hand to each object. All are inactive by default and activated at runtime only when the gripper closes on an object.

---

## 7. Perception: Detection and Color Classification

Detection is implemented in `src/detection/detector.py` and supports three backends selected by config (`detection.mode`):

- `segmentation` (default when the simulator is available) - uses MuJoCo's segmentation rendering for pixel-accurate object masks, then classifies color from the RGB pixels.
- `color` - pure HSV thresholding on the RGB image (fallback).
- `yolo` - custom-trained YOLOv8 weights (wired in but not the active path).

### Why segmentation
Raw color thresholding is unreliable under the scene lighting (flat surfaces wash out toward white). Segmentation gives a clean mask per object; color is then judged from the pixels inside that mask, which mirrors how a real color-sorting vision system separates "where is the object" from "what color is it."

### Color classification (HSV hue voting)
For each object mask, pixels are converted to HSV. Only saturated, well-lit pixels vote (saturation > 40, value between 25 and 253). If fewer than 8% of the pixels are colored, the object is labeled `unknown` (this captures white, black, and grey items). Otherwise each colored pixel is binned by hue:

| Hue range (OpenCV 0-179) | Category |
|--------------------------|----------|
| <= 10 or >= 160 | red |
| 11 - 23 | unknown (orange) |
| 24 - 35 | yellow |
| 36 - 85 | green |
| 86 - 132 | blue |
| >= 133 | unknown (purple / magenta) |

The winning category must hold at least 45% of the colored pixels; otherwise the object is `unknown`. The result is one of `red`, `blue`, `green`, `yellow`, or `unknown`, and that category (not the object's true identity) decides the destination bin. This is what lets the arm correctly send a novel purple or orange item to the trash bin.

During perception the arm first moves to a dedicated scan pose so it does not occlude the overhead camera.

---

## 8. 3D Localization

Localization is handled in `src/localization/localizer.py`.

For each detection, the masked region's depth is read from the depth buffer and the pixel is back-projected through the standard pinhole model:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

with the principal point at the image center and focal length derived from the camera's field of view (640x480). The resulting camera-frame point is transformed to world coordinates using the known camera extrinsics (exact in simulation).

A pick-zone filter rejects points outside the working area (approximately x in [0.30, 0.70], y in [-0.26, 0.12], z in [0.28, 0.50]), which discards bin positions and stray geometry so only the staged product is targeted.

---

## 9. Robot Control

### Arm controller (`arm_controller.py`)
- **Inverse kinematics:** damped least-squares, targeting a Cartesian position with a downward-orientation constraint for clean top-down grasps.
- **Move acceptance:** a Cartesian move is considered successful when the final end-effector position is within 0.08 m of the target.
- **Home pose (qpos):** `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]`
- **Scan pose (qpos):** `[0, -1.5, 0, -2.5, 0, 1.5, 0.785]` - tucks the arm back so it does not block the camera.
- **Workspace bounds:** x in [-0.1, 0.85], y in [-0.6, 0.6], z in [0.0, 1.0].

### Pick-and-place state machine (`pick_place.py`)
```
APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> LOWER -> RELEASE -> RETREAT
```

Motion heights (relative to object Z or absolute world Z):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Approach height | 0.15 | Above the object before descending |
| Grasp Z offset | 0.058 | Hand-to-fingertip offset (measured from the model) |
| Lift height | 0.56 | Absolute Z after lifting |
| Transport height | 0.56 | Absolute Z during transport; the object hangs ~9 cm below and clears the 0.40 m bin walls |
| Place height offset | 0.10 | Release above the bin floor so the item drops in cleanly |
| Retreat height | 0.56 | Absolute Z after release |

The grasp step includes a retry: if the gripper closes on nothing, it re-opens, re-descends, and tries once more. If a motion fails, the arm returns to the scan pose so the next item starts from a known-good configuration (this prevents cascading failures).

---

## 10. Grasping Mechanism

Grasping went through several iterations before arriving at the current approach. The final mechanism is **contact-triggered weld grasping**, implemented across `gripper_controller.py` and `environment.py`.

**Sequence:**
1. The gripper closes physically on the object.
2. The controller identifies the object between the fingers by matching on horizontal (XY) distance within a vertical window beneath the hand. This is important: the object's center sits roughly 8-9 cm below the hand's end-effector reference, so a naive 3D distance check never triggers.
3. A MuJoCo weld equality constraint (hand <-> that object) is activated at the object's current pose relative to the hand. The relative transform is computed as `hand^-1 * object` and written into the constraint's `eq_data` (anchor, relative position, relative quaternion, torque scale).
4. The weld holds the object rigidly during lift and transport, like a firm grip.
5. Opening the gripper deactivates the weld and the object drops into the bin.

This is genuine physics (a rigid constraint solved by the engine), not a teleport or attach hack. It is the standard technique used in MuJoCo manipulation research and eliminates the slipping and lag that pure friction grasping produced for small objects during motion.

---

## 11. Conveyor Feed

Implemented in `environment.py` as `feed_object`.

1. The target product spawns at the belt entrance, roughly (0.5, -0.55, 0.35), with a small random X offset and a random yaw for orientation variety.
2. It is driven forward along +Y at 0.45 m/s until it reaches the staging gate near (0.5, -0.12, 0.33).
3. On arrival its velocity is zeroed and it is damped to a complete stop, then allowed to settle, so the perceived position is stable for both detection and grasping.
4. A fallback places the item directly at the staging point if the belt loop runs out, so an item is never left stranded mid-belt.

Feeding one item at a time mimics a real sorting line and avoids bin-collision confusion during perception.

---

## 12. Sorting Logic and Bins

Sorting rules live in `src/simulation/scene_builder.py`.

**Color category -> bin:**

| Detected color | Bin |
|----------------|-----|
| red | Red |
| blue | Blue |
| green | Green |
| yellow | Yellow |
| unknown | Trash |

**Bin release points (`DESTINATIONS`, z = 0.35):**

| Bin | (x, y) |
|-----|--------|
| red | (0.30, -0.30) |
| blue | (0.30, 0.30) |
| green | (0.62, -0.30) |
| yellow | (0.62, 0.30) |
| trash | (0.46, 0.31) |

---

## 13. Evaluation and Metrics

Evaluation is intentionally honest. Success is not "the arm finished its motion" - it is whether the object physically ended up inside the target bin.

After each place, `run_pipeline.py` reads the object's actual world position and checks that its XY is within tolerance of the bin center (about 0.08 m, matching the inner bin half-width) and its Z is below the bin rim. Two outcomes are tracked separately:

- **Physical placement success** - the item is verified inside the correct bin.
- **Sort accuracy** - among placed items, the fraction routed to the correct bin.

Failed items are automatically parked off-table so a dropped object cannot corrupt the next detection or localization cycle.

---

## 14. Custom Labels

MuJoCo cannot render arbitrary text as geometry, so labels are created as image textures.

- `scripts/make_labels.py` uses Pillow to generate two PNGs into `assets/textures/`:
  - `label_trash.png` - "TRASH" in bold black on a cream sign with a red border.
  - `label_rabah.png` - "RABAH" in white on a dark plate with a blue border.
- Each image is mapped onto a thin, non-colliding `plane` geom:
  - The TRASH sign sits on the front wall of the trash bin, facing the workspace.
  - The RABAH nameplate is mounted on the robot base, standing at the front where it stays visible during operation.

A `plane` geom is used rather than a `box` because box UV mapping unwraps a single texture across all six faces, showing only a fraction per face; a plane maps one full image cleanly with `texrepeat="1 1"` and `texuniform="false"`. The labels are visual only (`contype="0" conaffinity="0"`) and do not affect physics.

---

## 15. Engineering Decisions and Notable Fixes

This section documents the reasoning behind the design and the key problems solved during development.

### Grasping: weld over friction over teleport
Pure friction grasping of small objects was too fragile - items slipped and lagged during motion. The earlier teleport/attach approach was rejected as unrealistic. The contact-triggered weld is both robust and legitimate: it only engages when the fingers actually close on an object, and it is real constraint-based physics.

### Perception: segmentation over thresholding and over pretrained YOLO
Color thresholding alone was unreliable under the lighting, and a pretrained YOLO does not recognize custom colored shapes. Segmentation provides reliable masks; color is then classified from the RGB pixels, which is both reliable and faithful to how color-sorting systems work.

### Spheres rolled away
The spheres originally used `condim="4"`, which enables slide and torsion friction but ignores rolling friction. They rolled out of the gripper and across the table, and a loose ball corrupted downstream detection. Switching the spheres to `condim="6"` with rolling friction (`1.5 0.05 0.02`) fixed this.

### Objects missing bins (the decisive fix)
The welded object hangs roughly 9 cm below the hand. At the original transport height (z = 0.50) that put the object at about z = 0.41 - level with the 0.40 m bin walls - so items clipped the walls of bins they flew over during transit and were knocked loose. Raising lift and transport to z = 0.56 gave the dangling object clearance over the walls, which raised physical placement from about 46% to about 94%.

### Cascading failures
A failed grasp used to leave the object sitting at the staging point; the next feed dropped another item on top, causing collisions and misdetections that snowballed across an episode. Two changes fixed this: failed items are parked off-table, and the arm returns to the scan pose after any failure.

### Stable feed
Objects were still moving when perceived, so the grasp targeted a stale position. The feed now brings each item to a full stop and lets it settle before perception.

### Contact model
Elliptic friction with a high impedance ratio caused round-object instabilities (spheres flew off). Switching to a pyramidal cone with `impratio="1"` stabilized the contacts.

---

## 16. Verified Results

Measured over 6 episodes of 6 randomized products each, with physical landing verified per item:

| Metric | Value |
|--------|-------|
| Physical placement success | ~94% (item lands inside the target bin) |
| Sort accuracy | 100% (correct bin for every placed item) |
| Cycle time | ~0.8 s per item |
| Failure handling | failed items auto-cleared so they do not disrupt the next pick |

The remaining ~6% are occasional grasp misses, which is realistic for a physical system.

---

## 17. Limitations and Future Work

### Limitations
- Simulation only - no physical robot or real camera, and no sim-to-real transfer validation.
- Detection uses the simulator's ground-truth segmentation masks; color is then classified from real RGB pixels. A learned detector is wired in but is not the active path.

### Future work
- Train YOLOv8 on synthetic data rendered from the simulator with domain randomization (textures, lighting, object size, camera noise).
- Add visual servoing on the final few centimeters of approach to close the remaining grasp-miss gap.
- Grasp quality estimation (antipodal analysis) to reject and re-plan poor grasps.
- A realistic depth-sensor noise model to test localization robustness.
- Dynamic picking off a moving belt (currently items stop at a staging gate before picking).

---

## 18. Configuration Reference

Configuration is split into per-module YAML files under `config/`. These are loaded and merged at startup.

| File | Controls |
|------|----------|
| `camera.yaml` | Camera pose, resolution (640x480), intrinsics, depth range |
| `detection.yaml` | Detection mode, HSV ranges, class mapping, visualization |
| `robot_control.yaml` | IK solver, pick-and-place phase parameters, workspace bounds, home pose |
| `simulation.yaml` | Simulation-level settings |
| `task.yaml` | Task-level settings |
| `tracking.yaml` | Tracker parameters |
| `evaluation.yaml` | Metrics settings |

Note that some values in the YAML files are authoritative for perception (HSV ranges, camera intrinsics), while the motion heights that proved decisive for reliable placement are defined as constants in `pick_place.py` and `arm_controller.py`.

---

## 19. Attribution and License

The Franka Emika Panda model in `assets/franka/` is from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) by Google DeepMind,
used under its BSD-3-Clause license (see `assets/franka/LICENSE`).

Project code is released under the MIT license.

---

*Documentation authored by Rabah Bouguezel.*
