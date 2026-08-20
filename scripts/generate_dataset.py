"""
Synthetic dataset generator for the learned YOLO detector (Phase 2).

Renders randomized RGB frames from the existing MuJoCo scene and derives
ground-truth bounding-box labels from the segmentation masks. Products are
scattered across the table; the bins and arm remain in the scene as unlabeled
distractors so the detector learns to ignore them. Deterministic given a seed.

Output (Ultralytics YOLO layout):
  <output_dir>/images/{train,val}/*.png
  <output_dir>/labels/{train,val}/*.txt      (class cx cy w h, normalized)
  <output_dir>/data.yaml

Usage:
  python scripts/generate_dataset.py                 # uses config/dataset.yaml [dataset]
  python scripts/generate_dataset.py --config-key dataset_ood
  python scripts/generate_dataset.py --preview       # also save a labeled preview image
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import cv2
import yaml
import mujoco

from src.simulation.environment import SimulationEnvironment

CLASS_NAMES = ["red", "blue", "green", "yellow", "unknown"]


def color_of(name: str) -> str:
    for c in ("red", "blue", "green", "yellow"):
        if name.startswith(c):
            return c
    return "unknown"


def set_object_pose(env, name, x, y, z, yaw):
    jid = env._object_joint_ids[name]
    addr = env.model.jnt_qposadr[jid]
    env.data.qpos[addr:addr + 3] = [x, y, z]
    env.data.qpos[addr + 3:addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    dof = env.model.jnt_dofadr[jid]
    env.data.qvel[dof:dof + 6] = 0


def jitter_camera(env, rng, cfg, base_pos, base_fovy):
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.camera_name)
    j = cfg["camera_pos_jitter_m"]
    env.model.cam_pos[cam_id] = base_pos + rng.uniform(-j, j, size=3)
    fj = cfg["camera_fovy_jitter_deg"]
    env.model.cam_fovy[cam_id] = base_fovy + rng.uniform(-fj, fj)


def randomize_lighting(env, rng, cfg):
    lo, hi = cfg["light_intensity"]
    pj = cfg["light_pos_jitter_m"]
    for i in range(env.model.nlight):
        env.model.light_diffuse[i] = rng.uniform(lo, hi, size=3)
        env.model.light_pos[i] = env.model.light_pos[i] + rng.uniform(-pj, pj, size=3)


def randomize_background(env, rng, cfg):
    for mat_name, rng_key in (("floor_mat", "floor_gray"), ("table_mat", "table_gray")):
        mid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        if mid >= 0:
            g = rng.uniform(*cfg[rng_key])
            env.model.mat_rgba[mid] = [g, g, g * rng.uniform(0.95, 1.05), 1.0]


def jitter_object_colors(env, rng, names, amount, base_rgba):
    for name in names:
        gid = [g for g, n in env._object_geom_ids.items() if n == name]
        if not gid:
            continue
        gid = gid[0]
        base = base_rgba[gid]
        noise = rng.uniform(-amount, amount, size=3)
        env.model.geom_rgba[gid, :3] = np.clip(base[:3] + noise, 0.02, 1.0)


def add_image_noise(img, rng, std):
    if std <= 0:
        return img
    noise = rng.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def bbox_from_mask(mask, min_pixels):
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def generate(cfg, split_counts, preview=False):
    out = Path(cfg["output_dir"])
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    env = SimulationEnvironment()
    env.initialize(headless=True)
    W, H = env.render_width, env.render_height

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.camera_name)
    base_cam_pos = env.model.cam_pos[cam_id].copy()
    base_fovy = float(env.model.cam_fovy[cam_id])
    base_geom_rgba = env.model.geom_rgba.copy()

    rc = cfg["randomize"]
    all_objects = list(env._object_joint_ids.keys())
    lo_n, hi_n = cfg["objects_per_image"]
    ws = cfg["workspace"]
    rng = np.random.default_rng(cfg["seed"])

    preview_saved = False
    idx_global = 0
    for split, count in split_counts.items():
        for i in range(count):
            env.park_all_objects()
            # restore appearance baseline, then randomize
            env.model.geom_rgba[:] = base_geom_rgba

            n = int(rng.integers(lo_n, hi_n + 1))
            chosen = list(rng.choice(all_objects, size=min(n, len(all_objects)), replace=False))

            placed = []
            for k, name in enumerate(chosen):
                if placed and rng.random() < rc["cluster_prob"]:
                    ax, ay = placed[rng.integers(len(placed))]
                    r = rc["cluster_radius_m"]
                    x = np.clip(ax + rng.uniform(-r, r), *ws["x"])
                    y = np.clip(ay + rng.uniform(-r, r), *ws["y"])
                else:
                    x = rng.uniform(*ws["x"])
                    y = rng.uniform(*ws["y"])
                set_object_pose(env, name, x, y, ws["z"], rng.uniform(0, 2 * np.pi))
                placed.append((x, y))

            jitter_camera(env, rng, rc, base_cam_pos, base_fovy)
            randomize_lighting(env, rng, rc)
            randomize_background(env, rng, rc)
            jitter_object_colors(env, rng, chosen, rc["object_rgba_jitter"], base_geom_rgba)
            mujoco.mj_forward(env.model, env.data)

            rgb = env.render_rgb()          # RGB
            seg = env.render_segmentation()  # geom-id map

            labels = []
            for name in chosen:
                gid = [g for g, n in env._object_geom_ids.items() if n == name]
                if not gid:
                    continue
                box = bbox_from_mask(seg == gid[0], cfg["min_visible_pixels"])
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                cls = CLASS_NAMES.index(color_of(name))
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = add_image_noise(bgr, rng, rc["image_gaussian_std"])

            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(out / f"images/{split}/{stem}.png"), bgr)
            with open(out / f"labels/{split}/{stem}.txt", "w") as f:
                f.write("\n".join(labels))

            if preview and not preview_saved and labels and len(chosen) >= 4:
                prev = bgr.copy()
                for ln in labels:
                    c, cx, cy, bw, bh = ln.split()
                    c = int(c); cx = float(cx) * W; cy = float(cy) * H
                    bw = float(bw) * W; bh = float(bh) * H
                    p1 = (int(cx - bw / 2), int(cy - bh / 2))
                    p2 = (int(cx + bw / 2), int(cy + bh / 2))
                    cv2.rectangle(prev, p1, p2, (0, 255, 0), 2)
                    cv2.putText(prev, CLASS_NAMES[c], (p1[0], p1[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                Path("docs/figures").mkdir(parents=True, exist_ok=True)
                cv2.imwrite("docs/figures/fig_dataset_samples.png", prev)
                preview_saved = True

            idx_global += 1
            if idx_global % 100 == 0:
                print(f"  generated {idx_global} images")

    env.shutdown()

    # data.yaml for Ultralytics
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"Dataset written to {out}  (train={split_counts['train']}, val={split_counts['val']})")
    print(f"data.yaml: {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--config-key", default="dataset")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)[args.config_key]

    split_counts = {"train": cfg["num_train"], "val": cfg["num_val"]}
    generate(cfg, split_counts, preview=args.preview)


if __name__ == "__main__":
    main()
