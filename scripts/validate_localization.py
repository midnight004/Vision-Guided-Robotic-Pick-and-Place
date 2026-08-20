"""
Camera calibration and 3D localization validation (Phase 5).

Makes the camera geometry explicit (intrinsics + extrinsics) and measures
localization accuracy by placing objects at known world positions across the
pick zone, running the full detect -> depth -> back-project -> world pipeline,
and comparing the estimate against the simulator's ground-truth position.

Reports X / Y / Z / Euclidean errors (mean, median, max) and the systematic
Z bias (the depth back-projection estimates the top surface, so Z sits above
the body center by ~half the object height).

Usage:
  python scripts/validate_localization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from run_pipeline import load_config
from src.simulation.environment import SimulationEnvironment
from src.camera.camera_interface import CameraInterface
from src.camera.depth_processor import DepthProcessor
from src.detection.detector import ObjectDetector
from src.localization.localizer import ObjectLocalizer

OUT = Path("results/experiments"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("docs/figures"); FIG.mkdir(parents=True, exist_ok=True)

PROBE_OBJECTS = ["red_box", "green_cylinder", "yellow_sphere", "blue_capsule"]


def set_pose(env, name, x, y, z):
    jid = env._object_joint_ids[name]
    addr = env.model.jnt_qposadr[jid]
    env.data.qpos[addr:addr + 3] = [x, y, z]
    env.data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]
    dof = env.model.jnt_dofadr[jid]
    env.data.qvel[dof:dof + 6] = 0


def main():
    cfg = load_config()
    cfg["detection"]["mode"] = "segmentation"

    env = SimulationEnvironment()
    env.initialize(headless=True)
    camera = CameraInterface(env)
    depth_proc = DepthProcessor(camera.intrinsic_matrix, depth_range=(0.05, 3.0))
    detector = ObjectDetector(cfg, sim_env=env)
    localizer = ObjectLocalizer(cfg, depth_proc, sim_env=env)

    K = camera.intrinsic_matrix
    cam_pos, cam_rot = env.get_camera_extrinsics()

    print("=" * 60)
    print("  CAMERA GEOMETRY")
    print("=" * 60)
    print(f"  Resolution: {env.render_width} x {env.render_height}")
    print(f"  Intrinsics K:\n{np.round(K, 2)}")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  Camera position (world): {np.round(cam_pos, 4)}")
    print(f"  Camera rotation (world, rows=cam axes):\n{np.round(cam_rot, 3)}")

    xs = np.linspace(0.37, 0.63, 6)
    ys = np.linspace(-0.22, 0.10, 5)

    errs = {"x": [], "y": [], "z": [], "e3d": [], "z_signed": []}
    samples = []

    for name in PROBE_OBJECTS:
        for x in xs:
            for y in ys:
                env.park_all_objects()
                set_pose(env, name, x, y, 0.33)
                mujoco.mj_forward(env.model, env.data)
                # let it rest on the table
                for _ in range(40):
                    mujoco.mj_step(env.model, env.data)
                mujoco.mj_forward(env.model, env.data)

                true_pos = env.get_object_position(name)
                frame = camera.capture_frame(sim_time=env.data.time)
                det = detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
                if not det.detections or not frame.has_depth:
                    continue
                localized = localizer.localize_all(det.detections, frame.depth)
                if not localized:
                    continue
                # nearest estimate to the true position
                best = min(localized, key=lambda o: np.linalg.norm(o.position_world[:2] - true_pos[:2]))
                est = best.position_world
                e = est - true_pos
                errs["x"].append(abs(e[0])); errs["y"].append(abs(e[1]))
                errs["z"].append(abs(e[2])); errs["z_signed"].append(e[2])
                errs["e3d"].append(float(np.linalg.norm(e)))
                samples.append({"object": name, "true": true_pos.tolist(),
                                "est": est.tolist(), "e3d_cm": float(np.linalg.norm(e) * 100)})

    env.shutdown()

    def stats(a):
        a = np.array(a)
        return {"mean_cm": float(a.mean() * 100), "median_cm": float(np.median(a) * 100),
                "max_cm": float(a.max() * 100)}

    report = {
        "camera": {"width": env.render_width, "height": env.render_height,
                   "K": K.tolist(), "position": cam_pos.tolist(), "rotation": cam_rot.tolist()},
        "n_samples": len(errs["e3d"]),
        "error_x": stats(errs["x"]), "error_y": stats(errs["y"]), "error_z": stats(errs["z"]),
        "error_3d": stats(errs["e3d"]),
        "z_bias_cm": float(np.mean(errs["z_signed"]) * 100),
    }
    import json
    (OUT / "localization_validation.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("  LOCALIZATION ACCURACY  (n=%d)" % report["n_samples"])
    print("=" * 60)
    for ax in ("error_x", "error_y", "error_z", "error_3d"):
        s = report[ax]
        print(f"  {ax:9s} mean={s['mean_cm']:.2f} cm  median={s['median_cm']:.2f} cm  max={s['max_cm']:.2f} cm")
    print(f"  Z bias (signed): {report['z_bias_cm']:+.2f} cm  "
          f"(depth estimates top surface, above body center)")

    # Error scatter over the workspace
    fig, ax = plt.subplots(figsize=(7, 5))
    tx = [s["true"][0] for s in samples]; ty = [s["true"][1] for s in samples]
    ec = [s["e3d_cm"] for s in samples]
    sc = ax.scatter(tx, ty, c=ec, cmap="viridis", s=60, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, label="3D error (cm)")
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)")
    ax.set_title("Localization 3D error across the pick zone")
    fig.tight_layout(); fig.savefig(FIG / "fig_localization.png", dpi=160); plt.close(fig)
    print(f"\n  Saved: {OUT / 'localization_validation.json'}")
    print(f"  Saved: {FIG / 'fig_localization.png'}")


if __name__ == "__main__":
    main()
