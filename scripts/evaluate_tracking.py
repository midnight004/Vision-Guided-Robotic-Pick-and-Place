"""
Tracking evaluation (Phase 6).

Tracking is intentionally OFF in the sorting pipeline: each product is fed and
allowed to settle to rest before perception, so at the perception instant there
is a single static object and nothing to track.

This script evaluates the tracker where it IS meaningful: while a product moves
along the conveyor. It drives an object down the belt, captures frames, runs
detection + the ByteTrack-style tracker each frame, and measures whether the
object keeps a single consistent ID.

Reports (measured, not assumed): tracked trials, single-ID rate, ID switches,
mean frames tracked, and detection-without-confirmed-track fragmentation.

Usage:
  python scripts/evaluate_tracking.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
import mujoco

from run_pipeline import load_config
from src.simulation.environment import SimulationEnvironment
from src.camera.camera_interface import CameraInterface
from src.detection.detector import ObjectDetector
from src.tracking.tracker import ObjectTracker

OUT = Path("results/experiments"); OUT.mkdir(parents=True, exist_ok=True)

TRIAL_OBJECTS = ["red_box", "blue_box", "green_cylinder", "yellow_sphere",
                 "red_can", "blue_capsule", "green_box", "yellow_bottle"]

SPAWN = np.array([0.5, -0.55, 0.35])
STAGING_Y = -0.12
BELT_SPEED = 0.30
CAPTURE_EVERY = 8       # physics steps between captured frames
MAX_FRAMES = 40


def place_on_belt(env, name):
    jid = env._object_joint_ids[name]
    addr = env.model.jnt_qposadr[jid]
    env.data.qpos[addr:addr + 3] = SPAWN
    env.data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]
    dof = env.model.jnt_dofadr[jid]
    env.data.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    return jid, addr, dof


def main():
    cfg = load_config()
    cfg["detection"]["mode"] = "segmentation"
    cfg["tracking"]["enabled"] = True

    env = SimulationEnvironment()
    env.initialize(headless=True)
    camera = CameraInterface(env)
    detector = ObjectDetector(cfg, sim_env=env)
    tracker = ObjectTracker(cfg)

    trials = []
    for name in TRIAL_OBJECTS:
        env.park_all_objects()
        jid, addr, dof = place_on_belt(env, name)
        tracker.reset()

        ids_seen = []          # track id assigned to this object per captured frame
        frames_detected = 0
        frames_with_track = 0
        body_id = env._object_body_ids[name]

        step = 0
        captured = 0
        while captured < MAX_FRAMES:
            ypos = env.data.xpos[body_id][1]
            if ypos < STAGING_Y:
                env.data.qvel[dof + 1] = BELT_SPEED
            mujoco.mj_step(env.model, env.data)
            step += 1
            if step % CAPTURE_EVERY != 0:
                continue
            captured += 1

            frame = camera.capture_frame(sim_time=env.data.time)
            det = detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
            # find our object's detection
            ours = [d for d in det.detections if getattr(d, "true_name", "") == name]
            if ours:
                frames_detected += 1
            tracked = tracker.update(det)
            tid_for_obj = None
            for d, tid in tracked:
                if getattr(d, "true_name", "") == name:
                    tid_for_obj = tid
                    break
            if tid_for_obj is not None:
                frames_with_track += 1
                ids_seen.append(tid_for_obj)

            if env.data.xpos[body_id][1] >= STAGING_Y:
                break

        distinct = sorted(set(ids_seen))
        switches = sum(1 for i in range(1, len(ids_seen)) if ids_seen[i] != ids_seen[i - 1])
        trials.append({
            "object": name,
            "frames_detected": frames_detected,
            "frames_with_track": frames_with_track,
            "distinct_ids": distinct,
            "id_switches": switches,
            "single_id": len(distinct) <= 1 and len(ids_seen) > 0,
        })

    env.shutdown()

    n = len(trials)
    single_id = sum(1 for t in trials if t["single_id"])
    total_switches = sum(t["id_switches"] for t in trials)
    mean_tracked = float(np.mean([t["frames_with_track"] for t in trials]))
    mean_detected = float(np.mean([t["frames_detected"] for t in trials]))
    frag = sum(max(0, t["frames_detected"] - t["frames_with_track"]) for t in trials)

    report = {
        "scenario": "moving object on conveyor",
        "trials": n,
        "single_id_rate": single_id / n,
        "total_id_switches": total_switches,
        "mean_frames_detected": mean_detected,
        "mean_frames_tracked": mean_tracked,
        "detection_without_confirmed_track": frag,
        "per_trial": trials,
    }
    (OUT / "tracking_eval.json").write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print("  TRACKING EVALUATION (moving conveyor object)")
    print("=" * 60)
    print(f"  Trials:                 {n}")
    print(f"  Single-ID rate:         {report['single_id_rate']:.1%}")
    print(f"  Total ID switches:      {total_switches}")
    print(f"  Mean frames detected:   {mean_detected:.1f}")
    print(f"  Mean frames tracked:    {mean_tracked:.1f}")
    print(f"  Detect w/o conf. track: {frag}  (mostly the min_hits warm-up)")
    print("=" * 60)
    print(f"  Saved: {OUT / 'tracking_eval.json'}")


if __name__ == "__main__":
    main()
