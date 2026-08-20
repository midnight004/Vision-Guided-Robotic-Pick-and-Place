"""Render figures for the report: cell screenshots and an overhead detection overlay."""

import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import cv2
import mujoco
import yaml

from src.simulation.environment import SimulationEnvironment
from src.simulation.scene_builder import SceneBuilder
from src.camera.camera_interface import CameraInterface
from src.detection.detector import ObjectDetector
from src.detection.detection_visualizer import DetectionVisualizer
from src.localization.localizer import ObjectLocalizer
from src.camera.depth_processor import DepthProcessor
from src.robot_control.arm_controller import ArmController
from src.robot_control.gripper_controller import GripperController
from src.robot_control.pick_place import PickPlaceExecutor

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1200


def load_config():
    cfg = {}
    for f in Path("config").glob("*.yaml"):
        with open(f) as fh:
            d = yaml.safe_load(fh)
            if d:
                cfg.update(d)
    return cfg


def free_shot(renderer, data, path, lookat, azimuth, elevation, distance):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", path)


def manual_pick_capture(env, arm, grip, renderer, item, bin_xy):
    """Feed one item and drive it through a pick, capturing the transport moment."""
    pos = env.feed_object(item, render=False)
    grip.open(steps=60, render=False)
    approach = pos.copy(); approach[2] += 0.15
    arm.move_to_cartesian(approach, duration=1.2, render=False)
    grasp = pos.copy(); grasp[2] += 0.058
    arm.move_to_cartesian(grasp, duration=1.0, render=False)
    grip.close(steps=250, render=False)
    lift = pos.copy(); lift[2] = 0.56
    arm.move_to_cartesian(lift, duration=1.5, render=False)
    transport = np.array([bin_xy[0], bin_xy[1], 0.56])
    arm.move_to_cartesian(transport, duration=2.0, render=False)
    # Capture the arm holding the object above the target bin
    free_shot(renderer, env.data, FIG_DIR / "fig_action.png",
              lookat=[0.34, -0.18, 0.42], azimuth=125, elevation=-18, distance=0.95)
    # Finish the placement
    lower = np.array([bin_xy[0], bin_xy[1], 0.45])
    arm.move_to_cartesian(lower, duration=1.2, render=False)
    grip.open(steps=100, render=False)
    env.step_n(150, render=False)


def main():
    cfg = load_config()
    env = SimulationEnvironment()
    env.initialize(headless=True)

    arm = ArmController(env)
    grip = GripperController(env)
    pp = PickPlaceExecutor(arm, grip)

    renderer = mujoco.Renderer(env.model, H, W)

    env.reset_objects()
    arm.move_to_scan_pose(render=False)

    # Hero pick: red_box to the red bin, capturing the transport moment
    manual_pick_capture(env, arm, grip, renderer, "red_box",
                        SceneBuilder.get_destination("red")[:2])

    # Populate more bins so the overview shows a working cell
    for item, color in [("blue_box", "blue"), ("green_cylinder", "green"),
                         ("yellow_sphere", "yellow"), ("purple_box", "unknown")]:
        pos = env.feed_object(item, render=False)
        dest = SceneBuilder.get_destination(color)
        pp.execute(pick_position=pos, place_position=dest, object_name=item)

    arm.move_to_scan_pose(render=False)
    env.step_n(50, render=False)

    # Wide overview of the whole cell
    free_shot(renderer, env.data, FIG_DIR / "fig_overview.png",
              lookat=[0.45, 0.0, 0.34], azimuth=138, elevation=-24, distance=1.5)

    # Close view of the labeled bins with sorted objects
    free_shot(renderer, env.data, FIG_DIR / "fig_bins.png",
              lookat=[0.46, 0.0, 0.34], azimuth=70, elevation=-32, distance=1.05)

    # Overhead detection overlay
    camera = CameraInterface(env)
    depth_proc = DepthProcessor(camera.intrinsic_matrix, depth_range=(0.05, 3.0))
    detector = ObjectDetector(cfg, sim_env=env)
    localizer = ObjectLocalizer(cfg, depth_proc, sim_env=env)
    visualizer = DetectionVisualizer()

    env.feed_object("blue_capsule", render=False)
    frame = camera.capture_frame(sim_time=env.data.time)
    det = detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
    localized = []
    if frame.has_depth and det.detections:
        localized = localizer.localize_all(det.detections, frame.depth)
    overlay = visualizer.draw_detections(
        frame.rgb, det.detections, None,
        [o.position_world for o in localized] if localized else None)
    cv2.imwrite(str(FIG_DIR / "fig_detection.png"), overlay)
    print("wrote", FIG_DIR / "fig_detection.png")

    env.shutdown()


if __name__ == "__main__":
    main()
