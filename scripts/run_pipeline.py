"""
Factory Pick-and-Place Pipeline
Menagerie Franka | Real Physics Grasping | 8 Products | 4 Sorting Bins
"""

import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.simulation.environment import SimulationEnvironment
from src.simulation.scene_builder import SceneBuilder, PRODUCT_TO_COLOR, DESTINATIONS
from src.camera.camera_interface import CameraInterface
from src.camera.depth_processor import DepthProcessor
from src.detection.detector import ObjectDetector
from src.detection.detection_visualizer import DetectionVisualizer
from src.tracking.tracker import ObjectTracker
from src.localization.localizer import ObjectLocalizer
from src.robot_control.arm_controller import ArmController
from src.robot_control.gripper_controller import GripperController
from src.robot_control.pick_place import PickPlaceExecutor
from src.evaluation.metrics import MetricsCollector
from src.utils.logger import setup_logger

logger = setup_logger("pipeline", log_to_file=True)


def load_config() -> dict:
    config = {}
    for f in Path("config").glob("*.yaml"):
        with open(f) as fh:
            d = yaml.safe_load(fh)
            if d:
                config.update(d)
    return config


class Pipeline:
    def __init__(self, headless: bool = False, num_objects: int = 4):
        self.config = load_config()
        self.headless = headless
        self.num_objects = num_objects

        logger.info("=" * 60)
        logger.info("  FACTORY PICK-AND-PLACE")
        logger.info("  Menagerie Franka | Real Physics | MuJoCo")
        logger.info("=" * 60)

        # Simulation
        self.sim = SimulationEnvironment()
        self.sim.initialize(headless=headless)

        # Camera
        self.camera = CameraInterface(self.sim)
        self.depth_proc = DepthProcessor(self.camera.intrinsic_matrix, depth_range=(0.05, 3.0))

        # Detection (segmentation-based, uses sim ground-truth masks)
        self.detector = ObjectDetector(self.config, sim_env=self.sim)
        self.visualizer = DetectionVisualizer()

        # Tracking
        self.tracker = ObjectTracker(self.config)

        # Localization
        self.localizer = ObjectLocalizer(self.config, self.depth_proc, sim_env=self.sim)

        # Robot
        self.arm = ArmController(self.sim)
        self.gripper = GripperController(self.sim)
        self.pick_place = PickPlaceExecutor(self.arm, self.gripper)

        # Metrics
        self.metrics = MetricsCollector()

        logger.info("Pipeline ready!")

    def run_episode(self, episode_id: int = 0, randomize: bool = True) -> dict:
        logger.info(f"\n{'='*40} EPISODE {episode_id} {'='*40}")
        episode_start = time.time()

        # Reset
        gt_positions = self.sim.reset_objects(randomize=randomize, num_objects=self.num_objects)
        self.arm.move_to_home(render=True)
        self.tracker.reset()
        self.sim.step_n(200, render=True)

        logger.info(f"Products on table: {list(gt_positions.keys())[:self.num_objects]}")

        objects_completed = 0
        objects_failed = 0
        handled = set()
        max_picks = self.num_objects + 2  # safety cap

        for pick_iter in range(max_picks):
            # SCAN: tuck arm aside so the camera sees the whole table
            self.arm.move_to_scan_pose(render=True)
            self.sim.step_n(30, render=not self.headless)

            # PERCEPTION: detect + localize all objects on the table
            frame = self.camera.capture_frame(sim_time=self.sim.data.time)
            det_result = self.detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
            tracked = self.tracker.update(det_result)
            detections = [d for d, _ in tracked]
            track_ids = [t for _, t in tracked]

            localized = []
            if frame.has_depth and detections:
                localized = self.localizer.localize_all(detections, frame.depth, track_ids)

            # Visualization
            if not self.headless:
                det_image = self.visualizer.draw_detections(
                    frame.rgb, detections, track_ids,
                    [obj.position_world for obj in localized] if localized else None
                )
                det_image = self.visualizer.draw_info_panel(det_image, det_result, {
                    'Episode': episode_id,
                    'Sorted': f"{objects_completed}/{self.num_objects}",
                    'Detected': f"{len(localized)}",
                })
                cv2.imshow("Vision Pipeline", det_image)
                cv2.waitKey(1)

            # Choose closest un-handled object
            target_obj = None
            best_dist = 1e9
            scan_pos = self.arm.get_ee_position()
            for obj in localized:
                key = f"{obj.class_name}_{round(obj.position_world[0],2)}_{round(obj.position_world[1],2)}"
                if obj.class_name in handled:
                    continue
                if SceneBuilder.get_destination(obj.class_name) is None:
                    continue
                d = np.linalg.norm(obj.position_world[:2] - scan_pos[:2])
                if d < best_dist:
                    best_dist = d
                    target_obj = obj

            if target_obj is None:
                logger.info("  All detected objects sorted - episode complete!")
                break

            # Execute pick and place
            dest = SceneBuilder.get_destination(target_obj.class_name)
            color = SceneBuilder.get_color_category(target_obj.class_name)
            logger.info(f"\n  Target: {target_obj.class_name} -> {color} bin")

            success = self.pick_place.execute(
                pick_position=target_obj.position_world,
                place_position=dest,
                object_name=target_obj.class_name,
            )

            handled.add(target_obj.class_name)
            if success:
                objects_completed += 1
            else:
                objects_failed += 1

            self.metrics.record_task_attempt(success, success, time.time() - episode_start)

        # Park at the end
        self.arm.move_to_scan_pose(render=True)

        elapsed = time.time() - episode_start
        total = objects_completed + objects_failed
        success_rate = objects_completed / max(1, total)

        result = {
            'episode_id': episode_id,
            'duration': elapsed,
            'completed': objects_completed,
            'failed': objects_failed,
            'success_rate': success_rate,
        }
        logger.info(f"\n  Episode {episode_id}: {objects_completed}/{total} sorted ({success_rate:.0%}), {elapsed:.1f}s")
        return result

    def run(self, num_episodes: int = 1) -> dict:
        all_results = []
        for ep in range(num_episodes):
            result = self.run_episode(episode_id=ep, randomize=(ep > 0))
            all_results.append(result)

            if not self.sim.is_viewer_alive():
                break

            if not self.headless and ep < num_episodes - 1:
                time.sleep(1.5)

        self.metrics.print_summary()
        self.metrics.save_results("factory_results.json")

        avg = np.mean([r['success_rate'] for r in all_results])
        logger.info(f"\nFINAL: {num_episodes} episodes, avg success: {avg:.1%}")
        return {'episodes': all_results, 'avg_success': avg}

    def shutdown(self):
        if not self.headless:
            cv2.destroyAllWindows()
        self.sim.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Factory Pick-and-Place")
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--objects', type=int, default=4, help='Objects per episode (max 8)')
    args = parser.parse_args()

    pipeline = Pipeline(headless=args.headless, num_objects=min(args.objects, 8))
    try:
        pipeline.run(num_episodes=args.episodes)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
