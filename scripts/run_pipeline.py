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
from src.simulation.scene_builder import SceneBuilder, DESTINATIONS
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

        # Reset: park everything, arm to scan pose
        self.sim.reset_objects()
        self.arm.move_to_scan_pose(render=True)
        self.tracker.reset()

        # Build a randomized queue for this episode (mix of known + unknown items)
        queue = self.sim.build_episode_queue(self.num_objects, unknown_ratio=0.3)
        logger.info(f"Conveyor queue ({len(queue)} items): {queue}")

        objects_completed = 0
        objects_failed = 0
        correct_sorts = 0

        for item_idx, item_name in enumerate(queue):
            # FEED: conveyor delivers the next item to the staging area
            self.arm.move_to_scan_pose(render=not self.headless)
            self.sim.feed_object(item_name, render=not self.headless)

            # PERCEIVE: capture frame, detect + classify by color, localize in 3D
            # (single-shot detection per fed item; no tracker needed)
            frame = self.camera.capture_frame(sim_time=self.sim.data.time)
            det_result = self.detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
            detections = det_result.detections

            localized = []
            if frame.has_depth and detections:
                localized = self.localizer.localize_all(detections, frame.depth)

            if not self.headless:
                det_image = self.visualizer.draw_detections(
                    frame.rgb, detections, None,
                    [o.position_world for o in localized] if localized else None
                )
                det_image = self.visualizer.draw_info_panel(det_image, det_result, {
                    'Episode': episode_id,
                    'Item': f"{item_idx+1}/{len(queue)}",
                    'Sorted': f"{objects_completed}",
                })
                cv2.imshow("Vision Pipeline", det_image)
                cv2.waitKey(1)

            # Pick the object in the staging zone (closest to staging point)
            target_obj = None
            best_dist = 1e9
            staging = np.array([0.5, -0.12])
            for obj in localized:
                d = np.linalg.norm(obj.position_world[:2] - staging)
                if d < best_dist:
                    best_dist = d
                    target_obj = obj

            if target_obj is None:
                logger.warning(f"  Item {item_name} not detected in staging - skipping")
                objects_failed += 1
                continue

            color_class = target_obj.class_name           # detected color category
            bin_name = SceneBuilder.get_bin_name(color_class)
            dest = SceneBuilder.get_destination(color_class)
            true_name = getattr(target_obj, 'true_name', item_name)

            logger.info(f"\n  Item: {true_name} -> detected '{color_class}' -> {bin_name.upper()} bin")

            success = self.pick_place.execute(
                pick_position=target_obj.position_world,
                place_position=dest,
                object_name=f"{true_name}({color_class})",
            )

            # Verify the object physically landed inside the target bin
            landed = self._landed_in_bin(true_name, dest)
            expected_bin = self._expected_bin(true_name)
            routed_right = (bin_name == expected_bin)

            if success and landed:
                objects_completed += 1
                if routed_right:
                    correct_sorts += 1
                logger.info(f"  LANDED in {bin_name.upper()} bin"
                            f" ({'correct' if routed_right else 'wrong bin'})")
            else:
                objects_failed += 1
                reason = "arm motion failed" if not success else "missed bin"
                logger.warning(f"  NOT PLACED ({reason})")
                # Clear the failed item off the table so it doesn't corrupt the
                # next detection/localization cycle.
                self.sim.park_object(true_name)

            self.metrics.record_task_attempt(success, success, time.time() - episode_start)

        self.arm.move_to_scan_pose(render=True)

        elapsed = time.time() - episode_start
        total = objects_completed + objects_failed
        success_rate = objects_completed / max(1, len(queue))
        sort_accuracy = correct_sorts / max(1, objects_completed)

        result = {
            'episode_id': episode_id,
            'duration': elapsed,
            'completed': objects_completed,
            'failed': objects_failed,
            'success_rate': success_rate,
            'sort_accuracy': sort_accuracy,
        }
        logger.info(f"\n  Episode {episode_id}: {objects_completed}/{len(queue)} handled, "
                    f"sort accuracy {sort_accuracy:.0%}, {elapsed:.1f}s")
        return result

    def _landed_in_bin(self, object_name: str, dest, xy_tol: float = 0.08,
                       z_max: float = 0.45) -> bool:
        """Confirm the object's actual final position is inside the target bin."""
        pos = self.sim.get_object_position(object_name)
        if pos is None:
            return False
        dxy = float(np.linalg.norm(pos[:2] - np.asarray(dest)[:2]))
        return dxy <= xy_tol and pos[2] <= z_max

    @staticmethod
    def _expected_bin(true_name: str) -> str:
        """Ground-truth bin for evaluating sort correctness."""
        if true_name.startswith("red"):
            return "red"
        if true_name.startswith("blue"):
            return "blue"
        if true_name.startswith("green"):
            return "green"
        if true_name.startswith("yellow"):
            return "yellow"
        return "trash"  # purple, orange, white, black

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
    parser.add_argument('--objects', type=int, default=6, help='Items fed per episode (max 14)')
    args = parser.parse_args()

    pipeline = Pipeline(headless=args.headless, num_objects=min(args.objects, 14))
    try:
        pipeline.run(num_episodes=args.episodes)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
