"""
Main Pipeline - Vision-Guided Robotic Pick-and-Place
=====================================================
Runs the complete autonomous pick-and-place pipeline in MuJoCo.

Pipeline:
    Camera (RGB-D) -> YOLO Detection -> ByteTrack Tracking ->
    3D Localization -> Task Planning -> Robot Control -> Pick & Place

Usage:
    # Full pipeline with 3D viewer
    python scripts/run_pipeline.py

    # Headless (no viewer, faster for evaluation)
    python scripts/run_pipeline.py --headless

    # Multiple episodes for evaluation
    python scripts/run_pipeline.py --episodes 10
"""

import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.simulation.environment import SimulationEnvironment
from src.camera.camera_interface import CameraInterface
from src.camera.depth_processor import DepthProcessor
from src.detection.detector import ObjectDetector
from src.detection.detection_visualizer import DetectionVisualizer
from src.tracking.tracker import ObjectTracker
from src.localization.localizer import ObjectLocalizer
from src.robot_control.arm_controller import ArmController
from src.robot_control.gripper_controller import GripperController
from src.robot_control.pick_place import PickPlaceExecutor
from src.task_logic.sorting_rules import SortingRules
from src.task_logic.task_planner import TaskPlanner
from src.evaluation.metrics import MetricsCollector
from src.utils.logger import setup_logger
from src.utils.visualization import overlay_info, create_pipeline_display

logger = setup_logger("pipeline", log_to_file=True)


def load_config() -> dict:
    """Load all YAML configs into one dictionary."""
    config = {}
    for yaml_file in Path("config").glob("*.yaml"):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if data:
                config.update(data)
    return config


class Pipeline:
    """
    Main pipeline orchestrating all components.
    
    Architecture:
        SimulationEnvironment (MuJoCo)
         -> CameraInterface -> DepthProcessor
         -> ObjectDetector (YOLO)
         -> ObjectTracker (ByteTrack)
         -> ObjectLocalizer (3D from depth)
         -> TaskPlanner (color sorting)
         -> ArmController + GripperController
         -> PickPlaceExecutor
    """
    
    def __init__(self, headless: bool = False):
        self.config = load_config()
        self.headless = headless
        
        logger.info("=" * 60)
        logger.info("  VISION-GUIDED ROBOTIC PICK-AND-PLACE")
        logger.info("  MuJoCo Simulation | YOLO | OpenCV | ByteTrack")
        logger.info("=" * 60)
        
        # 1. Simulation
        logger.info("[1/8] Initializing MuJoCo simulation...")
        self.sim = SimulationEnvironment()
        self.sim.initialize(headless=headless)
        
        # 2. Camera
        logger.info("[2/8] Setting up camera...")
        self.camera = CameraInterface(self.sim)
        self.depth_proc = DepthProcessor(
            self.camera.intrinsic_matrix,
            depth_range=(0.05, 3.0)
        )
        
        # 3. Detection
        logger.info("[3/8] Loading YOLO detector...")
        self.detector = ObjectDetector(self.config)
        self.visualizer = DetectionVisualizer()
        
        # 4. Tracking
        logger.info("[4/8] Initializing tracker...")
        self.tracker = ObjectTracker(self.config)
        
        # 5. Localization
        logger.info("[5/8] Setting up 3D localizer...")
        self.localizer = ObjectLocalizer(self.config, self.depth_proc)
        
        # 6. Robot control
        logger.info("[6/8] Initializing robot controller...")
        self.arm = ArmController(self.sim)
        self.gripper = GripperController(self.sim)
        self.pick_place = PickPlaceExecutor(self.arm, self.gripper)
        
        # 7. Task logic
        logger.info("[7/8] Loading task planner...")
        self.sorting = SortingRules(self.config)
        self.planner = TaskPlanner(self.config, self.sorting)
        
        # 8. Evaluation
        logger.info("[8/8] Metrics collector ready")
        self.metrics = MetricsCollector()
        
        logger.info("")
        logger.info("Pipeline ready!")
        logger.info("=" * 60)
    
    def run_episode(self, episode_id: int = 0, randomize: bool = True) -> dict:
        """
        Run a single pick-and-place episode.
        
        Args:
            episode_id: Episode number
            randomize: Randomize object positions
            
        Returns:
            Episode results
        """
        logger.info(f"\n--- EPISODE {episode_id} ---")
        episode_start = time.time()
        
        # Reset scene
        gt_positions = self.sim.reset_objects(randomize=randomize)
        self.arm.move_to_home(render=True)
        self.tracker.reset()
        self.planner.start_task()
        
        logger.info(f"Objects placed: {list(gt_positions.keys())}")
        for name, pos in gt_positions.items():
            logger.info(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Wait for scene to settle
        self.sim.step_n(100, render=True)
        
        # ===== PERCEPTION-MANIPULATION LOOP =====
        objects_completed = 0
        objects_failed = 0
        max_iterations = 4  # Max objects to handle
        
        for obj_iter in range(max_iterations):
            if self.planner.is_complete():
                break
            
            # --- PERCEPTION ---
            logger.info(f"\n  [PERCEPTION] Capturing frame...")
            frame = self.camera.capture_frame(sim_time=self.sim.data.time)
            
            # Detection
            det_result = self.detector.detect(frame.rgb, frame.timestamp, frame.frame_id)
            logger.info(f"  Detected {det_result.num_detections} objects")
            
            # Tracking
            tracked = self.tracker.update(det_result)
            detections = [d for d, _ in tracked]
            track_ids = [t for _, t in tracked]
            
            # 3D Localization
            localized = []
            if frame.has_depth and detections:
                localized = self.localizer.localize_all(detections, frame.depth, track_ids)
                logger.info(f"  Localized {len(localized)} objects in 3D")
                for obj in localized:
                    logger.info(f"    {obj.class_name}: world=[{obj.position_world[0]:.3f}, "
                              f"{obj.position_world[1]:.3f}, {obj.position_world[2]:.3f}]")
            
            # Evaluate detection/localization
            self.metrics.record_detection(
                len(detections), len(gt_positions),
                det_result.inference_time_ms,
                [d.confidence for d in detections],
                matched_count=min(len(detections), len(gt_positions))
            )
            
            if localized:
                for obj in localized:
                    if obj.class_name in gt_positions:
                        self.metrics.record_localization_error(
                            obj.position_world, gt_positions[obj.class_name]
                        )
            
            # --- VISUALIZATION ---
            if not self.headless:
                det_image = self.visualizer.draw_detections(
                    frame.rgb, detections, track_ids,
                    [obj.position_world for obj in localized] if localized else None
                )
                det_image = self.visualizer.draw_info_panel(det_image, det_result, {
                    'Episode': episode_id,
                    'Object': f"{obj_iter+1}/{max_iterations}",
                })
                cv2.imshow("Vision Pipeline", det_image)
                cv2.waitKey(1)
            
            # --- TASK PLANNING ---
            if not localized:
                logger.warning("  No localized objects, skipping...")
                continue
            
            robot_pos = self.arm.get_ee_position()
            target = self.planner.select_next_target(localized, robot_pos)
            
            if target is None:
                logger.info("  No more targets")
                break
            
            # --- MANIPULATION ---
            logger.info(f"\n  [MANIPULATION] Target: {target.object_name}")
            pick_start = time.time()
            
            success = self.pick_place.execute(
                pick_position=target.pick_position,
                place_position=target.place_position,
                object_name=target.object_name,
            )
            
            cycle_time = time.time() - pick_start
            self.metrics.record_task_attempt(success, success, cycle_time)
            self.planner.report_result(success)
            
            if success:
                objects_completed += 1
            else:
                objects_failed += 1
            
            # Return to home between picks
            self.arm.move_to_home(render=True)
        
        # Episode results
        elapsed = time.time() - episode_start
        total = objects_completed + objects_failed
        success_rate = objects_completed / max(1, total)
        
        result = {
            'episode_id': episode_id,
            'duration': elapsed,
            'objects_completed': objects_completed,
            'objects_failed': objects_failed,
            'success_rate': success_rate,
        }
        
        logger.info(f"\n  Episode {episode_id} done: {objects_completed}/{total} successful ({success_rate:.0%}), {elapsed:.1f}s")
        
        return result
    
    def run(self, num_episodes: int = 1) -> dict:
        """
        Run multiple episodes and collect results.
        
        Args:
            num_episodes: Number of episodes
            
        Returns:
            Aggregated results
        """
        all_results = []
        
        for ep in range(num_episodes):
            result = self.run_episode(episode_id=ep, randomize=(ep > 0))
            all_results.append(result)
            
            # Pause between episodes so viewer can show the reset
            if not self.headless and ep < num_episodes - 1:
                logger.info("  --- Next episode in 2s ---")
                time.sleep(2.0)
            
            # Check viewer is still open
            if not self.sim.is_viewer_alive():
                logger.info("Viewer closed, stopping")
                break
        
        # Print summary
        self.metrics.print_summary()
        self.metrics.save_results("pipeline_results.json")
        
        avg_success = np.mean([r['success_rate'] for r in all_results])
        logger.info(f"\n{'='*60}")
        logger.info(f"  FINAL: {num_episodes} episodes, avg success: {avg_success:.1%}")
        logger.info(f"{'='*60}")
        
        return {'episodes': all_results, 'summary': self.metrics.get_summary()}
    
    def shutdown(self):
        """Clean shutdown."""
        if not self.headless:
            cv2.destroyAllWindows()
        self.sim.shutdown()
        logger.info("Pipeline shut down")


def main():
    parser = argparse.ArgumentParser(description="Vision-Guided Robotic Pick-and-Place")
    parser.add_argument('--headless', action='store_true', help='No visualization')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    args = parser.parse_args()
    
    pipeline = Pipeline(headless=args.headless)
    
    try:
        pipeline.run(num_episodes=args.episodes)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
