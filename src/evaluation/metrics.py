"""
Metrics Collector
==================
Collects and computes performance metrics for the entire pipeline.
Measures detection accuracy, localization error, and task success.

All metrics come from actual measurements during simulation.
No values are fabricated.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("metrics")


@dataclass
class DetectionMetrics:
    """Detection performance metrics for one evaluation run."""
    total_frames: int = 0
    total_detections: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    inference_times_ms: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def avg_inference_ms(self) -> float:
        return np.mean(self.inference_times_ms) if self.inference_times_ms else 0.0
    
    @property
    def fps(self) -> float:
        avg = self.avg_inference_ms
        return 1000.0 / avg if avg > 0 else 0.0
    
    @property
    def avg_confidence(self) -> float:
        return np.mean(self.confidences) if self.confidences else 0.0


@dataclass
class LocalizationMetrics:
    """3D localization error metrics."""
    errors_x: List[float] = field(default_factory=list)
    errors_y: List[float] = field(default_factory=list)
    errors_z: List[float] = field(default_factory=list)
    errors_3d: List[float] = field(default_factory=list)  # Euclidean
    
    @property
    def mae_x(self) -> float:
        return np.mean(np.abs(self.errors_x)) if self.errors_x else 0.0
    
    @property
    def mae_y(self) -> float:
        return np.mean(np.abs(self.errors_y)) if self.errors_y else 0.0
    
    @property
    def mae_z(self) -> float:
        return np.mean(np.abs(self.errors_z)) if self.errors_z else 0.0
    
    @property
    def mae_3d(self) -> float:
        return np.mean(self.errors_3d) if self.errors_3d else 0.0
    
    @property
    def rmse_3d(self) -> float:
        return np.sqrt(np.mean(np.array(self.errors_3d)**2)) if self.errors_3d else 0.0
    
    @property
    def max_error(self) -> float:
        return np.max(self.errors_3d) if self.errors_3d else 0.0
    
    @property
    def num_samples(self) -> int:
        return len(self.errors_3d)


@dataclass
class TaskMetrics:
    """Pick-and-place task performance metrics."""
    total_episodes: int = 0
    total_pick_attempts: int = 0
    successful_picks: int = 0
    successful_places: int = 0
    complete_cycles: int = 0  # Full pick AND place
    total_objects: int = 0
    cycle_times: List[float] = field(default_factory=list)  # seconds
    
    @property
    def pick_success_rate(self) -> float:
        return self.successful_picks / max(1, self.total_pick_attempts)
    
    @property
    def place_success_rate(self) -> float:
        return self.successful_places / max(1, self.successful_picks)
    
    @property
    def complete_task_rate(self) -> float:
        return self.complete_cycles / max(1, self.total_objects)
    
    @property
    def avg_cycle_time(self) -> float:
        return np.mean(self.cycle_times) if self.cycle_times else 0.0
    
    @property
    def objects_per_minute(self) -> float:
        avg = self.avg_cycle_time
        return 60.0 / avg if avg > 0 else 0.0


@dataclass
class TrackingMetrics:
    """Object tracking performance metrics."""
    total_frames: int = 0
    total_tracks_created: int = 0
    id_switches: int = 0
    track_fragmentation: int = 0
    avg_track_length: float = 0.0
    
    @property
    def id_consistency_rate(self) -> float:
        if self.total_tracks_created == 0:
            return 0.0
        return 1.0 - (self.id_switches / max(1, self.total_frames))


class MetricsCollector:
    """
    Centralized metrics collection for the full pipeline.
    Collects actual measurements from simulation runs.
    """
    
    def __init__(self, output_dir: str = "results/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detection = DetectionMetrics()
        self.localization = LocalizationMetrics()
        self.task = TaskMetrics()
        self.tracking = TrackingMetrics()
        
        self.episode_results: List[Dict] = []
        self.start_time = time.time()
        
        logger.info("MetricsCollector initialized")
    
    def record_detection(
        self,
        num_detected: int,
        num_ground_truth: int,
        inference_time_ms: float,
        confidences: List[float],
        iou_threshold: float = 0.5,
        matched_count: int = 0,
    ) -> None:
        """
        Record detection metrics for a single frame.
        
        Args:
            num_detected: Number of detections in this frame
            num_ground_truth: Number of actual objects in scene
            inference_time_ms: Inference time in milliseconds
            confidences: List of confidence scores
            iou_threshold: IoU threshold for TP/FP determination
            matched_count: Number of detections matched to GT (TPs)
        """
        self.detection.total_frames += 1
        self.detection.total_detections += num_detected
        self.detection.inference_times_ms.append(inference_time_ms)
        self.detection.confidences.extend(confidences)
        
        # Compute TP/FP/FN
        tp = matched_count
        fp = num_detected - matched_count
        fn = num_ground_truth - matched_count
        
        self.detection.true_positives += tp
        self.detection.false_positives += fp
        self.detection.false_negatives += fn
    
    def record_localization_error(
        self,
        estimated_position: np.ndarray,
        ground_truth_position: np.ndarray,
    ) -> None:
        """
        Record localization error for a single object.
        
        Args:
            estimated_position: [x, y, z] estimated world position
            ground_truth_position: [x, y, z] true world position from simulator
        """
        error = estimated_position - ground_truth_position
        error_3d = np.linalg.norm(error)
        
        self.localization.errors_x.append(float(error[0]))
        self.localization.errors_y.append(float(error[1]))
        self.localization.errors_z.append(float(error[2]))
        self.localization.errors_3d.append(float(error_3d))
    
    def record_task_attempt(
        self,
        pick_success: bool,
        place_success: bool,
        cycle_time: float,
    ) -> None:
        """
        Record results of a single pick-and-place attempt.
        
        Args:
            pick_success: Whether pick succeeded
            place_success: Whether place succeeded
            cycle_time: Total cycle time in seconds
        """
        self.task.total_pick_attempts += 1
        self.task.total_objects += 1
        
        if pick_success:
            self.task.successful_picks += 1
        if place_success:
            self.task.successful_places += 1
        if pick_success and place_success:
            self.task.complete_cycles += 1
        
        self.task.cycle_times.append(cycle_time)
    
    def record_episode(self, episode_result: Dict) -> None:
        """Record results from a complete episode."""
        self.task.total_episodes += 1
        self.episode_results.append(episode_result)
    
    def get_summary(self) -> Dict:
        """Get complete metrics summary."""
        return {
            'detection': {
                'precision': self.detection.precision,
                'recall': self.detection.recall,
                'f1_score': self.detection.f1_score,
                'avg_confidence': self.detection.avg_confidence,
                'avg_inference_ms': self.detection.avg_inference_ms,
                'fps': self.detection.fps,
                'total_frames': self.detection.total_frames,
            },
            'localization': {
                'mae_3d_m': self.localization.mae_3d,
                'mae_3d_cm': self.localization.mae_3d * 100,
                'rmse_3d_m': self.localization.rmse_3d,
                'mae_x_cm': self.localization.mae_x * 100,
                'mae_y_cm': self.localization.mae_y * 100,
                'mae_z_cm': self.localization.mae_z * 100,
                'max_error_cm': self.localization.max_error * 100,
                'num_samples': self.localization.num_samples,
            },
            'task': {
                'pick_success_rate': self.task.pick_success_rate,
                'place_success_rate': self.task.place_success_rate,
                'complete_task_rate': self.task.complete_task_rate,
                'avg_cycle_time_s': self.task.avg_cycle_time,
                'objects_per_minute': self.task.objects_per_minute,
                'total_episodes': self.task.total_episodes,
                'total_attempts': self.task.total_pick_attempts,
            },
            'tracking': {
                'id_consistency_rate': self.tracking.id_consistency_rate,
                'total_tracks': self.tracking.total_tracks_created,
                'id_switches': self.tracking.id_switches,
            },
        }
    
    def save_results(self, filename: str = "pipeline_metrics.json") -> str:
        """Save all metrics to JSON file."""
        output_path = self.output_dir / filename
        
        results = {
            'summary': self.get_summary(),
            'episodes': self.episode_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_runtime_s': time.time() - self.start_time,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {output_path}")
        return str(output_path)
    
    def print_summary(self) -> None:
        """Print formatted metrics summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("  PIPELINE PERFORMANCE METRICS")
        print("=" * 70)
        
        print("\n--- DETECTION ---")
        det = summary['detection']
        print(f"  Precision:    {det['precision']:.3f}")
        print(f"  Recall:       {det['recall']:.3f}")
        print(f"  F1 Score:     {det['f1_score']:.3f}")
        print(f"  Avg Conf:     {det['avg_confidence']:.3f}")
        print(f"  Inference:    {det['avg_inference_ms']:.1f}ms ({det['fps']:.1f} FPS)")
        print(f"  Total Frames: {det['total_frames']}")
        
        print("\n--- LOCALIZATION ---")
        loc = summary['localization']
        print(f"  MAE (3D):     {loc['mae_3d_cm']:.2f} cm")
        print(f"  RMSE (3D):    {loc['rmse_3d_m']*100:.2f} cm")
        print(f"  MAE (X):      {loc['mae_x_cm']:.2f} cm")
        print(f"  MAE (Y):      {loc['mae_y_cm']:.2f} cm")
        print(f"  MAE (Z):      {loc['mae_z_cm']:.2f} cm")
        print(f"  Max Error:    {loc['max_error_cm']:.2f} cm")
        print(f"  Samples:      {loc['num_samples']}")
        
        print("\n--- TASK PERFORMANCE ---")
        task = summary['task']
        print(f"  Pick Success:    {task['pick_success_rate']:.1%}")
        print(f"  Place Success:   {task['place_success_rate']:.1%}")
        print(f"  Complete Rate:   {task['complete_task_rate']:.1%}")
        print(f"  Avg Cycle Time:  {task['avg_cycle_time_s']:.2f}s")
        print(f"  Objects/min:     {task['objects_per_minute']:.1f}")
        print(f"  Episodes:        {task['total_episodes']}")
        
        print("\n--- TRACKING ---")
        trk = summary['tracking']
        print(f"  ID Consistency:  {trk['id_consistency_rate']:.1%}")
        print(f"  ID Switches:     {trk['id_switches']}")
        
        print("\n" + "=" * 70)
