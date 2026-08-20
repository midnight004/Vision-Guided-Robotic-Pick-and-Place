"""
Pipeline Evaluator
====================
Runs systematic evaluation of the full perception-manipulation pipeline.
Performs multiple episodes with randomized configurations and collects metrics.
"""

import numpy as np
from typing import Dict, List, Optional
import time
import json
from pathlib import Path

from .metrics import MetricsCollector
from ..utils.logger import setup_logger

logger = setup_logger("evaluator")


class PipelineEvaluator:
    """
    Evaluates the full pipeline across multiple episodes.
    Handles setup, execution, and results aggregation.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Evaluation configuration
        """
        self.config = config['evaluation']
        self.metrics = MetricsCollector()
        
        # Evaluation parameters
        robustness_cfg = self.config['robustness']
        self.num_trials = robustness_cfg['num_trials']
        self.randomize_positions = robustness_cfg['randomize_positions']
        self.randomize_orientations = robustness_cfg['randomize_orientations']
        
        self.results: List[Dict] = []
        
        logger.info(f"PipelineEvaluator initialized ({self.num_trials} trials)")
    
    def evaluate_detection(
        self,
        detections: List,
        ground_truths: Dict[str, np.ndarray],
        inference_time_ms: float,
        iou_threshold: float = 0.5,
    ) -> None:
        """
        Evaluate detection performance for a single frame.
        
        Args:
            detections: List of Detection objects
            ground_truths: Dict of object_name → world_position
            inference_time_ms: Inference time
            iou_threshold: IoU threshold for matching
        """
        num_detected = len(detections)
        num_gt = len(ground_truths)
        
        # Simple matching: count detections that correspond to GT objects
        # by class name
        gt_classes = set()
        for name in ground_truths.keys():
            # Extract class from object name (e.g., "red_box" from "red_box")
            gt_classes.add(name)
        
        matched = 0
        for det in detections:
            if det.class_name in gt_classes:
                matched += 1
                gt_classes.discard(det.class_name)  # One-to-one matching
        
        confidences = [det.confidence for det in detections]
        
        self.metrics.record_detection(
            num_detected=num_detected,
            num_ground_truth=num_gt,
            inference_time_ms=inference_time_ms,
            confidences=confidences,
            matched_count=matched,
        )
    
    def evaluate_localization(
        self,
        estimated_positions: Dict[str, np.ndarray],
        ground_truth_positions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate 3D localization accuracy.
        
        Args:
            estimated_positions: Dict of object_name → estimated [x,y,z]
            ground_truth_positions: Dict of object_name → true [x,y,z]
            
        Returns:
            Dictionary of per-object errors
        """
        errors = {}
        
        for name, gt_pos in ground_truth_positions.items():
            if name in estimated_positions:
                est_pos = estimated_positions[name]
                self.metrics.record_localization_error(est_pos, gt_pos)
                errors[name] = float(np.linalg.norm(est_pos - gt_pos))
        
        return errors
    
    def evaluate_episode(
        self,
        episode_id: int,
        results: Dict,
    ) -> Dict:
        """
        Record results from a complete episode evaluation.
        
        Args:
            episode_id: Episode number
            results: Episode results dict
            
        Returns:
            Episode summary
        """
        episode_summary = {
            'episode_id': episode_id,
            'timestamp': time.strftime("%H:%M:%S"),
            **results,
        }
        
        self.metrics.record_episode(episode_summary)
        self.results.append(episode_summary)
        
        logger.info(f"Episode {episode_id} evaluated: {results.get('success_rate', 0):.1%}")
        
        return episode_summary
    
    def generate_report(self) -> str:
        """Generate and save the full evaluation report."""
        self.metrics.print_summary()
        
        report_path = self.metrics.save_results("evaluation_results.json")
        
        # Also save a human-readable report
        report_text = self._generate_text_report()
        text_path = Path("results/metrics/evaluation_report.txt")
        text_path.write_text(report_text)
        
        logger.info(f"Full report saved to {text_path}")
        return report_text
    
    def _generate_text_report(self) -> str:
        """Generate human-readable evaluation report."""
        summary = self.metrics.get_summary()
        
        report = []
        report.append("=" * 70)
        report.append("VISION-GUIDED ROBOTIC PICK-AND-PLACE")
        report.append("EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Episodes: {summary['task']['total_episodes']}")
        report.append(f"Total Runtime: {time.time() - self.metrics.start_time:.1f}s")
        report.append("")
        report.append("-" * 40)
        report.append("OBJECT DETECTION")
        report.append("-" * 40)
        report.append(f"  Precision:       {summary['detection']['precision']:.4f}")
        report.append(f"  Recall:          {summary['detection']['recall']:.4f}")
        report.append(f"  F1 Score:        {summary['detection']['f1_score']:.4f}")
        report.append(f"  Mean Confidence: {summary['detection']['avg_confidence']:.4f}")
        report.append(f"  Inference Time:  {summary['detection']['avg_inference_ms']:.1f}ms")
        report.append(f"  FPS:             {summary['detection']['fps']:.1f}")
        report.append("")
        report.append("-" * 40)
        report.append("3D LOCALIZATION")
        report.append("-" * 40)
        report.append(f"  Mean Absolute Error (3D): {summary['localization']['mae_3d_cm']:.2f} cm")
        report.append(f"  RMSE (3D):                {summary['localization']['rmse_3d_m']*100:.2f} cm")
        report.append(f"  Per-axis MAE:")
        report.append(f"    X: {summary['localization']['mae_x_cm']:.2f} cm")
        report.append(f"    Y: {summary['localization']['mae_y_cm']:.2f} cm")
        report.append(f"    Z: {summary['localization']['mae_z_cm']:.2f} cm")
        report.append(f"  Maximum Error:            {summary['localization']['max_error_cm']:.2f} cm")
        report.append("")
        report.append("-" * 40)
        report.append("PICK-AND-PLACE TASK")
        report.append("-" * 40)
        report.append(f"  Pick Success Rate:       {summary['task']['pick_success_rate']:.1%}")
        report.append(f"  Place Success Rate:      {summary['task']['place_success_rate']:.1%}")
        report.append(f"  Complete Task Rate:      {summary['task']['complete_task_rate']:.1%}")
        report.append(f"  Average Cycle Time:      {summary['task']['avg_cycle_time_s']:.2f}s")
        report.append(f"  Throughput:              {summary['task']['objects_per_minute']:.1f} objects/min")
        report.append("")
        report.append("-" * 40)
        report.append("TRACKING")
        report.append("-" * 40)
        report.append(f"  ID Consistency:  {summary['tracking']['id_consistency_rate']:.1%}")
        report.append(f"  ID Switches:     {summary['tracking']['id_switches']}")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
