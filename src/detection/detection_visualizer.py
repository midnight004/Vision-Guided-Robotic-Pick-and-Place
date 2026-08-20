"""
Detection Visualizer
=====================
Draws detection results on camera frames for demonstration and debugging.
Creates portfolio-quality visualizations with bounding boxes, labels,
confidence scores, and tracking IDs.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .detector import Detection, DetectionResult
from ..utils.logger import setup_logger

logger = setup_logger("detection_viz")


# Class-specific colors (BGR)
CLASS_COLORS = {
    'red_box': (0, 0, 220),
    'blue_box': (220, 0, 0),
    'green_cylinder': (0, 200, 0),
    'yellow_sphere': (0, 220, 220),
    'unknown': (128, 128, 128),
}


class DetectionVisualizer:
    """
    Visualizes detection results on images.
    Supports bounding boxes, labels, confidence, tracking IDs, and 3D position overlay.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Optional visualization config from detection.yaml
        """
        self.config = config or {}
        self.box_thickness = self.config.get('box_thickness', 2)
        self.font_scale = self.config.get('font_scale', 0.6)
        self.draw_centers = self.config.get('draw_centers', True)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        track_ids: Optional[List[int]] = None,
        world_positions: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Draw all detections on an image.
        
        Args:
            image: BGR image to draw on
            detections: List of Detection objects
            track_ids: Optional list of tracking IDs (same length as detections)
            world_positions: Optional list of 3D world positions
            
        Returns:
            Annotated image (copy)
        """
        vis = image.copy()
        
        for i, det in enumerate(detections):
            color = CLASS_COLORS.get(det.class_name, CLASS_COLORS['unknown'])
            track_id = track_ids[i] if track_ids and i < len(track_ids) else None
            world_pos = world_positions[i] if world_positions and i < len(world_positions) else None
            
            vis = self._draw_single_detection(vis, det, color, track_id, world_pos)
        
        return vis
    
    def _draw_single_detection(
        self,
        image: np.ndarray,
        det: Detection,
        color: Tuple[int, int, int],
        track_id: Optional[int] = None,
        world_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw a single detection with all annotations."""
        x1, y1, x2, y2 = det.bbox
        
        # Bounding box with rounded corners effect
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Corner accents (portfolio quality)
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        thick = self.box_thickness + 1
        # Top-left
        cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Top-right
        cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, thick)
        # Bottom-left
        cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, thick)
        # Bottom-right
        cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, thick)
        
        # Build label text
        label_parts = []
        if track_id is not None:
            label_parts.append(f"#{track_id}")
        label_parts.append(det.class_name)
        label_parts.append(f"{det.confidence:.0%}")
        label_text = " | ".join(label_parts)
        
        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, self.font, self.font_scale, 1
        )
        label_y = max(y1 - 8, text_h + 5)
        
        # Semi-transparent label background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 10, label_y + 3),
            color, -1
        )
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Label text
        cv2.putText(
            image, label_text,
            (x1 + 5, label_y - 2),
            self.font, self.font_scale,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Center point
        if self.draw_centers:
            cx, cy = det.center
            cv2.circle(image, (cx, cy), 5, color, -1)
            cv2.circle(image, (cx, cy), 7, (255, 255, 255), 1)
        
        # 3D world position annotation
        if world_pos is not None:
            pos_text = f"3D: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})m"
            cv2.putText(
                image, pos_text,
                (x1, y2 + 18),
                self.font, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA
            )
        
        return image
    
    def draw_info_panel(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        additional_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw an information panel with detection statistics.
        
        Args:
            image: Image to draw on
            detection_result: Detection results
            additional_info: Extra info to display
            
        Returns:
            Image with info panel
        """
        vis = image.copy()
        
        # Info lines
        info_lines = [
            f"Frame: {detection_result.frame_id}",
            f"Detections: {detection_result.num_detections}",
            f"Inference: {detection_result.inference_time_ms:.1f}ms",
        ]
        
        # Class breakdown
        class_counts = {}
        for det in detection_result.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        for cls_name, count in class_counts.items():
            info_lines.append(f"  {cls_name}: {count}")
        
        if additional_info:
            info_lines.append("---")
            for key, value in additional_info.items():
                info_lines.append(f"{key}: {value}")
        
        # Draw panel
        panel_x, panel_y = 10, 10
        line_height = 22
        
        # Background
        panel_h = len(info_lines) * line_height + 15
        panel_w = 250
        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 5, panel_y - 5),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0), -1
        )
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
        
        # Text
        for i, line in enumerate(info_lines):
            cv2.putText(
                vis, line,
                (panel_x, panel_y + 15 + i * line_height),
                self.font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return vis
    
    def create_detection_grid(
        self,
        image: np.ndarray,
        detections: List[Detection],
        grid_size: Tuple[int, int] = (200, 200)
    ) -> np.ndarray:
        """
        Create a grid of cropped detection images.
        Useful for portfolio display.
        """
        if not detections:
            return np.zeros((*grid_size, 3), dtype=np.uint8)
        
        crops = []
        for det in detections[:4]:  # Max 4 crops
            x1, y1, x2, y2 = det.bbox
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (grid_size[0] // 2, grid_size[1] // 2))
                # Add label
                cv2.putText(
                    crop_resized, det.class_name,
                    (5, 15), self.font, 0.4,
                    CLASS_COLORS.get(det.class_name, (255, 255, 255)),
                    1, cv2.LINE_AA
                )
                crops.append(crop_resized)
        
        # Arrange in 2x2 grid
        while len(crops) < 4:
            crops.append(np.zeros_like(crops[0]) if crops else 
                        np.zeros((grid_size[1]//2, grid_size[0]//2, 3), dtype=np.uint8))
        
        row1 = np.hstack(crops[:2])
        row2 = np.hstack(crops[2:4])
        grid = np.vstack([row1, row2])
        
        return grid
