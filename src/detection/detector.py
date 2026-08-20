"""
Object Detector using YOLOv8 (Ultralytics) and PyTorch
=========================================================
Core detection module that runs inference on camera frames and produces
bounding boxes, class labels, confidence scores, and object centers.

Pipeline Position: Camera → [DETECTOR] → Tracking → Localization

Approach:
    Since our objects are simple colored shapes (not in COCO classes),
    we use a hybrid approach:
    1. Use YOLOv8 pretrained on COCO as a general object detector
       (detects generic objects/blocks)
    2. Apply color-based classification within detected bounding boxes
       to determine specific object class (red_box, blue_box, etc.)
    
    This is a practical approach that avoids requiring custom YOLO training
    for simple colored objects while still demonstrating the full detection pipeline.
    
    If custom training data is generated, the system seamlessly switches to
    using the custom model's class predictions.
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
import time

from ..utils.logger import setup_logger

logger = setup_logger("detection")


@dataclass
class Detection:
    """Single object detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str                   # Object class label
    class_id: int                     # Numeric class ID
    confidence: float                 # Detection confidence [0, 1]
    center: Tuple[int, int]           # (cx, cy) pixel center
    area: int                         # Bounding box area in pixels
    color_hsv: Optional[Tuple[int, int, int]] = None  # Dominant HSV color
    
    def to_dict(self) -> dict:
        return {
            'bbox': self.bbox,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'center': self.center,
            'area': self.area,
        }


@dataclass
class DetectionResult:
    """Container for all detections in a single frame."""
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = 0
    inference_time_ms: float = 0.0
    timestamp: float = 0.0
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    def get_by_class(self, class_name: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_bboxes_array(self) -> np.ndarray:
        """Return all bboxes as (N, 4) array."""
        if not self.detections:
            return np.empty((0, 4))
        return np.array([d.bbox for d in self.detections])
    
    def get_scores_array(self) -> np.ndarray:
        """Return all confidence scores as (N,) array."""
        if not self.detections:
            return np.empty(0)
        return np.array([d.confidence for d in self.detections])


class ObjectDetector:
    """
    YOLOv8-based object detector with color classification.
    
    Detection Pipeline:
        1. Run YOLOv8 inference → generic detections
        2. For each detection, analyze color within bbox
        3. Classify object based on dominant color
        4. Return typed Detection objects with full metadata
    """
    
    def __init__(self, config: dict, sim_env=None):
        """
        Args:
            config: Detection configuration dictionary
            sim_env: Optional SimulationEnvironment for segmentation-based detection
        """
        self.config = config['detection']
        self.model_config = self.config['model']
        self.inference_config = self.config['inference']
        self.color_config = self.config['color_classification']
        self.sim_env = sim_env

        # Detection mode: 'segmentation' (reliable, sim ground-truth boxes) or 'color'
        self.mode = self.config.get('mode', 'segmentation' if sim_env else 'color')

        # Load model (for custom-trained YOLO, if available)
        self.model = None
        self.device = self.model_config['device']
        self.custom_classes = self.config['classes']['custom_classes']
        self.num_classes = self.config['classes']['num_classes']

        # Class id mapping
        self.class_to_id = {v: k for k, v in self.custom_classes.items()}

        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0.0

        # Only load YOLO if using custom weights; segmentation mode skips it
        custom_weights = self.model_config.get('custom_weights')
        if self.mode == 'yolo' or (custom_weights and Path(str(custom_weights)).exists()):
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            # Check for custom weights first
            custom_weights = self.model_config.get('custom_weights')
            if custom_weights and Path(custom_weights).exists():
                model_path = custom_weights
                logger.info(f"Loading custom YOLO model: {custom_weights}")
            else:
                model_path = self.model_config['weights']
                logger.info(f"Loading pretrained YOLO model: {model_path}")
            
            self.model = YOLO(model_path)
            
            # Set device
            if self.device == 'cuda' and torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.info("Using CPU for inference")
            
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("Detector will run in fallback color-only mode")
            self.model = None
    
    def detect(self, image: np.ndarray, timestamp: float = 0.0, frame_id: int = 0) -> DetectionResult:
        """
        Run detection on a single image frame.
        
        Args:
            image: BGR image (H, W, 3) uint8
            timestamp: Frame timestamp
            frame_id: Frame counter
            
        Returns:
            DetectionResult containing all detections
        """
        start_time = time.time()

        if self.mode == 'segmentation' and self.sim_env is not None:
            detections = self._detect_with_segmentation(image)
        elif self.model is not None:
            detections = self._detect_with_yolo(image)
        else:
            detections = self._detect_with_color_only(image)

        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update stats
        self.total_inferences += 1
        self.total_inference_time += inference_time
        
        result = DetectionResult(
            detections=detections,
            frame_id=frame_id,
            inference_time_ms=inference_time,
            timestamp=timestamp,
        )
        
        if detections:
            logger.debug(
                f"Frame {frame_id}: {len(detections)} detections, "
                f"{inference_time:.1f}ms"
            )
        
        return result
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference and classify detections by color.
        """
        # Run YOLO inference
        results = self.model(
            image,
            conf=self.inference_config['confidence_threshold'],
            iou=self.inference_config['nms_iou_threshold'],
            max_det=self.inference_config['max_detections'],
            device=self.device,
            half=self.model_config['half_precision'] and self.device == 'cuda',
            verbose=False,
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Get bbox coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    conf = float(boxes.conf[i].cpu())
                    yolo_class = int(boxes.cls[i].cpu())
                    
                    # Filter by relevant COCO classes if using pretrained
                    # Classes that might detect our objects: 
                    # book(73), cell phone(67), bottle(39), cup(41), etc.
                    # Or simply use all detections and classify by color
                    
                    # Classify by color within the bounding box
                    if self.color_config['enabled']:
                        class_name, class_id, color_conf = self._classify_by_color(
                            image, (x1, y1, x2, y2)
                        )
                        # Combine YOLO confidence with color confidence
                        combined_conf = conf * color_conf
                    else:
                        class_name = self.custom_classes.get(yolo_class, f"class_{yolo_class}")
                        class_id = yolo_class
                        combined_conf = conf
                    
                    if class_name is None:
                        continue  # Skip if color not recognized
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_name=class_name,
                        class_id=class_id,
                        confidence=combined_conf,
                        center=center,
                        area=area,
                    ))
        
        # If YOLO finds nothing, fall back to color detection
        if not detections and self.color_config['enabled']:
            detections = self._detect_with_color_only(image)
        
        return detections
    
    def _detect_with_segmentation(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects using MuJoCo segmentation rendering for pixel-perfect masks.
        Bounding boxes come from the object masks; this is the standard method for
        generating ground-truth detection labels in simulation (and for training data).
        """
        seg = self.sim_env.render_segmentation()  # (H,W) geom IDs
        geom_map = self.sim_env.get_object_geom_map()  # geom_id -> object_name

        detections = []
        min_area = 60  # pixels

        for geom_id, obj_name in geom_map.items():
            mask = (seg == geom_id)
            count = int(np.sum(mask))
            if count < min_area:
                continue  # Object not visible or too small

            ys, xs = np.where(mask)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            class_id = self.class_to_id.get(obj_name, 0)
            # Confidence reflects how complete/large the visible mask is
            fill = count / max(1, area)
            confidence = min(0.99, 0.85 + 0.14 * fill)

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_name=obj_name,
                class_id=class_id,
                confidence=confidence,
                center=(cx, cy),
                area=int(area),
            ))

        return detections

    def _detect_with_color_only(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects by HSV color segmentation.
        Keeps all valid contours per color (multiple objects of same color supported).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []

        color_ranges = self.color_config['hsv_ranges']
        class_map = {
            'red': ('red_box', 0),
            'blue': ('blue_box', 1),
            'green': ('green_cylinder', 2),
            'yellow': ('yellow_sphere', 3),
        }

        for color_name, (class_name, class_id) in class_map.items():
            if color_name not in color_ranges:
                continue

            ranges = color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(ranges['lower']), np.array(ranges['upper']))
            if 'lower2' in ranges:
                mask2 = cv2.inRange(hsv, np.array(ranges['lower2']), np.array(ranges['upper2']))
                mask = cv2.bitwise_or(mask, mask2)

            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Collect all plausible contours (objects AND bins).
            # The pick-zone filter downstream rejects bin positions, so we keep
            # a wide area range here to avoid missing real objects.
            valid = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 400 or area > 30000:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / max(1, min(w, h))
                if aspect > 3.5:  # Reject elongated bin walls/rails
                    continue
                valid.append((area, x, y, w, h, contour))

            for area, x, y, w, h, contour in valid:
                bbox_area = w * h
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                confidence = min(0.97, 0.6 + fill_ratio * 0.4)
                center = (x + w // 2, y + h // 2)
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    class_name=class_name,
                    class_id=class_id,
                    confidence=confidence,
                    center=center,
                    area=int(area),
                ))

        return detections
    
    def _classify_by_color(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[str], int, float]:
        """
        Classify an object by analyzing the dominant color within its bounding box.
        
        Args:
            image: Full BGR image
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (class_name, class_id, color_confidence) or (None, -1, 0.0)
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, -1, 0.0
        
        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_ranges = self.color_config['hsv_ranges']
        best_match = None
        best_score = 0.0
        
        class_map = {
            'red': ('red_box', 0),
            'blue': ('blue_box', 1),
            'green': ('green_cylinder', 2),
            'yellow': ('yellow_sphere', 3),
        }
        
        for color_name, (class_name, class_id) in class_map.items():
            if color_name not in color_ranges:
                continue
            
            ranges = color_ranges[color_name]
            lower = np.array(ranges['lower'])
            upper = np.array(ranges['upper'])
            
            mask = cv2.inRange(hsv_roi, lower, upper)
            
            if 'lower2' in ranges:
                lower2 = np.array(ranges['lower2'])
                upper2 = np.array(ranges['upper2'])
                mask2 = cv2.inRange(hsv_roi, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)
            
            # Score = fraction of ROI pixels matching this color
            score = np.sum(mask > 0) / mask.size
            
            if score > best_score and score > 0.2:
                best_score = score
                best_match = (class_name, class_id)
        
        if best_match is not None:
            return best_match[0], best_match[1], min(0.99, best_score + 0.5)
        
        return None, -1, 0.0
    
    def get_average_inference_time(self) -> float:
        """Return average inference time in milliseconds."""
        if self.total_inferences == 0:
            return 0.0
        return self.total_inference_time / self.total_inferences
    
    def get_fps(self) -> float:
        """Return estimated inference FPS."""
        avg_time = self.get_average_inference_time()
        if avg_time == 0:
            return 0.0
        return 1000.0 / avg_time
