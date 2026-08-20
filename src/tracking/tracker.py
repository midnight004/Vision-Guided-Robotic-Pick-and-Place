"""
Multi-Object Tracker (ByteTrack-inspired)
===========================================
Maintains object identity across frames using IoU-based association
and Kalman filtering for motion prediction.

Pipeline Position: Detection → [TRACKER] → Localization

Features:
    - ID assignment and management
    - Kalman filter for motion prediction
    - IoU-based association (Hungarian algorithm)
    - Track lifecycle management (tentative → confirmed → lost → deleted)
    - Modular enable/disable for static scenes
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import deque

from ..detection.detector import Detection, DetectionResult
from ..utils.logger import setup_logger

logger = setup_logger("tracking")


@dataclass
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    state: str = "tentative"  # tentative, confirmed, lost
    hits: int = 0             # Number of successful associations
    age: int = 0              # Total frames since creation
    time_since_update: int = 0  # Frames since last detection match
    
    # Motion state (Kalman filter state)
    # State: [cx, cy, w, h, vx, vy, vw, vh]
    kf_state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    kf_covariance: np.ndarray = field(default_factory=lambda: np.eye(8) * 10)
    
    # History
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    @property
    def center(self) -> Tuple[int, int]:
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        return (cx, cy)
    
    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"


class KalmanBoxTracker:
    """
    Simple Kalman filter for bounding box tracking.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    
    # Process noise
    Q_std = np.array([1, 1, 1, 1, 5, 5, 2, 2], dtype=np.float64)
    
    # Measurement noise
    R_std = np.array([2, 2, 5, 5], dtype=np.float64)
    
    def __init__(self):
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1  # cx += vx
        self.F[1, 5] = 1  # cy += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        
        # Observation matrix
        self.H = np.eye(4, 8, dtype=np.float64)
        
        # Process noise covariance
        self.Q = np.diag(self.Q_std ** 2)
        
        # Measurement noise covariance
        self.R = np.diag(self.R_std ** 2)
    
    def predict(self, state: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        state_pred = self.F @ state
        P_pred = self.F @ P @ self.F.T + self.Q
        return state_pred, P_pred
    
    def update(self, state: np.ndarray, P: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement."""
        # Innovation
        y = measurement - self.H @ state
        S = self.H @ P @ self.H.T + self.R
        
        # Kalman gain
        K = P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state
        state_new = state + K @ y
        P_new = (np.eye(8) - K @ self.H) @ P
        
        return state_new, P_new


class ObjectTracker:
    """
    Multi-object tracker using IoU association and Kalman filtering.
    Inspired by ByteTrack for robustness.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Tracking configuration dictionary
        """
        self.config = config['tracking']
        self.enabled = self.config['enabled']
        
        # ByteTrack params
        bt_cfg = self.config['bytetrack']
        self.track_thresh = bt_cfg['track_thresh']
        self.track_buffer = bt_cfg['track_buffer']
        self.match_thresh = bt_cfg['match_thresh']
        self.min_box_area = bt_cfg['min_box_area']
        
        # Track management params
        mgmt_cfg = self.config['management']
        self.max_age = mgmt_cfg['max_age']
        self.min_hits = mgmt_cfg['min_hits']
        self.iou_threshold = mgmt_cfg['iou_threshold']
        
        # Track storage
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Kalman filter
        self.kf = KalmanBoxTracker()
        
        logger.info(f"Tracker initialized (enabled={self.enabled})")
    
    def update(self, detection_result: DetectionResult) -> List[Tuple[Detection, int]]:
        """
        Update tracker with new detections.
        
        Args:
            detection_result: DetectionResult from detector
            
        Returns:
            List of (Detection, track_id) tuples for confirmed tracks
        """
        if not self.enabled:
            # Pass-through: assign temporary IDs
            return [(det, i + 1) for i, det in enumerate(detection_result.detections)]
        
        self.frame_count += 1
        detections = detection_result.detections
        
        # Predict existing tracks forward
        self._predict_tracks()
        
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matched:
            self._update_track(self.tracks[track_idx], detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det.area >= self.min_box_area:
                self._create_track(det)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
            if self.tracks[track_idx].time_since_update > self.max_age:
                self.tracks[track_idx].state = "lost"
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.state != "lost"]
        
        # Return confirmed tracks with their detections
        result = []
        for det_idx, track_idx in matched:
            track = self.tracks[track_idx]
            if track.is_confirmed or track.hits >= 1:
                result.append((detections[det_idx], track.track_id))
        
        # Also include newly created tracks (first frame scenario)
        if not result and detections:
            for track in self.tracks:
                if track.time_since_update == 0:
                    for det in detections:
                        if self._iou(np.array(det.bbox), track.bbox) > 0.5:
                            if not any(d.bbox == det.bbox for d, _ in result):
                                result.append((det, track.track_id))
                            break
        
        return result
    
    def _predict_tracks(self) -> None:
        """Predict all tracks forward one step using Kalman filter."""
        for track in self.tracks:
            track.kf_state, track.kf_covariance = self.kf.predict(
                track.kf_state, track.kf_covariance
            )
            # Update bbox from predicted state
            cx, cy, w, h = track.kf_state[:4]
            track.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            track.age += 1
    
    def _associate(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """
        Associate detections to existing tracks using IoU.
        Uses Hungarian algorithm for optimal assignment.
        """
        if not self.tracks or not detections:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            return [], unmatched_dets, unmatched_tracks
        
        # Compute IoU cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets))
        
        for t in range(num_tracks):
            for d in range(num_dets):
                iou_matrix[t, d] = self._iou(
                    self.tracks[t].bbox,
                    np.array(detections[d].bbox)
                )
        
        # Hungarian algorithm (minimize cost = 1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        if min(cost_matrix.shape) > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        else:
            track_indices = np.empty(0, dtype=int)
            det_indices = np.empty(0, dtype=int)
        
        # Filter by IoU threshold
        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(num_tracks))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched.append((d_idx, t_idx))
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _update_track(self, track: Track, detection: Detection) -> None:
        """Update a track with a matched detection."""
        bbox = np.array(detection.bbox, dtype=np.float64)
        
        # Measurement: [cx, cy, w, h]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([cx, cy, w, h])
        
        # Kalman update
        track.kf_state, track.kf_covariance = self.kf.update(
            track.kf_state, track.kf_covariance, measurement
        )
        
        # Update track attributes
        track.bbox = bbox
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.bbox_history.append(bbox.copy())
        
        # Promote to confirmed
        if track.hits >= self.min_hits:
            track.state = "confirmed"
    
    def _create_track(self, detection: Detection) -> None:
        """Create a new track from an unmatched detection."""
        bbox = np.array(detection.bbox, dtype=np.float64)
        
        # Initialize Kalman state
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        
        track = Track(
            track_id=self.next_id,
            bbox=bbox,
            class_name=detection.class_name,
            confidence=detection.confidence,
            kf_state=state,
            kf_covariance=np.eye(8) * 10,
        )
        track.hits = 1
        track.bbox_history.append(bbox.copy())
        
        self.tracks.append(track)
        self.next_id += 1
        
        logger.debug(f"New track #{track.track_id}: {detection.class_name}")
    
    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def reset(self) -> None:
        """Reset tracker state for a new episode."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
    
    def get_active_tracks(self) -> List[Track]:
        """Return all confirmed active tracks."""
        return [t for t in self.tracks if t.is_confirmed]
    
    def get_track_count(self) -> int:
        """Return number of active confirmed tracks."""
        return sum(1 for t in self.tracks if t.is_confirmed)
