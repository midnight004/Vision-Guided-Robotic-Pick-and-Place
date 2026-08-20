"""
3D Object Localizer (MuJoCo)
================================
Estimates 3D world position of detected objects using RGB-D data.
Uses the camera intrinsics and extrinsics from MuJoCo.

Pipeline: Detection + Depth -> [LOCALIZER] -> Robot Target

MuJoCo Camera Convention:
    Camera frame: X-right, Y-down, Z-forward (OpenGL-like)
    World frame: X-forward, Y-left, Z-up
    
Transformation:
    pixel (u,v) + depth -> camera point -> world point
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from ..detection.detector import Detection
from ..camera.depth_processor import DepthProcessor
from ..utils.logger import setup_logger

logger = setup_logger("localization")


@dataclass
class LocalizedObject:
    """Object with estimated 3D position."""
    class_name: str
    class_id: int
    confidence: float
    track_id: Optional[int]
    
    # 2D info
    pixel_center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    
    # 3D positions
    position_camera: np.ndarray   # [X, Y, Z] in camera frame
    position_world: np.ndarray    # [X, Y, Z] in world frame
    
    # Depth
    depth: float  # meters
    
    def to_dict(self) -> dict:
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'track_id': self.track_id,
            'position_world': self.position_world.tolist(),
            'depth': self.depth,
        }


class ObjectLocalizer:
    """
    Localizes detected objects in 3D using depth images.
    
    Pipeline:
        1. Get object center pixel (u, v) from detection
        2. Get depth d at that pixel from depth image
        3. Back-project to camera frame: P_cam = K^{-1} * [u,v,1] * d
        4. Transform to world frame using camera extrinsics
    """
    
    def __init__(self, config: dict, depth_processor: DepthProcessor, 
                 sim_env=None):
        """
        Args:
            config: Configuration dictionary
            depth_processor: DepthProcessor instance
            sim_env: SimulationEnvironment (for getting camera pose)
        """
        self.depth_processor = depth_processor
        self.sim_env = sim_env
        
        # Intrinsics
        self.K = depth_processor.K.copy()
        self.K_inv = np.linalg.inv(self.K)
        
        # Camera pose (will be updated each frame from sim)
        self.cam_pos = np.array([0.5, 0.0, 1.3])
        self.cam_rot = np.eye(3)
        self.T_cam_to_world = np.eye(4)
        
        logger.info("ObjectLocalizer initialized")
        logger.info(f"  Intrinsics: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
    
    def update_camera_transform(self, cam_pos: np.ndarray, cam_rot: np.ndarray) -> None:
        """
        Update camera-to-world transformation.
        
        MuJoCo camera convention (OpenGL-like):
            Camera looks along its -Z axis
            cam_xmat rows are the camera axes in world frame
            
            cam_xmat row 0 = camera X axis (right) in world
            cam_xmat row 1 = camera Y axis (down) in world  
            cam_xmat row 2 = camera Z axis (backward, -viewing dir) in world
            
        But for back-projection we need:
            A point in camera frame [Xc, Yc, Zc] maps to world as:
            P_world = R_cam_to_world * P_cam + t_cam
            
        The camera intrinsics project: [Xc, Yc, Zc] where Zc is depth along viewing direction.
        In OpenGL/MuJoCo, the camera looks along -Z, so the actual depth axis is -Z.
        But MuJoCo's depth renderer returns positive distance, so effectively depth = -Zc.
        
        For back-projection from pixel (u,v) + depth d:
            Xc = (u - cx) / fx * d
            Yc = (v - cy) / fy * d
            Zc = d (depth is positive, along viewing direction)
            
        Then to go to world: P_world = cam_pos + R * [Xc, -Yc, -Zc]
        (negate Y and Z because MuJoCo camera Y points up in image but
         image v goes down, and Z points backward)
         
        Actually, let's be precise. The cam_xmat gives us the camera frame.
        MuJoCo Renderer produces images with:
            - u increases rightward (= camera +X direction)
            - v increases downward (= camera -Y direction, since MuJoCo cam Y is up)
            - depth is positive distance along camera -Z (viewing direction)
            
        So pixel (u,v) with depth d corresponds to camera-frame point:
            Xc = (u - cx) / fx * d   (right)
            Yc = -(v - cy) / fy * d  (up, note negation because v is flipped)
            Zc = -d                   (forward = -Z in camera frame)
            
        Then: P_world = cam_pos + cam_xmat^T * [Xc, Yc, Zc]
        (cam_xmat^T because cam_xmat rows are camera axes in world)
        """
        # cam_rot is the 3x3 matrix where rows = camera axes in world coords
        # To transform from camera frame to world: multiply by cam_rot.T
        # (cam_rot.T columns = camera axes in world = maps camera coords to world)
        
        self.cam_pos = cam_pos.copy()
        self.cam_rot = cam_rot.copy()  # rows = camera X,Y,Z in world
        
        # Build 4x4 transform
        # cam_rot rows are camera axes in world, so cam_rot.T converts cam->world
        self.T_cam_to_world = np.eye(4)
        self.T_cam_to_world[:3, :3] = cam_rot.T  # transpose to get cam-to-world rotation
        self.T_cam_to_world[:3, 3] = cam_pos
    
    def localize_detection(
        self,
        detection: Detection,
        depth_image: np.ndarray,
        track_id: Optional[int] = None,
    ) -> Optional[LocalizedObject]:
        """
        Localize a single detection in 3D.
        
        Args:
            detection: Detection with bbox
            depth_image: (H, W) depth in meters
            track_id: Optional tracking ID
            
        Returns:
            LocalizedObject or None
        """
        # Get depth at detection center
        depth = self.depth_processor.get_depth_at_bbox(
            depth_image, detection.bbox, method="center_region"
        )
        
        if depth <= 0:
            return None
        
        # Back-project to camera frame using MuJoCo conventions
        u, v = detection.center
        
        # In MuJoCo's rendered image:
        #   u (rightward) corresponds to camera +X
        #   v (downward) corresponds to camera -Y (cam Y points up)
        #   depth is along camera -Z (cam Z points backward)
        
        # Camera frame point:
        Xc = (u - self.K[0, 2]) / self.K[0, 0] * depth   # right
        Yc = -(v - self.K[1, 2]) / self.K[1, 1] * depth  # up (negated because v is down)
        Zc = -depth                                         # forward (negated because cam Z is backward)
        
        point_camera = np.array([Xc, Yc, Zc])
        
        # Transform to world frame
        # cam_rot.T converts camera coordinates to world coordinates
        point_world = self.cam_pos + self.cam_rot.T @ point_camera
        
        return LocalizedObject(
            class_name=detection.class_name,
            class_id=detection.class_id,
            confidence=detection.confidence,
            track_id=track_id,
            pixel_center=detection.center,
            bbox=detection.bbox,
            position_camera=point_camera,
            position_world=point_world,
            depth=depth,
        )
    
    def localize_all(
        self,
        detections: List[Detection],
        depth_image: np.ndarray,
        track_ids: Optional[List[int]] = None,
    ) -> List[LocalizedObject]:
        """
        Localize all detections in a frame.
        Filters out objects not in the pick zone (rejects objects in bins or off table).
        
        Pick zone: center of table where objects spawn
            x: [0.35, 0.6], y: [-0.18, 0.18], z: [0.3, 0.45]
        """
        # Update camera transform
        if self.sim_env is not None:
            cam_pos, cam_rot = self.sim_env.get_camera_extrinsics()
            self.update_camera_transform(cam_pos, cam_rot)
        
        localized = []
        for i, det in enumerate(detections):
            tid = track_ids[i] if track_ids and i < len(track_ids) else None
            obj = self.localize_detection(det, depth_image, tid)
            if obj is not None:
                # PICK ZONE FILTER:
                # Only accept objects in the central pick area
                # This prevents picking objects already placed in bins
                px, py, pz = obj.position_world
                in_pick_zone = (
                    0.35 <= px <= 0.60 and
                    -0.18 <= py <= 0.18 and
                    0.30 <= pz <= 0.45
                )
                if in_pick_zone:
                    localized.append(obj)
        
        return localized
