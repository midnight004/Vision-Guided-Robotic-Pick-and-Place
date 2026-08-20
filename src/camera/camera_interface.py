"""
Camera Interface (MuJoCo)
===========================
Wraps MuJoCo's built-in rendering to provide RGB and depth images.
Computes camera intrinsics from MuJoCo's field-of-view parameters.

Pipeline Position: Camera -> [Detection, Localization]
"""

import numpy as np
import cv2
import mujoco
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger("camera")


@dataclass
class CameraFrame:
    """Container for a single camera frame with RGB and depth data."""
    rgb: np.ndarray              # (H, W, 3) uint8 BGR image (OpenCV format)
    depth: Optional[np.ndarray]  # (H, W) float32 depth in meters
    timestamp: float             # Simulation time
    frame_id: int                # Frame counter
    
    @property
    def height(self) -> int:
        return self.rgb.shape[0]
    
    @property
    def width(self) -> int:
        return self.rgb.shape[1]
    
    @property
    def has_depth(self) -> bool:
        return self.depth is not None


class CameraInterface:
    """
    Interface to MuJoCo's camera rendering system.
    Provides synchronized RGB-D frames for the perception pipeline.
    
    MuJoCo Camera Model:
        - Pinhole camera with vertical field of view (fovy)
        - Square pixels (fx == fy)
        - No lens distortion (simulation)
        - Position and orientation defined in scene XML
    """
    
    def __init__(self, sim_env, camera_name: str = "overhead_cam"):
        """
        Args:
            sim_env: SimulationEnvironment instance
            camera_name: Name of camera in MuJoCo scene XML
        """
        self.sim_env = sim_env
        self.camera_name = camera_name
        self.width = sim_env.render_width
        self.height = sim_env.render_height
        self.frame_count = 0
        
        # Compute intrinsic matrix from MuJoCo camera parameters
        self.intrinsic_matrix = sim_env.get_camera_intrinsics()
        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        
        # No distortion in simulation
        self.distortion_coeffs = np.zeros(5, dtype=np.float64)
        
        # Depth settings
        self.depth_enabled = True
        self.depth_min = 0.05   # meters
        self.depth_max = 5.0    # meters
        
        logger.info(f"Camera initialized: {self.width}x{self.height}")
        logger.info(f"  Name: {camera_name}")
        logger.info(f"  Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        logger.info(f"  Depth: enabled, range=[{self.depth_min}, {self.depth_max}]m")
    
    def capture_frame(self, sim_time: float = 0.0) -> CameraFrame:
        """
        Capture a single RGB-D frame from the MuJoCo camera.
        
        Args:
            sim_time: Current simulation timestamp
            
        Returns:
            CameraFrame with RGB (BGR format) and depth data
        """
        # Render RGB
        rgb = self.sim_env.render_rgb(camera=self.camera_name)
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Render depth
        depth = None
        if self.depth_enabled:
            raw_depth = self.sim_env.render_depth(camera=self.camera_name)
            depth = self._process_depth(raw_depth)
        
        self.frame_count += 1
        
        return CameraFrame(
            rgb=bgr,
            depth=depth,
            timestamp=sim_time,
            frame_id=self.frame_count,
        )
    
    def _process_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """
        Process raw MuJoCo depth buffer.
        
        MuJoCo returns depth as distance from camera to surface point
        along the camera ray (not Z-buffer depth). Values are in meters.
        
        Args:
            raw_depth: Raw depth from MuJoCo renderer
            
        Returns:
            Processed depth image (meters), invalid pixels = 0
        """
        depth = raw_depth.astype(np.float32)
        
        # Clip to valid range
        invalid = (depth < self.depth_min) | (depth > self.depth_max)
        depth[invalid] = 0.0
        
        return depth
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K."""
        return self.intrinsic_matrix.copy()
    
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and rotation in world frame.
        
        Returns:
            (position [3], rotation_matrix [3x3]) - camera to world
        """
        return self.sim_env.get_camera_extrinsics()
    
    def get_camera_info(self) -> Dict:
        """Return camera info dictionary."""
        pos, rot = self.get_camera_pose()
        return {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'K': self.intrinsic_matrix.flatten().tolist(),
            'D': self.distortion_coeffs.tolist(),
            'position': pos.tolist(),
            'rotation': rot.tolist(),
            'frame_id': 'camera_frame',
        }
