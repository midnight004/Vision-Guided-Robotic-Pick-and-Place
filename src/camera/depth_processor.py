"""
Depth Image Processor
======================
Processes depth images for 3D point estimation and visualization.
Provides utilities for depth-to-3D conversion used by the localization module.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

from ..utils.logger import setup_logger

logger = setup_logger("depth_processor")


class DepthProcessor:
    """
    Processes depth images and provides 3D point computation from pixel coordinates.
    """
    
    def __init__(self, intrinsic_matrix: np.ndarray, depth_range: Tuple[float, float] = (0.1, 5.0)):
        """
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix
            depth_range: (min, max) valid depth range in meters
        """
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
        self.depth_min, self.depth_max = depth_range
    
    def pixel_to_camera_point(
        self,
        u: int,
        v: int,
        depth: float
    ) -> np.ndarray:
        """
        Convert a single pixel coordinate + depth to a 3D point in camera frame.
        
        Uses the pinhole camera model:
            X_cam = (u - cx) * depth / fx
            Y_cam = (v - cy) * depth / fy
            Z_cam = depth
        
        Args:
            u: Pixel x-coordinate (column)
            v: Pixel y-coordinate (row)
            depth: Depth value in meters
            
        Returns:
            3D point [X, Y, Z] in camera coordinate frame (meters)
        """
        if depth <= 0 or depth < self.depth_min or depth > self.depth_max:
            logger.warning(f"Invalid depth {depth:.3f}m at pixel ({u}, {v})")
            return np.array([0.0, 0.0, 0.0])
        
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        
        return np.array([X, Y, Z])
    
    def get_depth_at_bbox(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        method: str = "center_region"
    ) -> float:
        """
        Extract a robust depth estimate for an object given its bounding box.
        
        Args:
            depth_image: (H, W) depth image in meters
            bbox: (x1, y1, x2, y2) bounding box
            method: Depth estimation method:
                - "center_pixel": Single center pixel
                - "center_region": Median of center 25% region
                - "min_region": Minimum depth in center region (closest point)
                
        Returns:
            Estimated depth in meters
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Ensure bounds are valid
        h, w = depth_image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox after clipping: ({x1},{y1},{x2},{y2})")
            return 0.0
        
        if method == "center_pixel":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return float(depth_image[cy, cx])
        
        elif method == "center_region":
            # Use center 50% of bounding box for robustness
            margin_x = (x2 - x1) // 4
            margin_y = (y2 - y1) // 4
            
            region = depth_image[
                y1 + margin_y : y2 - margin_y,
                x1 + margin_x : x2 - margin_x
            ]
            
            # Filter valid depths
            valid = region[(region > self.depth_min) & (region < self.depth_max)]
            
            if len(valid) == 0:
                logger.warning(f"No valid depth in bbox center region")
                return 0.0
            
            return float(np.median(valid))
        
        elif method == "min_region":
            # Center region, take minimum (closest surface point)
            margin_x = (x2 - x1) // 4
            margin_y = (y2 - y1) // 4
            
            region = depth_image[
                y1 + margin_y : y2 - margin_y,
                x1 + margin_x : x2 - margin_x
            ]
            
            valid = region[(region > self.depth_min) & (region < self.depth_max)]
            
            if len(valid) == 0:
                return 0.0
            
            return float(np.min(valid))
        
        else:
            raise ValueError(f"Unknown depth method: {method}")
    
    def depth_to_colormap(
        self,
        depth_image: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Convert depth image to a colored visualization.
        
        Args:
            depth_image: (H, W) depth in meters
            colormap: OpenCV colormap constant
            
        Returns:
            (H, W, 3) BGR colorized depth image
        """
        # Normalize to 0-255 range
        valid_mask = (depth_image > self.depth_min) & (depth_image < self.depth_max)
        
        depth_norm = np.zeros_like(depth_image)
        if valid_mask.any():
            d_min = depth_image[valid_mask].min()
            d_max = depth_image[valid_mask].max()
            if d_max > d_min:
                depth_norm[valid_mask] = (
                    (depth_image[valid_mask] - d_min) / (d_max - d_min) * 255
                )
        
        depth_uint8 = depth_norm.astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, colormap)
        
        # Black out invalid pixels
        colored[~valid_mask] = 0
        
        return colored
    
    def compute_point_cloud(
        self,
        depth_image: np.ndarray,
        downsample: int = 4
    ) -> np.ndarray:
        """
        Compute a point cloud from the full depth image.
        
        Args:
            depth_image: (H, W) depth in meters
            downsample: Skip factor for speed
            
        Returns:
            (N, 3) array of 3D points in camera frame
        """
        h, w = depth_image.shape
        
        # Create pixel grid
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Get depths at grid points
        depths = depth_image[v_grid, u_grid]
        
        # Mask invalid
        valid = (depths > self.depth_min) & (depths < self.depth_max)
        
        u_valid = u_grid[valid].astype(np.float64)
        v_valid = v_grid[valid].astype(np.float64)
        d_valid = depths[valid].astype(np.float64)
        
        # Back-project to 3D
        X = (u_valid - self.cx) * d_valid / self.fx
        Y = (v_valid - self.cy) * d_valid / self.fy
        Z = d_valid
        
        points = np.stack([X, Y, Z], axis=-1)
        
        return points
