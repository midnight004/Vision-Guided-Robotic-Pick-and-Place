"""
Visualization utilities for the perception pipeline.
Provides drawing functions for debug and portfolio demonstration.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def draw_3d_axes(
    image: np.ndarray,
    origin: Tuple[int, int],
    rotation_matrix: np.ndarray,
    camera_matrix: np.ndarray,
    axis_length: float = 0.1,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw 3D coordinate axes on an image for visualization.
    
    Args:
        image: Input image (BGR)
        origin: 2D pixel origin point
        rotation_matrix: 3x3 rotation matrix
        camera_matrix: 3x3 camera intrinsic matrix
        axis_length: Length of axes in meters
        thickness: Line thickness
        
    Returns:
        Image with drawn axes
    """
    img = image.copy()
    
    # Define 3D axis endpoints
    axes_3d = np.float32([
        [axis_length, 0, 0],   # X - Red
        [0, axis_length, 0],   # Y - Green
        [0, 0, axis_length],   # Z - Blue
    ])
    
    # Project to 2D (simplified for overlay purposes)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB -> BGR
    labels = ['X', 'Y', 'Z']
    
    for i, (axis_pt, color, label) in enumerate(zip(axes_3d, colors, labels)):
        # Simple projection for visualization
        end_pt = (
            int(origin[0] + axis_pt[0] * 500),
            int(origin[1] - axis_pt[1] * 500)
        )
        cv2.arrowedLine(img, origin, end_pt, color, thickness, tipLength=0.2)
        cv2.putText(img, label, end_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, thickness)
    
    return img


def overlay_info(
    image: np.ndarray,
    info_lines: List[str],
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    alpha: float = 0.7
) -> np.ndarray:
    """
    Overlay text information on an image with semi-transparent background.
    
    Args:
        image: Input image (BGR)
        info_lines: List of text strings to display
        position: Top-left corner of text block
        font_scale: Font size
        color: Text color (BGR)
        bg_color: Background color (BGR), None for no background
        alpha: Background transparency
        
    Returns:
        Image with text overlay
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = int(25 * font_scale / 0.5)
    
    if bg_color is not None and info_lines:
        # Calculate background rectangle
        max_width = 0
        for line in info_lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)
        
        total_height = line_height * len(info_lines) + 10
        x, y = position
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 5, y - line_height),
                      (x + max_width + 10, y + total_height - line_height),
                      bg_color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw text lines
    x, y = position
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img


def draw_detection_overlay(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    track_id: Optional[int] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a single detection with bounding box, label, and confidence.
    
    Args:
        image: Input image (BGR)
        bbox: (x1, y1, x2, y2) bounding box
        label: Object class name
        confidence: Detection confidence [0, 1]
        track_id: Optional tracking ID
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with detection overlay
    """
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Build label text
    text = f"{label} {confidence:.0%}"
    if track_id is not None:
        text = f"#{track_id} {text}"
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, 1
    )
    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1),
                  color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 5),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw center point
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), 4, color, -1)
    
    return img


def draw_world_position(
    image: np.ndarray,
    pixel_center: Tuple[int, int],
    world_pos: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """
    Annotate an object with its estimated 3D world position.
    
    Args:
        image: Input image (BGR)
        pixel_center: (x, y) pixel center of object
        world_pos: [x, y, z] world position in meters
        color: Text color (BGR)
        
    Returns:
        Image with position annotation
    """
    img = image.copy()
    cx, cy = pixel_center
    
    # Draw position text below the object
    pos_text = f"({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})m"
    cv2.putText(img, pos_text, (cx - 80, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    return img


def create_pipeline_display(
    camera_view: np.ndarray,
    detection_view: np.ndarray,
    info_panel: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """
    Create a composite display showing the full pipeline for demonstration.
    
    Args:
        camera_view: Raw camera image
        detection_view: Image with detection overlays
        info_panel: Optional info panel image
        target_size: Output resolution (width, height)
        
    Returns:
        Composite display image
    """
    h, w = target_size[1], target_size[0]
    
    # Resize views
    half_w = w // 2
    half_h = h if info_panel is None else h * 2 // 3
    
    cam_resized = cv2.resize(camera_view, (half_w, half_h))
    det_resized = cv2.resize(detection_view, (half_w, half_h))
    
    # Combine horizontally
    top_row = np.hstack([cam_resized, det_resized])
    
    if info_panel is not None:
        panel_h = h - half_h
        panel_resized = cv2.resize(info_panel, (w, panel_h))
        display = np.vstack([top_row, panel_resized])
    else:
        display = top_row
    
    # Add title bar
    cv2.putText(display, "RAW CAMERA", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "DETECTION + TRACKING", (half_w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display
