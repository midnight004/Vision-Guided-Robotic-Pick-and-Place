"""
Scene Builder - Factory configuration and object management.
Handles 8 product types sorted into 4 color-coded bins.
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("scene_builder")

# Color classification: each product maps to a color category -> bin
PRODUCT_TO_COLOR = {
    "red_box": "red",
    "red_can": "red",
    "blue_box": "blue",
    "blue_capsule": "blue",
    "green_cylinder": "green",
    "green_box": "green",
    "yellow_sphere": "yellow",
    "yellow_bottle": "yellow",
}

# Bin positions (matching scene.xml - at z=0.30 table height, release from above)
DESTINATIONS = {
    "red": np.array([0.35, -0.28, 0.35]),
    "blue": np.array([0.35, 0.28, 0.35]),
    "green": np.array([0.6, -0.28, 0.35]),
    "yellow": np.array([0.6, 0.28, 0.35]),
}


class SceneBuilder:
    """Manages product placement, sorting rules, and scene configuration."""

    @staticmethod
    def get_destination(class_name: str) -> Optional[np.ndarray]:
        """Get the bin position for a detected object class."""
        color = PRODUCT_TO_COLOR.get(class_name)
        if color and color in DESTINATIONS:
            return DESTINATIONS[color].copy()
        return None

    @staticmethod
    def get_color_category(class_name: str) -> Optional[str]:
        """Get the color category for a product."""
        return PRODUCT_TO_COLOR.get(class_name)

    @staticmethod
    def get_all_products() -> List[str]:
        return list(PRODUCT_TO_COLOR.keys())

    @staticmethod
    def generate_random_positions(count: int, x_range=(0.35, 0.65), y_range=(-0.2, 0.2), min_sep=0.06):
        positions = []
        for _ in range(count):
            for attempt in range(200):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) >= min_sep for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)]))
        return positions
