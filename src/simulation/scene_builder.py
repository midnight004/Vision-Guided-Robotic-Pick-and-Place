"""
Scene Builder - Factory sorting configuration.
Maps detected color categories to destination bins.
Known colors -> color bins; unknown colors -> trash bin.
"""

import numpy as np
from typing import Dict, Optional, List
from ..utils.logger import setup_logger

logger = setup_logger("scene_builder")

# Bin positions (must match scene.xml). Release happens above these points.
DESTINATIONS = {
    "red":    np.array([0.30, -0.30, 0.35]),
    "blue":   np.array([0.30,  0.30, 0.35]),
    "green":  np.array([0.62, -0.30, 0.35]),
    "yellow": np.array([0.62,  0.30, 0.35]),
    "trash":  np.array([0.46,  0.31, 0.35]),
}

# Color category -> bin
COLOR_TO_BIN = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "unknown": "trash",
}


class SceneBuilder:
    """Sorting rules: detected color category -> destination bin position."""

    @staticmethod
    def get_destination(color_class: str) -> Optional[np.ndarray]:
        bin_name = COLOR_TO_BIN.get(color_class, "trash")
        return DESTINATIONS[bin_name].copy()

    @staticmethod
    def get_bin_name(color_class: str) -> str:
        return COLOR_TO_BIN.get(color_class, "trash")
