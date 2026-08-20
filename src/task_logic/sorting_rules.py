"""
Sorting Rules - Maps products to destination bins by color.
8 products -> 4 color bins (red, blue, green, yellow)
"""

import numpy as np
from typing import Dict, Optional
from ..simulation.scene_builder import SceneBuilder, PRODUCT_TO_COLOR, DESTINATIONS
from ..utils.logger import setup_logger

logger = setup_logger("sorting_rules")


class SortingRules:
    def __init__(self, config: dict = None):
        self.rules = PRODUCT_TO_COLOR.copy()
        self.destinations = DESTINATIONS.copy()

        logger.info("Sorting rules:")
        for product, color in self.rules.items():
            logger.info(f"  {product} -> {color} bin")

    def get_destination(self, class_name: str) -> Optional[np.ndarray]:
        return SceneBuilder.get_destination(class_name)

    def get_color(self, class_name: str) -> Optional[str]:
        return PRODUCT_TO_COLOR.get(class_name)

    def reset(self) -> None:
        pass
