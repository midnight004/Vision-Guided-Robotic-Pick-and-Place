"""
Sorting Rules
===============
Defines the mapping from detected object classes to their destination positions.
This is the "intelligence" layer that determines WHERE each object should go.

Task: Color-based sorting
    - red_box → red_destination (left)
    - blue_box → blue_destination (right)
    - green_cylinder → green_destination (far)
    - yellow_sphere → red_destination (grouped with red)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger("sorting_rules")


@dataclass
class Destination:
    """Defines a placement destination."""
    name: str
    position: np.ndarray  # [x, y, z] world coordinates
    color: np.ndarray     # [r, g, b] for visualization
    capacity: int = 5     # Max objects
    current_count: int = 0


class SortingRules:
    """
    Maps detected object classes to destination positions.
    Implements the decision logic for the pick-and-place task.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Task and simulation configuration
        """
        task_cfg = config['task']
        sim_cfg = config['simulation']
        
        # Load sorting rules
        self.rules: Dict[str, str] = task_cfg['sorting_rules']
        
        # Load destination positions from simulation config
        place_areas = sim_cfg['workspace']['place_areas']
        self.destinations: Dict[str, Destination] = {}
        
        for dest_name, dest_info in place_areas.items():
            self.destinations[dest_name] = Destination(
                name=dest_name,
                position=np.array(dest_info['center']),
                color=np.array(dest_info['color']),
            )
        
        logger.info("Sorting rules loaded:")
        for obj_class, dest in self.rules.items():
            dest_pos = self.destinations[dest].position if dest in self.destinations else "UNKNOWN"
            logger.info(f"  {obj_class} -> {dest} @ {dest_pos}")
    
    def get_destination(self, class_name: str) -> Optional[np.ndarray]:
        """
        Get the destination position for a given object class.
        
        Args:
            class_name: Detected object class name
            
        Returns:
            [x, y, z] destination world position, or None if no rule exists
        """
        dest_name = self.rules.get(class_name)
        
        if dest_name is None:
            logger.warning(f"No sorting rule for class '{class_name}'")
            return None
        
        destination = self.destinations.get(dest_name)
        if destination is None:
            logger.warning(f"Destination '{dest_name}' not found")
            return None
        
        # Slightly offset position for multiple objects at same destination
        offset = self._compute_stack_offset(destination)
        position = destination.position.copy() + offset
        
        destination.current_count += 1
        
        return position
    
    def _compute_stack_offset(self, destination: Destination) -> np.ndarray:
        """Compute small offset to avoid stacking objects exactly on top of each other."""
        count = destination.current_count
        if count == 0:
            return np.zeros(3)
        
        # Arrange in a small grid pattern
        col = count % 3
        row = count // 3
        offset_x = (col - 1) * 0.05  # 5cm spacing
        offset_y = (row - 0.5) * 0.05
        
        return np.array([offset_x, offset_y, 0.0])
    
    def get_destination_name(self, class_name: str) -> Optional[str]:
        """Get destination name for an object class."""
        return self.rules.get(class_name)
    
    def reset(self) -> None:
        """Reset destination counts for a new episode."""
        for dest in self.destinations.values():
            dest.current_count = 0
    
    def get_all_rules(self) -> Dict[str, str]:
        """Return complete rule mapping."""
        return self.rules.copy()
