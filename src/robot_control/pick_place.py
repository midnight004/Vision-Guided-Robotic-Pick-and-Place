"""
Pick and Place Executor (MuJoCo)
==================================
Orchestrates complete pick-and-place operations using arm + gripper.

Sequence:
    1. APPROACH: Move above object (pre-grasp)
    2. DESCEND: Lower to grasp height
    3. GRASP: Close gripper
    4. LIFT: Raise object
    5. TRANSPORT: Move above destination
    6. LOWER: Descend to place height
    7. RELEASE: Open gripper
    8. RETREAT: Rise above placed object
"""

import numpy as np
from typing import Optional
from enum import Enum
import time

from .arm_controller import ArmController
from .gripper_controller import GripperController
from ..utils.logger import setup_logger

logger = setup_logger("pick_place")


class PickPlaceState(Enum):
    IDLE = "idle"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    COMPLETE = "complete"
    FAILED = "failed"


class PickPlaceExecutor:
    """
    Executes pick-and-place using the arm and gripper.
    Uses a state machine for clear sequencing and debugging.
    """
    
    # Heights (meters, world Z coordinate)
    APPROACH_HEIGHT = 0.10    # How high above object for approach
    GRASP_HEIGHT_OFFSET = -0.02  # EE goes slightly BELOW object center for better grasp
    LIFT_HEIGHT = 0.50        # Absolute world Z height after lifting
    TRANSPORT_HEIGHT = 0.47   # Absolute world Z height during transport
    PLACE_HEIGHT_OFFSET = 0.02 # Release just barely above the bin (2cm)
    RETREAT_HEIGHT = 0.50     # Absolute world Z height after releasing
    
    def __init__(self, arm: ArmController, gripper: GripperController):
        """
        Args:
            arm: ArmController instance
            gripper: GripperController instance
        """
        self.arm = arm
        self.gripper = gripper
        
        # Link gripper to arm so arm motions update the grasp
        self.arm._gripper = gripper
        
        # State
        self.state = PickPlaceState.IDLE
        self.current_pick_target = None
        self.current_place_target = None
        
        # Statistics
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        
        logger.info("PickPlaceExecutor initialized")
    
    def execute(
        self,
        pick_position: np.ndarray,
        place_position: np.ndarray,
        object_name: str = "unknown",
    ) -> bool:
        """
        Execute a complete pick-and-place operation.
        
        Args:
            pick_position: [x, y, z] world position of object
            place_position: [x, y, z] world position of destination
            object_name: Name of object (for logging)
            
        Returns:
            True if successful
        """
        self.total_attempts += 1
        self.current_pick_target = pick_position
        self.current_place_target = place_position
        
        logger.info(f"=== PICK AND PLACE: {object_name} ===")
        logger.info(f"  Pick:  [{pick_position[0]:.3f}, {pick_position[1]:.3f}, {pick_position[2]:.3f}]")
        logger.info(f"  Place: [{place_position[0]:.3f}, {place_position[1]:.3f}, {place_position[2]:.3f}]")
        
        start_time = time.time()
        
        try:
            # 1. APPROACH: Move above the object
            self.state = PickPlaceState.APPROACH
            approach_pos = pick_position.copy()
            approach_pos[2] += self.APPROACH_HEIGHT
            logger.debug(f"  [APPROACH] -> {approach_pos}")
            
            if not self.arm.move_to_cartesian(approach_pos, duration=1.5):
                return self._fail("Cannot reach approach position")
            
            # 2. Open gripper
            self.gripper.open(steps=50)
            
            # 3. DESCEND: Lower to grasp height
            self.state = PickPlaceState.DESCEND
            grasp_pos = pick_position.copy()
            grasp_pos[2] += self.GRASP_HEIGHT_OFFSET
            logger.debug(f"  [DESCEND] -> {grasp_pos}")
            
            if not self.arm.move_to_cartesian(grasp_pos, duration=1.0):
                return self._fail("Cannot reach grasp position")
            
            # 4. GRASP: Close gripper
            self.state = PickPlaceState.GRASP
            logger.debug("  [GRASP] Closing gripper")
            self.gripper.close(steps=250)
            
            # Verify grasp
            if not self.gripper.is_object_grasped():
                return self._fail("Failed to grasp object")
            self.successful_picks += 1
            
            # 5. LIFT: Raise object
            self.state = PickPlaceState.LIFT
            lift_pos = grasp_pos.copy()
            lift_pos[2] = self.LIFT_HEIGHT
            logger.debug(f"  [LIFT] -> {lift_pos}")
            
            if not self.arm.move_to_cartesian(lift_pos, duration=1.0):
                return self._fail("Cannot lift")
            
            # 6. TRANSPORT: Move above destination
            self.state = PickPlaceState.TRANSPORT
            transport_pos = place_position.copy()
            transport_pos[2] = self.TRANSPORT_HEIGHT
            logger.debug(f"  [TRANSPORT] -> {transport_pos}")
            
            if not self.arm.move_to_cartesian(transport_pos, duration=1.5):
                return self._fail("Cannot reach transport position")
            
            # 7. LOWER: Descend to place height
            self.state = PickPlaceState.LOWER
            lower_pos = place_position.copy()
            lower_pos[2] += self.PLACE_HEIGHT_OFFSET
            logger.debug(f"  [LOWER] -> {lower_pos}")
            
            if not self.arm.move_to_cartesian(lower_pos, duration=1.0):
                return self._fail("Cannot lower to place position")
            
            # 8. RELEASE: Open gripper
            self.state = PickPlaceState.RELEASE
            logger.debug("  [RELEASE] Opening gripper")
            self.gripper.open(steps=80)
            
            # 9. RETREAT: Rise above
            self.state = PickPlaceState.RETREAT
            retreat_pos = lower_pos.copy()
            retreat_pos[2] = self.RETREAT_HEIGHT
            self.arm.move_to_cartesian(retreat_pos, duration=0.8)
            
            # SUCCESS
            self.state = PickPlaceState.COMPLETE
            self.successful_places += 1
            
            elapsed = time.time() - start_time
            logger.info(f"  Pick-and-place COMPLETE ({elapsed:.1f}s)")
            
            return True
            
        except Exception as e:
            return self._fail(f"Exception: {e}")
    
    def _fail(self, reason: str) -> bool:
        """Handle failure."""
        self.state = PickPlaceState.FAILED
        logger.error(f"  FAILED: {reason}")
        return False
    
    def get_statistics(self) -> dict:
        """Get performance statistics."""
        return {
            'total_attempts': self.total_attempts,
            'successful_picks': self.successful_picks,
            'successful_places': self.successful_places,
            'pick_success_rate': self.successful_picks / max(1, self.total_attempts),
            'place_success_rate': self.successful_places / max(1, self.total_attempts),
        }
    
    def reset_statistics(self) -> None:
        """Reset counters."""
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        self.state = PickPlaceState.IDLE
