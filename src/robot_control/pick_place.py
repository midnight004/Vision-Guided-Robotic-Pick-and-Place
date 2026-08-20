"""
Pick and Place Executor - Real Physics (No teleportation)
State machine: APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> LOWER -> RELEASE -> RETREAT
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
    # Heights relative to object / world Z
    APPROACH_HEIGHT = 0.15     # Above object for approach
    GRASP_Z_OFFSET = 0.058    # Hand-to-fingertip offset (measured from Menagerie model)
    LIFT_HEIGHT = 0.58         # Absolute Z after lift (high to clear bin walls)
    TRANSPORT_HEIGHT = 0.58    # Absolute Z during transport (stays high)
    PLACE_HEIGHT_OFFSET = 0.18 # Release well above the bin so object drops in cleanly
    RETREAT_HEIGHT = 0.58      # Absolute Z after release

    def __init__(self, arm: ArmController, gripper: GripperController):
        self.arm = arm
        self.gripper = gripper
        self.state = PickPlaceState.IDLE
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        logger.info("PickPlaceExecutor initialized (real physics)")

    def execute(self, pick_position: np.ndarray, place_position: np.ndarray, object_name: str = "unknown") -> bool:
        self.total_attempts += 1
        logger.info(f"=== PICK AND PLACE: {object_name} ===")
        logger.info(f"  Pick:  [{pick_position[0]:.3f}, {pick_position[1]:.3f}, {pick_position[2]:.3f}]")
        logger.info(f"  Place: [{place_position[0]:.3f}, {place_position[1]:.3f}, {place_position[2]:.3f}]")

        start_time = time.time()

        try:
            # 1. Open gripper
            self.gripper.open(steps=80)

            # 2. APPROACH above object
            self.state = PickPlaceState.APPROACH
            approach_pos = pick_position.copy()
            approach_pos[2] += self.APPROACH_HEIGHT
            if not self.arm.move_to_cartesian(approach_pos, duration=1.5):
                return self._fail("Cannot reach approach")

            # 3. DESCEND to grasp height
            self.state = PickPlaceState.DESCEND
            grasp_pos = pick_position.copy()
            grasp_pos[2] += self.GRASP_Z_OFFSET
            if not self.arm.move_to_cartesian(grasp_pos, duration=1.2):
                return self._fail("Cannot descend")

            # 4. GRASP - close fingers around object
            self.state = PickPlaceState.GRASP
            self.gripper.close(steps=250)
            self.successful_picks += 1
            # Real physics: if object is between fingers, friction holds it
            # If not, it will fall during lift (natural failure)

            # 5. LIFT
            self.state = PickPlaceState.LIFT
            lift_pos = pick_position.copy()
            lift_pos[2] = self.LIFT_HEIGHT
            if not self.arm.move_to_cartesian(lift_pos, duration=1.2):
                return self._fail("Cannot lift")

            # 6. TRANSPORT above destination
            self.state = PickPlaceState.TRANSPORT
            transport_pos = place_position.copy()
            transport_pos[2] = self.TRANSPORT_HEIGHT
            if not self.arm.move_to_cartesian(transport_pos, duration=1.5):
                return self._fail("Cannot transport")

            # 7. LOWER to place height
            self.state = PickPlaceState.LOWER
            lower_pos = place_position.copy()
            lower_pos[2] += self.PLACE_HEIGHT_OFFSET
            if not self.arm.move_to_cartesian(lower_pos, duration=1.0):
                return self._fail("Cannot lower")

            # 8. RELEASE
            self.state = PickPlaceState.RELEASE
            self.gripper.open(steps=100)

            # 9. RETREAT
            self.state = PickPlaceState.RETREAT
            retreat_pos = lower_pos.copy()
            retreat_pos[2] = self.RETREAT_HEIGHT
            self.arm.move_to_cartesian(retreat_pos, duration=0.8)

            self.state = PickPlaceState.COMPLETE
            self.successful_places += 1
            elapsed = time.time() - start_time
            logger.info(f"  COMPLETE ({elapsed:.1f}s)")
            return True

        except Exception as e:
            return self._fail(f"Exception: {e}")

    def _fail(self, reason: str) -> bool:
        self.state = PickPlaceState.FAILED
        logger.error(f"  FAILED: {reason}")
        # Open gripper on failure to release any partial grasp
        self.gripper.open(steps=50)
        return False

    def get_statistics(self) -> dict:
        return {
            'total_attempts': self.total_attempts,
            'successful_picks': self.successful_picks,
            'successful_places': self.successful_places,
            'pick_rate': self.successful_picks / max(1, self.total_attempts),
            'place_rate': self.successful_places / max(1, self.total_attempts),
        }

    def reset_statistics(self) -> None:
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        self.state = PickPlaceState.IDLE
