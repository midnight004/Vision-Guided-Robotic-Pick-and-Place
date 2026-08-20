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
    LIFT_HEIGHT = 0.56         # Absolute Z after lift
    TRANSPORT_HEIGHT = 0.56    # Absolute Z during transport (object hangs ~9cm below, clears 0.40 bin walls)
    PLACE_HEIGHT_OFFSET = 0.10 # Release above bin floor (object drops in cleanly)
    RETREAT_HEIGHT = 0.56      # Absolute Z after release

    def __init__(self, arm: ArmController, gripper: GripperController):
        self.arm = arm
        self.gripper = gripper
        self.state = PickPlaceState.IDLE
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        logger.info("PickPlaceExecutor initialized (real physics)")

    def execute(self, pick_position: np.ndarray, place_position: np.ndarray,
                object_name: str = "unknown", relocalize_fn=None) -> bool:
        """
        Run one pick-and-place cycle.

        relocalize_fn (optional, Phase 10 visual servoing): a callback invoked
        once from the approach pose that returns a refined pick position (or
        None). When provided, the descend target is corrected using a fresh
        visual observation before grasping. Default None reproduces the exact
        baseline behavior.
        """
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

            # 2b. VISUAL SERVOING (optional): refine the pick target from a fresh
            #     observation taken at the approach pose, then correct XY.
            if relocalize_fn is not None:
                refined = relocalize_fn()
                if refined is not None:
                    corr = float(np.linalg.norm(refined[:2] - pick_position[:2]))
                    pick_position = np.array([refined[0], refined[1], pick_position[2]])
                    logger.info(f"  Visual servo correction: {corr*100:.1f} mm")

            # 3. DESCEND to grasp height
            self.state = PickPlaceState.DESCEND
            grasp_pos = pick_position.copy()
            grasp_pos[2] += self.GRASP_Z_OFFSET
            if not self.arm.move_to_cartesian(grasp_pos, duration=1.2):
                return self._fail("Cannot descend")

            # 4. GRASP - close fingers around object. If nothing is captured,
            #    re-open, re-descend and retry once before giving up.
            self.state = PickPlaceState.GRASP
            grasped = self.gripper.close(steps=250)
            if not grasped:
                logger.info("  Empty grasp - realigning and retrying")
                self.gripper.open(steps=60)
                self.arm.move_to_cartesian(grasp_pos, duration=1.0)
                grasped = self.gripper.close(steps=250)
            if not grasped:
                return self._fail("Nothing grasped")
            self.successful_picks += 1

            # 5. LIFT (slow, straight up, to keep a firm grip)
            self.state = PickPlaceState.LIFT
            lift_pos = pick_position.copy()
            lift_pos[2] = self.LIFT_HEIGHT
            if not self.arm.move_to_cartesian(lift_pos, duration=2.0):
                return self._fail("Cannot lift")

            # 6. TRANSPORT above destination (slow to avoid slipping)
            self.state = PickPlaceState.TRANSPORT
            transport_pos = place_position.copy()
            transport_pos[2] = self.TRANSPORT_HEIGHT
            if not self.arm.move_to_cartesian(transport_pos, duration=2.5):
                return self._fail("Cannot transport")

            # 7. LOWER to place height (slow)
            self.state = PickPlaceState.LOWER
            lower_pos = place_position.copy()
            lower_pos[2] += self.PLACE_HEIGHT_OFFSET
            if not self.arm.move_to_cartesian(lower_pos, duration=1.5):
                return self._fail("Cannot lower")

            # 8. RELEASE and let the object settle into the bin
            self.state = PickPlaceState.RELEASE
            self.gripper.open(steps=120)

            # 9. RETREAT straight up (don't drag through the object)
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
        # Release any partial grasp and return to a clean pose so the next
        # attempt starts from a known-good configuration (avoids cascading IK failures).
        self.gripper.open(steps=50)
        self.arm.move_to_scan_pose(render=True)
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
