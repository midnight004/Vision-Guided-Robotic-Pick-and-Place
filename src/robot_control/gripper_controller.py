"""
Gripper Controller - Real Physics Grasping (Menagerie Franka)
The Menagerie Franka has detailed fingertip pad collision geometries
that enable real friction-based grasping without any teleportation hacks.

Gripper control: actuator8 (tendon coupling both fingers)
  ctrl=255 -> fully open (0.04m per finger)
  ctrl=0   -> fully closed (fingers together)
"""

import numpy as np
import mujoco
import time
from typing import Optional

from ..utils.logger import setup_logger

logger = setup_logger("gripper")

OPEN_VALUE = 255.0    # Fully open
CLOSED_VALUE = 0.0    # Fully closed (no object)
GRASP_VALUE = 10.0    # Closed with force (squeeze on object)


class GripperController:
    """
    Real physics-based gripper using Menagerie Franka's fingertip pads.
    No teleportation, no attach hacks. Objects are held by friction forces.
    """

    def __init__(self, sim_env):
        self.sim = sim_env
        self.model = sim_env.model
        self.data = sim_env.data
        self.is_gripping = False
        logger.info("Gripper initialized (real physics grasping)")

    def open(self, steps: int = 150, render: bool = True) -> bool:
        """Open gripper fully."""
        logger.debug("Opening gripper")
        self.sim.set_gripper(OPEN_VALUE)
        self.is_gripping = False
        self._animate(steps, render)
        return True

    def close(self, steps: int = 200, render: bool = True) -> bool:
        """Close gripper with force to grasp object via friction."""
        logger.debug("Closing gripper")
        # Close with firm force
        self.sim.set_gripper(GRASP_VALUE)
        self.is_gripping = True
        self._animate(steps, render)

        # Check if we actually grasped something
        width = self.sim.get_gripper_width()
        if width > 0.005:  # Fingers didn't fully close -> object between them
            logger.info(f"  GRASPED (finger width: {width*1000:.1f}mm)")
            return True
        else:
            logger.warning(f"  Grasp may be empty (width: {width*1000:.1f}mm)")
            return True  # Still proceed

    def _animate(self, steps: int, render: bool) -> None:
        """Step physics with rendering for smooth visual."""
        render_interval = 8
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if render and self.sim.viewer is not None and i % render_interval == 0:
                self.sim.viewer.sync()
                time.sleep(0.012)

    def is_object_grasped(self) -> bool:
        """Check if fingers stopped by an object (width > small threshold)."""
        if not self.is_gripping:
            return False
        width = self.sim.get_gripper_width()
        # With Menagerie fingers: > 1mm means something might be between them
        return width > 0.001

    def get_width(self) -> float:
        return self.sim.get_gripper_width()

    def get_state(self) -> dict:
        return {
            'is_gripping': self.is_gripping,
            'width_mm': self.sim.get_gripper_width() * 1000,
            'object_detected': self.is_object_grasped(),
        }
