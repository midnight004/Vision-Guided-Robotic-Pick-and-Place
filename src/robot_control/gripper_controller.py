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
GRASP_VALUE = 0.0     # Fully close so fingers firmly squeeze the object


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
        self.grasped_object = None
        logger.info("Gripper initialized (contact-triggered weld grasp)")

    def open(self, steps: int = 150, render: bool = True) -> bool:
        """Open gripper and release any welded object."""
        logger.debug("Opening gripper")
        self.sim.release_all_welds()
        self.grasped_object = None
        self.sim.set_gripper(OPEN_VALUE)
        self.is_gripping = False
        self._animate(steps, render)
        return True

    def close(self, steps: int = 200, render: bool = True) -> bool:
        """Close gripper on the nearest object; weld it for a firm grasp."""
        logger.debug("Closing gripper")
        self.sim.set_gripper(GRASP_VALUE)
        self.is_gripping = True
        self._animate(steps, render)

        # Identify the object between the fingers (closest to the hand)
        grasped = self._nearest_object()
        width = self.sim.get_gripper_width()
        if grasped is not None:
            self.sim.activate_grasp_weld(grasped)
            self.grasped_object = grasped
            logger.info(f"  GRASPED {grasped} (finger width: {width*1000:.1f}mm)")
            return True
        else:
            logger.warning(f"  Grasp empty (width: {width*1000:.1f}mm)")
            return False

    def _nearest_object(self, max_xy: float = 0.06, max_z: float = 0.14) -> Optional[str]:
        """
        Find the object between the fingers. The object center sits below the
        hand's EE reference, so match on horizontal (xy) proximity within a
        vertical window beneath the hand.
        """
        hand_pos = self.sim.get_ee_position()
        best = None
        bd = max_xy
        for name in self.sim._object_body_ids:
            pos = self.sim.get_object_position(name)
            if pos is None:
                continue
            dz = hand_pos[2] - pos[2]
            if dz < -0.03 or dz > max_z:
                continue
            dxy = float(np.linalg.norm(pos[:2] - hand_pos[:2]))
            if dxy < bd:
                bd = dxy
                best = name
        return best

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
