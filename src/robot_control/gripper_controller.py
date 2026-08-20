"""
Gripper Controller (Franka Panda - MuJoCo)
============================================
Controls the gripper and implements reliable grasping by attaching/detaching
objects to the hand body (constraint-based grasp).

This approach is standard in MuJoCo manipulation research:
- Animate finger closing for visual effect
- Attach object to hand using mocap/weld when "grasped"
- Detach on release
"""

import numpy as np
import mujoco
import time
from typing import Optional

from ..utils.logger import setup_logger

logger = setup_logger("gripper")


class GripperController:
    """
    Gripper controller that physically attaches objects for reliable grasping.
    """
    
    def __init__(self, sim_env):
        self.sim = sim_env
        self.model = sim_env.model
        self.data = sim_env.data
        
        self.is_gripping = False
        self.grasped_object: Optional[str] = None
        self._grasped_joint_id: Optional[int] = None
        self._grasp_offset = np.zeros(3)
        
        logger.info("Gripper initialized (constraint-based grasp)")
    
    def open(self, steps: int = 80, render: bool = True) -> bool:
        """Open the gripper and release any attached object."""
        logger.debug("Opening gripper")
        
        # Release object if holding one
        if self.grasped_object is not None:
            self._release_object()
        
        # Animate fingers opening (visual)
        self.sim.set_gripper(0.0)  # 0 = fingers at outer position (open)
        
        render_interval = 8
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if render and self.sim.viewer is not None and i % render_interval == 0:
                self.sim.viewer.sync()
                time.sleep(0.012)
        
        self.is_gripping = False
        return True
    
    def close(self, steps: int = 120, render: bool = True) -> bool:
        """Close the gripper and attach the nearest object."""
        logger.debug("Closing gripper")
        
        # Animate fingers closing (visual)
        self.sim.set_gripper(0.04)  # 0.04 = fingers at inner position (closed)
        
        render_interval = 8
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if render and self.sim.viewer is not None and i % render_interval == 0:
                self.sim.viewer.sync()
                time.sleep(0.012)
        
        # Find and attach nearest object
        self._attach_nearest_object()
        
        self.is_gripping = True
        return True
    
    def _attach_nearest_object(self) -> None:
        """Find the nearest object to the EE and attach it (centered between fingers)."""
        ee_pos = self.sim.get_ee_position()
        
        best_name = None
        best_dist = 0.12  # Max 12cm to consider "graspable"
        
        for name, body_id in self.sim._object_body_ids.items():
            obj_pos = self.data.xpos[body_id]
            dist = np.linalg.norm(ee_pos - obj_pos)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        if best_name is not None:
            self.grasped_object = best_name
            joint_id = self.sim._object_joint_ids[best_name]
            self._grasped_joint_id = joint_id
            
            # Object snaps to EE center (no offset) for precise placement
            self._grasp_offset = np.array([0.0, 0.0, 0.0])
            
            logger.info(f"  GRASPED: {best_name} (dist={best_dist*100:.1f}cm)")
        else:
            logger.warning(f"  No object within grasp range (nearest > 6cm)")
    
    def _release_object(self) -> None:
        """Release the currently held object."""
        if self.grasped_object:
            logger.info(f"  RELEASED: {self.grasped_object}")
            self.grasped_object = None
            self._grasped_joint_id = None
            self._grasp_offset = np.zeros(3)
    
    def update_grasp(self) -> None:
        """
        Called every physics step to keep the grasped object attached to the EE.
        This simulates the object being held by the gripper.
        """
        if self.grasped_object is None or self._grasped_joint_id is None:
            return
        
        # Move object to follow the EE
        ee_pos = self.sim.get_ee_position()
        target_pos = ee_pos + self._grasp_offset
        
        # Set object position directly via qpos (free joint: x,y,z,qw,qx,qy,qz)
        addr = self.model.jnt_qposadr[self._grasped_joint_id]
        self.data.qpos[addr:addr+3] = target_pos
        # Keep orientation upright
        self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
        # Zero velocity
        dof_addr = self.model.jnt_dofadr[self._grasped_joint_id]
        self.data.qvel[dof_addr:dof_addr+6] = 0
    
    def is_object_grasped(self) -> bool:
        return self.grasped_object is not None
    
    def get_state(self) -> dict:
        return {
            'is_gripping': self.is_gripping,
            'grasped_object': self.grasped_object,
        }
