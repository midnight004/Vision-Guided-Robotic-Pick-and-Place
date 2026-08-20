"""
Robotic Arm Controller (Franka Panda - MuJoCo)
=================================================
Controls the simulated Franka Panda using MuJoCo's built-in actuators.
Provides joint-space and Cartesian-space motion primitives.

Control Strategy:
    - Position control via PD-like actuators (defined in scene XML)
    - Inverse kinematics using MuJoCo's built-in IK (mj_jac + damped least squares)
    - Smooth trajectory interpolation between waypoints
"""

import numpy as np
import mujoco
from typing import Optional, Tuple
import time

from ..utils.logger import setup_logger

logger = setup_logger("arm_controller")


class ArmController:
    """
    Controller for the Franka Panda arm in MuJoCo.
    Provides both joint-space and Cartesian-space control.
    """
    
    # Franka Panda joint limits (radians)
    JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    # Home configuration (gripper facing downward toward table)
    HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    def __init__(self, sim_env):
        """
        Args:
            sim_env: SimulationEnvironment instance (MuJoCo)
        """
        self.sim = sim_env
        self.model = sim_env.model
        self.data = sim_env.data
        self._gripper = None  # Set by PickPlaceExecutor
        
        # IK parameters
        self.ik_damping = 1e-3
        self.ik_max_iterations = 300
        self.ik_tolerance = 0.005  # 5mm position tolerance
        
        # Motion parameters
        self.interpolation_steps = 100  # Steps for smooth motion
        self.position_tolerance = 0.02  # 2cm for "reached" check
        
        logger.info("ArmController initialized (MuJoCo)")
    
    def move_to_joint_positions(
        self,
        target: np.ndarray,
        duration: float = 2.0,
        render: bool = True,
    ) -> bool:
        """
        Move to target joint configuration with smooth interpolation.
        Renders every N steps for smooth visual motion.
        Calls gripper.update_grasp() to keep held objects attached.
        """
        target = np.clip(target, self.JOINT_LIMITS_LOWER, self.JOINT_LIMITS_UPPER)
        
        current = self.sim.get_joint_positions()
        dt = self.model.opt.timestep
        num_steps = max(100, int(duration / dt))
        
        render_interval = max(1, int(1.0 / (60.0 * dt)))
        
        for step in range(num_steps):
            alpha = min(1.0, (step + 1) / (num_steps * 0.7))
            interpolated = current + alpha * (target - current)
            self.sim.set_joint_targets(interpolated)
            mujoco.mj_step(self.model, self.data)
            
            # Keep grasped object attached
            if self._gripper is not None:
                self._gripper.update_grasp()
            
            if render and self.sim.viewer is not None and step % render_interval == 0:
                self.sim.viewer.sync()
                time.sleep(dt * render_interval)
        
        # Settling
        self.sim.set_joint_targets(target)
        for i in range(200):
            mujoco.mj_step(self.model, self.data)
            if self._gripper is not None:
                self._gripper.update_grasp()
            if render and self.sim.viewer is not None and i % (render_interval * 2) == 0:
                self.sim.viewer.sync()
        
        if render and self.sim.viewer is not None:
            self.sim.viewer.sync()
        
        final = self.sim.get_joint_positions()
        error = np.max(np.abs(final - target))
        
        if error < 0.1:
            return True
        else:
            logger.warning(f"Joint target error: {error:.4f} rad")
            return error < 0.3
    
    def move_to_cartesian(
        self,
        target_pos: np.ndarray,
        target_orient: Optional[np.ndarray] = None,
        duration: float = 2.0,
        render: bool = True,
    ) -> bool:
        """
        Move end-effector to target Cartesian position using IK.
        
        Args:
            target_pos: [x, y, z] target position in world frame
            target_orient: Optional target orientation (3x3 rotation matrix)
            duration: Motion duration
            render: Whether to render during motion
            
        Returns:
            True if target reached within tolerance
        """
        # Validate workspace
        if not self._is_in_workspace(target_pos):
            logger.error(f"Target {target_pos} outside workspace!")
            return False
        
        # Solve IK for target pose
        target_joints = self._solve_ik(target_pos, target_orient)
        
        if target_joints is None:
            logger.warning(f"IK failed for target {target_pos}")
            return False
        
        # Move to IK solution
        success = self.move_to_joint_positions(target_joints, duration=duration, render=render)
        
        # Verify Cartesian position
        final_pos = self.sim.get_ee_position()
        error = np.linalg.norm(final_pos - target_pos)
        
        if error < self.position_tolerance:
            logger.debug(f"Cartesian target reached (error: {error*100:.2f}cm)")
            return True
        elif error < 0.05:
            logger.debug(f"Cartesian target close enough (error: {error*100:.2f}cm)")
            return True
        else:
            # Try additional settling
            self.sim.set_joint_targets(target_joints)
            for _ in range(500):
                mujoco.mj_step(self.model, self.data)
            
            final_pos = self.sim.get_ee_position()
            error = np.linalg.norm(final_pos - target_pos)
            logger.info(f"After extra settling: error={error*100:.2f}cm")
            return error < 0.08  # Accept within 8cm after extra settling
    
    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_orient: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using damped least squares.
        Targets position AND downward orientation for the gripper.
        
        Args:
            target_pos: [x, y, z] target EE position
            target_orient: Optional orientation (not used, always top-down)
            
        Returns:
            (7,) joint angles or None if IK fails
        """
        # Work on a copy, starting from HOME config for reliable IK
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]
        
        ee_site_id = self.sim._ee_site_id
        
        # Joint DOF indices
        joint_ids = self.sim._joint_ids
        dof_indices = np.array([self.model.jnt_dofadr[jid] for jid in joint_ids])
        
        # Reset arm joints to home for better IK starting point
        for i, jid in enumerate(joint_ids):
            addr = self.model.jnt_qposadr[jid]
            ik_data.qpos[addr] = self.HOME_QPOS[i]
        ik_data.qvel[:] = 0
        mujoco.mj_forward(self.model, ik_data)
        
        # Desired orientation: Z-axis of EE should point DOWN [0, 0, -1]
        # This means the gripper fingers point toward the table
        desired_z_axis = np.array([0.0, 0.0, -1.0])
        
        for iteration in range(self.ik_max_iterations):
            current_pos = ik_data.site_xpos[ee_site_id].copy()
            current_mat = ik_data.site_xmat[ee_site_id].reshape(3, 3)
            current_z = current_mat[:, 2]  # Z-axis of EE frame
            
            # Position error
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            # Orientation error (cross product gives rotation axis * sin(angle))
            orient_error = np.cross(current_z, desired_z_axis)
            orient_error_norm = np.linalg.norm(orient_error)
            
            # Check convergence
            if pos_error_norm < self.ik_tolerance and orient_error_norm < 0.1:
                result = np.zeros(7)
                for i, jid in enumerate(joint_ids):
                    result[i] = ik_data.qpos[self.model.jnt_qposadr[jid]]
                return result
            
            # Compute Jacobians
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, ee_site_id)
            
            # Extract for our 7 joints
            Jp = jacp[:, dof_indices]  # (3 x 7) position Jacobian
            Jr = jacr[:, dof_indices]  # (3 x 7) rotation Jacobian
            
            # Combined error: position (weight=1.0) + orientation (weight=0.5)
            error_6d = np.concatenate([pos_error, orient_error * 0.5])
            J_6d = np.vstack([Jp, Jr * 0.5])  # (6 x 7)
            
            # Damped least squares
            JJT = J_6d @ J_6d.T + self.ik_damping * np.eye(6)
            dq = J_6d.T @ np.linalg.solve(JJT, error_6d)
            
            # Scale step
            max_step = 0.1
            step_scale = min(1.0, max_step / (np.max(np.abs(dq)) + 1e-6))
            dq *= step_scale
            
            # Apply
            for i, jid in enumerate(joint_ids):
                addr = self.model.jnt_qposadr[jid]
                ik_data.qpos[addr] += dq[i]
                ik_data.qpos[addr] = np.clip(
                    ik_data.qpos[addr],
                    self.JOINT_LIMITS_LOWER[i],
                    self.JOINT_LIMITS_UPPER[i]
                )
            
            mujoco.mj_forward(self.model, ik_data)
        
        # Return best attempt even if not fully converged
        result = np.zeros(7)
        for i, jid in enumerate(joint_ids):
            result[i] = ik_data.qpos[self.model.jnt_qposadr[jid]]
        
        final_pos = ik_data.site_xpos[ee_site_id]
        final_error = np.linalg.norm(target_pos - final_pos)
        
        if final_error < 0.05:
            return result
        
        logger.warning(f"IK failed: pos_error={final_error*100:.1f}cm")
        return None
    
    def move_to_home(self, render: bool = True) -> bool:
        """Move robot to home configuration."""
        logger.info("Moving to home position")
        return self.move_to_joint_positions(self.HOME_QPOS, duration=2.0, render=render)
    
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        return self.sim.get_ee_position()
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """Check if position is within robot workspace."""
        # Approximate workspace bounds for Franka Panda
        return (
            -0.1 <= pos[0] <= 0.9 and
            -0.6 <= pos[1] <= 0.6 and
            0.02 <= pos[2] <= 1.2
        )
    
    def reset(self) -> None:
        """Reset controller state."""
        self.sim.reset_to_home()
        logger.info("Arm controller reset to home")
