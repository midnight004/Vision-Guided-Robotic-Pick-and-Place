"""
Franka Panda Arm Controller (Menagerie model)
Uses the official actuator scheme: general actuators with gravity compensation.
Joint names: joint1-joint7. Gripper: actuator8 (tendon, 0-255 range).
"""

import numpy as np
import mujoco
from typing import Optional
import time

from ..utils.logger import setup_logger

logger = setup_logger("arm_controller")

HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
HOME_CTRL = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0])

# Scan pose: arm tucked back near its base so it doesn't occlude the
# overhead camera's view of the table during perception.
SCAN_QPOS = np.array([0.0, -1.5, 0.0, -2.5, 0.0, 1.5, 0.785])

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class ArmController:
    def __init__(self, sim_env):
        self.sim = sim_env
        self.model = sim_env.model
        self.data = sim_env.data
        self._gripper = None

        self.ik_damping = 1e-3
        self.ik_max_iterations = 300
        self.ik_tolerance = 0.005

        logger.info("ArmController initialized (Menagerie Franka)")

    def move_to_joint_positions(self, target: np.ndarray, duration: float = 2.0, render: bool = True) -> bool:
        target = np.clip(target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        current = self.sim.get_joint_positions()
        dt = self.model.opt.timestep
        num_steps = max(200, int(duration / dt))
        render_interval = max(1, int(1.0 / (60.0 * dt)))

        for step in range(num_steps):
            alpha = min(1.0, (step + 1) / (num_steps * 0.7))
            interpolated = current + alpha * (target - current)
            self.sim.set_joint_targets(interpolated)
            mujoco.mj_step(self.model, self.data)

            if render and self.sim.viewer is not None and step % render_interval == 0:
                self.sim.viewer.sync()
                time.sleep(dt * render_interval)

        # Settle
        self.sim.set_joint_targets(target)
        for i in range(300):
            mujoco.mj_step(self.model, self.data)
            if render and self.sim.viewer is not None and i % (render_interval * 3) == 0:
                self.sim.viewer.sync()

        if render and self.sim.viewer is not None:
            self.sim.viewer.sync()

        final = self.sim.get_joint_positions()
        error = np.max(np.abs(final - target))
        return error < 0.3

    def move_to_cartesian(self, target_pos: np.ndarray, duration: float = 2.0, render: bool = True) -> bool:
        if not self._is_in_workspace(target_pos):
            logger.error(f"Target {target_pos} outside workspace!")
            return False

        target_joints = self._solve_ik(target_pos)
        if target_joints is None:
            logger.warning(f"IK failed for target {target_pos}")
            return False

        success = self.move_to_joint_positions(target_joints, duration=duration, render=render)

        final_pos = self.sim.get_ee_position()
        error = np.linalg.norm(final_pos - target_pos)
        return error < 0.08

    def _solve_ik(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        """Damped least-squares IK targeting position + downward orientation."""
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]
        ik_data.qvel[:] = 0

        joint_ids = self.sim._joint_ids
        dof_indices = np.array([self.model.jnt_dofadr[jid] for jid in joint_ids])
        hand_body_id = self.sim._hand_body_id

        # Start from home
        for i, jid in enumerate(joint_ids):
            ik_data.qpos[self.model.jnt_qposadr[jid]] = HOME_QPOS[i]
        mujoco.mj_forward(self.model, ik_data)

        desired_z = np.array([0.0, 0.0, -1.0])

        for _ in range(self.ik_max_iterations):
            current_pos = ik_data.xpos[hand_body_id].copy()
            current_mat = ik_data.xmat[hand_body_id].reshape(3, 3)
            current_z = current_mat[:, 2]

            pos_error = target_pos - current_pos
            orient_error = np.cross(current_z, desired_z)

            if np.linalg.norm(pos_error) < self.ik_tolerance and np.linalg.norm(orient_error) < 0.15:
                result = np.zeros(7)
                for i, jid in enumerate(joint_ids):
                    result[i] = ik_data.qpos[self.model.jnt_qposadr[jid]]
                return result

            # Jacobian for the hand body
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, ik_data, jacp, jacr, hand_body_id)

            Jp = jacp[:, dof_indices]
            Jr = jacr[:, dof_indices]

            error_6d = np.concatenate([pos_error, orient_error * 0.4])
            J_6d = np.vstack([Jp, Jr * 0.4])

            JJT = J_6d @ J_6d.T + self.ik_damping * np.eye(6)
            dq = J_6d.T @ np.linalg.solve(JJT, error_6d)

            step_scale = min(1.0, 0.15 / (np.max(np.abs(dq)) + 1e-6))
            dq *= step_scale

            for i, jid in enumerate(joint_ids):
                addr = self.model.jnt_qposadr[jid]
                ik_data.qpos[addr] = np.clip(
                    ik_data.qpos[addr] + dq[i],
                    JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]
                )

            mujoco.mj_forward(self.model, ik_data)

        # Return best attempt
        final_pos = ik_data.xpos[hand_body_id]
        if np.linalg.norm(target_pos - final_pos) < 0.06:
            result = np.zeros(7)
            for i, jid in enumerate(joint_ids):
                result[i] = ik_data.qpos[self.model.jnt_qposadr[jid]]
            return result

        logger.warning(f"IK failed: error={np.linalg.norm(target_pos - final_pos)*100:.1f}cm")
        return None

    def move_to_home(self, render: bool = True) -> bool:
        logger.info("Moving to home")
        return self.move_to_joint_positions(HOME_QPOS, duration=2.0, render=render)

    def move_to_scan_pose(self, render: bool = True) -> bool:
        """Tuck the arm back so the camera can see the whole table."""
        return self.move_to_joint_positions(SCAN_QPOS, duration=1.5, render=render)

    def get_ee_position(self) -> np.ndarray:
        return self.sim.get_ee_position()

    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        return (-0.1 <= pos[0] <= 0.85 and -0.6 <= pos[1] <= 0.6 and 0.0 <= pos[2] <= 1.0)

    def reset(self) -> None:
        self.sim.reset_to_home()
