"""
MuJoCo Factory Simulation Environment
Uses the official Menagerie Franka Panda model with real physics grasping.
"""

import numpy as np
import mujoco
import mujoco.viewer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import time

from ..utils.logger import setup_logger

logger = setup_logger("simulation")

SCENE_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "scene.xml"

# Menagerie Franka joint/actuator names
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 9)]  # 8th = gripper tendon
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]

# Full product pool: 14 items (8 known-color + 6 unknown)
OBJECT_NAMES = [
    "red_box", "red_can", "blue_box", "blue_capsule",
    "green_cylinder", "green_box", "yellow_sphere", "yellow_bottle",
    "purple_box", "orange_sphere", "white_cylinder", "black_box",
    "purple_cylinder", "orange_box",
]

# Known-color items go to color bins; unknown items go to trash
KNOWN_OBJECTS = [
    "red_box", "red_can", "blue_box", "blue_capsule",
    "green_cylinder", "green_box", "yellow_sphere", "yellow_bottle",
]
UNKNOWN_OBJECTS = [
    "purple_box", "orange_sphere", "white_cylinder", "black_box",
    "purple_cylinder", "orange_box",
]

OBJECT_JOINT_NAMES = {
    "red_box": "red_box_joint",
    "red_can": "red_can_joint",
    "blue_box": "blue_box_joint",
    "blue_capsule": "blue_cap_joint",
    "green_cylinder": "green_cyl_joint",
    "green_box": "green_box_joint",
    "yellow_sphere": "yellow_sph_joint",
    "yellow_bottle": "yellow_bot_joint",
    "purple_box": "purple_box_joint",
    "orange_sphere": "orange_sph_joint",
    "white_cylinder": "white_cyl_joint",
    "black_box": "black_box_joint",
    "purple_cylinder": "purple_cyl_joint",
    "orange_box": "orange_box_joint",
}

OBJECT_GEOM_NAMES = {
    "red_box": "red_box_geom",
    "red_can": "red_can_geom",
    "blue_box": "blue_box_geom",
    "blue_capsule": "blue_cap_geom",
    "green_cylinder": "green_cyl_geom",
    "green_box": "green_box_geom",
    "yellow_sphere": "yellow_sph_geom",
    "yellow_bottle": "yellow_bot_geom",
    "purple_box": "purple_box_geom",
    "orange_sphere": "orange_sph_geom",
    "white_cylinder": "white_cyl_geom",
    "black_box": "black_box_geom",
    "purple_cylinder": "purple_cyl_geom",
    "orange_box": "orange_box_geom",
}

# Conveyor feed geometry
CONVEYOR_SPAWN = np.array([0.5, -0.55, 0.35])   # where items appear on the belt
STAGING_POINT = np.array([0.5, -0.12, 0.33])    # where items settle for picking
PARK_BASE = np.array([2.0, 2.0, 0.05])          # off-screen parking

# Home configuration (from Menagerie)
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
HOME_CTRL = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0])


class SimulationEnvironment:
    """Factory simulation with Menagerie Franka Panda, conveyor belt, and 8 products."""

    def __init__(self, config_path: str = "config/simulation.yaml"):
        self.config = self._load_config(config_path)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.depth_renderer: Optional[mujoco.Renderer] = None
        self.seg_renderer: Optional[mujoco.Renderer] = None
        self.viewer = None
        self._object_geom_ids: Dict[int, str] = {}

        self.render_width = 640
        self.render_height = 480
        self.camera_name = "overhead_cam"

        self._object_body_ids: Dict[str, int] = {}
        self._object_joint_ids: Dict[str, int] = {}
        self._joint_ids: List[int] = []
        self._actuator_ids: List[int] = []
        self._gripper_actuator_id: int = -1
        self._hand_body_id: int = -1
        self._finger_joint_ids: List[int] = []

        self.is_initialized = False
        self.step_count = 0

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def initialize(self, scene_path: str = None, headless: bool = False) -> None:
        xml_path = Path(scene_path) if scene_path else SCENE_XML_PATH
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        logger.info(f"Loading factory scene: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.depth_renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.depth_renderer.enable_depth_rendering()
        self.seg_renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.seg_renderer.enable_segmentation_rendering()

        self._cache_ids()
        self.reset_to_home()

        if not headless:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                logger.info("Viewer launched")
            except Exception as e:
                logger.warning(f"Viewer failed: {e}")
                self.viewer = None

        self.is_initialized = True
        logger.info(f"Factory initialized: {model_info(self.model)}")

    def _cache_ids(self) -> None:
        # Robot joints
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_ids.append(jid)

        # Actuators (1-7 for arm, 8 for gripper tendon)
        for i, name in enumerate(ACTUATOR_NAMES):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 7:
                self._actuator_ids.append(aid)
            else:
                self._gripper_actuator_id = aid

        # Finger joints
        for name in FINGER_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._finger_joint_ids.append(jid)

        # Hand body (EE reference)
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Object bodies and joints
        for name in OBJECT_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._object_body_ids[name] = bid
            joint_name = OBJECT_JOINT_NAMES.get(name)
            if joint_name:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    self._object_joint_ids[name] = jid

        # Object geom IDs (for segmentation-based detection)
        for name in OBJECT_NAMES:
            geom_name = OBJECT_GEOM_NAMES.get(name)
            if geom_name:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if gid >= 0:
                    self._object_geom_ids[gid] = name

    def reset_to_home(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "factory_home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Set control
        for i in range(8):
            self.data.ctrl[i] = HOME_CTRL[i]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

    def step(self, render: bool = True) -> None:
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if render and self.viewer is not None:
            self.viewer.sync()

    def step_n(self, n: int, render: bool = True) -> None:
        render_interval = max(1, n // 30)
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            if render and self.viewer is not None and i % render_interval == 0:
                self.viewer.sync()
                time.sleep(0.01)

    def render_rgb(self, camera: str = None) -> np.ndarray:
        cam = camera or self.camera_name
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_depth(self, camera: str = None) -> np.ndarray:
        cam = camera or self.camera_name
        self.depth_renderer.update_scene(self.data, camera=cam)
        return self.depth_renderer.render().copy().astype(np.float32)

    def render_segmentation(self, camera: str = None) -> np.ndarray:
        """Render per-pixel geom-id segmentation. Returns (H,W) int array of geom IDs."""
        cam = camera or self.camera_name
        self.seg_renderer.update_scene(self.data, camera=cam)
        seg = self.seg_renderer.render()  # (H,W,2): [:,:,0]=geom id
        return seg[:, :, 0].copy()

    def get_object_geom_map(self) -> Dict[int, str]:
        """Return mapping of geom ID -> object name for detection."""
        return self._object_geom_ids.copy()

    def activate_grasp_weld(self, object_name: str) -> bool:
        """
        Activate the weld between the hand and the given object at their current
        relative pose (represents a firm grasp). Returns True on success.
        """
        eq_name = f"grasp_{object_name}"
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        if eq_id < 0:
            return False

        hand_id = self._hand_body_id
        obj_id = self._object_body_ids.get(object_name)
        if obj_id is None:
            return False

        # Compute pose of object relative to hand: rel = hand^-1 * obj
        p_h = self.data.xpos[hand_id].copy()
        q_h = self.data.xquat[hand_id].copy()
        p_o = self.data.xpos[obj_id].copy()
        q_o = self.data.xquat[obj_id].copy()

        q_h_inv = np.zeros(4)
        mujoco.mju_negQuat(q_h_inv, q_h)
        dp = p_o - p_h
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, dp, q_h_inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, q_h_inv, q_o)

        # Weld eq_data layout: [anchor(3), relpose pos(3) + quat(4), torquescale(1)]
        data = np.zeros(11)
        data[0:3] = 0.0            # anchor at body2 origin
        data[3:6] = rel_pos        # relative position
        data[6:10] = rel_quat      # relative orientation
        data[10] = 1.0             # torque scale
        self.model.eq_data[eq_id] = data
        self.data.eq_active[eq_id] = 1
        mujoco.mj_forward(self.model, self.data)
        return True

    def release_all_welds(self) -> None:
        """Deactivate every grasp weld (release any held object)."""
        for name in self._object_joint_ids.keys():
            eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{name}")
            if eq_id >= 0:
                self.data.eq_active[eq_id] = 0
        mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        return self.data.xpos[self._hand_body_id].copy()

    def get_ee_orientation(self) -> np.ndarray:
        return self.data.xmat[self._hand_body_id].reshape(3, 3).copy()

    def get_joint_positions(self) -> np.ndarray:
        positions = np.zeros(7)
        for i, jid in enumerate(self._joint_ids):
            positions[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return positions

    def set_joint_targets(self, targets: np.ndarray) -> None:
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = targets[i]

    def set_gripper(self, value: float) -> None:
        """Set gripper: 0=closed, 255=fully open (Menagerie convention)."""
        self.data.ctrl[self._gripper_actuator_id] = np.clip(value, 0, 255)

    def get_gripper_width(self) -> float:
        """Get total finger opening (sum of both finger joint positions)."""
        total = 0.0
        for jid in self._finger_joint_ids:
            total += self.data.qpos[self.model.jnt_qposadr[jid]]
        return total

    def get_object_positions(self) -> Dict[str, np.ndarray]:
        positions = {}
        for name, body_id in self._object_body_ids.items():
            positions[name] = self.data.xpos[body_id].copy()
        return positions

    def get_object_position(self, name: str) -> Optional[np.ndarray]:
        body_id = self._object_body_ids.get(name)
        if body_id is not None:
            return self.data.xpos[body_id].copy()
        return None

    def park_all_objects(self) -> None:
        """Move every product off-screen (resting on the floor, far from the cell)."""
        for idx, name in enumerate(self._object_joint_ids.keys()):
            jid = self._object_joint_ids[name]
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr:addr+3] = [PARK_BASE[0] + idx * 0.15, PARK_BASE[1], PARK_BASE[2]]
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
            dof_addr = self.model.jnt_dofadr[jid]
            self.data.qvel[dof_addr:dof_addr+6] = 0
        mujoco.mj_forward(self.model, self.data)

    def park_object(self, name: str) -> None:
        """Move a single product off-screen (used to clear a failed pick)."""
        jid = self._object_joint_ids.get(name)
        if jid is None:
            return
        self.release_all_welds()
        idx = list(self._object_joint_ids.keys()).index(name)
        addr = self.model.jnt_qposadr[jid]
        self.data.qpos[addr:addr+3] = [PARK_BASE[0] + idx * 0.15, PARK_BASE[1], PARK_BASE[2]]
        self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
        dof_addr = self.model.jnt_dofadr[jid]
        self.data.qvel[dof_addr:dof_addr+6] = 0
        mujoco.mj_forward(self.model, self.data)

    def build_episode_queue(self, num_objects: int, unknown_ratio: float = 0.3) -> List[str]:
        """
        Build a randomized queue of products for one episode.
        A fraction are unknown items (destined for the trash bin).
        """
        num_unknown = int(round(num_objects * unknown_ratio))
        num_unknown = min(num_unknown, len(UNKNOWN_OBJECTS))
        num_known = num_objects - num_unknown

        known = list(np.random.choice(KNOWN_OBJECTS, size=min(num_known, len(KNOWN_OBJECTS)),
                                      replace=False))
        unknown = list(np.random.choice(UNKNOWN_OBJECTS, size=num_unknown, replace=False))
        queue = known + unknown
        np.random.shuffle(queue)
        return queue

    def feed_object(self, name: str, render: bool = True) -> np.ndarray:
        """
        Deliver one product via the conveyor: spawn it at the belt entrance and
        slide it forward (+Y) into the staging area where the arm can pick it.
        Returns the settled world position.
        """
        jid = self._object_joint_ids[name]
        addr = self.model.jnt_qposadr[jid]
        dof_addr = self.model.jnt_dofadr[jid]

        # Spawn on the belt with a small random x offset for variety
        x0 = 0.5 + np.random.uniform(-0.03, 0.03)
        self.data.qpos[addr:addr+3] = [x0, CONVEYOR_SPAWN[1], CONVEYOR_SPAWN[2]]
        # Random yaw so items arrive in varied orientations
        yaw = np.random.uniform(0, 2 * np.pi)
        self.data.qpos[addr+3:addr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.data.qvel[dof_addr:dof_addr+6] = 0
        mujoco.mj_forward(self.model, self.data)

        belt_speed = 0.45  # m/s forward
        body_id = self._object_body_ids[name]
        reached = False

        # Slide forward until the item reaches the staging line (y >= STAGING_POINT[1])
        for step in range(700):
            ypos = self.data.xpos[body_id][1]
            if ypos < STAGING_POINT[1] - 0.02:
                # Drive the belt: keep forward velocity while on the belt/table
                self.data.qvel[dof_addr + 1] = belt_speed
            # Let physics run
            mujoco.mj_step(self.model, self.data)
            if render and self.viewer is not None and step % 8 == 0:
                self.viewer.sync()
                time.sleep(0.01)
            if self.data.xpos[body_id][1] >= STAGING_POINT[1] - 0.02:
                reached = True
                break

        # Fallback: if the belt didn't carry it all the way, place it at staging
        # so the item is never left stranded mid-belt.
        if not reached:
            self.data.qpos[addr:addr+3] = STAGING_POINT
            self.data.qvel[dof_addr:dof_addr+6] = 0
            mujoco.mj_forward(self.model, self.data)

        # Stop the item at the staging gate and let it settle to rest so the
        # perceived position is stable for grasping.
        self.data.qvel[dof_addr:dof_addr+6] = 0
        mujoco.mj_forward(self.model, self.data)
        for step in range(200):
            mujoco.mj_step(self.model, self.data)
            # Damp any residual drift/rolling so it comes fully to rest
            if step % 20 == 0:
                self.data.qvel[dof_addr:dof_addr+6] *= 0.3
            if render and self.viewer is not None and step % 10 == 0:
                self.viewer.sync()
                time.sleep(0.005)
        self.data.qvel[dof_addr:dof_addr+6] = 0
        mujoco.mj_forward(self.model, self.data)

        return self.data.xpos[body_id].copy()

    def reset_objects(self, randomize: bool = True, num_objects: int = 8) -> Dict[str, np.ndarray]:
        """Compatibility shim: park everything and return empty (feed happens per-item)."""
        self.park_all_objects()
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return {}

    def _generate_random_positions(self, count, x_range, y_range, min_sep):
        positions = []
        for _ in range(count):
            for attempt in range(200):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) >= min_sep for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)]))
        return positions

    def get_camera_intrinsics(self) -> np.ndarray:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        fovy = self.model.cam_fovy[cam_id]
        fy = self.render_height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        fx = fy
        cx = self.render_width / 2.0
        cy = self.render_height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def get_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        pos = self.data.cam_xpos[cam_id].copy()
        rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        return pos, rot

    def is_viewer_alive(self) -> bool:
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def shutdown(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()
        if self.seg_renderer is not None:
            self.seg_renderer.close()
        logger.info("Simulation shut down")


def model_info(model):
    return f"{model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, {model.ngeom} geoms"
