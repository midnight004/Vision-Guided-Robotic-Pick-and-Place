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

# 8 products in the factory
OBJECT_NAMES = [
    "red_box", "blue_box", "green_cylinder", "yellow_sphere",
    "red_can", "blue_capsule", "green_box", "yellow_bottle"
]

OBJECT_JOINT_NAMES = {
    "red_box": "red_box_joint",
    "blue_box": "blue_box_joint",
    "green_cylinder": "green_cyl_joint",
    "yellow_sphere": "yellow_sph_joint",
    "red_can": "red_can_joint",
    "blue_capsule": "blue_cap_joint",
    "green_box": "green_box_joint",
    "yellow_bottle": "yellow_bot_joint",
}

OBJECT_GEOM_NAMES = {
    "red_box": "red_box_geom",
    "blue_box": "blue_box_geom",
    "green_cylinder": "green_cyl_geom",
    "yellow_sphere": "yellow_sph_geom",
    "red_can": "red_can_geom",
    "blue_capsule": "blue_cap_geom",
    "green_box": "green_box_geom",
    "yellow_bottle": "yellow_bot_geom",
}

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

    def reset_objects(self, randomize: bool = True, num_objects: int = 8) -> Dict[str, np.ndarray]:
        """Reset product positions on the table."""
        table_z = 0.295 + 0.035  # table surface + half object (bigger objects)
        x_range = (0.38, 0.62)
        y_range = (-0.16, 0.16)
        min_separation = 0.14  # More spacing so objects don't merge in detection

        all_names = list(self._object_joint_ids.keys())

        # Pick a diverse subset: one from each color if possible
        if num_objects <= 4:
            # Use the 4 primary distinct-color products
            active_names = ["red_box", "blue_box", "green_cylinder", "yellow_sphere"][:num_objects]
        else:
            active_names = all_names[:num_objects]

        if randomize:
            positions = self._generate_random_positions(len(active_names), x_range, y_range, min_separation)
        else:
            positions = [
                np.array([0.42, -0.12]), np.array([0.58, 0.12]),
                np.array([0.42, 0.12]), np.array([0.58, -0.12]),
                np.array([0.5, 0.0]), np.array([0.5, 0.15]),
                np.array([0.45, -0.05]), np.array([0.55, 0.05]),
            ][:len(active_names)]

        # Place active objects on the table
        for i, name in enumerate(active_names):
            jid = self._object_joint_ids[name]
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr:addr+3] = [positions[i][0], positions[i][1], table_z]
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
            dof_addr = self.model.jnt_dofadr[jid]
            self.data.qvel[dof_addr:dof_addr+6] = 0

        # Park unused objects far off-screen (under the floor, out of camera view)
        for name in all_names:
            if name not in active_names:
                jid = self._object_joint_ids[name]
                addr = self.model.jnt_qposadr[jid]
                # Hide far away below the floor
                idx = all_names.index(name)
                self.data.qpos[addr:addr+3] = [2.0 + idx * 0.1, 2.0, -0.5]
                self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
                dof_addr = self.model.jnt_dofadr[jid]
                self.data.qvel[dof_addr:dof_addr+6] = 0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Return only active object positions
        return {name: self.data.xpos[self._object_body_ids[name]].copy() for name in active_names}

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
