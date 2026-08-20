"""
MuJoCo Simulation Environment
================================
Sets up and manages the complete MuJoCo simulation world:
physics, Franka Panda robot, table, objects, camera, and lighting.

This replaces the Isaac Sim backend with MuJoCo, which:
- Works on any GPU (AMD, Intel, NVIDIA) or CPU
- Provides accurate physics simulation
- Includes built-in RGB and depth rendering
- Is the standard in manipulation research (DeepMind, OpenAI)
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

# Default scene path
SCENE_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "scene.xml"


class SimulationEnvironment:
    """
    MuJoCo-based simulation environment manager.
    Handles model loading, physics stepping, rendering, and object management.
    """
    
    # Object names in the scene
    OBJECT_NAMES = ["red_box", "blue_box", "green_cylinder", "yellow_sphere"]
    
    # Home joint configuration for Franka Panda
    HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    GRIPPER_OPEN = np.array([0.04, 0.04])
    GRIPPER_CLOSED = np.array([0.0, 0.0])
    
    def __init__(self, config_path: str = "config/simulation.yaml"):
        """
        Initialize the simulation environment.
        
        Args:
            config_path: Path to simulation configuration YAML
        """
        self.config = self._load_config(config_path)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.depth_renderer: Optional[mujoco.Renderer] = None
        self.viewer = None
        
        # Rendering parameters
        self.render_width = 640
        self.render_height = 480
        self.camera_name = "overhead_cam"
        
        # Object body IDs (cached after load)
        self._object_body_ids: Dict[str, int] = {}
        self._object_joint_ids: Dict[str, int] = {}
        
        # Robot joint and actuator IDs
        self._joint_ids: List[int] = []
        self._actuator_ids: List[int] = []
        self._finger_actuator_ids: List[int] = []
        
        # EE site
        self._ee_site_id: int = -1
        
        # State
        self.is_initialized = False
        self.step_count = 0
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def initialize(self, scene_path: str = None, headless: bool = False) -> None:
        """
        Load the MuJoCo model and initialize rendering.
        
        Args:
            scene_path: Path to scene XML (uses default if None)
            headless: If True, skip viewer creation
        """
        xml_path = Path(scene_path) if scene_path else SCENE_XML_PATH
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")
        
        logger.info(f"Loading MuJoCo scene: {xml_path}")
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Set up renderers
        self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.depth_renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.depth_renderer.enable_depth_rendering()
        
        # Cache body/joint/actuator IDs
        self._cache_ids()
        
        # Reset to home configuration
        self.reset_to_home()
        
        # Create viewer for visualization (unless headless)
        if not headless:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                logger.info("MuJoCo viewer launched")
            except Exception as e:
                logger.warning(f"Could not launch viewer: {e}")
                self.viewer = None
        
        self.is_initialized = True
        
        logger.info(f"Simulation initialized:")
        logger.info(f"  Bodies: {self.model.nbody}, Joints: {self.model.njnt}")
        logger.info(f"  Actuators: {self.model.nu}, DOF: {self.model.nv}")
        logger.info(f"  Objects: {len(self._object_body_ids)}")
        logger.info(f"  Render: {self.render_width}x{self.render_height}")
    
    def _cache_ids(self) -> None:
        """Cache frequently-used body, joint, and actuator IDs."""
        # Object bodies
        for name in self.OBJECT_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._object_body_ids[name] = body_id
            
            # Joint IDs for objects (free joints)
            joint_name = name.replace("cylinder", "cyl").replace("sphere", "sph") + "_joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self._object_joint_ids[name] = joint_id
        
        # Robot joint IDs
        for i in range(1, 8):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i}")
            self._joint_ids.append(jid)
        
        # Robot actuator IDs (joints)
        for i in range(1, 8):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint{i}")
            self._actuator_ids.append(aid)
        
        # Finger actuators
        for i in range(1, 3):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_finger{i}")
            self._finger_actuator_ids.append(aid)
        
        # EE site
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    
    def reset_to_home(self) -> None:
        """Reset robot to home configuration with gripper open."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        
        # Set control to home position
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = self.HOME_QPOS[i]
        
        # Open gripper (0 = open in our convention)
        for aid in self._finger_actuator_ids:
            self.data.ctrl[aid] = 0.0
        
        self.step_count = 0
    
    def step(self, render: bool = True) -> None:
        """
        Advance simulation by one timestep.
        
        Args:
            render: Whether to sync the viewer
        """
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        
        if render and self.viewer is not None:
            self.viewer.sync()
    
    def step_n(self, n: int, render: bool = True) -> None:
        """Step simulation n times with optional real-time rendering."""
        render_interval = max(1, n // 30)  # Show ~30 frames during the stepping
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            if render and self.viewer is not None and i % render_interval == 0:
                self.viewer.sync()
                time.sleep(0.01)
    
    def render_rgb(self, camera: str = None) -> np.ndarray:
        """
        Render an RGB image from the specified camera.
        
        Args:
            camera: Camera name (default: overhead_cam)
            
        Returns:
            (H, W, 3) uint8 RGB image
        """
        cam = camera or self.camera_name
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()
    
    def render_depth(self, camera: str = None) -> np.ndarray:
        """
        Render a depth image from the specified camera.
        
        Args:
            camera: Camera name (default: overhead_cam)
            
        Returns:
            (H, W) float32 depth image in meters
        """
        cam = camera or self.camera_name
        self.depth_renderer.update_scene(self.data, camera=cam)
        return self.depth_renderer.render().copy().astype(np.float32)
    
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position [x, y, z]."""
        return self.data.site_xpos[self._ee_site_id].copy()
    
    def get_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as rotation matrix (3x3)."""
        return self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current 7 joint positions."""
        positions = np.zeros(7)
        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            positions[i] = self.data.qpos[addr]
        return positions
    
    def set_joint_targets(self, targets: np.ndarray) -> None:
        """
        Set joint position targets for the robot arm.
        
        Args:
            targets: (7,) array of target joint angles
        """
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = targets[i]
    
    def set_gripper(self, width: float) -> None:
        """
        Set gripper opening.
        
        Args:
            width: Joint value (0=open/spread, 0.04=closed/together)
        """
        width = np.clip(width, 0.0, 0.04)
        for aid in self._finger_actuator_ids:
            self.data.ctrl[aid] = width
    
    def get_object_positions(self) -> Dict[str, np.ndarray]:
        """
        Get current world positions of all objects.
        Used as ground truth for evaluation.
        
        Returns:
            Dict mapping object name to [x, y, z] position
        """
        positions = {}
        for name, body_id in self._object_body_ids.items():
            positions[name] = self.data.xpos[body_id].copy()
        return positions
    
    def get_object_position(self, name: str) -> Optional[np.ndarray]:
        """Get position of a specific object."""
        body_id = self._object_body_ids.get(name)
        if body_id is not None:
            return self.data.xpos[body_id].copy()
        return None
    
    def reset_objects(self, randomize: bool = True) -> Dict[str, np.ndarray]:
        """
        Reset object positions for a new episode.
        
        Args:
            randomize: If True, place objects at random positions on table
            
        Returns:
            Dict of object_name → ground truth position
        """
        # Table surface height
        table_z = 0.32 + 0.025  # table top + half object height
        
        # Workspace bounds - PICK ZONE only (center of table, away from bins)
        # Bins are at: x=[0.3,0.55], y=[-0.25,0.25]
        # Objects spawn in center: x=[0.4,0.55], y=[-0.12,0.12]
        x_range = (0.40, 0.55)
        y_range = (-0.12, 0.12)
        min_separation = 0.06
        
        if randomize:
            positions = self._generate_random_positions(
                len(self.OBJECT_NAMES), x_range, y_range, min_separation
            )
        else:
            # Default grid layout
            positions = [
                np.array([0.45, -0.1]),
                np.array([0.55, 0.1]),
                np.array([0.4, 0.15]),
                np.array([0.6, -0.05]),
            ]
        
        # Set object positions via qpos
        for i, (name, joint_id) in enumerate(self._object_joint_ids.items()):
            addr = self.model.jnt_qposadr[joint_id]
            # Free joint qpos: [x, y, z, qw, qx, qy, qz]
            self.data.qpos[addr:addr+3] = [positions[i][0], positions[i][1], table_z]
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]  # No rotation
            # Zero velocities
            dof_addr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_addr:dof_addr+6] = 0
        
        # Forward dynamics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        # Step a bit to let objects settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Return ground truth positions
        return self.get_object_positions()
    
    def _generate_random_positions(
        self,
        count: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        min_sep: float,
    ) -> List[np.ndarray]:
        """Generate non-overlapping random positions on the table."""
        positions = []
        max_attempts = 200
        
        for _ in range(count):
            for attempt in range(max_attempts):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                pos = np.array([x, y])
                
                # Check separation
                valid = all(
                    np.linalg.norm(pos - p) >= min_sep for p in positions
                )
                if valid:
                    positions.append(pos)
                    break
            else:
                # Fallback
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                positions.append(np.array([x, y]))
        
        return positions
    
    def get_camera_intrinsics(self) -> np.ndarray:
        """
        Compute camera intrinsic matrix from MuJoCo camera parameters.
        
        MuJoCo uses vertical field of view (fovy) and image dimensions.
        
        Returns:
            3x3 intrinsic matrix K
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        fovy = self.model.cam_fovy[cam_id]
        
        # Compute focal length from fov
        # fy = height / (2 * tan(fovy/2))
        fy = self.render_height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        fx = fy  # Square pixels
        
        cx = self.render_width / 2.0
        cy = self.render_height / 2.0
        
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        
        return K
    
    def get_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and rotation matrix in world frame.
        
        Returns:
            (position [3], rotation_matrix [3x3])
            Rotation is camera-to-world.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        
        # Camera position in world
        pos = self.data.cam_xpos[cam_id].copy()
        
        # Camera rotation matrix (3x3, row-major in MuJoCo)
        rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        
        return pos, rot
    
    def is_viewer_alive(self) -> bool:
        """Check if the viewer window is still open."""
        if self.viewer is None:
            return True  # Headless mode, always "alive"
        return self.viewer.is_running()
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()
        logger.info("Simulation shut down")
