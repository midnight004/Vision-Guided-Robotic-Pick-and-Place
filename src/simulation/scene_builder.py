"""
Scene Builder (MuJoCo)
========================
Utility functions for dynamic scene manipulation:
- Adding/removing objects at runtime
- Modifying object properties
- Generating scene variations for robustness testing

Note: The base scene is defined in assets/scene.xml.
This module handles runtime modifications via MjData manipulation.
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("scene_builder")


class SceneBuilder:
    """
    Dynamic scene manipulation for the MuJoCo simulation.
    Handles object positioning, randomization, and scene configuration.
    """
    
    # Object spawn configurations
    OBJECT_CONFIGS = {
        "red_box": {
            "size": np.array([0.02, 0.02, 0.02]),
            "color": np.array([0.85, 0.12, 0.1]),
            "shape": "box",
        },
        "blue_box": {
            "size": np.array([0.02, 0.02, 0.02]),
            "color": np.array([0.1, 0.15, 0.85]),
            "shape": "box",
        },
        "green_cylinder": {
            "size": np.array([0.018, 0.018, 0.025]),
            "color": np.array([0.1, 0.75, 0.15]),
            "shape": "cylinder",
        },
        "yellow_sphere": {
            "size": np.array([0.02, 0.02, 0.02]),
            "color": np.array([0.92, 0.85, 0.1]),
            "shape": "sphere",
        },
    }
    
    # Destination positions (centers of the bins)
    DESTINATIONS = {
        "red_destination": np.array([0.3, -0.25, 0.35]),
        "blue_destination": np.array([0.3, 0.25, 0.35]),
        "green_destination": np.array([0.55, -0.25, 0.35]),
        "yellow_destination": np.array([0.55, 0.25, 0.35]),
    }
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
    
    @staticmethod
    def generate_scene_config(
        num_objects: int = 4,
        randomize_positions: bool = True,
        randomize_orientations: bool = False,
    ) -> Dict:
        """
        Generate a scene configuration for an episode.
        
        Args:
            num_objects: Number of objects to place
            randomize_positions: Randomize XY positions
            randomize_orientations: Randomize object rotations
            
        Returns:
            Configuration dictionary with object placements
        """
        config = {
            "objects": {},
            "timestamp": None,
        }
        
        # Table workspace bounds
        x_range = (0.35, 0.65)
        y_range = (-0.25, 0.25)
        table_z = 0.415  # Table surface + half object
        min_sep = 0.08
        
        positions = []
        object_names = list(SceneBuilder.OBJECT_CONFIGS.keys())[:num_objects]
        
        for name in object_names:
            # Generate position
            if randomize_positions:
                for _ in range(100):
                    x = np.random.uniform(*x_range)
                    y = np.random.uniform(*y_range)
                    pos = np.array([x, y])
                    if all(np.linalg.norm(pos - p) >= min_sep for p in positions):
                        positions.append(pos)
                        break
                else:
                    positions.append(np.array([
                        np.random.uniform(*x_range),
                        np.random.uniform(*y_range)
                    ]))
            else:
                # Default positions from scene XML
                defaults = {
                    "red_box": np.array([0.45, -0.1]),
                    "blue_box": np.array([0.55, 0.1]),
                    "green_cylinder": np.array([0.4, 0.15]),
                    "yellow_sphere": np.array([0.6, -0.05]),
                }
                positions.append(defaults[name])
            
            # Orientation
            if randomize_orientations:
                yaw = np.random.uniform(0, 2 * np.pi)
                quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            config["objects"][name] = {
                "position": np.array([positions[-1][0], positions[-1][1], table_z]),
                "orientation": quat,
            }
        
        return config
    
    def apply_config(self, config: Dict) -> None:
        """
        Apply a scene configuration to the simulation.
        
        Args:
            config: Configuration from generate_scene_config()
        """
        for name, obj_cfg in config["objects"].items():
            joint_name = name.replace("cylinder", "cyl").replace("sphere", "sph") + "_joint"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            
            if joint_id < 0:
                logger.warning(f"Joint not found for {name}")
                continue
            
            addr = self.model.jnt_qposadr[joint_id]
            pos = obj_cfg["position"]
            quat = obj_cfg["orientation"]
            
            self.data.qpos[addr:addr+3] = pos
            self.data.qpos[addr+3:addr+7] = quat
            
            # Zero velocity
            dof_addr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_addr:dof_addr+6] = 0
        
        mujoco.mj_forward(self.model, self.data)
        logger.debug(f"Applied scene config with {len(config['objects'])} objects")
    
    @staticmethod
    def get_destination_for_class(class_name: str) -> Optional[np.ndarray]:
        """Get the destination position for a given object class."""
        mapping = {
            "red_box": "red_destination",
            "blue_box": "blue_destination",
            "green_cylinder": "green_destination",
            "yellow_sphere": "yellow_destination",
        }
        dest_name = mapping.get(class_name)
        if dest_name:
            return SceneBuilder.DESTINATIONS[dest_name].copy()
        return None
