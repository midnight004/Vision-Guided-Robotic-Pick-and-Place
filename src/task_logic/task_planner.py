"""
Task Planner
==============
Orchestrates the object sorting task by selecting which object to pick next
and managing the overall task state.

Responsibilities:
    - Maintain task state across frames
    - Select next target object based on priority rules
    - Track which objects have been handled
    - Determine when task is complete
    - Handle failure recovery (retry logic)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ..localization.localizer import LocalizedObject
from .sorting_rules import SortingRules
from ..utils.logger import setup_logger

logger = setup_logger("task_planner")


class TaskState(Enum):
    """Overall task state."""
    IDLE = "idle"
    SELECTING = "selecting_target"
    EXECUTING = "executing_pick_place"
    WAITING = "waiting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TaskTarget:
    """A planned pick-and-place target."""
    object_name: str
    class_name: str
    pick_position: np.ndarray
    place_position: np.ndarray
    track_id: Optional[int] = None
    attempts: int = 0
    max_attempts: int = 3
    
    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts


class TaskPlanner:
    """
    Plans and manages the pick-and-place sorting task.
    
    Algorithm:
        1. Receive localized objects from perception
        2. Filter out already-handled objects
        3. Select highest-priority unhandled object
        4. Determine destination from sorting rules
        5. Issue pick-and-place command
        6. Track success/failure
        7. Repeat until all objects handled or timeout
    """
    
    def __init__(self, config: dict, sorting_rules: SortingRules):
        """
        Args:
            config: Task configuration
            sorting_rules: SortingRules instance
        """
        self.config = config['task']
        self.sorting_rules = sorting_rules
        
        # Task parameters
        exec_cfg = self.config['execution']
        self.max_attempts = exec_cfg['max_attempts_per_object']
        self.timeout_per_object = exec_cfg['timeout_per_object']
        self.total_timeout = exec_cfg['total_timeout']
        self.pause_between = exec_cfg['pause_between_picks']
        
        # Priority settings
        priority_cfg = self.config['priority']
        self.priority_method = priority_cfg['method']
        self.class_order = priority_cfg.get('class_order', [])
        
        # State
        self.state = TaskState.IDLE
        self.handled_objects: List[str] = []  # Track IDs or names of completed objects
        self.current_target: Optional[TaskTarget] = None
        self.task_queue: List[TaskTarget] = []
        self.task_start_time: Optional[float] = None
        
        # Results
        self.completed_targets: List[TaskTarget] = []
        self.failed_targets: List[TaskTarget] = []
        
        logger.info(f"TaskPlanner initialized (priority={self.priority_method})")
    
    def start_task(self) -> None:
        """Start a new task episode."""
        self.state = TaskState.SELECTING
        self.handled_objects.clear()
        self.task_queue.clear()
        self.completed_targets.clear()
        self.failed_targets.clear()
        self.current_target = None
        self.task_start_time = time.time()
        self.sorting_rules.reset()
        
        logger.info("Task started")
    
    def select_next_target(
        self,
        localized_objects: List[LocalizedObject],
        robot_position: Optional[np.ndarray] = None,
    ) -> Optional[TaskTarget]:
        """
        Select the next object to pick based on priority and availability.
        
        Args:
            localized_objects: Currently visible/localized objects
            robot_position: Current robot end-effector position (for nearest priority)
            
        Returns:
            TaskTarget if an object is selected, None if task complete
        """
        if self._check_timeout():
            self.state = TaskState.FAILED
            logger.warning("Task timeout reached!")
            return None
        
        # Filter objects that haven't been handled yet
        # Track by class name (we have one of each class)
        available = []
        for obj in localized_objects:
            if obj.class_name not in self.handled_objects:
                available.append(obj)
        
        if not available:
            self.state = TaskState.COMPLETE
            logger.info("All objects handled - task complete!")
            return None
        
        # Select by priority
        selected = self._select_by_priority(available, robot_position)
        
        if selected is None:
            return None
        
        # Determine destination
        destination = self.sorting_rules.get_destination(selected.class_name)
        if destination is None:
            logger.warning(f"No destination for {selected.class_name}")
            return None
        
        # Create target
        target = TaskTarget(
            object_name=f"{selected.class_name}_{selected.track_id or 0}",
            class_name=selected.class_name,
            pick_position=selected.position_world.copy(),
            place_position=destination,
            track_id=selected.track_id,
            max_attempts=self.max_attempts,
        )
        
        self.current_target = target
        self.state = TaskState.EXECUTING
        
        logger.info(
            f"Selected target: {target.object_name} "
            f"-> pick at {target.pick_position} "
            f"-> place at {target.place_position}"
        )
        
        return target
    
    def report_result(self, success: bool) -> None:
        """
        Report the result of a pick-and-place attempt.
        
        Args:
            success: Whether the operation succeeded
        """
        if self.current_target is None:
            return
        
        self.current_target.attempts += 1
        
        if success:
            # Mark this class as handled
            self.handled_objects.append(self.current_target.class_name)
            self.completed_targets.append(self.current_target)
            logger.info(f"Target {self.current_target.class_name} completed successfully")
        else:
            if not self.current_target.can_retry:
                self.failed_targets.append(self.current_target)
                self.handled_objects.append(self.current_target.class_name)
                logger.warning(
                    f"Target {self.current_target.class_name} failed "
                    f"after {self.current_target.attempts} attempts"
                )
            else:
                logger.info(
                    f"Retrying target {self.current_target.object_name} "
                    f"(attempt {self.current_target.attempts + 1})"
                )
                return  # Don't clear target, will retry
        
        self.current_target = None
        self.state = TaskState.SELECTING
    
    def _select_by_priority(
        self,
        objects: List[LocalizedObject],
        robot_position: Optional[np.ndarray],
    ) -> Optional[LocalizedObject]:
        """Select object based on priority method."""
        if not objects:
            return None
        
        if self.priority_method == "nearest" and robot_position is not None:
            # Pick closest to robot
            distances = [
                np.linalg.norm(obj.position_world - robot_position)
                for obj in objects
            ]
            return objects[np.argmin(distances)]
        
        elif self.priority_method == "class_order":
            # Pick by class priority order
            for cls in self.class_order:
                for obj in objects:
                    if obj.class_name == cls:
                        return obj
            return objects[0]
        
        elif self.priority_method == "largest":
            # Pick largest bounding box (most confident detection)
            areas = [
                (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
                for obj in objects
            ]
            return objects[np.argmax(areas)]
        
        else:
            # Default: first available
            return objects[0]
    
    def _check_timeout(self) -> bool:
        """Check if total task timeout has been reached."""
        if self.task_start_time is None:
            return False
        elapsed = time.time() - self.task_start_time
        return elapsed > self.total_timeout
    
    def is_complete(self) -> bool:
        """Check if task is complete or failed."""
        return self.state in (TaskState.COMPLETE, TaskState.FAILED)
    
    def get_progress(self) -> Dict:
        """Get current task progress."""
        total = len(self.completed_targets) + len(self.failed_targets)
        elapsed = 0
        if self.task_start_time:
            elapsed = time.time() - self.task_start_time
        
        return {
            'state': self.state.value,
            'completed': len(self.completed_targets),
            'failed': len(self.failed_targets),
            'remaining': max(0, 4 - total),  # Assuming ~4 objects
            'elapsed_time': elapsed,
            'success_rate': (
                len(self.completed_targets) / max(1, total)
            ),
        }
