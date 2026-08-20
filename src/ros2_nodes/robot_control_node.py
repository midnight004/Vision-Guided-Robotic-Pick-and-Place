"""
ROS2 Robot Control Node
=========================
Subscribes to target poses from the perception pipeline and executes robot motions.

Topics Subscribed:
    /robot/target_pose       - Pose of next pick target
    /robot/command           - String commands (pick, place, home)
    /vision/localized_objects - PoseArray from perception

Topics Published:
    /robot/state             - Current robot state
    /robot/ee_pose           - Current end-effector pose
    /robot/task_result       - Task execution results
"""

import numpy as np
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Pose, PoseStamped, PoseArray
    from std_msgs.msg import String, Bool
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class RobotControlNode:
    """
    ROS2 node that interfaces between the task planner and robot controller.
    
    Receives target poses and commands, executes robot motions, and publishes state.
    Acts as the bridge between ROS2 topics and the Isaac Sim robot controller.
    """
    
    def __init__(self):
        """Initialize robot control node."""
        self.node = None
        
        # State
        self.current_target = None
        self.current_command = None
        self.is_executing = False
        
        # Callbacks
        self.on_target_received = None
        self.on_command_received = None
        
        if ROS2_AVAILABLE:
            self._init_ros2()
    
    def _init_ros2(self) -> None:
        """Initialize ROS2 node, subscribers, and publishers."""
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node('robot_controller')
        
        # Subscribers
        self.sub_target = self.node.create_subscription(
            PoseStamped, '/robot/target_pose',
            self._target_callback, 10
        )
        self.sub_command = self.node.create_subscription(
            String, '/robot/command',
            self._command_callback, 10
        )
        
        # Publishers
        self.pub_state = self.node.create_publisher(
            String, '/robot/state', 10
        )
        self.pub_ee_pose = self.node.create_publisher(
            PoseStamped, '/robot/ee_pose', 10
        )
        self.pub_result = self.node.create_publisher(
            String, '/robot/task_result', 10
        )
        self.pub_busy = self.node.create_publisher(
            Bool, '/robot/busy', 10
        )
        
        self.node.get_logger().info("Robot control node initialized")
    
    def _target_callback(self, msg) -> None:
        """Handle incoming target pose."""
        pos = msg.pose.position
        self.current_target = np.array([pos.x, pos.y, pos.z])
        
        if self.on_target_received:
            self.on_target_received(self.current_target)
        
        if self.node:
            self.node.get_logger().info(
                f"Target received: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
            )
    
    def _command_callback(self, msg) -> None:
        """Handle incoming command."""
        self.current_command = msg.data
        
        if self.on_command_received:
            self.on_command_received(msg.data)
        
        if self.node:
            self.node.get_logger().info(f"Command received: {msg.data}")
    
    def publish_state(self, state: str) -> None:
        """Publish robot state."""
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        msg = String()
        msg.data = state
        self.pub_state.publish(msg)
    
    def publish_ee_pose(self, position: np.ndarray, orientation: np.ndarray = None) -> None:
        """Publish current end-effector pose."""
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        if orientation is not None:
            msg.pose.orientation.w = float(orientation[0])
            msg.pose.orientation.x = float(orientation[1])
            msg.pose.orientation.y = float(orientation[2])
            msg.pose.orientation.z = float(orientation[3])
        else:
            msg.pose.orientation.w = 1.0
        
        self.pub_ee_pose.publish(msg)
    
    def publish_task_result(self, success: bool, details: str = "") -> None:
        """Publish task execution result."""
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        msg = String()
        msg.data = f"{'SUCCESS' if success else 'FAILED'}: {details}"
        self.pub_result.publish(msg)
    
    def publish_busy(self, busy: bool) -> None:
        """Publish busy state."""
        if not ROS2_AVAILABLE or self.node is None:
            return
        msg = Bool()
        msg.data = busy
        self.pub_busy.publish(msg)
    
    def spin_once(self) -> None:
        """Process callbacks."""
        if ROS2_AVAILABLE and self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.001)
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        if ROS2_AVAILABLE and self.node is not None:
            self.node.destroy_node()
