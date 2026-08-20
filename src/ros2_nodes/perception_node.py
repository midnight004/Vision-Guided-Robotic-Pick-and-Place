"""
ROS2 Perception Node
======================
Publishes detection and localization results to ROS2 topics.
Subscribes to camera data and runs the perception pipeline.

Topics Published:
    /vision/detections           - DetectionArray (custom message)
    /vision/localized_objects    - PoseArray of localized objects
    /vision/detection_image      - Image with detection overlays
    /camera/rgb                  - Raw camera RGB image
    /camera/depth                - Depth image

Topics Subscribed:
    (Camera data comes directly from Isaac Sim in this implementation)

TF2 Frames Published:
    camera_frame → world_frame
    object_frame → world_frame (for each detected object)
"""

import numpy as np
from typing import Optional, List

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, TransformStamped
    from std_msgs.msg import Header, String
    from visualization_msgs.msg import MarkerArray, Marker
    import tf2_ros
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class PerceptionNode:
    """
    ROS2 node that runs the perception pipeline and publishes results.
    
    This node:
        1. Receives camera frames (from Isaac Sim or ROS2 topic)
        2. Runs detection (YOLO)
        3. Runs tracking (ByteTrack)
        4. Runs localization (depth → 3D)
        5. Publishes results to ROS2 topics
        6. Broadcasts TF2 transforms (camera frame, object frames)
    """
    
    def __init__(self):
        """Initialize the perception node."""
        self.node = None
        self.bridge = None
        self.tf_broadcaster = None
        
        # Publishers
        self.pub_detection_image = None
        self.pub_rgb = None
        self.pub_depth = None
        self.pub_poses = None
        self.pub_markers = None
        self.pub_status = None
        
        if ROS2_AVAILABLE:
            self._init_ros2()
    
    def _init_ros2(self) -> None:
        """Initialize ROS2 node and publishers."""
        rclpy.init()
        self.node = rclpy.create_node('vision_perception')
        self.bridge = CvBridge()
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.pub_rgb = self.node.create_publisher(
            Image, '/camera/rgb', sensor_qos)
        self.pub_depth = self.node.create_publisher(
            Image, '/camera/depth', sensor_qos)
        self.pub_detection_image = self.node.create_publisher(
            Image, '/vision/detection_image', sensor_qos)
        self.pub_poses = self.node.create_publisher(
            PoseArray, '/vision/localized_objects', 10)
        self.pub_markers = self.node.create_publisher(
            MarkerArray, '/vision/markers', 10)
        self.pub_status = self.node.create_publisher(
            String, '/vision/status', 10)
        
        # TF2 broadcaster for coordinate frames
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        
        self.node.get_logger().info("Perception node initialized")
    
    def publish_camera_frame(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        Publish raw camera data to ROS2 topics.
        
        Args:
            rgb_image: BGR image (H, W, 3)
            depth_image: Depth image (H, W) float32, meters
            timestamp: Frame timestamp
        """
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        
        # Publish RGB
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        rgb_msg.header = header
        self.pub_rgb.publish(rgb_msg)
        
        # Publish depth
        if depth_image is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(
                depth_image.astype(np.float32), encoding='32FC1'
            )
            depth_msg.header = header
            self.pub_depth.publish(depth_msg)
    
    def publish_detections(
        self,
        detection_image: np.ndarray,
        localized_objects: List,
        timestamp: float = 0.0,
    ) -> None:
        """
        Publish detection results and localized object poses.
        
        Args:
            detection_image: Image with detection overlays
            localized_objects: List of LocalizedObject instances
            timestamp: Frame timestamp
        """
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = 'world'
        
        # Publish detection image
        det_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')
        det_msg.header = header
        det_msg.header.frame_id = 'camera_frame'
        self.pub_detection_image.publish(det_msg)
        
        # Publish pose array
        pose_array = PoseArray()
        pose_array.header = header
        
        for obj in localized_objects:
            pose = Pose()
            pose.position = Point(
                x=float(obj.position_world[0]),
                y=float(obj.position_world[1]),
                z=float(obj.position_world[2]),
            )
            pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            pose_array.poses.append(pose)
        
        self.pub_poses.publish(pose_array)
        
        # Publish markers for visualization in RViz
        markers = self._create_markers(localized_objects, header)
        self.pub_markers.publish(markers)
    
    def broadcast_camera_tf(
        self,
        camera_position: np.ndarray,
        camera_orientation: np.ndarray,
    ) -> None:
        """
        Broadcast camera-to-world TF2 transform.
        
        Args:
            camera_position: [x, y, z] camera position in world frame
            camera_orientation: [w, x, y, z] quaternion
        """
        if not ROS2_AVAILABLE or self.tf_broadcaster is None:
            return
        
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera_frame'
        
        t.transform.translation.x = float(camera_position[0])
        t.transform.translation.y = float(camera_position[1])
        t.transform.translation.z = float(camera_position[2])
        
        t.transform.rotation.w = float(camera_orientation[0])
        t.transform.rotation.x = float(camera_orientation[1])
        t.transform.rotation.y = float(camera_orientation[2])
        t.transform.rotation.z = float(camera_orientation[3])
        
        self.tf_broadcaster.sendTransform(t)
    
    def broadcast_object_tf(
        self,
        object_name: str,
        position: np.ndarray,
    ) -> None:
        """Broadcast a detected object's position as a TF frame."""
        if not ROS2_AVAILABLE or self.tf_broadcaster is None:
            return
        
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = f'object_{object_name}'
        
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
    
    def _create_markers(self, localized_objects: List, header) -> 'MarkerArray':
        """Create RViz markers for detected objects."""
        marker_array = MarkerArray()
        
        color_map = {
            'red_box': (1.0, 0.0, 0.0),
            'blue_box': (0.0, 0.0, 1.0),
            'green_cylinder': (0.0, 1.0, 0.0),
            'yellow_sphere': (1.0, 1.0, 0.0),
        }
        
        for i, obj in enumerate(localized_objects):
            marker = Marker()
            marker.header = header
            marker.ns = 'detected_objects'
            marker.id = i
            marker.type = Marker.CUBE if 'box' in obj.class_name else Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position = Point(
                x=float(obj.position_world[0]),
                y=float(obj.position_world[1]),
                z=float(obj.position_world[2]),
            )
            marker.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            
            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.04
            
            color = color_map.get(obj.class_name, (0.5, 0.5, 0.5))
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8
            
            marker.lifetime.sec = 1
            
            marker_array.markers.append(marker)
        
        return marker_array
    
    def publish_status(self, status: str) -> None:
        """Publish a status message."""
        if not ROS2_AVAILABLE or self.node is None:
            return
        
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)
    
    def spin_once(self) -> None:
        """Process ROS2 callbacks once."""
        if ROS2_AVAILABLE and self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.001)
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        if ROS2_AVAILABLE and self.node is not None:
            self.node.destroy_node()
            rclpy.shutdown()
