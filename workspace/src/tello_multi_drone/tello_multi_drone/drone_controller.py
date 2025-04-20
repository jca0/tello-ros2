import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
from tf_transformations import euler_from_quaternion
import math

class DroneController(Node):
    def __init__(self, drone_id):
        super().__init__(f'drone_controller_{drone_id}')
        self.drone_id = drone_id
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'/drone_{drone_id}/cmd_vel', 10)
        self.local_map_pub = self.create_publisher(
            OccupancyGrid, f'/drone_{drone_id}/local_map', 10)
        
        # Subscribers
        self.create_subscription(
            PoseStamped, f'/drone_{drone_id}/pose',
            self.pose_callback, 10)
        
        # Current state
        self.current_pose = None
        self.target_frontier = None
        
        # Navigation parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.goal_tolerance = 0.3  # meters
        self.obstacle_threshold = 0.5  # meters
        self.other_drone_threshold = 1.0  # meters
        
        # Subscribe to other drones' poses
        self.other_drone_poses = {}
        for i in range(3):  # Assuming 3 drones
            if i != drone_id:
                self.create_subscription(
                    PoseStamped,
                    f'/drone_{i}/pose',
                    lambda msg, drone_id=i: self.other_drone_pose_callback(msg, drone_id),
                    10
                )
    
    def pose_callback(self, msg):
        self.current_pose = msg
        
    def other_drone_pose_callback(self, msg, other_drone_id):
        self.other_drone_poses[other_drone_id] = msg
        
    def set_target_frontier(self, frontier):
        self.target_frontier = frontier
    
    def get_yaw(self):
        if self.current_pose is None:
            return 0.0
        orientation = self.current_pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        return yaw
    
    def distance_to_point(self, point):
        if self.current_pose is None:
            return float('inf')
        dx = point[0] - self.current_pose.pose.position.x
        dy = point[1] - self.current_pose.pose.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def angle_to_point(self, point):
        if self.current_pose is None:
            return 0.0
        dx = point[0] - self.current_pose.pose.position.x
        dy = point[1] - self.current_pose.pose.position.y
        target_angle = math.atan2(dy, dx)
        current_yaw = self.get_yaw()
        angle_diff = target_angle - current_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return angle_diff
    
    def check_collision_risk(self):
        if not self.other_drone_poses:
            return False
            
        for other_pose in self.other_drone_poses.values():
            dx = other_pose.pose.position.x - self.current_pose.pose.position.x
            dy = other_pose.pose.position.y - self.current_pose.pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.other_drone_threshold:
                return True
        return False
    
    def get_avoidance_vector(self):
        if not self.other_drone_poses:
            return (0.0, 0.0)
            
        avoid_x, avoid_y = 0.0, 0.0
        for other_pose in self.other_drone_poses.values():
            dx = self.current_pose.pose.position.x - other_pose.pose.position.x
            dy = self.current_pose.pose.position.y - other_pose.pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.other_drone_threshold:
                # Add repulsive vector, stronger when closer
                strength = (self.other_drone_threshold - distance) / self.other_drone_threshold
                avoid_x += dx * strength / distance
                avoid_y += dy * strength / distance
                
        return (avoid_x, avoid_y)
    
    def update(self):
        if self.target_frontier is None or self.current_pose is None:
            # No target or no pose information - hover in place
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return
            
        # Calculate distance and angle to target
        distance = self.distance_to_point(self.target_frontier)
        angle = self.angle_to_point(self.target_frontier)
        
        # Check if we've reached the target
        if distance < self.goal_tolerance:
            self.target_frontier = None
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return
            
        # Initialize command velocity
        cmd_vel = Twist()
        
        # Check for collision risks
        if self.check_collision_risk():
            # Get avoidance vector
            avoid_x, avoid_y = self.get_avoidance_vector()
            
            # Blend avoidance with goal-seeking behavior
            blend_factor = 0.7  # Prioritize collision avoidance
            cmd_vel.linear.x = (1 - blend_factor) * (self.max_linear_speed * math.cos(angle)) + \
                              blend_factor * avoid_x
            cmd_vel.linear.y = (1 - blend_factor) * (self.max_linear_speed * math.sin(angle)) + \
                              blend_factor * avoid_y
        else:
            # Normal navigation towards target
            # First align with the target
            if abs(angle) > 0.1:  # Small angle threshold
                cmd_vel.angular.z = self.max_angular_speed * (angle / math.pi)
            else:
                # Move towards target
                cmd_vel.linear.x = self.max_linear_speed * math.cos(angle)
                cmd_vel.linear.y = self.max_linear_speed * math.sin(angle)
        
        # Limit velocities
        cmd_vel.linear.x = max(min(cmd_vel.linear.x, self.max_linear_speed), -self.max_linear_speed)
        cmd_vel.linear.y = max(min(cmd_vel.linear.y, self.max_linear_speed), -self.max_linear_speed)
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        # Publish command velocity
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Update local map (assuming we have sensor data processing elsewhere)
        # This would typically involve processing sensor data and updating the occupancy grid
        # self.update_local_map()
        