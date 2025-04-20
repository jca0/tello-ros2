import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math

class MapManager(Node):
    def __init__(self):
        super().__init__('map_manager')
        
        # Map parameters
        self.map_size = (100, 100)  # Adjustable size
        self.resolution = 0.1  # meters per cell
        self.map = np.full(self.map_size, -1)  # -1: unknown, 0: free, 100: occupied
        
        # Map origin (center of the map)
        self.origin_x = -(self.map_size[0] * self.resolution) / 2
        self.origin_y = -(self.map_size[1] * self.resolution) / 2
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE)
        
        # Publisher for the merged map
        self.map_pub = self.create_publisher(
            OccupancyGrid, 
            '/global_map', 
            qos_profile)
        
        # Store local maps from each drone
        self.drone_maps = {}
        self.drone_poses = {}
        
        # Subscribers for each drone's local map and pose
        for i in range(3):  # For 3 drones
            self.create_subscription(
                OccupancyGrid,
                f'/drone_{i}/local_map',
                lambda msg, drone_id=i: self.map_callback(msg, drone_id),
                qos_profile)
            
            self.create_subscription(
                PoseStamped,
                f'/drone_{i}/pose',
                lambda msg, drone_id=i: self.pose_callback(msg, drone_id),
                qos_profile)
        
        # Timer for periodic map publishing
        self.create_timer(0.1, self.publish_map)  # 10Hz update rate
        
        # TF broadcaster for map frame
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Map update parameters
        self.occupied_threshold = 0.65  # Probability threshold for occupied cells
        self.free_threshold = 0.35  # Probability threshold for free cells
        self.log_odds_map = np.zeros(self.map_size)  # Log odds representation
        
    def pose_callback(self, msg: PoseStamped, drone_id: int):
        """Store drone pose for coordinate transformation."""
        self.drone_poses[drone_id] = msg.pose
    
    def world_to_map(self, x: float, y: float) -> tuple:
        """Convert world coordinates to map cell coordinates."""
        map_x = int((x - self.origin_x) / self.resolution)
        map_y = int((y - self.origin_y) / self.resolution)
        
        # Ensure coordinates are within map bounds
        map_x = max(0, min(map_x, self.map_size[0] - 1))
        map_y = max(0, min(map_y, self.map_size[1] - 1))
        
        return map_x, map_y
    
    def map_to_world(self, map_x: int, map_y: int) -> tuple:
        """Convert map cell coordinates to world coordinates."""
        world_x = map_x * self.resolution + self.origin_x
        world_y = map_y * self.resolution + self.origin_y
        return world_x, world_y
    
    def map_callback(self, msg: OccupancyGrid, drone_id: int):
        """Process incoming local map from a drone."""
        if drone_id not in self.drone_poses:
            return  # Skip if we don't have the drone's pose yet
            
        # Store the local map
        self.drone_maps[drone_id] = msg
        
        # Update global map
        self.merge_maps()
    
    def update_log_odds(self, x: int, y: int, occupied: bool):
        """Update the log odds value for a cell."""
        if not (0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]):
            return
            
        # Log odds update values
        l_occ = 0.85  # log odds for occupied observation
        l_free = -0.4  # log odds for free observation
        
        # Update log odds
        if occupied:
            self.log_odds_map[x, y] += l_occ
        else:
            self.log_odds_map[x, y] += l_free
            
        # Clamp values to prevent overflow
        self.log_odds_map[x, y] = max(-10.0, min(10.0, self.log_odds_map[x, y]))
    
    def merge_maps(self):
        """Merge all drone maps into the global map."""
        if not self.drone_maps:
            return
            
        # Reset temporary map for merging
        temp_map = np.full(self.map_size, -1)
        
        for drone_id, local_map in self.drone_maps.items():
            if drone_id not in self.drone_poses:
                continue
                
            drone_pose = self.drone_poses[drone_id]
            
            # Get drone position in map coordinates
            drone_x, drone_y = self.world_to_map(
                drone_pose.position.x,
                drone_pose.position.y
            )
            
            # Process each cell in the local map
            local_map_data = np.array(local_map.data).reshape(
                (local_map.info.height, local_map.info.width)
            )
            
            for i in range(local_map.info.height):
                for j in range(local_map.info.width):
                    # Get world coordinates of local map cell
                    cell_world_x = (j - local_map.info.width/2) * local_map.info.resolution + drone_pose.position.x
                    cell_world_y = (i - local_map.info.height/2) * local_map.info.resolution + drone_pose.position.y
                    
                    # Convert to global map coordinates
                    map_x, map_y = self.world_to_map(cell_world_x, cell_world_y)
                    
                    if not (0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]):
                        continue
                    
                    cell_value = local_map_data[i, j]
                    if cell_value >= 0:  # Only update if cell has valid data
                        self.update_log_odds(map_x, map_y, cell_value > 50)
        
        # Convert log odds to probability and update global map
        prob_map = 1 - (1 / (1 + np.exp(self.log_odds_map)))
        self.map = np.where(prob_map > self.occupied_threshold, 100,
                           np.where(prob_map < self.free_threshold, 0, -1))
    
    def publish_map(self):
        """Publish the merged global map."""
        if not hasattr(self, 'map'):
            return
            
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.info.resolution = self.resolution
        msg.info.width = self.map_size[0]
        msg.info.height = self.map_size[1]
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        
        # Flatten map to 1D array
        msg.data = self.map.flatten().tolist()
        
        # Publish the map
        self.map_pub.publish(msg)
        
        # Broadcast map transform
        self.broadcast_map_transform()
    
    def broadcast_map_transform(self):
        """Broadcast the transform between map and world frames."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'
        
        # Set transform to identity (map frame origin)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        
        self.tf_broadcaster.sendTransform(t)
    
    def get_frontiers(self):
        """Find frontier cells (boundaries between explored and unexplored areas)."""
        frontiers = []
        for i in range(1, self.map_size[0] - 1):
            for j in range(1, self.map_size[1] - 1):
                if self.is_frontier(i, j):
                    frontiers.append((i, j))
        return frontiers
    
    def is_frontier(self, i, j):
        """Check if cell is at frontier (unknown space adjacent to free space)."""
        if self.map[i, j] != 0:  # Must be free space
            return False
        # Check if any adjacent cell is unknown (-1)
        return any(self.map[ni, nj] == -1 
                  for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)])
    
    def get_map_data(self):
        """Return current map data for external use."""
        return {
            'map': self.map.copy(),
            'resolution': self.resolution,
            'origin': (self.origin_x, self.origin_y),
            'size': self.map_size
        }
    
    def get_exploration_progress(self):
        """Calculate and return exploration progress."""
        total_cells = self.map_size[0] * self.map_size[1]
        unknown_cells = np.sum(self.map == -1)
        explored_cells = total_cells - unknown_cells
        return (explored_cells / total_cells) * 100
    