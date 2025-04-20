import rclpy
from rclpy.node import Node

class CoordinationNode(Node):
    def __init__(self):
        super().__init__('coordination_node')
        self.map_manager = MapManager()
        self.frontier_explorer = FrontierExplorer()
        self.drone_controllers = [
            DroneController(i) for i in range(3)
        ]
        
        # Timer for coordination loop
        self.create_timer(0.1, self.coordination_loop)
        
    def coordination_loop(self):
        # Get current map and frontiers
        frontiers = self.map_manager.get_frontiers()
        
        # Get current drone positions
        drone_positions = [
            controller.current_pose.pose.position 
            for controller in self.drone_controllers
        ]
        
        # Assign frontiers to drones
        assignments = self.frontier_explorer.assign_frontiers(
            frontiers, drone_positions)
            
        # Update drone targets
        for drone_id, frontier in assignments.items():
            self.drone_controllers[drone_id].set_target_frontier(frontier)