import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import linear_sum_assignment

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.num_drones = 3
        
        # Keep track of drone assignments
        self.drone_assignments = {i: None for i in range(self.num_drones)}
        
        # Parameters for frontier clustering and selection
        self.min_frontier_size = 5  # Minimum number of cells to consider a frontier
        self.cluster_distance = 3.0  # Maximum distance between cells in a cluster
        self.min_frontier_distance = 2.0  # Minimum distance between frontier centroids
        
    def find_frontiers(self, occupancy_grid):
        """Find frontier cells in the occupancy grid."""
        frontiers = []
        height, width = occupancy_grid.shape
        
        # Define neighbor offsets for 8-connectivity
        neighbors = [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1),  (1,0),  (1,1)]
        
        # Find frontier cells
        for i in range(1, height-1):
            for j in range(1, width-1):
                if self.is_frontier_cell(occupancy_grid, i, j, neighbors):
                    frontiers.append((i, j))
        
        return self.cluster_frontiers(frontiers)
    
    def is_frontier_cell(self, grid, i, j, neighbors):
        """Check if a cell is a frontier cell."""
        # Free cell (0) adjacent to unexplored (-1)
        if grid[i,j] != 0:  # Must be free space
            return False
            
        # Check if any neighbors are unexplored
        return any(grid[i+di,j+dj] == -1 
                  for di, dj in neighbors 
                  if 0 <= i+di < grid.shape[0] and 0 <= j+dj < grid.shape[1])
    
    def cluster_frontiers(self, frontier_cells):
        """Cluster frontier cells into distinct frontiers."""
        if not frontier_cells:
            return []
            
        clusters = []
        visited = set()
        
        for cell in frontier_cells:
            if cell in visited:
                continue
                
            # Start a new cluster
            cluster = []
            queue = [cell]
            visited.add(cell)
            
            # Grow cluster
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                
                # Check neighbors
                for neighbor in frontier_cells:
                    if neighbor not in visited and self.distance(current, neighbor) <= self.cluster_distance:
                        queue.append(neighbor)
                        visited.add(neighbor)
            
            if len(cluster) >= self.min_frontier_size:
                centroid = self.calculate_centroid(cluster)
                clusters.append(centroid)
        
        # Filter clusters that are too close to each other
        return self.filter_close_centroids(clusters)
    
    def calculate_centroid(self, cluster):
        """Calculate the centroid of a cluster of cells."""
        if not cluster:
            return None
        
        sum_i = sum(cell[0] for cell in cluster)
        sum_j = sum(cell[1] for cell in cluster)
        return (sum_i / len(cluster), sum_j / len(cluster))
    
    def filter_close_centroids(self, centroids):
        """Filter out centroids that are too close to each other."""
        if not centroids:
            return []
            
        filtered = []
        for centroid in centroids:
            if not any(self.distance(centroid, existing) < self.min_frontier_distance 
                      for existing in filtered):
                filtered.append(centroid)
        return filtered
    
    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def assign_frontiers(self, frontiers, drone_positions):
        """Assign frontiers to drones using Hungarian algorithm."""
        if not frontiers:
            return {}
            
        # Create cost matrix
        cost_matrix = np.zeros((len(drone_positions), len(frontiers)))
        for i, drone_pos in enumerate(drone_positions):
            for j, frontier in enumerate(frontiers):
                cost_matrix[i, j] = self.calculate_cost(drone_pos, frontier)
        
        # Assign frontiers to minimize total cost
        assignments = self.hungarian_assignment(cost_matrix)
        
        # Convert assignments to dictionary
        assignment_dict = {}
        for drone_id, frontier_idx in enumerate(assignments[0]):
            if frontier_idx < len(frontiers):  # Ensure valid frontier index
                assignment_dict[drone_id] = frontiers[frontier_idx]
        
        return assignment_dict
    
    def calculate_cost(self, drone_pos, frontier):
        """Calculate cost (distance) between drone and frontier."""
        return np.sqrt((drone_pos[0] - frontier[0])**2 + 
                      (drone_pos[1] - frontier[1])**2)
    
    def hungarian_assignment(self, cost_matrix):
        """Implement Hungarian algorithm using scipy's linear_sum_assignment."""
        return linear_sum_assignment(cost_matrix)
    
    def update_assignments(self, occupancy_grid, drone_positions):
        """Main update function to reassign frontiers to drones."""
        # Find new frontiers
        frontiers = self.find_frontiers(occupancy_grid)
        
        if not frontiers:
            self.get_logger().info('No frontiers found. Exploration complete.')
            return {}
        
        # Assign frontiers to drones
        new_assignments = self.assign_frontiers(frontiers, drone_positions)
        
        # Update stored assignments
        self.drone_assignments = new_assignments
        
        return new_assignments
    
    def get_exploration_status(self):
        """Get the current exploration status."""
        return {
            'active_assignments': self.drone_assignments,
            'num_active_drones': len([x for x in self.drone_assignments.values() if x is not None])
        }
    
    def is_exploration_complete(self, occupancy_grid):
        """Check if exploration is complete."""
        # Count unexplored cells
        unexplored = np.sum(occupancy_grid == -1)
        total_cells = occupancy_grid.size
        explored_percentage = (total_cells - unexplored) / total_cells * 100
        
        return explored_percentage >= 95  # Consider exploration complete at 95% coverage
    