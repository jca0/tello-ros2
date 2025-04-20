from dynamic_grid import DynamicGrid
from utils import *

class MultiDroneExplorer:
    def __init__(self, drone_ids):
        self.map     = DynamicGrid()
        self.drones  = {i: {'pos': (0,0)} for i in drone_ids}

    def update_drone(self, drone_id, x, y, observations):
        self.drones[drone_id]['pos'] = (x,y)
        self.map.update(x, y, observations)

    def explore_step(self):
        # 1) detect frontiers
        fronts = detect_frontiers(self.map.grid)
        if not fronts:
            return False  # done

        # 2) get current drone cells
        cells = [ self.map.world_to_cell(*self.drones[i]['pos']) 
                  for i in self.drones ]

        # 3) assign frontiers
        assigns = assign_frontiers(cells, fronts)

        # 4) plan & dispatch
        for drone_idx, frontier_cell in assigns.items():
            start = cells[drone_idx]
            path  = astar(self.map.grid, start, frontier_cell)
            # send path down to drone (RPC / ROS action / etc)
            self.send_path(drone_idx, path)

        return True

    def run(self):
        while True:
            # wait for all drones to publish new map observations...
            if not self.explore_step():
                print("Exploration complete.")
                break