import numpy as np
from scipy.ndimage import binary_dilation

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

class DynamicGrid:
    def __init__(self, size=100, resolution=0.1):
        """
        size is initial size of grid
        resolution is meters per cell
        """
        self.resolution = resolution
        self.grid = np.full((size, size), UNKNOWN, dtype=int)
        self.origin = np.array([0, 0])    

    def world_to_cell(self, x, y):
        cell_x = int((x - self.origin[0]) / self.resolution)
        cell_y = int((y - self.origin[1]) / self.resolution)
        return cell_x, cell_y                           


def expand_map():
    pass

def detect_frontiers():
    pass

def assign_drones():
    pass
