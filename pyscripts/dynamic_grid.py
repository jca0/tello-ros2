import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1
# i, j is cell coordinates
# x, y is world coordinates

class DynamicGrid:
    def __init__(self, size=100, resolution=0.1):
        """
        size is initial size of grid
        resolution is meters per cell
        """
        self.resolution = resolution
        self.grid = np.full((size, size), UNKNOWN, dtype=int)
        self.origin = (-size//2, -size//2) 

    def world_to_cell(self, x, y):
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)
        return i, j       
    
    def expand_if_needed(self, i, j, expand_size=5):
        hi, hj = self.grid.shape
        top = max(0, -i + expand_size)
        left = max(0, -j + expand_size)
        bottom = max(0, i + expand_size - hi)
        right = max(0, j + expand_size - hj)
        if any([top, left, bottom, right]):
            self.grid = np.pad(self.grid, 
                               ((top, bottom), (left, right)),
                               constant_values=UNKNOWN)
            self.origin = (self.origin[0] - top,
                           self.origin[1] - left)
            
    def update(self, x, y, observations):
        """
        observations: list of (rel_x, rel_y, state) from drone sensor
        """
        for dx, dy, state in observations:
            wx, wy = x + dx, y + dy
            i, j = self.world_to_cell(wx, wy)
            self.expand_if_needed(i, j)
            self.grid[i, j] = state

