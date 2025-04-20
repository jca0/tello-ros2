import heapq
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

def detect_frontiers(grid):
    free_mask = grid == FREE
    unknown_mask = grid == UNKNOWN
    border = binary_dilation(free_mask, structure=np.ones((3,3))) & unknown_mask
    frontiers = np.argwhere(border)
    return [tuple(coord) for coord in frontiers]

def assign_frontiers(drone_cells, frontiers):
    cost = cdist(drone_cells, frontiers, metric='euclidean')
    row_idx, col_idx = linear_sum_assignment(cost)
    assignment = {}
    for r, c in zip(row_idx, col_idx):
        assignment[r] = frontiers[c]
    return assignment

def astar(grid, start, goal):
    """Return list of cell-coords from start to goal."""
    h = lambda a,b: np.hypot(b[0]-a[0], b[1]-a[1])
    open_set = [(h(start,goal), 0, start, None)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, g, cur, parent = heapq.heappop(open_set)
        if cur == goal:
            # reconstruct
            path = [cur]
            while parent:
                path.append(parent)
                parent = came_from[parent]
            return path[::-1]

        if cur in came_from: continue
        came_from[cur] = parent

        for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            nbr = (cur[0]+di, cur[1]+dj)
            if (0 <= nbr[0] < grid.shape[0] and
                0 <= nbr[1] < grid.shape[1] and
                grid[nbr] == FREE):
                new_cost = g + 1
                if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                    cost_so_far[nbr] = new_cost
                    heapq.heappush(open_set, 
                                   (new_cost + h(nbr,goal),
                                    new_cost, nbr, cur))
    return []  # no path