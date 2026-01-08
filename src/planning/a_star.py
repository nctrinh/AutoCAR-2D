import numpy as np 
import heapq
import time
from typing import List, Tuple, Optional
import sys
from pathlib import Path as pathlib_Path

sys.path.append(str(pathlib_Path(__file__).resolve().parents[2]))

from src.planning.base_planner import BasePlanner, PathPoint, Path
from src.core.map import Map2D

class AStarPlanner(BasePlanner):
    def __init__(self, map_env: Map2D, grid_resolution: float = 0.5, 
                 max_iterations: int = 10000, heuristic_weight: float = 1.0,
                 spacing: float = 3.0):
        self.map_env = map_env
        self.grid_resolution = grid_resolution
        self.max_iterations = max_iterations
        self.heuristic_weight = heuristic_weight
        self.spacing = spacing
        
        self.grid_width = int(np.ceil(self.map_env.width / self.grid_resolution))
        self.grid_height = int(np.ceil(self.map_env.height / self.grid_resolution))

        self.occupancy_grid = self._create_occupancy_grid()
    
    def _create_occupancy_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                wx, wy = self.grid_to_world(x, y)
                if self.is_valid_position(wx, wy):
                    grid[x, y] = True
        return grid

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int(np.floor(x / self.grid_resolution))
        grid_y = int(np.floor(y / self.grid_resolution))
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = (grid_x + 0.5) * self.grid_resolution
        y = (grid_y + 0.5) * self.grid_resolution
        return (x, y)
    
    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> Optional[Path]:
        start_time = time.time()
        
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        # Validate Start/Goal
        if not self.occupancy_grid[start_grid[0], start_grid[1]]:
            print("A*: Start position is invalid!")
            return None
        
        if not self.occupancy_grid[goal_grid[0], goal_grid[1]]:
            print("A*: Goal position is invalid!")
            return None
        
        # Priority Queue: (f_score, counter, node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        closed_set = set()
        counter = 0
        self.iterations = 0

        w = self.heuristic_weight
        # Chi phí di chuyển: Thẳng = 1.0, Chéo = 1.414
        COST_STRAIGHT = 1.0 * self.grid_resolution
        COST_DIAGONAL = 1.414 * self.grid_resolution 
        
        while open_set and self.iterations < self.max_iterations:
            self.iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal_grid:
                path = self._reconstruct_path(came_from, current, start, goal)
                self.planning_time = time.time() - start_time
                print(f"Path found! Length: {path.length:.2f}m, Time: {self.planning_time:.3f}s, Iterations: {self.iterations}")
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            cx, cy = current

            neighbors_check = [
                (0, 1, COST_STRAIGHT), (0, -1, COST_STRAIGHT), 
                (1, 0, COST_STRAIGHT), (-1, 0, COST_STRAIGHT),
                (1, 1, COST_DIAGONAL), (1, -1, COST_DIAGONAL),
                (-1, 1, COST_DIAGONAL), (-1, -1, COST_DIAGONAL)
            ]

            for dx, dy, move_cost in neighbors_check:
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue
                if not self.occupancy_grid[nx, ny]:
                    continue
                if (nx, ny) in closed_set:
                    continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Heuristic optimization
                    h = self.heuristic(neighbor, goal_grid) * self.grid_resolution
                    f = tentative_g + w * h
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        self.planning_time = time.time() - start_time
        print(f"A*: No path found after {self.iterations} iterations!")
        return None

    def _get_grid_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue

                world_x, world_y = self.grid_to_world(nx, ny)
                if self.is_valid_position(world_x, world_y):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int], 
                          real_start: Tuple[float, float],
                          real_goal: Tuple[float, float]) -> Path:
        grid_path = [current]
        while current in came_from:
            current = came_from[current]
            grid_path.append(current)
        
        grid_path.reverse()
        
        path_points = []
        
        path_points.append(PathPoint(real_start[0], real_start[1]))
        
        for i in range(1, len(grid_path) - 1):
            gx, gy = grid_path[i]
            wx, wy = self.grid_to_world(gx, gy)
            path_points.append(PathPoint(wx, wy))
            
        path_points.append(PathPoint(real_goal[0], real_goal[1]))
        
        raw_path = Path(path_points)
        
        smoothed_path = self._smooth_path(raw_path)
        smoothed_path = self._resample_path(smoothed_path)
        
        return smoothed_path

    def _smooth_path(self, path: Path) -> Path:
        if len(path.points) <= 2:
            return path
        
        path_points = path.points
        grid_points = [self.world_to_grid(p.x, p.y) for p in path_points]
        
        smoothed_points = [path_points[0]]
        current_idx = 0
        
        while current_idx < len(path_points) - 1:
            furthest_idx = current_idx + 1
            for i in range(len(path_points) - 1, current_idx, -1):
                if self._check_line_of_sight_grid(grid_points[current_idx], grid_points[i]):
                    furthest_idx = i
                    break
            
            smoothed_points.append(path_points[furthest_idx])
            current_idx = furthest_idx
        
        return Path(smoothed_points)

    def _check_line_of_sight_grid(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        x0, y0 = p1
        x1, y1 = p2
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x = x0
        y = y0
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while True:
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                return False
            if not self.occupancy_grid[x, y]: 
                return False
                
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True
    
    def _resample_path(self, path: Path) -> Path:
        if not path.points or len(path.points) < 2:
            return path
        
        points_arr = path.to_array()
        x = points_arr[:, 0]
        y = points_arr[:, 1]

        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.hypot(dx, dy)
        
        cum_dist = np.concatenate(([0], np.cumsum(dist)))
        total_length = cum_dist[-1]
        
        if total_length < self.spacing:
            return path

        n_points = int(total_length / self.spacing)
        new_cum_dist = np.linspace(0, total_length, n_points + 1)

        new_x = np.interp(new_cum_dist, cum_dist, x)
        new_y = np.interp(new_cum_dist, cum_dist, y)

        new_points = [PathPoint(nx, ny) for nx, ny in zip(new_x, new_y)]
        
        original_goal = path.points[-1]
        new_points[-1] = PathPoint(original_goal.x, original_goal.y)

        return Path(new_points)

    def is_path_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], num_samples: int = 10) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            
            if not self.is_valid_position(x, y):
                return False
        return True
    
    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
