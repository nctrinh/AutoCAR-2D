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
                 max_iterations: int = 100000, heuristic_weight: float = 1.0):
        self.map_env = map_env
        self.grid_resolution = grid_resolution
        self.max_iterations = max_iterations
        self.heuristic_weight = heuristic_weight
        
        self.grid_width = int(np.ceil(self.map_env.width / self.grid_resolution))
        self.grid_height = int(np.ceil(self.map_env.height / self.grid_resolution))

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
        if not self.is_valid_position(start[0], start[1]):
            print("A*: Start position is invalid!")
            return None
        
        if not self.is_valid_position(goal[0], goal[1]):
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

        while open_set and self.iterations < self.max_iterations:
            self.iterations += 1
            _, _, current = heapq.heappop(open_set)

            # Check goal condition
            if current == goal_grid:
                path = self._reconstruct_path(came_from, current, start, goal)
                
                self.planning_time = time.time() - start_time
                self.path = path
                
                print(f"A*: Path found! Length: {path.length:.2f}m, "
                      f"Time: {self.planning_time:.3f}s, Iterations: {self.iterations}")
                return path

            closed_set.add(current)

            for neighbor in self._get_grid_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Euclidean distance on grid
                dist = np.sqrt((neighbor[0]-current[0])**2 + (neighbor[1]-current[1])**2)
                move_cost = dist * self.grid_resolution
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    h = self.heuristic(neighbor, goal_grid) * self.grid_resolution
                    f_score[neighbor] = tentative_g + self.heuristic_weight * h
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        self.planning_time = time.time() - start_time
        print(f"A*: No path found after {self.iterations} iterations!")
        return None

    def _get_grid_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        neighbors = []

        # 8-connectivity
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue

                # Check collision
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
        
        return smoothed_path

    def _smooth_path(self, path: Path) -> Path:
        if len(path.points) <= 2:
            return path
        
        smoothed = [path.points[0]]
        current_idx = 0
        
        while current_idx < len(path.points) - 1:
            furthest_idx = current_idx + 1
            
            for i in range(len(path.points) - 1, current_idx, -1):
                p1 = path.points[current_idx].to_tuple()
                p2 = path.points[i].to_tuple()
                
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
                step_size = 0.2
                num_samples = int(max(5, dist / step_size))
                
                if self.is_path_valid(p1, p2, num_samples=num_samples):
                    furthest_idx = i
                    break
            
            current_idx = furthest_idx
            smoothed.append(path.points[current_idx])
        
        return Path(smoothed)
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
