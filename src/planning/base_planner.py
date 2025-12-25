from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.map import Map2D

class PathPoint:
    '''A point in the path'''

    def __init__(self, x: float, y: float, theta: Optional[float] = None):
        self.x = x
        self.y = y
        self.theta = theta
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'PathPoint') -> float:
        dx = other.x - self.x
        dy = other.y - self.y
        return np.sqrt(dx**2 + dy**2)
    
    def __repr__(self) -> str:
        if self.theta is not None:
            return f"PathPoint({self.x:.2f}, {self.y:.2f}, θ={np.degrees(self.theta):.1f}°)"
        return f"PathPoint({self.x:.2f}, {self.y:.2f})"
    
class Path:
    '''A planned path'''

    def __init__(self, points: List[PathPoint]):
        self.points = points
        self.length = self._calculate_length()

    def _calculate_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.points) - 1):
            total_length += self.points[i].distance_to(self.points[i+1])
        return total_length
    
    def get_point_at_distance(self, distance: float) -> PathPoint:
        if distance <= 0.0:
            return self.points[0] if self.points else None
        if distance >= self.length:
            return self.points[-1] if self.points else None
        
        accumulated_dist = 0.0
        for i in range(len(self.points) - 1):
            segment_length = self.points[i].distance_to(self.points[i+1])
            if accumulated_dist + segment_length >= distance:
                t = (distance - accumulated_dist) / segment_length
                x = self.points[i].x + t * (self.points[i+1].x - self.points[i].x) 
                y = self.points[i].y + t * (self.points[i+1].y - self.points[i].y) 
                return PathPoint(x, y)
            accumulated_dist += segment_length
        return self.points[-1]
    
    def to_array(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.points])
    
    def smooth(self, window_size: int = 5) -> 'Path':
        if len(self.points) < window_size:
            return self
        
        arr = self.to_array()
        smoothed = np.copy(arr)

        half_window = window_size // 2

        for i in range(half_window, len(arr) - half_window):
            smoothed[i] = np.mean(arr[i - half_window:i + half_window + 1], axis=0)
        smoothed = [PathPoint(x, y) for x, y in smoothed]
        return Path(smoothed)

    def __len__(self) -> int:
        return len(self.points)
    
    def __repr__(self) -> str:
        return f"Path(points={len(self.points)}, length={self.length:.2f}m)"
    
class BasePlanner(ABC):
    def __init__(self, map_env: Map2D):
        self.map_env = map_env
        self.path: Optional[Path] = None
        self.planning_time: float = 0.0
        self.iterations: int = 0
    
    @abstractmethod
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], **kwargs) -> Optional[Path]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Path object if successful, None otherwise
        """
        pass

    def is_valid_position(self, x: float, y: float) -> bool:
        if not (0 <= x <= self.map_env.width and
                0 <= y <= self.map_env.height):
            return False
        
        return not self.map_env.is_collision(x, y)
    
    def is_valid_path(self, p1: Tuple[float, float],
                      p2: Tuple[float, float], 
                      num_samples: int = 10) -> bool:
        return self.map_env.is_path_collision_free(p1[0], p1[1], p2[0], p2[1], num_samples)
    
    def heuristic(self, p1: Tuple[float, float],
                  p2: Tuple[float, float],
                  heuristic_type: str = "euclidean") -> float:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        if heuristic_type == "euclidean":
            return np.sqrt(dx**2 + dy**2)
        elif heuristic_type == "manhattan":
            return dx + dy
        elif heuristic_type == "octile":
            return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
        else:
            return np.sqrt(dx**2 + dy**2)
        
    def reconstruct_path(self, came_from: dict,
                         current: Tuple[float, float]) -> Path:
        path_points = [PathPoint(current[0], current[1])]

        while current in came_from:
            current = came_from[current]
            path_points.append(PathPoint(current[0], current[1]))

        path_points.reverse()
        return Path(path_points)
    
    def get_neighbors(self, point: Tuple[float, float],
                      step_size: int = 1,
                      connectivity: int = 8) -> List[Tuple[float, float]]:
        x, y = point
        neighbors = []

        directions = [
            (step_size, 0), (-step_size, 0), (0, step_size), [0, -step_size]
        ]

        if connectivity == 8:
            directions.extend([
                (step_size, step_size), (step_size, -step_size),
                (-step_size, step_size), (-step_size, -step_size)
            ])
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def get_stats(self) -> dict:
        """Get planning statistics."""
        return {
            "planning_time": self.planning_time,
            "iterations": self.iterations,
            "path_length": self.path.length if self.path else 0.0,
            "path_points": len(self.path.points) if self.path else 0
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(map={self.map_env})"
    

if __name__ == '__main__':
    from src.core.map import Map2D, CircleObstacle

    map_env = Map2D(width=100, height=100)
    map_env.add_obstacle(CircleObstacle(x=50, y=50, radius=10))

    class DummyPlanner(BasePlanner):
        def plan(self, start, goal, **kwargs):
            # Just return straight line
            points = [
                PathPoint(start[0], start[1]),
                PathPoint(goal[0], goal[1])
            ]
            return Path(points)
    
    planner = DummyPlanner(map_env)
    print(planner)
    
    # Test path
    path = planner.plan((10, 10), (90, 90))
    print(path)
