import numpy as np
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum



class ObstacleType(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"


@dataclass
class Obstacle:
    type: ObstacleType = field(init=False)

    def contains_point(self, x: float, y: float) -> bool:
        raise NotImplementedError
    
    def distance_to_point(self, x: float, y: float) -> float:
        raise NotImplementedError
    

@dataclass
class RectangleObstacle(Obstacle):
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0

    def __post_init__(self):
        self.type = ObstacleType.RECTANGLE
    
    def contains_point(self, x: float, y: float) -> bool:
        dx = x - self.x
        dy = y - self.y

        if self.angle != 0:
            cos_a = np.cos(-self.angle)
            sin_a = np.sin(-self.angle)
            local_x = cos_a * dx - sin_a * dy
            local_y = sin_a * dx + cos_a * dy
        else:
            local_x, local_y = dx, dy
        
        return (abs(local_x) <= self.width/2 and abs(local_y) <= self.height/2) 
    
    def distance_to_point(self, x: float, y: float) -> float:
        dx = x - self.x
        dy = y - self.y

        if self.angle != 0:
            cos_a = np.cos(-self.angle)
            sin_a = np.sin(-self.angle)
            local_x = cos_a * dx - sin_a * dy
            local_y = sin_a * dx + cos_a * dy
        else:
            local_x, local_y = dx, dy

        closest_x = np.clip(local_x, -self.width/2, self.width/2)
        closest_y = np.clip(local_y, -self.height/2, self.height/2)        

        dist_x = local_x - closest_x
        dist_y = local_y - closest_y

        return np.sqrt(dist_x**2 + dist_y**2)
    
    def get_corners(self) -> np.ndarray:
        hw, hh = self.width/2, self.height/2
        corners = np.array([
            [hw, hh], [hw, -hh], [-hw, -hh], [-hw, hh]
        ])

        if self.angle != 0:
            # BUG FIX: Sử dụng +self.angle để transform từ Local ra World
            cos_a = np.cos(self.angle) 
            sin_a = np.sin(self.angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rotation.T
        
        corners[:, 0] += self.x
        corners[:, 1] += self.y
        return corners
    

@dataclass
class CircleObstacle(Obstacle):
    x: float
    y: float
    radius: float

    def __post_init__(self):
        self.type = ObstacleType.CIRCLE

    def contains_point(self, x: float, y: float):
        dx = x - self.x
        dy = y - self.y
        return np.sqrt(dx**2 + dy**2) <= self.radius
    
    def distance_to_point(self, x: float, y: float):
        dx = x - self.x
        dy = y - self.y
        dist_to_center = np.sqrt(dx**2 + dy**2)
        return max(0, dist_to_center - self.radius)
    

@dataclass
class PolygonObstacle(Obstacle):
    vertices: np.ndarray 

    def __post_init__(self):
        self.type = ObstacleType.POLYGON
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.array(self.vertices)

    def contains_point(self, x: float, y: float) -> bool:
        n = len(self.vertices)
        inside = False
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y < max(p1y, p2y):
                    if x < max(p1x, p2x):
                        x_inner = p1x + (y - p1y) * (p2x - p1x) / (p2y - p1y)
                        if x <= x_inner:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def distance_to_point(self, x: float, y: float) -> float:
        if self.contains_point(x, y):
            return 0.0
            
        min_dist =  float('inf')
        point = np.array([x, y])
        
        n = len(self.vertices)
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1) % n]
            dist = self._point_to_segment_distance(point, v1, v2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    @staticmethod
    def _point_to_segment_distance(point, seg_start, seg_end):
        segment = seg_end - seg_start
        point_vec = point - seg_start
        segment_length_sq = np.dot(segment, segment)
        if segment_length_sq < 1e-10:
            return np.linalg.norm(point_vec)
        t = np.clip(np.dot(point_vec, segment) / segment_length_sq, 0, 1)
        projection = seg_start + t * segment
        return np.linalg.norm(point - projection)

class Map2D:
    """
    2D map environment with obstacles.
    Handles collision detection and map representation.
    """
    def __init__(self, width: float, height: float, safety_margin: float = 1.0):
        self.width = width
        self.height = height
        self.safety_margin = safety_margin
        self.obstacles: List[Obstacle] = []
        self.start: Optional[Tuple[float, float]] = None
        self.goal: Optional[Tuple[float, float]] = None

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to map."""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int):
        """Remove obstacle by index."""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
        
    def set_start(self, x: float, y: float):
        """Set start position."""
        self.start = (x, y)
    
    def set_goal(self, x: float, y: float):
        """Set goal position."""
        self.goal = (x, y)
    
    def is_collision(self, x: float, y: float, safety_margin: Optional[float] = None) -> bool:
        """
        Check if point collides with any obstacle.
        
        Args:
            x, y: Point coordinates
            
        Returns:
            True if collision detected
        """
        #Check map boundaries
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return True
        if safety_margin is None:
            safety_margin = self.safety_margin
        # Check obstacles
        for obstacle in self.obstacles:
            if safety_margin > 0:
                dist = obstacle.distance_to_point(x, y)
                if dist < safety_margin:
                    return True
            else:
                if obstacle.contains_point(x, y):
                    return True
        
        return False
    
    def is_path_collision_free(self, x1: float, y1: float,
                              x2: float, y2: float,
                              num_samples: int = 10) -> bool:
        """
        Check if line segment from (x1,y1) to (x2,y2) is collision-free.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            num_samples: Number of points to sample along path
            
        Returns:
            True if path is collision-free
        """
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.is_collision(x, y):
                return False
        return True
        
    def get_nearest_obstacle_distance(self, x: float, y: float) -> float:
        """Get distance to nearest obstacle."""
        if not self.obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obstacle in self.obstacles:
            dist = obstacle.distance_to_point(x, y)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def save_to_json(self, filename: str):
        """Save map to JSON file."""
        data = {
            "width": self.width,
            "height": self.height,
            "safety_margin": self.safety_margin,
            "start": self.start,
            "goal": self.goal,
            "obstacles": []
        }
        
        for obs in self.obstacles:
            if isinstance(obs, RectangleObstacle):
                data["obstacles"].append({
                    "type": "rectangle",
                    "x": obs.x,
                    "y": obs.y,
                    "width": obs.width,
                    "height": obs.height,
                    "angle": obs.angle
                })
            elif isinstance(obs, CircleObstacle):
                data["obstacles"].append({
                    "type": "circle",
                    "x": obs.x,
                    "y": obs.y,
                    "radius": obs.radius
                })
            elif isinstance(obs, PolygonObstacle):
                data["obstacles"].append({
                    "type": "polygon",
                    "vertices": obs.vertices.tolist()
                })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filename: str) -> 'Map2D':
        """Load map from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        safety = data.get("safety_margin", 0.0)
        map_obj = cls(data["width"], data["height"], safety_margin=safety)
        
        if "start" in data and data["start"]:
            map_obj.start = tuple(data["start"])
        if "goal" in data and data["goal"]:
            map_obj.goal = tuple(data["goal"])
        
        for obs_data in data.get("obstacles", []):
            obs_type = obs_data["type"]
            
            if obs_type == "rectangle":
                obstacle = RectangleObstacle(
                    x=obs_data["x"],
                    y=obs_data["y"],
                    width=obs_data["width"],
                    height=obs_data["height"],
                    angle=obs_data.get("angle", 0.0)
                )
            elif obs_type == "circle":
                obstacle = CircleObstacle(
                    x=obs_data["x"],
                    y=obs_data["y"],
                    radius=obs_data["radius"]
                )
            elif obs_type == "polygon":
                obstacle = PolygonObstacle(
                    vertices=np.array(obs_data["vertices"])
                )
            else:
                continue
            
            map_obj.add_obstacle(obstacle)
        
        return map_obj
    
    def __repr__(self) -> str:
        return (f"Map2D({self.width}x{self.height}, "
                f"{len(self.obstacles)} obstacles)")
    

if __name__ == "__main__":
    map_env = Map2D(width=100, height=100)
    
    map_env.add_obstacle(RectangleObstacle(x=30, y=30, width=20, height=10))
    map_env.add_obstacle(CircleObstacle(x=70, y=70, radius=8))
    map_env.add_obstacle(PolygonObstacle(vertices=np.array([
        [50, 10], [60, 20], [55, 30], [45, 25]
    ])))
    
    map_env.set_start(10, 10)
    map_env.set_goal(90, 90)
    
    print("Collision at (30, 30):", map_env.is_collision(30, 30))
    print("Collision at (50, 50):", map_env.is_collision(50, 50))
    
    map_env.save_to_json("./maps/test_map.json")
    print("Map saved!")
    
    loaded_map = Map2D.load_from_json("test_map.json")
    print("Loaded:", loaded_map)