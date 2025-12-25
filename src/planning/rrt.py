from typing import Optional, List, Tuple
import numpy as np
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.planning.base_planner import PathPoint, Path, BasePlanner
from src.core.map import Map2D



class RRTNode:
    '''Node in RRT tree'''

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent : Optional[RRTNode] = None
        self.cost = 0.0

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, another: 'RRTNode') -> float:
        dx = self.x - another.x
        dy = self.y - another.y
        return np.sqrt(dx**2 + dy**2)
    
    def __repr__(self) -> float:
        return f"RRTNode({self.x:.2f}, {self.y:.2f})"
    
    
class RRTPlanner(BasePlanner):
    """
    RRT (Rapidly-exploring Random Tree) planner.
    
    Randomly samples the space and grows a tree from start to goal.
    Fast but produces suboptimal paths.
    """

    def __init__(self, map_env: Map2D, max_iterations: int = 1000, 
                step_size: float = 2.0, goal_sample_rate: float = 0.1,
                goal_threshold: float = 2.0, rewire_radius: float = 5.0):
        super().__init__(map_env)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius
        self.nodes : List[RRTNode] = []

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> Optional[Path]:
        """
        Plan path using RRT algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            step_size: Maximum distance to extend tree
            goal_sample_rate: Probability of sampling goal directly
            goal_threshold: Distance to goal considered as reached
            
        Returns:
            Path object if successful, None otherwise
        """

        start_time = time.time()
        
        # Check validity
        if not self.is_valid_position(start[0], start[1]):
            print("RRT: Start position is invalid!")
            return None
        
        if not self.is_valid_position(goal[0], goal[1]):
            print("RRT: Goal position is invalid!")
            return None
        
        for i in range(self.max_iterations):
            self.iterations = i + 1
            
            # Sample random point (with bias toward goal)
            if np.random.random() < self.goal_sample_rate:
                random_point = RRTNode(goal[0], goal[1])
            else:
                random_point = self._sample_random_point()
            
            self.nodes = [RRTNode(start[0], start[1])]
        self.iterations = 0
        
        for i in range(self.max_iterations):
            self.iterations = i + 1
            
            # Sample random point (with bias toward goal)
            if np.random.random() < self.goal_sample_rate:
                random_point = RRTNode(goal[0], goal[1])
            else:
                random_point = self._sample_random_point()
            
            # Find nearest node in tree
            nearest_node = self._get_nearest_node(random_point)
            
            # Steer toward random point with step_size limit
            new_node = self._steer(nearest_node, random_point, self.step_size)
            
            # Check if path to new node is collision-free
            if self._is_path_collision_free(nearest_node, new_node):
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + nearest_node.distance_to(new_node)
                self.nodes.append(new_node)
                
                # Check if goal is reached
                if new_node.distance_to(RRTNode(goal[0], goal[1])) <= self.goal_threshold:
                    # Try to connect directly to goal
                    goal_node = RRTNode(goal[0], goal[1])
                    if self._is_path_collision_free(new_node, goal_node):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + new_node.distance_to(goal_node)
                        self.nodes.append(goal_node)
                        
                        # Path found!
                        path = self._extract_path(goal_node)
                        self.planning_time = time.time() - start_time
                        self.path = path
                        
                        print(f"RRT: Path found! Length: {path.length:.2f}m, "
                              f"Time: {self.planning_time:.3f}s, Iterations: {self.iterations}")
                        return path
        
        self.planning_time = time.time() - start_time
        print(f"RRT: No path found after {self.max_iterations} iterations!")
        return None
    
    def _sample_random_point(self) -> RRTNode:
        """Sample random point in free space."""
        x = np.random.uniform(0, self.map_env.width)
        y = np.random.uniform(0, self.map_env.height)
        return RRTNode(x, y)
    
    def _get_nearest_node(self, point: RRTNode) -> RRTNode:
        """Find nearest node in tree to given point."""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = node.distance_to(point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: RRTNode, to_node: RRTNode, 
               step_size: float) -> RRTNode:
        """
        Steer from from_node toward to_node with maximum step_size.
        
        Returns new node that is at most step_size away from from_node.
        """
        dist = from_node.distance_to(to_node)
        
        if dist <= step_size:
            return RRTNode(to_node.x, to_node.y)
        
        # Interpolate
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + step_size * np.cos(theta)
        new_y = from_node.y + step_size * np.sin(theta)
        
        return RRTNode(new_x, new_y)
    
    def _is_path_collision_free(self, node1: RRTNode, node2: RRTNode) -> bool:
        """Check if straight line between two nodes is collision-free."""
        num_samples = int(np.ceil(node1.distance_to(node2) / 0.5))
        num_samples = max(num_samples, 5)  # At least 5 samples
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            
            if not self.is_valid_position(x, y):
                return False
        
        return True
    
    def _extract_path(self, goal_node: RRTNode) -> Path:
        """Extract path from start to goal by backtracking through parents."""
        path_points = []
        current = goal_node
        
        while current is not None:
            path_points.append(PathPoint(current.x, current.y))
            current = current.parent
        
        path_points.reverse()
        return Path(path_points)
    
    
class RRTStarPlanner(RRTPlanner):
    """
    RRT* (optimal RRT) planner.
    
    Rewires tree to find better paths. Asymptotically optimal.
    """
    
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float]) -> Optional[Path]:
        """
        Plan path using RRT* algorithm with tree rewiring.
        
        Args:
            rewire_radius: Radius to search for rewiring neighbors
            Other args: Same as RRT
        """
        start_time = time.time()
        
        if not self.is_valid_position(start[0], start[1]):
            print("RRT*: Start position is invalid!")
            return None
        
        if not self.is_valid_position(goal[0], goal[1]):
            print("RRT*: Goal position is invalid!")
            return None
        
        self.nodes = [RRTNode(start[0], start[1])]
        self.iterations = 0
        goal_node = None

        for i in range(self.max_iterations):
            self.iterations = i + 1
            
            # Sample
            if np.random.random() < self.goal_sample_rate:
                random_point = RRTNode(goal[0], goal[1])
            else:
                random_point = self._sample_random_point()
            
            # Find nearest
            nearest_node = self._get_nearest_node(random_point)
            
            # Steer
            new_node = self._steer(nearest_node, random_point, self.step_size)
            
            # Check collision
            if not self._is_path_collision_free(nearest_node, new_node):
                continue
            
            # Find neighbors for rewiring
            neighbors = self._get_neighbors(new_node, self.rewire_radius)
            
            # Choose best parent (lowest cost)
            best_parent = nearest_node
            min_cost = nearest_node.cost + nearest_node.distance_to(new_node)
            
            for neighbor in neighbors:
                cost = neighbor.cost + neighbor.distance_to(new_node)
                if cost < min_cost and self._is_path_collision_free(neighbor, new_node):
                    best_parent = neighbor
                    min_cost = cost
            
            # Add new node with best parent
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire tree
            for neighbor in neighbors:
                new_cost = new_node.cost + new_node.distance_to(neighbor)
                if new_cost < neighbor.cost and self._is_path_collision_free(new_node, neighbor):
                    neighbor.parent = new_node
                    neighbor.cost = new_cost
            
            # Check goal
            if new_node.distance_to(RRTNode(goal[0], goal[1])) <= self.goal_threshold:
                goal_node_temp = RRTNode(goal[0], goal[1])
                if self._is_path_collision_free(new_node, goal_node_temp):
                    if goal_node is None:
                        goal_node = goal_node_temp
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + new_node.distance_to(goal_node)
                        self.nodes.append(goal_node)
                    else:
                        # Update goal if found better path
                        new_cost = new_node.cost + new_node.distance_to(goal_node)
                        if new_cost < goal_node.cost:
                            goal_node.parent = new_node
                            goal_node.cost = new_cost
        
        self.planning_time = time.time() - start_time
        
        if goal_node is not None:
            path = self._extract_path(goal_node)
            self.path = path
            print(f"RRT*: Path found! Length: {path.length:.2f}m, "
                  f"Time: {self.planning_time:.3f}s, Iterations: {self.iterations}")
            return path
        
        print(f"RRT*: No path found after {self.max_iterations} iterations!")
        return None
    
    def _get_neighbors(self, node: RRTNode, radius: float) -> List[RRTNode]:
        """Get all nodes within radius of given node."""
        neighbors = []
        for n in self.nodes:
            if n.distance_to(node) <= radius and n != node:
                neighbors.append(n)
        return neighbors
    

if __name__ == "__main__":
    from src.core.map import Map2D, CircleObstacle, RectangleObstacle
    
    # Create map
    map_env = Map2D(width=100, height=100)
    map_env.add_obstacle(CircleObstacle(x=30, y=30, radius=8))
    map_env.add_obstacle(RectangleObstacle(x=60, y=50, width=15, height=30))
    map_env.add_obstacle(CircleObstacle(x=70, y=70, radius=6))
    
    start = (10, 10)
    goal = (90, 90)
    
    print("="*60)
    print("Testing RRT")
    print("="*60)
    
    # Test RRT
    rrt = RRTPlanner(map_env)
    path_rrt = rrt.plan(start, goal)
    
    if path_rrt:
        print(f"RRT Path: {path_rrt.length:.2f}m, {len(path_rrt.points)} waypoints")
    
    print("\n" + "="*60)
    print("Testing RRT*")
    print("="*60)
    
    # Test RRT*
    rrt_star = RRTStarPlanner(map_env)
    path_rrt_star = rrt_star.plan(start, goal)
    
    if path_rrt_star:
        print(f"RRT* Path: {path_rrt_star.length:.2f}m, {len(path_rrt_star.points)} waypoints")
    
    # Compare
    if path_rrt and path_rrt_star:
        print("\n" + "="*60)
        print("Comparison:")
        print(f"  RRT:  {path_rrt.length:.2f}m in {rrt.planning_time:.3f}s")
        print(f"  RRT*: {path_rrt_star.length:.2f}m in {rrt_star.planning_time:.3f}s")
        print(f"  Improvement: {(1 - path_rrt_star.length/path_rrt.length)*100:.1f}%")
