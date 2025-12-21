import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle
from src.planning.base_planner import Path as PlannedPath, PathPoint



class PurePursuitController:
    def __init__(self, 
                 lookahead_distance: float = 5.0,
                 lookahead_gain: float = 0.5,
                 min_lookahead: float = 2.0,
                 max_lookahead: float = 10.0,
                 target_speed: float = 5.0,
                 speed_kp: float = 0.5):
        """
        Initialize Pure Pursuit controller.
        
        Args:
            lookahead_distance: Base lookahead distance (meters)
            lookahead_gain: Gain for velocity-dependent lookahead
            min_lookahead: Minimum lookahead distance
            max_lookahead: Maximum lookahead distance
            target_speed: Desired speed along path
            speed_kp: Proportional gain for speed control
        """

        self.base_lookahead = lookahead_distance
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.target_speed = target_speed
        self.speed_kp = speed_kp
        
        self.path: Optional[PlannedPath] = None
        self.current_target_idx = 0

    def set_path(self, path: PlannedPath):
        self.path = path
        self.current_target_idx = 0

    def control(self, vehicle: Vehicle, dt: float = 0.1) -> Tuple[float, float]:
        """
        Compute control commands using Pure Pursuit.
        
        Args:
            vehicle: Current vehicle state
            dt: Time step
            
        Returns:
            (acceleration, steering_angle) control commands
        """
        if self.path is None or len(self.path.points) < 2:
            return 0.0, 0.0

        current_speed = max(vehicle.state.velocity, 1.0)
        lookahead = self.base_lookahead + self.lookahead_gain * current_speed
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)

        lookahead_point = self._find_lookahead_point(vehicle, lookahead)

        if lookahead_point is None:
            lookahead_point = self.path.points[-1].to_tuple()

        steering_angle = self._compute_steering_angle(vehicle, lookahead_point, lookahead)

        acceleration = self._compute_acceleration(vehicle)

        return acceleration, steering_angle
    
    def _find_lookahead_point(self, vehicle: Vehicle, lookahead: float) -> Optional[Tuple[float, float]]:
        """
        Find the lookahead point on the path.
        
        Searches for a point on the path that is approximately lookahead
        distance away from the vehicle.
        """

        vehicle_pos = vehicle.get_position()

        best_point = None
        best_distance_diff = float('inf')

        for i in range(self.current_target_idx, len(self.path.points)):
            point = self.path.points[i]
            distance = np.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)

            distance_diff = abs(distance - lookahead)

            if distance_diff < best_distance_diff:
                best_distance_diff = distance_diff
                best_point = (point.x, point.y)
                self.current_target_idx = i

            if distance >= lookahead:
                break
        
        return best_point
    
    def _compute_steering_angle(self, vehicle: Vehicle, lookahead_point: Tuple[float, float], lookahead: float) -> float:
        """
        Compute steering angle using Pure Pursuit algorithm.
        
        Pure Pursuit formula:
        steering_angle = atan(2 * L * sin(alpha) / lookahead)
        
        where:
        - L is the wheelbase
        - alpha is the angle between vehicle heading and lookahead point
        """

        vehicle_pos = vehicle.get_position()
        vehicle_theta = vehicle.state.theta

        dx = lookahead_point[0] - vehicle_pos[0]
        dy = lookahead_point[1] - vehicle_pos[1]

        alpha = np.arctan2(dy, dx) - vehicle_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        wheelbase = vehicle.config.wheelbase

        if abs(lookahead) < 1e-3:
            return 0.0

        steering_angle = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead)

        max_steering_angle = vehicle.config.max_steering_angle

        steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

        return steering_angle
    
    def _compute_acceleration(self, vehicle: Vehicle) -> float:
        """
        Compute acceleration for speed control.
        
        Simple proportional control to reach target speed.
        """
        current_speed = vehicle.state.velocity
        speed_error = self.target_speed - current_speed

        acceleration = self.speed_kp * speed_error

        max_accel = vehicle.config.max_acceleration
        max_decel = vehicle.config.max_deceleration

        acceleration = np.clip(acceleration, max_decel, max_accel)

        return acceleration
    
    def is_goal_reached(self, vehicle: Vehicle, threshold: float = 1.0) -> bool:
        if self.path is None or len(self.path.points) == 0:
            return True
        
        goal = self.path.points[-1]
        distance = vehicle.distance_to(goal.x, goal.y)

        return distance < threshold
    
    def get_lookahead_point(self, vehicle: Vehicle) -> Optional[Tuple[float, float]]:
        current_speed = max(vehicle.state.velocity, 1.0)
        lookahead = self.base_lookahead + self.lookahead_gain * current_speed
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)

        return self._find_lookahead_point(vehicle, lookahead)
    

class AdaptivePurePursuitController(PurePursuitController):
    """
    Adaptive Pure Pursuit controller.
    
    Adjusts speed based on path curvature and obstacles.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_curvature_speed = 3.0
        self.curvature_lookahead = 10.0

    def control(self, vehicle: Vehicle, dt: float = 0.1) -> Tuple[float, float]:
        """Control with adaptive speed based on path curvature."""
        if self.path is None or len(self.path.points) < 2:
            return 0.0, 0.0
        
        curvature = self._estimate_curvature(vehicle)

        if curvature > 0.1:
            self.target_speed = self.max_curvature_speed
        else:
            pass

        return super().control(vehicle, dt)
    
    def _estimate_curvature(self, vehicle: Vehicle) -> float:
        """
        Estimate path curvature ahead of vehicle.
        
        Returns:
            Estimated curvature (1/radius)
        """
        if self.path is None or len(self.path.points) < 3:
            return 0.0
        
        vehicle_pos = vehicle.get_position()

        points_ahead = []

        for i in range(self.current_target_idx, min(len(self.path.points), self.current_target_idx + 3)):
            point = self.path.points[i]
            distance = np.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)

            if distance < self.curvature_lookahead:
                points_ahead.append((point.x, point.y))
        
        if len(points_ahead) < 3:
            return 0.0
        
        p1, p2, p3 = points_ahead[:3]

        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        # Vectors
        ab = b - a
        bc = c - b
        
        # Angle change
        angle1 = np.arctan2(ab[1], ab[0])
        angle2 = np.arctan2(bc[1], bc[0])
        angle_diff = abs(angle2 - angle1)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)

        total_dist = np.linalg.norm(ab) + np.linalg.norm(bc)
        
        if total_dist < 1e-3:
            return 0.0
        
        # Approximate curvature
        curvature = angle_diff / total_dist
        
        return curvature
    
if __name__ == "__main__":
    from src.core.vehicle import Vehicle
    from src.planning.base_planner import PathPoint, Path as PlannedPath
    
    # Create vehicle
    vehicle = Vehicle()
    vehicle.reset(x=0, y=0, theta=0)
    
    # Create curved path
    path_points = []
    for i in range(50):
        t = i / 10.0
        x = t
        y = 5 * np.sin(t * 0.5)  # Sinusoidal path
        path_points.append(PathPoint(x, y))
    
    path = PlannedPath(path_points)
    
    # Create Pure Pursuit controller
    controller = PurePursuitController(
        lookahead_distance=3.0,
        lookahead_gain=0.3,
        target_speed=5.0
    )
    controller.set_path(path)
    
    print("Simulating Pure Pursuit on curved path...")
    print("="*60)
    
    dt = 0.1
    max_steps = 1000
    
    for step in range(max_steps):
        # Get control
        acceleration, steering = controller.control(vehicle, dt)
        
        # Update vehicle
        vehicle.update(acceleration, steering, dt)
        
        # Print progress
        if step % 10 == 0:
            pos = vehicle.get_position()
            lookahead = controller.get_lookahead_point(vehicle)
            print(f"Step {step}:")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"  Speed: {vehicle.state.velocity:.2f} m/s")
            print(f"  Steering: {np.degrees(steering):.1f}°")
            if lookahead:
                print(f"  Lookahead: ({lookahead[0]:.2f}, {lookahead[1]:.2f})")
        
        # Check goal
        if controller.is_goal_reached(vehicle):
            print(f"\nGoal reached at step {step}!")
            break
    
    print("\nSimulation complete!")