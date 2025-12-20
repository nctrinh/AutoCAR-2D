import numpy as np
from typing import Optional, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle
from src.planning.base_planner import Path as PlannedPath



class PIDController:
    """
    PID controller for vehicle control.
    
    Controls both longitudinal (speed) and lateral (steering) motion.
    """

    def __init__(self, 
                 kp_lateral: float = 1.0,
                 ki_lateral: float = 0.1,
                 kd_lateral: float = 0.5,
                 kp_longitudinal: float = 0.5,
                 ki_longitudinal: float = 0.05,
                 kd_longitudinal: float = 0.1,
                 max_steering: float = 0.6,
                 max_acceleration: float = 3.0):
        """
        Initialize PID controller.
        
        Args:
            kp_lateral: Proportional gain for steering
            ki_lateral: Integral gain for steering
            kd_lateral: Derivative gain for steering
            kp_longitudinal: Proportional gain for speed
            ki_longitudinal: Integral gain for speed
            kd_longitudinal: Derivative gain for speed
            max_steering: Maximum steering angle output
            max_acceleration: Maximum acceleration output
        """

        # Lateral (steering) PID parameters
        self.kp_lat = kp_lateral
        self.ki_lat = ki_lateral
        self.kd_lat = kd_lateral
        
        # Longitudinal (speed) PID parameters
        self.kp_long = kp_longitudinal
        self.ki_long = ki_longitudinal
        self.kd_long = kd_longitudinal

        self.max_steering = max_steering
        self.max_acceleration = max_acceleration        

        # State variables
        self.integral_error_lat = 0.0
        self.prev_error_lat = 0.0
        self.integral_error_long = 0.0
        self.prev_error_long = 0.0
        
        # Anti-windup limits
        self.integral_limit = 10.0

    def reset(self):
        """Reset controller state."""
        self.integral_error_lat = 0.0
        self.prev_error_lat = 0.0
        self.integral_error_long = 0.0
        self.prev_error_long = 0.0

    def control(self, vehicle: Vehicle, 
                target_point: Tuple[float, float],
                target_speed: float,
                dt: float = 0.1) -> Tuple[float, float]:
        """
        Compute control commands (acceleration and steering).
        
        Args:
            vehicle: Current vehicle state
            target_point: Target point to track (x, y)
            target_speed: Desired speed
            dt: Time step
            
        Returns:
            (acceleration, steering_angle) control commands
        """
        # Lateral control (steering)
        steering_angle = self._lateral_control(vehicle, target_point, dt)
        
        # Longitudinal control (speed)
        acceleration = self._longitudinal_control(vehicle, target_speed, dt)
        
        return acceleration, steering_angle

    def _lateral_control(self, vehicle: Vehicle, 
                        target_point: Tuple[float, float],
                        dt: float) -> float:
        """
        Compute steering angle using PID control.
        
        Uses cross-track error and heading error.
        """
        # Calculate errors
        vehicle_x, vehicle_y = vehicle.get_position()
        vehicle_theta = vehicle.state.theta
        
        # Cross-track error (lateral distance to target)
        dx = target_point[0] - vehicle_x
        dy = target_point[1] - vehicle_y
        
        # Transform to vehicle frame
        error_front_axle = -np.sin(vehicle_theta) * dx + np.cos(vehicle_theta) * dy
        
        # Heading error
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - vehicle_theta
        
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Combined error (weighted)
        error = error_front_axle + 0.5 * heading_error
        
        # PID calculation
        self.integral_error_lat += error * dt
        
        # Anti-windup
        self.integral_error_lat = np.clip(
            self.integral_error_lat,
            -self.integral_limit,
            self.integral_limit
        )
        
        derivative_error = (error - self.prev_error_lat) / dt if dt > 0 else 0
        
        # PID output
        steering = (self.kp_lat * error + 
                   self.ki_lat * self.integral_error_lat + 
                   self.kd_lat * derivative_error)
        
        # Update previous error
        self.prev_error_lat = error
        
        # Clip to limits
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        return steering
    
    def _longitudinal_control(self, vehicle: Vehicle, 
                             target_speed: float,
                             dt: float) -> float:
        """
        Compute acceleration using PID control.
        
        Controls vehicle speed to match target speed.
        """
        # Speed error
        current_speed = vehicle.state.velocity
        error = target_speed - current_speed
        
        # PID calculation
        self.integral_error_long += error * dt
        
        # Anti-windup
        self.integral_error_long = np.clip(
            self.integral_error_long,
            -self.integral_limit,
            self.integral_limit
        )
        
        derivative_error = (error - self.prev_error_long) / dt if dt > 0 else 0
        
        # PID output
        acceleration = (self.kp_long * error + 
                       self.ki_long * self.integral_error_long + 
                       self.kd_long * derivative_error)
        
        # Update previous error
        self.prev_error_long = error
        
        # Clip to limits
        acceleration = np.clip(
            acceleration,
            -self.max_acceleration,
            self.max_acceleration
        )
        
        return acceleration


class PathFollowingPID:
    """
    Path following controller using PID.
    
    Tracks a planned path by finding the closest point on the path
    and using it as the target.
    """
    
    def __init__(self, 
                 pid_controller: Optional[PIDController] = None,
                 lookahead_distance: float = 5.0,
                 target_speed: float = 5.0):
        """
        Initialize path following controller.
        
        Args:
            pid_controller: PID controller instance (creates default if None)
            lookahead_distance: Distance ahead on path to target
            target_speed: Desired speed along path
        """
        self.pid = pid_controller or PIDController()
        self.lookahead_distance = lookahead_distance
        self.target_speed = target_speed
        self.path: Optional[PlannedPath] = None
        self.current_path_distance = 0.0
    
    def set_path(self, path: PlannedPath):
        """Set the path to follow."""
        self.path = path
        self.current_path_distance = 0.0
        self.pid.reset()
    
    def control(self, vehicle: Vehicle, dt: float = 0.1) -> Tuple[float, float]:
        """
        Compute control commands to follow path.
        
        Args:
            vehicle: Current vehicle state
            dt: Time step
            
        Returns:
            (acceleration, steering_angle) control commands
        """
        if self.path is None or len(self.path.points) == 0:
            return 0.0, 0.0
        
        # Find target point on path
        target_point = self._find_target_point(vehicle)
        
        # Adjust speed based on path curvature (optional)
        target_speed = self._compute_target_speed(vehicle, target_point)
        
        # Compute control
        acceleration, steering = self.pid.control(
            vehicle, target_point, target_speed, dt
        )
        
        return acceleration, steering
    
    def _find_target_point(self, vehicle: Vehicle) -> Tuple[float, float]:
        """
        Find target point on path using lookahead distance.
        
        Finds closest point on path, then looks ahead by lookahead_distance.
        """
        vehicle_pos = vehicle.get_position()
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.path.points):
            dist = np.sqrt((point.x - vehicle_pos[0])**2 + 
                          (point.y - vehicle_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Calculate distance along path to closest point
        path_dist = 0.0
        for i in range(closest_idx):
            path_dist += self.path.points[i].distance_to(self.path.points[i + 1])
        
        # Look ahead
        target_dist = path_dist + self.lookahead_distance
        
        # Get point at target distance
        target_point = self.path.get_point_at_distance(target_dist)
        
        if target_point is None:
            # Use last point if beyond path end
            target_point = self.path.points[-1]
        
        return target_point.to_tuple()
    
    def _compute_target_speed(self, vehicle: Vehicle, 
                             target_point: Tuple[float, float]) -> float:
        """
        Compute target speed based on path curvature.
        
        Slows down for sharp turns.
        """
        # Simple version: constant speed
        # Advanced: reduce speed for high curvature
        
        # Calculate heading change rate (approximation of curvature)
        heading_to_target = vehicle.heading_to(target_point[0], target_point[1])
        
        # Reduce speed for large heading errors (sharp turns)
        speed_reduction = 1.0 - min(abs(heading_to_target) / np.pi, 0.5)
        
        return self.target_speed * speed_reduction
    
    def is_goal_reached(self, vehicle: Vehicle, threshold: float = 1.0) -> bool:
        """Check if vehicle has reached the end of path."""
        if self.path is None:
            return True
        
        goal = self.path.points[-1]
        distance = vehicle.distance_to(goal.x, goal.y)
        
        return distance < threshold
    

if __name__ == "__main__":
    from src.core.vehicle import Vehicle, VehicleConfig
    from src.planning.base_planner import PathPoint, Path as PlannedPath
    
    # Create vehicle
    vehicle = Vehicle()
    vehicle.reset(x=0, y=0, theta=0)
    
    # Create simple path (straight line with turn)
    path_points = [
        PathPoint(0, 0),
        PathPoint(10, 0),
        PathPoint(20, 10),
        PathPoint(30, 20),
        PathPoint(40, 20)
    ]
    path = PlannedPath(path_points)
    
    # Create path following controller
    controller = PathFollowingPID(
        lookahead_distance=5.0,
        target_speed=5.0
    )
    controller.set_path(path)
    
    # Simulate
    print("Simulating path following...")
    dt = 0.1
    max_steps = 10000
    
    for step in range(max_steps):
        # Get control commands
        acceleration, steering = controller.control(vehicle, dt)
        
        # Update vehicle
        vehicle.update(acceleration, steering, dt)
        
        # Print progress every 50 steps
        if step % 50 == 0:
            pos = vehicle.get_position()
            print(f"Step {step}: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                  f"v={vehicle.state.velocity:.2f}m/s, "
                  f"steering={np.degrees(steering):.1f}°")
        
        # Check if goal reached
        if controller.is_goal_reached(vehicle, threshold=1.0):
            print(f"\nGoal reached at step {step}!")
            print(f"Final position: {vehicle.get_position()}")
            print(f"Target position: ({path.points[-1].x}, {path.points[-1].y})")
            break
    else:
        print("\nMax steps reached!")
        