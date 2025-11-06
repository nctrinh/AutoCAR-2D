import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class VehicleConfig:
    length: float = 4.0 # meters
    width: float = 2.0 # meters
    wheelbase: float = 2.5 # distance between front and rear axles
    max_velocity: float =  10.0 # m/s
    max_acceleration: float = 3.0 # m/s^2
    max_deceleration: float = -5.0 # m/s^2
    max_steering_angle: float = 0.6 # radian

@dataclass
class VehicleState:
    x: float = 0.0 # position x
    y: float = 0.0 # position y
    theta: float = 0.0 # heading angle
    velocity: float = 0.0 # foward verlocity
    steering_angle: float = 0.0 # current steering angle

class Vehicle:
    """
    2D Vehicle with bicycle kinematics model.
    
    The bicycle model simplifies the vehicle to two wheels:
    - One front wheel (steerable)
    - One rear wheel (fixed direction)
    
    State: [x, y, theta, v]
    Control: [acceleration, steering_angle]
    """

    def __init__(self, config: Optional[VehicleConfig] = None):
        """
        Initialize vehicle.
        
        Args:
            config: Vehicle configuration. Uses default if None.
        """
        self.config = config or VehicleConfig()
        self.state = VehicleState()
        self.dt = 0.1 # time step for simulation (seconds)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.state = VehicleState(x=x, y=y, theta=theta)

    def update(self, acceleration: float, steering_angle: float, dt: Optional[float] = None):
        """
        Update vehicle state using bicycle model.
        
        Bicycle model equations:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = (v / L) * tan(steering_angle)
        v_dot = acceleration
        
        Args:
            acceleration: Desired acceleration (m/s^2)
            steering_angle: Desired steering angle (radians)
            dt: Time step (uses self.dt if None)
        """
        dt = dt or self.dt

        acceleration = np.clip(
            acceleration, 
            self.config.max_deceleration, 
            self.config.max_acceleration
        )
        steering_angle = np.clip(
            steering_angle,
            -self.config.max_steering_angle,
            self.config.max_steering_angle
        )

        new_velocity = self.state.velocity + acceleration * dt
        new_velocity = np.clip(new_velocity, 0, self.config.max_velocity)

        if abs(new_velocity) >= 1e-3:
            theta_dot = (new_velocity / self.config.wheelbase) * np.tan(steering_angle)

            new_theta = self.state.theta + theta_dot * dt
            new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta)) # [-pi, pi]

            new_x = self.state.x + new_velocity * np.cos(self.state.theta) * dt
            new_y = self.state.y + new_velocity * np.sin(self.state.theta) * dt
        else:
            new_x, new_y, new_theta = self.state.x, self.state.y, self.state.theta

        self.state.x = new_x
        self.state.y = new_y
        self.state.theta = new_theta
        self.state.velocity = new_velocity
        self.state.steering_angle = steering_angle

    def get_position(self) -> Tuple[float, float]:
        return (self.state.x, self.state.y)
    

    def get_state_array(self) -> np.ndarray:
        return np.array([
            self.state.x,
            self.state.y,
            self.state.theta,
            self.state.velocity
        ])
    

    def get_corners(self) -> np.ndarray:
        """
        Get vehicle's four corners in global coordinates.
        Useful for collision detection and visualization.
        
        Returns:
            Array of shape (4, 2) with corner positions
        """
        # Local coordinates (vehicle frame)
        half_length = self.config.length / 2
        half_width = self.config.width / 2
        
        corners_local = np.array([
            [half_length, half_width],    # front-right
            [half_length, -half_width],   # front-left
            [-half_length, -half_width],  # rear-left
            [-half_length, half_width]    # rear-right
        ])

        # Rotation matrix
        cos_theta = np.cos(self.state.theta)
        sin_theta = np.sin(self.state.theta)
        rotation = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Transform to global coordinates
        corners_global = corners_local @ rotation.T
        corners_global[:, 0] += self.state.x
        corners_global[:, 1] += self.state.y
        
        return corners_global
    
    def get_front_axle_position(self) -> Tuple[float, float]:
        front_x = self.state.x + (self.config.wheelbase / 2) * np.cos(self.state.theta)
        front_y = self.state.y + (self.config.wheelbase / 2) * np.sin(self.state.theta)
        return (front_x, front_y)
    
    def distance_to(self, x: float, y: float) -> float:
        dx = x - self.state.x
        dy = y - self.state.y

        target_heading = np.arctan2(dy, dx)

        diff = target_heading - self.state.theta
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff
    
    def __repr__(self) -> str:
        return (f"Vehicle(pos=({self.state.x:.2f}, {self.state.y:.2f}), "
                f"theta={np.degrees(self.state.theta):.1f}°, "
                f"v={self.state.velocity:.2f}m/s)")
    
if __name__ == '__main__':

    config = VehicleConfig(
        length=4.0,
        width=2.0,
        max_velocity=15.0
    )

    vehicle = Vehicle(config)

    vehicle.reset()

    print("Initial: ", vehicle)

    for _ in range(10):
        vehicle.update(acceleration=2.0, steering_angle=0.1)

    print("After 10 steps:", vehicle)
    print("Corners:", vehicle.get_corners())

