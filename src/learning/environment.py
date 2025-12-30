import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, Dict, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle, VehicleConfig
from src.core.map import Map2D, CircleObstacle, PolygonObstacle, RectangleObstacle


class AutonomousCarEnv(gym.Env):
    """
    Gymnasium environment for autonomous car navigation.
    
    Observation Space:
        - Vehicle state: [x, y, vx, vy, theta, v, steering]
        - Goal relative: [dx, dy, distance, angle]
        - LIDAR-like sensors: [distances to obstacles in N directions]
        - Total: 7 + 4 + N dimensions
    
    Action Space:
        - Continuous: [acceleration, steering_angle]
        - Both normalized to [-1, 1]
    
    Reward:
        - Progress toward goal: +reward
        - Collision: -large penalty
        - Goal reached: +large reward
        - Time penalty: small negative per step
        - Smooth control: penalty for large actions
    """

    meatadata = {'render_mode': {'human', 'rgb_array'}, 'render_fps': 60}

    def __init__(self, 
                 map_env: Optional[Map2D] = None,
                 vehicle: Vehicle = Vehicle(),
                 max_steps: int = 1000,
                 num_lidar_rays: int = 16,
                 lidar_range: float = 20.0,
                 render_mode: Optional[str] = None):
        """
        Initialize environment.
        
        Args:
            map_env: Map environment (creates default if None)
            max_steps: Maximum steps per episode
            num_lidar_rays: Number of LIDAR sensor rays
            lidar_range: Maximum LIDAR sensing distance
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()

        self.map_env = map_env or self._create_default_map()
        self.max_steps = max_steps
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = lidar_range
        self.render_mode = render_mode

        self.vehicle = vehicle

        self.steps = 0
        self.prev_distance_to_goal = 0.0
        self.total_reward = 0.0
        self.prev_action = np.array([0.0, 0.0])

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        obs_dim = 7 + 4 + self.num_lidar_rays

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.renderer = None
    
    def _create_default_map(self) -> Map2D:
        map_env = Map2D(width=100, height=100)

        for _ in range(5):
            x = np.random.uniform(20, 80)
            y = np.random.uniform(20, 80)
            radius = np.random.uniform(3, 8)
            map_env.add_obstacle(CircleObstacle(x, y, radius))
        
        map_env.set_start(10, 10)
        map_env.set_goal(90, 90)
        
        return map_env
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation, info
        """

        super().reset(seed=seed)

        if self.map_env.start:
            x, y = self.map_env.start
        else:
            x, y = 10, 10

        theta = np.random.uniform(-np.pi, np.pi) if options and options.get('random_heading') else 0
        
        self.vehicle.reset(x=x, y=y, theta=theta)
        
        # Reset episode state
        self.steps = 0
        self.total_reward = 0.0
        self.prev_action = np.array([0.0, 0.0])
        
        if self.map_env.goal:
            self.prev_distance_to_goal = self.vehicle.distance_to(*self.map_env.goal)
        else:
            self.prev_distance_to_goal = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [acceleration, steering_angle] normalized to [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """

        acceleration = action[0] * self.vehicle.config.max_acceleration
        steering = action[1] * self.vehicle.config.max_steering_angle
        
        # Update vehicle
        self.vehicle.update(acceleration, steering)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check collision
        pos = self.vehicle.get_position()
        if self.map_env.is_collision(pos[0], pos[1], safety_margin=1.0):
            reward -= 100.0  # Large penalty
            terminated = True
        
        # Check goal reached
        if self.map_env.goal:
            distance_to_goal = self.vehicle.distance_to(*self.map_env.goal)
            if distance_to_goal < 3.0:
                reward += 100.0  # Large reward
                terminated = True
        
        # Check max steps
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        
        # Update state
        self.total_reward += reward
        self.prev_action = action
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation vector of shape (obs_dim,)
        """
        # Vehicle state [7 dims]
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        v = self.vehicle.state.velocity
        steering = self.vehicle.state.steering_angle
        
        # Velocity components
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        
        # Normalize
        vehicle_state = np.array([
            pos[0] / self.map_env.width,
            pos[1] / self.map_env.height,
            vx / self.vehicle.config.max_velocity,
            vy / self.vehicle.config.max_velocity,
            np.sin(theta),
            np.cos(theta),
            steering / self.vehicle.config.max_steering_angle
        ], dtype=np.float32)
        
        if self.map_env.goal:
            goal_x, goal_y = self.map_env.goal
            dx = goal_x - pos[0]
            dy = goal_y - pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) - theta
            # Normalize angle to [-pi, pi]
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            
            goal_info = np.array([
                dx / self.map_env.width,
                dy / self.map_env.height,
                distance / np.sqrt(self.map_env.width**2 + self.map_env.height**2),
                angle / np.pi
            ], dtype=np.float32)
        else:
            goal_info = np.zeros(4, dtype=np.float32)
        
        lidar_data = self._get_lidar_readings()
        
        observation = np.concatenate([vehicle_state, goal_info, lidar_data])
        
        return observation

    def _get_lidar_readings(self) -> np.ndarray:
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        readings = np.ones(self.num_lidar_rays, dtype=np.float32)
        
        ray_angles = theta + np.linspace(0, 2*np.pi, self.num_lidar_rays, endpoint=False)
        
        step_size = 0.5 
        max_steps = int(self.lidar_range / step_size)
        
        for i, angle in enumerate(ray_angles):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for step in range(1, max_steps + 1):
                dist = step * step_size
                rx = pos[0] + dist * cos_a
                ry = pos[1] + dist * sin_a
                
                if not (0 <= rx <= self.map_env.width and 0 <= ry <= self.map_env.height):
                    readings[i] = dist / self.lidar_range
                    break
                
                # Check vật cản (Hàm này nên được tối ưu bên class Map2D)
                if self.map_env.is_collision(rx, ry):
                    readings[i] = dist / self.lidar_range
                    break
                    
        return readings
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        reward = 0.0
        
        if self.map_env.goal:
            current_distance = self.vehicle.distance_to(*self.map_env.goal)
            progress = self.prev_distance_to_goal - current_distance
            reward += progress
            self.prev_distance_to_goal = current_distance

        lidar_readings = self._get_lidar_readings()
        min_lidar = np.min(lidar_readings)
        
        if min_lidar < 0.2: 
            reward -= 2.0 * (0.2 - min_lidar) 

        pos = self.vehicle.get_position()
        if self.map_env.is_collision(pos[0], pos[1], safety_margin=1.0):
            reward -= 50.0
            return reward

        if self.map_env.goal and current_distance < 3.0:
            reward += 50.0
            return reward

        reward -= 0.5 * np.abs(action[1])
        
        action_diff = action - self.prev_action
        reward -= 0.2 * np.sum(action_diff**2)

        reward -= 0.05

        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
        }
        
        if self.map_env.goal:
            pos = self.vehicle.get_position()
            distance = self.vehicle.distance_to(*self.map_env.goal)
            info['distance_to_goal'] = distance
            info['position'] = pos
        
        return info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing using pygame."""
        if self.renderer is None:
            from src.simulation.renderer import Renderer
            self.renderer = Renderer(
                screen_width=800,
                screen_height=800,
                world_width=self.map_env.width,
                world_height=self.map_env.height,
                caption="RL Training"
            )
        
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        self.renderer.clear()
        self.renderer.draw_grid()
        self.renderer.draw_map(self.map_env)
        
        if self.map_env.start and self.map_env.goal:
            self.renderer.draw_start_goal(self.map_env.start, self.map_env.goal)
        
        self.renderer.draw_vehicle(self.vehicle)
        
        # Draw LIDAR rays
        if hasattr(self.renderer, 'show_sensors') and self.renderer.show_sensors:
            self._draw_lidar_rays()
        
        self.renderer.update()
    
    def _render_rgb_array(self):
        """Render as RGB array."""
        if self.renderer is None:
            from src.simulation.renderer import Renderer
            self.renderer = Renderer(
                screen_width=400,
                screen_height=400,
                world_width=self.map_env.width,
                world_height=self.map_env.height
            )
        
        self.renderer.clear()
        self.renderer.draw_map(self.map_env)
        self.renderer.draw_vehicle(self.vehicle)
        
        import pygame
        # Convert pygame surface to RGB array
        img = pygame.surfarray.array3d(self.renderer.screen)
        img = np.transpose(img, (1, 0, 2))
        
        return img
    
    def _draw_lidar_rays(self):
        """Draw LIDAR sensor rays."""
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        
        lidar_readings = self._get_lidar_readings()
        
        for i in range(self.num_lidar_rays):
            ray_angle = theta + (2 * np.pi * i / self.num_lidar_rays)
            distance = lidar_readings[i] * self.lidar_range
            
            end_x = pos[0] + distance * np.cos(ray_angle)
            end_y = pos[1] + distance * np.sin(ray_angle)
            
            # Draw ray (would need to add method to renderer)
            # self.renderer.draw_line(pos[0], pos[1], end_x, end_y, color=(255, 255, 0))
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Example usage and testing
if __name__ == "__main__":
    print("Testing RL Environment...")
    print("="*70)
    
    # Create environment
    env = AutonomousCarEnv(render_mode=None)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test random episode
    print("\nRunning random episode...")
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 20 == 0:
            print(f"  Step {steps}: reward={reward:.3f}, "
                  f"distance={info.get('distance_to_goal', 0):.2f}m")
    
    print(f"\nEpisode finished after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    
    # Test with Stable-Baselines3 check
    try:
        from stable_baselines3.common.env_checker import check_env
        print("\n" + "="*70)
        print("Checking environment compatibility with Stable-Baselines3...")
        check_env(env)
        print("✓ Environment is compatible with Stable-Baselines3!")
    except ImportError:
        print("\nStable-Baselines3 not installed. Skipping compatibility check.")
    except Exception as e:
        print(f"\n✗ Environment check failed: {e}")
    
    env.close()
    print("\n✓ Environment test complete!")
