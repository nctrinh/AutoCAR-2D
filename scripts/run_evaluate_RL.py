import sys
import argparse
from pathlib import Path
import numpy as np
import time
import pygame
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

sys.path.append(str(Path(__file__).parent.parent))

from src.core.vehicle import Vehicle
from src.learning.environment import AutonomousCarEnv
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
from src.utils.config_loader import ConfigLoader
from src.simulation.renderer import Renderer 
from src.planning.a_star import AStarPlanner
from src.control.pid_controller import PIDController

class FogOfWarDriver:
    def __init__(self, env: AutonomousCarEnv):
        self.env = env
        
        self.internal_map = Map2D(self.env.map_env.width, self.env.map_env.height, self.env.map_env.safety_margin)
        
        self.final_goal = env.map_env.goal
        self.internal_map.set_start(*env.map_env.start)
        self.internal_map.set_goal(*self.final_goal)
        
        self.current_path_points = []
        self.current_wp_idx = 0
        self.replan_cooldown = 0
        self.replan_interval = 1
        
        self.detected_obstacles_cache = [] 

    def update(self, lidar_data: np.ndarray =None):
        vehicle_pos = self.env.vehicle.get_position()
        
        self._update_map_from_lidar(lidar_data)

        self.replan_cooldown -= 1
        need_replan = False
        if not self.current_path_points:
            need_replan = True
        elif self.replan_cooldown <= 0:
            if self._is_path_blocked():
                need_replan = True
            self.replan_cooldown = self.replan_interval
        
        if need_replan:
            self._replan(vehicle_pos)
            self.replan_cooldown = self.replan_interval

        self._update_env_goal_for_rl(vehicle_pos)

    def _update_map_from_lidar(self, lidar_data: np.ndarray=None):
        if lidar_data is None:
            lidar_data, _ = self.env._get_observation()
        
        pos = self.env.vehicle.get_position()
        theta = self.env.vehicle.state.theta
        num_rays = self.env.num_lidar_rays
        max_range = self.env.lidar_range
        
        for i in range(num_rays):
            dist_norm = lidar_data[i]
            
            if dist_norm < 1.0: 
                real_dist = dist_norm * max_range
                angle = theta + (2 * np.pi * i / num_rays)
                
                ox = pos[0] + real_dist * np.cos(angle)
                oy = pos[1] + real_dist * np.sin(angle)
                
                if self._is_new_obstacle(ox, oy):
                    self.internal_map.add_obstacle(CircleObstacle(ox, oy, radius=0.25))
                    self.detected_obstacles_cache.append((ox, oy))

    def _is_new_obstacle(self, x, y):
        for (cx, cy) in self.detected_obstacles_cache:
            if np.hypot(cx - x, cy - y) < 2.0:
                return False
        return True

    def _is_path_blocked(self):
        if not self.current_path_points: return True
        
        check_range = min(len(self.current_path_points), self.current_wp_idx + 3)
        for i in range(self.current_wp_idx, check_range):
            p = self.current_path_points[i]
            if self.internal_map.is_collision(p.x, p.y):
                return True
        return False

    def _replan(self, start_pos):
        planner = AStarPlanner(self.internal_map, grid_resolution=1.0)
        
        path_obj = planner.plan(start_pos, self.final_goal, info=False)
        
        if path_obj:
            self.current_path_points = path_obj.points
            self.current_wp_idx = 0
        else:
            print("Warning: No path found")

    def _update_env_goal_for_rl(self, vehicle_pos):
        if not self.current_path_points:
            return

        target_p = self.current_path_points[self.current_wp_idx]
        dist = np.hypot(target_p.x - vehicle_pos[0], target_p.y - vehicle_pos[1])
        
        if dist < 3.0:
            if self.current_wp_idx < len(self.current_path_points) - 1:
                self.current_wp_idx += 1
        
        wp = self.current_path_points[self.current_wp_idx]
        
        self.env.map_env.set_goal(wp.x, wp.y)

def load_map(config: ConfigLoader) -> Map2D:
    """Load map environment từ config."""
    map_params = config.get_map_params()

    if map_params.get("evaluate_map_yaml_file") != '':
        map_env = Map2D.load_from_yaml(map_params.get("evaluate_map_yaml_file"))
        return map_env
    else:
        map_env = Map2D(100, 100, 4.0)

        start = [10, 10]
        goal = [90, 90]
        map_env.set_start(start[0], start[1])
        map_env.set_goal(goal[0], goal[1])

        map_env.add_obstacle(RectangleObstacle(x=25, y=20, width=10, height=40))
        map_env.add_obstacle(RectangleObstacle(x=50, y=40, width=10, height=40))
        map_env.add_obstacle(RectangleObstacle(x=75, y=20, width=10, height=40))
        map_env.add_obstacle(CircleObstacle(x=40, y=65, radius=8))

        return map_env

def evaluate_episode(model, env: AutonomousCarEnv, render: bool = False, deterministic: bool = True) -> dict:
    """
    Evaluate one episode using Hybrid Logic (A* + RL) without rendering.
    """
    driver = FogOfWarDriver(env)

    obs, info = env.reset()
    
    driver.update()
    
    _, obs = env._get_observation()

    done = False
    episode_reward = 0
    steps = 0
    start_time = time.time()
    
    is_success = False
    final_goal = driver.final_goal
    
    while not done:
        driver.update()
        _, obs = env._get_observation()
        
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        lidar_data = info['lidar_data']
        
        episode_reward += reward
        steps += 1
        vehicle_corners = env.vehicle.get_corners()
        pos = env.vehicle.get_position()
        if (env.map_env.is_collision(vehicle_corners[0][0], vehicle_corners[0][1]) 
            and env.map_env.is_collision(vehicle_corners[1][0], vehicle_corners[1][1])
            and env.map_env.is_collision(vehicle_corners[2][0], vehicle_corners[2][1])
            and env.map_env.is_collision(vehicle_corners[3][0], vehicle_corners[3][1])):
            done = True
            is_success = False
        
        dist_to_final = np.linalg.norm(pos - np.array(final_goal))
        
        if dist_to_final < 3.0:
            done = True
            is_success = True

        if steps >= env.max_steps:
            done = True
            
        if render:
            env.render()

    return {
        'reward': episode_reward,
        'steps': steps,
        'time': time.time() - start_time,
        'distance_to_goal': dist_to_final,
        'success': is_success,
    }

def visualize_episode_with_renderer(model, env: AutonomousCarEnv, config: ConfigLoader):
    """
    Visualize RL episode with 'Fog of War' logic:
    - Real-time Mapping (Lidar -> Internal Map)
    - Dynamic Re-planning (A* updates path when obstacles detected)
    """
    sim_params = config.get_simulation_params()
    map_params = config.get_map_params()
    
    renderer = Renderer(
        screen_width=sim_params.get('screen_width', 1200),
        screen_height=sim_params.get('screen_height', 800),
        world_width=map_params.get('width', 100),
        world_height=map_params.get('height', 100)
    )
    
    driver = FogOfWarDriver(env)

    print("Controls: SPACE=Pause, G=Grid, T=Trajectory, P=Path, I=Info, ESC=Quit")
    
    obs, info = env.reset()
    driver.update()
        
    done = False
    episode_reward = 0
    steps = 0
    paused = False
    clock = pygame.time.Clock()
    target_fps = 60
    render_every_n_frames = 1
    frame_counter = 0
    
    # Visualization toggles
    show_path = True
    show_internal_map = True
    
    renderer.trajectory = [] 
    
    running = True
    lidar_data, obs = env._get_observation()
    pid_controller = PIDController()
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE: 
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_g: renderer.show_grid = not renderer.show_grid
                elif event.key == pygame.K_t: renderer.show_trajectory = not renderer.show_trajectory
                elif event.key == pygame.K_i: renderer.show_info = not renderer.show_info
                elif event.key == pygame.K_p: show_path = not show_path # Toggle vẽ đường A*

        if not running: break
        if not paused and not done:
            # GET OBS (Refresh observation relative to new Waypoint)
            lidar_data, obs = env._get_observation()

            # DRIVER UPDATE (Sense -> Map -> Plan)
            driver.update(lidar_data) 
            
            # 3. RL PREDICT
            action, _ = model.predict(obs, deterministic=True)
            # action = pid_controller.control(env.vehicle, env.map_env.goal, 2.0)
            
            # 4. STEP
            obs, reward, terminated, truncated, info = env.step(action)
            lidar_data = info['lidar_data']
            
            episode_reward += reward
            steps += 1
            if terminated:
                print(f"\nCRASH! Vehicle collided at step {steps}. Reward: {episode_reward:.2f}")
                done = True
            # CHECK REAL GOAL
            dist_to_final = np.linalg.norm(env.vehicle.get_position() - np.array(driver.final_goal))
            if dist_to_final < 3.0:
                print(f"\nVICTORY! Reached Goal. Reward: {episode_reward:.2f}, Steps: {steps}")
                done = True
            elif steps >= env.max_steps:
                done = True

            pos = env.vehicle.get_position()
            renderer.add_trajectory_point(pos[0], pos[1])
        frame_counter += 1
        if frame_counter % render_every_n_frames == 0:
            # DRAWING
            renderer.clear()
            renderer.draw_grid()
            
            if env.map_env:
                renderer.draw_map(env.map_env)
                renderer.draw_start_goal(env.map_env.start, driver.final_goal)
                
            if show_internal_map:
                for obs_obj in driver.internal_map.obstacles:
                    if hasattr(obs_obj, 'radius'):
                        cx, cy = renderer.world_to_screen(obs_obj.x, obs_obj.y)
                        scale = (renderer.scale_x + renderer.scale_y) / 2
                        rad = int(obs_obj.radius * scale)
                        pygame.draw.circle(renderer.screen, (255, 50, 50), (cx, cy), rad, 1)

            if show_path and driver.current_path_points:
                points = [(p.x, p.y) for p in driver.current_path_points]
                if len(points) > 1:
                    screen_points = [renderer.world_to_screen(p[0], p[1]) for p in points]
                    pygame.draw.lines(renderer.screen, (255, 255, 0), False, screen_points, 2)
                    
                if driver.current_path_points:
                    try:
                        wp = driver.current_path_points[driver.current_wp_idx]
                        wx, wy = renderer.world_to_screen(wp.x, wp.y)
                        pygame.draw.circle(renderer.screen, (0, 255, 0), (wx, wy), 5)
                    except IndexError:
                        pass

            renderer.draw_lidar_zone(
                vehicle=env.vehicle,
                sensor_range=env.lidar_range,
                fov_deg=360,
                color=(0, 255, 255), 
                alpha=40
            )
            renderer.draw_trajectory()
            if env.vehicle:
                renderer.draw_vehicle(env.vehicle)
            
            additional_info = {
                'Steps': steps,
                'Real Dist': f"{dist_to_final:.2f}m"
            }
            renderer.draw_info_panel(env.vehicle, steps, clock.get_fps(), additional_info)
            
            renderer.draw_legend()
            renderer.update()
            clock.tick(target_fps)

    renderer.close()

def evaluate_multiple_episodes(model, env: AutonomousCarEnv, n_episodes: int = 10) -> dict:
    """Evaluate multiple episodes for statistics."""
    print(f"\nEvaluating over {n_episodes} episodes...")
    all_stats = []
    
    for episode in range(n_episodes):
        stats = evaluate_episode(model, env, render=False)
        all_stats.append(stats)
        print(f"  Ep {episode+1}: Reward={stats['reward']:.2f}, Steps={stats['steps']}, Success={'✓' if stats['success'] else '✗'}")
    
    rewards = [s['reward'] for s in all_stats]
    successes = [s['success'] for s in all_stats]
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': np.mean(successes),
        'all_episodes': all_stats
    }

def print_evaluation_summary(stats: dict):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes: {len(stats['all_episodes'])}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--model', type=str, default='trained_models/ppo/ppo_final.zip', help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/RL_config.yaml', help='Config path')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'])
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--visualize', action='store_true', help='Visualize with Renderer')
    
    args = parser.parse_args()
    config = ConfigLoader(args.config)
    
    print("=" * 70)
    print(f"RL AGENT EVALUATION ({args.algorithm.upper()})")
    print("=" * 70)
    
    try:
        if not Path(args.model).exists():
            print(f"Error: Model not found at {args.model}")
            return 1
        
        map_env = load_map(config)
        
        vehicle = Vehicle(config.get_vehicle_config())
        
        env = AutonomousCarEnv(
            map_env=map_env,
            vehicle=vehicle,
            max_steps=10000,
            num_lidar_rays=16,
            render_mode=None 
        )
        
        print(f"Loading model: {args.model}")
        if args.algorithm == 'ppo':
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
        elif args.algorithm == 'sac':
            from stable_baselines3 import SAC
            model = SAC.load(args.model)
            
        if args.visualize:
            visualize_episode_with_renderer(model, env, config)
        else:
            stats = evaluate_multiple_episodes(model, env, n_episodes=args.episodes)
            print_evaluation_summary(stats)
            
        env.close()
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())