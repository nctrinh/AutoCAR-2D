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
    """Evaluate one episode without Pygame rendering logic (Headless mode)."""
    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    start_time = time.time()
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        
        if render:
            env.render()

    return {
        'reward': episode_reward,
        'steps': steps,
        'time': time.time() - start_time,
        'distance_to_goal': info.get('distance_to_goal', 0),
        'success': info.get('distance_to_goal', float('inf')) < 3.0,
    }

def visualize_episode_with_renderer(model, env: AutonomousCarEnv, config: ConfigLoader):
    """
    Visualize RL episode using the high-quality Renderer from Simulation.
    """
    sim_params = config.get_simulation_params()
    map_params = config.get_map_params()
    
    # 1. Init Renderer
    renderer = Renderer(
        screen_width=sim_params.get('screen_width', 1200),
        screen_height=sim_params.get('screen_height', 800),
        world_width=map_params.get('width', 100),
        world_height=map_params.get('height', 100)
    )
    
    print("Controls: SPACE=Pause, G=Grid, T=Trajectory, I=Info, ESC=Quit")
    
    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    paused = False
    clock = pygame.time.Clock()
    target_fps = 60
    
    # Tracking data
    renderer.trajectory = [] # Reset trajectory
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_g:
                    renderer.show_grid = not renderer.show_grid
                elif event.key == pygame.K_t:
                    renderer.show_trajectory = not renderer.show_trajectory
                elif event.key == pygame.K_i:
                    renderer.show_info = not renderer.show_info

        if not running:
            break

        if not paused and not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            pos = env.vehicle.get_position()
            renderer.add_trajectory_point(pos[0], pos[1])
            
            if done:
                print(f"\nEpisode finished. Reward: {episode_reward:.2f}, Steps: {steps}")
                print("Press ESC to exit visualization.")

        renderer.clear()
        renderer.draw_grid()
        
        if env.map_env:
            renderer.draw_map(env.map_env)
            if env.map_env.start and env.map_env.goal:
                renderer.draw_start_goal(env.map_env.start, env.map_env.goal)
        renderer.draw_lidar_zone(
                vehicle=env.vehicle,
                sensor_range=env.lidar_range,
                fov_deg=360,
                color=(0, 255, 255), # Cyan
                alpha=40
            )
        renderer.draw_trajectory()
        
        if env.vehicle:
            renderer.draw_vehicle(env.vehicle)
        
        additional_info = {
            'Reward': f"{episode_reward:.1f}",
            'Steps': steps,
            'Action': f"[{action[0]:.2f}, {action[1]:.2f}]" if not done else "Done",
            'Dist to Goal': f"{info.get('distance_to_goal', 0):.2f}m"
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
            max_steps=1000,
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