import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(str(Path(__file__).parent.parent))

from src.learning.environment import AutonomousCarEnv
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
# Giả sử bạn đã có class ConfigLoader, hoặc dùng trực tiếp yaml như bên dưới

class ConfigLoader:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
    def get(self, section, key, default=None):
        return self.cfg.get(section, {}).get(key, default)
        
    def get_section(self, section):
        return self.cfg.get(section, {})

def load_map_from_config(config: ConfigLoader) -> Map2D:
    map_cfg = config.get_section("map")
    
    map_env = Map2D(
        width=map_cfg.get("width", 100), 
        height=map_cfg.get("height", 100),
        safety_margin=map_cfg.get("safety_margin", 1.0)
    )
    
    start = map_cfg.get("start", [10, 10])
    goal = map_cfg.get("goal", [90, 90])
    map_env.set_start(start[0], start[1])
    map_env.set_goal(goal[0], goal[1])
    
    template = map_cfg.get("train_template", "custom")
    print(f"Loading map template: {template}")

    if template == 'easy':
        map_env.add_obstacle(CircleObstacle(30, 30, 6))
        map_env.add_obstacle(CircleObstacle(70, 70, 6))
        
    elif template == 'medium':
        map_env.add_obstacle(CircleObstacle(30, 30, 8))
        map_env.add_obstacle(RectangleObstacle(50, 50, 15, 20))
        map_env.add_obstacle(CircleObstacle(70, 25, 6))
        map_env.add_obstacle(CircleObstacle(60, 75, 7))
        
    elif template == 'hard':
        for _ in range(20):
            map_env.add_obstacle(CircleObstacle(
                np.random.uniform(0, map_env.width), 
                np.random.uniform(0, map_env.height), 
                np.random.uniform(1, 4)
            ))
            
    elif template == 'maze':
        map_env.add_obstacle(RectangleObstacle(25, 20, 10, 40))
        map_env.add_obstacle(RectangleObstacle(50, 40, 10, 40))
        map_env.add_obstacle(RectangleObstacle(75, 20, 10, 40))
        map_env.add_obstacle(CircleObstacle(40, 65, 8))

    else:
        for obs in map_cfg.get("train_obstacles", []):
            if obs['type'] == 'circle':
                map_env.add_obstacle(CircleObstacle(obs['x'], obs['y'], obs['radius']))
            elif obs['type'] == 'rectangle':
                map_env.add_obstacle(RectangleObstacle(
                    obs['x'], obs['y'], obs['width'], obs['height'], obs.get('angle', 0)
                ))
            elif obs['type'] == "polygon":
                obstacle = PolygonObstacle(ertices=np.array(obs["vertices"]))
            else:
                print(f"Warning: Unknown obstacle type {obs['type']}, skipping...")
                continue
    return map_env

def train_model(env, config: ConfigLoader):
    train_cfg = config.get_section("training")
    model_cfg = config.get_section("model")
    
    algo = train_cfg.get("algorithm", "ppo").lower()
    save_dir = Path(train_cfg.get("save_dir", "trained_models"))
    log_dir = Path(train_cfg.get("log_dir", "logs"))
    ppo_model_dir = Path(train_cfg.get("ppo_model_dir", "trained_models/ppo"))
    sac_model_dir = Path(train_cfg.get("sac_model_dir", "trained_models/sac"))
    
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ppo_model_dir.mkdir(parents=True, exist_ok=True)
    sac_model_dir.mkdir(parents=True, exist_ok=True)

    if algo == 'ppo':
        save_dir = ppo_model_dir
    elif algo == 'sac':
        save_dir = sac_model_dir

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.get("save_freq", 50000),
        save_path=str(save_dir),
        name_prefix=f"{algo}_checkpoint"
    )
    
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"\nInitializing {algo.upper()} agent...")
    
    if algo == 'ppo':
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=model_cfg.get("learning_rate", 3e-4),
            n_steps=model_cfg.get("n_steps", 2048),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            ent_coef=model_cfg.get("ent_coef", 0.01),
            verbose=1,
            tensorboard_log=str(log_dir),
            device=train_cfg.get("device", "auto")
        )
    elif algo == 'sac':
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=model_cfg.get("learning_rate", 3e-4),
            buffer_size=model_cfg.get("buffer_size", 100000),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            tau=model_cfg.get("tau", 0.005),
            verbose=1,
            tensorboard_log=str(log_dir),
            device=train_cfg.get("device", "auto")
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(model.policy)
    print(f"\nStarting training for {train_cfg.get('timesteps')} steps...")
    
    model.learn(
        total_timesteps=train_cfg.get("timesteps"),
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    final_path = save_dir / f"{algo}_final.zip"
    model.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train RL agent with YAML config')
    parser.add_argument('--config', type=str, default='config/RL_config.yaml', help='Path to config file')
    args = parser.parse_args()

    print("="*70)
    print(f"LOADING CONFIGURATION: {args.config}")
    print("="*70)

    try:
        config = ConfigLoader(args.config)
        
        print("\nCreating environment...")
        map_env = load_map_from_config(config)
        
        env_cfg = config.get_section("environment")
        env = AutonomousCarEnv(
            map_env=map_env,
            max_steps=env_cfg.get("max_steps", 1000),
            num_lidar_rays=env_cfg.get("num_lidar_rays", 16),
            render_mode=None
        )
        print("✓ Environment created")

        model = train_model(env, config)
        
        print("\nRunning post-training evaluation...")
        eval_episodes = config.get("training", "eval_episodes", 5)
        obs, _ = env.reset()
        for i in range(eval_episodes):
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"  Eval Episode {i+1}: Reward = {total_reward:.2f}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())