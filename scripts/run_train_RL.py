import sys
import argparse
import yaml
import numpy as np
import glob
import os
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(str(Path(__file__).parent.parent))

from src.learning.environment import AutonomousCarEnv
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
from src.utils.config_loader import ConfigLoader

def load_map_from_yaml_file(yaml_path: Path) -> Map2D:
    print(f"Loading map from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    map_cfg = data.get("map", data) 
    
    map_env = Map2D(
        width=map_cfg.get("width", 100), 
        height=map_cfg.get("height", 100),
        safety_margin=map_cfg.get("safety_margin", 1.0)
    )
    
    start = map_cfg.get("start", [10, 10])
    goal = map_cfg.get("goal", [90, 90])
    map_env.set_start(start[0], start[1])
    map_env.set_goal(goal[0], goal[1])
    
    obstacles = map_cfg.get("obstacles", [])
    for obs in obstacles:
        if obs['type'] == 'circle':
            map_env.add_obstacle(CircleObstacle(obs['x'], obs['y'], obs['radius']))
        elif obs['type'] == 'rectangle':
            map_env.add_obstacle(RectangleObstacle(
                obs['x'], obs['y'], obs['width'], obs['height'], obs.get('angle', 0)
            ))
        elif obs['type'] == "polygon":
            map_env.add_obstacle(PolygonObstacle(vertices=np.array(obs["vertices"])))
        else:
            print(f"Warning: Unknown obstacle type {obs['type']}, skipping...")
            
    return map_env

def create_model(vec_env, train_cfg, model_cfg, algo):
    if algo == 'ppo':
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=model_cfg.get("learning_rate", 3e-4),
            n_steps=model_cfg.get("n_steps", 2048),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            ent_coef=model_cfg.get("ent_coef", 0.01),
            verbose=1,
            tensorboard_log=str(Path(train_cfg.get("log_dir", "logs"))),
            device=train_cfg.get("device", "cpu")
        )
    elif algo == 'sac':
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=model_cfg.get("learning_rate", 3e-4),
            buffer_size=model_cfg.get("buffer_size", 100000),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            tau=model_cfg.get("tau", 0.005),
            verbose=1,
            tensorboard_log=str(Path(train_cfg.get("log_dir", "logs"))),
            device=train_cfg.get("device", "cpu")
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description='Train RL agent with YAML config')
    parser.add_argument('--config', type=str, default='config/RL_config.yaml', help='Path to config file')
    args = parser.parse_args()

    print("="*70)
    print(f"LOADING MAIN CONFIGURATION: {args.config}")
    print("="*70)

    try:
        config = ConfigLoader(args.config)
        train_cfg = config.get_section("training", {})
        env_cfg = config.get_section("environment", {})
        map_cfg = config.get_section("map", {})
        
        algo = train_cfg.get("algorithm", "ppo").lower()
        save_dir = Path(train_cfg.get("save_dir", "trained_models")) / algo
        save_dir.mkdir(parents=True, exist_ok=True)
        
        map_folder_path = map_cfg.get("train_map_json")
        
        if not map_folder_path:
            print("Error: 'train_map_json' not defined in map section of config.")
            return 1

        map_files = sorted(glob.glob(os.path.join(map_folder_path, "*.yaml")))
        
        if not map_files:
            print(f"Error: No .yaml files found in {map_folder_path}")
            return 1
            
        print(f"Found {len(map_files)} maps for curriculum training: {[Path(p).name for p in map_files]}")

        model = None
        timesteps_per_map = train_cfg.get("timesteps", 10000)

        for i, map_file in enumerate(map_files):
            print(f"\n" + "-"*30)
            print(f"STARTING PHASE {i+1}/{len(map_files)}: Map {Path(map_file).name}")
            print("-"*30)
            
            current_map = load_map_from_yaml_file(Path(map_file))
            
            env = AutonomousCarEnv(
                map_env=current_map,
                max_steps=env_cfg.get("max_steps", 1000),
                num_lidar_rays=env_cfg.get("num_lidar_rays", 16),
                render_mode=None
            )
            vec_env = DummyVecEnv([lambda: env])
            
            if model is None:
                print("Initializing new model...")
                model = create_model(vec_env, train_cfg, config.get_section("model"), algo)
            else:
                print("Loading existing model into new map environment...")
                model.set_env(vec_env)

            checkpoint_callback = CheckpointCallback(
                save_freq=train_cfg.get("save_freq", 5000),
                save_path=str(save_dir),
                name_prefix=f"{algo}_map_{i+1}_{Path(map_file).stem}"
            )
            
            model.learn(
                total_timesteps=timesteps_per_map,
                callback=checkpoint_callback,
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            phase_save_path = save_dir / f"{algo}_finished_map_{i+1}.zip"
            model.save(phase_save_path)
            print(f"✓ Completed Map {i+1}. Model saved to {phase_save_path}")

        print("\n" + "="*70)
        print("ALL TRAINING PHASES COMPLETED")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())