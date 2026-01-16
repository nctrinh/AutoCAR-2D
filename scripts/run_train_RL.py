import sys
import argparse
import yaml
import numpy as np
import glob
import os
from pathlib import Path
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.append(str(Path(__file__).parent.parent))

# Giả định các module này đã tồn tại trong project của bạn
from src.learning.environment import AutonomousCarEnv
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
from src.utils.config_loader import ConfigLoader

def load_map_from_yaml(yaml_path: Path) -> Map2D:
    """
    Hàm helper để load map và cấu hình obstacles từ file YAML.
    Rất quan trọng cho việc tạo các kịch bản training (Scenario) khác nhau.
    """
    print(f"Loading map scenario from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Xử lý trường hợp file yaml có hoặc không có key root "map"
    map_cfg = data.get("map", data) 
    
    map_env = Map2D(
        width=map_cfg.get("width", 100), 
        height=map_cfg.get("height", 100),
        safety_margin=map_cfg.get("safety_margin", 1.0)
    )
    
    # Set Start/Goal cho kịch bản hiện tại
    start = map_cfg.get("start", [10, 10])
    goal = map_cfg.get("goal", [90, 90])
    map_env.set_start(start[0], start[1])
    map_env.set_goal(goal[0], goal[1])
    
    # Load Obstacles
    obstacles = map_cfg.get("obstacles", [])
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'circle':
            map_env.add_obstacle(CircleObstacle(obs['x'], obs['y'], obs['radius']))
        elif obs_type == 'rectangle':
            map_env.add_obstacle(RectangleObstacle(
                obs['x'], obs['y'], obs['width'], obs['height'], obs.get('angle', 0)
            ))
        elif obs_type == "polygon":
            map_env.add_obstacle(PolygonObstacle(vertices=np.array(obs["vertices"])))
        else:
            print(f"Warning: Unknown obstacle type {obs_type}, skipping...")
            
    return map_env

def make_env(map_path, env_cfg, rank=0, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        current_map = load_map_from_yaml(map_path)
        env = AutonomousCarEnv(
            map_env=current_map,
            max_steps=env_cfg.get("max_steps", 1000),
            num_lidar_rays=env_cfg.get("num_lidar_rays", 36), # Tăng lidar rays để nhận diện tốt hơn
            render_mode=None # Training không cần render
        )
        # Wrap environment với Monitor để ghi log cho EvalCallback
        log_file = os.path.join(env_cfg.get("log_dir", "logs"), str(rank))
        env = Monitor(env, log_file)
        env.reset(seed=seed + rank)
        return env
    return _init

def create_model(vec_env, train_cfg, model_cfg, algo, device):
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]) # Mạng sâu hơn chút cho bài toán control

    if algo == 'ppo':
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(model_cfg.get("learning_rate", 3e-4)),
            n_steps=model_cfg.get("n_steps", 2048),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            ent_coef=model_cfg.get("ent_coef", 0.005), # Giảm entropy khi map khó dần
            verbose=1,
            tensorboard_log=str(Path(train_cfg.get("log_dir", "logs"))),
            device=device,
            policy_kwargs=policy_kwargs
        )
    elif algo == 'sac':
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=float(model_cfg.get("learning_rate", 3e-4)),
            buffer_size=model_cfg.get("buffer_size", 100000),
            batch_size=model_cfg.get("batch_size", 256),
            gamma=model_cfg.get("gamma", 0.99),
            tau=model_cfg.get("tau", 0.005),
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=str(Path(train_cfg.get("log_dir", "logs"))),
            device=device,
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description='Train RL agent (Local Controller) with Curriculum')
    parser.add_argument('--config', type=str, default='config/RL_config.yaml', help='Path to config file')
    args = parser.parse_args()

    print("="*70)
    print(f"LOADING CONFIGURATION: {args.config}")
    print("="*70)

    try:
        config = ConfigLoader(args.config)
        train_cfg = config.get("training", {})
        env_cfg = config.get("environment", {})
        map_cfg = config.get("map", {})
        model_cfg = config.get("model", {})
        
        algo = train_cfg.get("algorithm", "ppo").lower()
        base_save_dir = Path(train_cfg.get("save_dir", "trained_models")) / algo
        log_dir = Path(train_cfg.get("log_dir", "logs"))
        
        base_save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Device detection
        device = train_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")
        
        # Load Maps for Curriculum
        map_folder_path = map_cfg.get("train_map_yaml_folder")
        if not map_folder_path:
            raise ValueError("'train_map_yaml_folder' not defined in config.")

        map_files = sorted(glob.glob(os.path.join(map_folder_path, "*.yaml")))
        if not map_files:
            raise FileNotFoundError(f"No .yaml files found in {map_folder_path}")
            
        print(f"Found {len(map_files)} curriculum scenarios: {[Path(p).name for p in map_files]}")

        model = None
        timesteps_per_map = train_cfg.get("timesteps", 200000)

        # --- TRAINING LOOP ---
        for i, map_file in enumerate(map_files):
            scenario_name = Path(map_file).stem
            print(f"\n" + "-"*50)
            print(f"PHASE {i+1}/{len(map_files)}: Scenario '{scenario_name}'")
            print("-"*50)
            
            # 1. Tạo môi trường Training
            # Sử dụng DummyVecEnv cho đơn giản, có thể dùng SubprocVecEnv nếu muốn train song song
            train_env = DummyVecEnv([make_env(map_file, env_cfg, rank=0)])
            
            # 2. Tạo môi trường Evaluation (Quan trọng: Dùng cùng map để test khả năng học)
            # Trong thực tế, có thể bạn muốn eval trên map khác để test độ tổng quát, 
            # nhưng với Curriculum, ta cần chắc chắn nó qua được bài này đã.
            eval_env = DummyVecEnv([make_env(map_file, env_cfg, rank=100)])

            # 3. Khởi tạo hoặc Load Model
            if model is None:
                print(f"Initializing new {algo.upper()} model...")
                model = create_model(train_env, train_cfg, model_cfg, algo, device)
            else:
                print("Transferring existing agent to new environment...")
                model.set_env(train_env)
                # Reset num_timesteps=False để Tensorboard vẽ đồ thị liên tục qua các phase
            
            # 4. Callbacks
            # Lưu checkpoint định kỳ
            checkpoint_callback = CheckpointCallback(
                save_freq=train_cfg.get("save_freq", 10000),
                save_path=str(base_save_dir / "checkpoints"),
                name_prefix=f"{algo}_{scenario_name}"
            )
            
            # Đánh giá model định kỳ và lưu model tốt nhất (BEST MODEL)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(base_save_dir / "best_model" / scenario_name),
                log_path=str(log_dir / "eval" / scenario_name),
                eval_freq=train_cfg.get("eval_freq", 5000),
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )

            # 5. Start Training
            try:
                model.learn(
                    total_timesteps=timesteps_per_map,
                    callback=[checkpoint_callback, eval_callback],
                    reset_num_timesteps=False, # Quan trọng để log liên tục
                    progress_bar=True,
                    tb_log_name=f"{algo}_{scenario_name}"
                )
            except KeyboardInterrupt:
                print("Training interrupted by user. Saving current model...")
                model.save(base_save_dir / f"{algo}_interrupted.zip")
                return 0

            # 6. Save Final Model của Phase này
            phase_save_path = base_save_dir / f"{algo}_finished_{scenario_name}.zip"
            model.save(phase_save_path)
            print(f"✓ Completed Phase {i+1}. Model saved to {phase_save_path}")
            
            # Đóng env để giải phóng resource
            train_env.close()
            eval_env.close()

        print("\n" + "="*70)
        print("ALL TRAINING PHASES COMPLETED SUCCESSFULLY")
        print("="*70)
        return 0

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())