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

from src.learning.environment import PathTrackingEnv
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
from src.planning.a_star import AStarPlanner
from src.utils.config_loader import ConfigLoader


def load_map_from_yaml(yaml_path: Path) -> Map2D:
    """Load map + obstacles từ file YAML kịch bản (dùng cho curriculum)."""
    print(f"Loading map scenario from: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    map_cfg = data.get("map", data)

    map_env = Map2D(
        width=map_cfg.get("width", 100),
        height=map_cfg.get("height", 100),
        safety_margin=map_cfg.get("safety_margin", 1.0),
    )

    start = map_cfg.get("start", [10, 10])
    goal = map_cfg.get("goal", [90, 90])
    map_env.set_start(start[0], start[1])
    map_env.set_goal(goal[0], goal[1])

    for obs in map_cfg.get("obstacles", []):
        obs_type = obs.get("type")
        if obs_type == "circle":
            map_env.add_obstacle(CircleObstacle(obs["x"], obs["y"], obs["radius"]))
        elif obs_type == "rectangle":
            map_env.add_obstacle(
                RectangleObstacle(
                    obs["x"], obs["y"], obs["width"], obs["height"], obs.get("angle", 0)
                )
            )
        elif obs_type == "polygon":
            map_env.add_obstacle(PolygonObstacle(vertices=np.array(obs["vertices"])))
        else:
            print(f"Warning: Unknown obstacle type {obs_type}, skipping...")

    return map_env


def plan_path_for_map(map_env: Map2D, grid_resolution: float = 1.0):
    """
    Chạy A* MỘT LẦN cho map này và tái sử dụng path đó xuyên suốt training.

    Đây là điểm khác biệt cốt lõi so với bản cũ: global planning (A*) và
    local tracking (RL) được TÁCH RIÊNG. RL không cần học lại việc tránh
    vật cản tĩnh trên toàn map — A* đã lo việc đó — RL chỉ cần học bám
    path này thật tốt.
    """
    planner = AStarPlanner(map_env, grid_resolution=grid_resolution)
    path = planner.plan(map_env.start, map_env.goal, info=True)
    if path is None:
        raise RuntimeError(
            f"A* không tìm được đường đi cho map (start={map_env.start}, "
            f"goal={map_env.goal}). Kiểm tra lại obstacles/safety_margin."
        )
    return path


def make_env(map_path, env_cfg, rank=0, seed=0):
    """Utility function for (vectorized) env creation."""

    def _init():
        current_map = load_map_from_yaml(map_path)
        fog_of_war = env_cfg.get("fog_of_war", False)

        if fog_of_war:
            # KHÔNG pre-plan path ở đây: PathTrackingEnv sẽ tự tính path ban
            # đầu trên internal_map RỖNG mỗi lần reset() (agent "mù" lúc bắt
            # đầu), rồi tự replan bằng A* khi LIDAR phát hiện obstacle mới
            # chặn path. Xem docstring fog-of-war trong path_tracking_env.py.
            path = None
        else:
            # Map hoàn toàn quan sát được: A* chạy 1 lần, path cố định suốt
            # episode -> RL chỉ học path-tracking thuần, hội tụ nhanh hơn.
            path = plan_path_for_map(
                current_map, grid_resolution=env_cfg.get("grid_resolution", 1.0)
            )

        env = PathTrackingEnv(
            map_env=current_map,
            path=path,
            max_steps=env_cfg.get("max_steps", 1000),
            num_lidar_rays=env_cfg.get("num_lidar_rays", 16),
            lidar_range=env_cfg.get("lidar_range", 20.0),
            goal_threshold=env_cfg.get("goal_threshold", 3.0),
            cte_fail_threshold=env_cfg.get("cte_fail_threshold", 8.0),
            render_mode=None,
            fog_of_war=fog_of_war,
            replan_lookahead_wp=env_cfg.get("replan_lookahead_wp", 5),
            replan_grid_resolution=env_cfg.get("replan_grid_resolution", 1.0),
            replan_obstacle_radius=env_cfg.get("replan_obstacle_radius", 0.5),
        )
        log_file = os.path.join(env_cfg.get("log_dir", "logs"), str(rank))
        env = Monitor(env, log_file)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_model(vec_env, train_cfg, model_cfg, algo, device):
    policy_kwargs = dict(net_arch=[256, 256])

    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(model_cfg.get("learning_rate", 3e-4)),
            n_steps=model_cfg.get("n_steps", 2048),
            batch_size=model_cfg.get("batch_size", 64),
            gamma=model_cfg.get("gamma", 0.99),
            ent_coef=model_cfg.get("ent_coef", 0.005),
            verbose=1,
            tensorboard_log=str(Path(train_cfg.get("log_dir", "logs"))),
            device=device,
            policy_kwargs=policy_kwargs,
        )
    elif algo == "sac":
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
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent as a LOCAL PATH-TRACKING controller "
        "(global path is generated by A*, RL only learns to follow it)"
    )
    parser.add_argument("--config", type=str, default="config/RL_config.yaml", help="Path to config file")
    args = parser.parse_args()

    print("=" * 70)
    print(f"LOADING CONFIGURATION: {args.config}")
    print("=" * 70)

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

        device = train_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")

        n_envs = train_cfg.get("n_envs", "auto")
        if n_envs == "auto":
            n_envs = max(1, (os.cpu_count() or 2) - 1)
        n_envs = int(n_envs)
        print(f"Parallel training environments: {n_envs} "
              f"({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")

        fog_of_war = env_cfg.get("fog_of_war", False)
        print(
            f"Fog-of-war online replanning: "
            f"{'ENABLED (path re-planned on the fly as LIDAR discovers obstacles)' if fog_of_war else 'disabled (path fixed from full-map A* at reset)'}"
        )

        map_folder_path = map_cfg.get("train_map_yaml_folder")
        if not map_folder_path:
            raise ValueError("'train_map_yaml_folder' not defined in config.")

        map_files = sorted(glob.glob(os.path.join(map_folder_path, "*.yaml")))
        if not map_files:
            raise FileNotFoundError(f"No .yaml files found in {map_folder_path}")

        print(f"Found {len(map_files)} curriculum scenarios: {[Path(p).name for p in map_files]}")

        model = None
        timesteps_per_map = train_cfg.get("timesteps", 200000)

        for i, map_file in enumerate(map_files):
            scenario_name = Path(map_file).stem
            print("\n" + "-" * 50)
            print(f"PHASE {i + 1}/{len(map_files)}: Scenario '{scenario_name}'")
            print("-" * 50)

            if n_envs > 1:
                train_env = SubprocVecEnv(
                    [make_env(map_file, env_cfg, rank=i) for i in range(n_envs)]
                )
            else:
                train_env = DummyVecEnv([make_env(map_file, env_cfg, rank=0)])
            # Eval env stays single-process: only a handful of deterministic
            # episodes per eval_freq, not worth the subprocess overhead.
            eval_env = DummyVecEnv([make_env(map_file, env_cfg, rank=100)])

            if model is None:
                print(f"Initializing new {algo.upper()} model...")
                model = create_model(train_env, train_cfg, model_cfg, algo, device)
            else:
                print("Transferring existing agent to new environment...")
                model.set_env(train_env)

            checkpoint_callback = CheckpointCallback(
                save_freq=train_cfg.get("save_freq", 10000),
                save_path=str(base_save_dir / "checkpoints"),
                name_prefix=f"{algo}_{scenario_name}",
            )

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(base_save_dir / "best_model" / scenario_name),
                log_path=str(log_dir / "eval" / scenario_name),
                eval_freq=train_cfg.get("eval_freq", 5000),
                deterministic=True,
                render=False,
                n_eval_episodes=5,
            )

            try:
                model.learn(
                    total_timesteps=timesteps_per_map,
                    callback=[checkpoint_callback, eval_callback],
                    reset_num_timesteps=False,
                    progress_bar=True,
                    tb_log_name=f"{algo}_{scenario_name}",
                )
            except KeyboardInterrupt:
                print("Training interrupted by user. Saving current model...")
                model.save(base_save_dir / f"{algo}_interrupted.zip")
                return 0

            phase_save_path = base_save_dir / f"{algo}_finished_{scenario_name}.zip"
            model.save(phase_save_path)
            print(f"Completed Phase {i + 1}. Model saved to {phase_save_path}")

            train_env.close()
            eval_env.close()

        print("\n" + "=" * 70)
        print("ALL TRAINING PHASES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())