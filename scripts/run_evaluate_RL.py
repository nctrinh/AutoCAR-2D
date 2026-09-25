"""
Đánh giá RL agent như một LOCAL PATH-TRACKING CONTROLLER, và (mặc định)
so sánh trực tiếp với PID / Pure Pursuit trên CÙNG một path do A* sinh ra.

Vì cả 3 bộ điều khiển giờ đều nhận cùng một Path và cùng một Vehicle model,
đây là so sánh công bằng (apple-to-apple):

    python scripts/run_evaluate_RL.py --model trained_models/ppo/ppo_finished_map_1.zip \
        --map maps/yaml/map_1.yaml --episodes 10 --compare

Nếu chỉ muốn xem RL chạy một mình (không so sánh):

    python scripts/run_evaluate_RL.py --model ... --map ... --episodes 10
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.core.vehicle import Vehicle
from src.core.map import Map2D
from src.planning.a_star import AStarPlanner
from src.control.pid_controller import PathFollowingPID
from src.control.pure_pursuit import PurePursuitController, AdaptivePurePursuitController
from src.learning.environment import PathTrackingEnv
from src.utils.config_loader import ConfigLoader


def load_map(map_yaml: str) -> Map2D:
    return Map2D.load_from_yaml(map_yaml)


def plan_path(map_env: Map2D, grid_resolution: float = 1.0):
    planner = AStarPlanner(map_env, grid_resolution=grid_resolution)
    path = planner.plan(map_env.start, map_env.goal, info=False)
    if path is None:
        raise RuntimeError("A* could not find a path for this map.")
    return path


def run_rl_episode(model, env: PathTrackingEnv, deterministic: bool = True, render: bool = False) -> dict:
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    cte_history = []
    start_time = time.time()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        cte_history.append(abs(info["cross_track_error"]))
        if render:
            env.render()
        done = terminated or truncated

    return {
        "controller": "RL",
        "success": bool(info.get("is_goal_reached", False)),
        "collided": bool(info.get("is_collision", False)),
        "steps": steps,
        "reward": episode_reward,
        "mean_abs_cte": float(np.mean(cte_history)) if cte_history else 0.0,
        "replan_count": info.get("replan_count", 0),
        "recovery_count": info.get("recovery_count", 0),
        "wall_time_s": time.time() - start_time,
    }


def run_classical_episode(controller, env: PathTrackingEnv, render: bool = False) -> dict:
    """
    Chạy PID hoặc Pure Pursuit trên CÙNG PathTrackingEnv, để termination
    condition / cross-track logging / collision check giống hệt RL,
    đảm bảo so sánh công bằng.

    Lưu ý fog-of-war: PID/PurePursuit (khác RL) không tự đọc self.path mỗi
    step -- chúng cache path nội bộ qua set_path(). Nếu env.fog_of_war=True
    và path bị replan giữa episode, ta phải gọi lại set_path() để controller
    dùng đúng path mới nhất; RL không cần việc này vì nó luôn tính lại
    cross-track-error/heading-error trực tiếp từ self.path mỗi step.
    """
    obs, info = env.reset()
    controller.vehicle = env.vehicle  # controller điều khiển đúng object Vehicle của env
    controller.set_path(env.path)
    last_replan_count = info.get("replan_count", 0)

    done = False
    episode_reward = 0.0
    steps = 0
    cte_history = []
    start_time = time.time()

    while not done:
        accel, steer = controller.control()
        action = np.array(
            [
                np.clip(accel / env.vehicle.config.max_acceleration, -1.0, 1.0),
                np.clip(steer / env.vehicle.config.max_steering_angle, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        cte_history.append(abs(info["cross_track_error"]))

        if env.fog_of_war and info.get("replan_count", 0) != last_replan_count:
            controller.set_path(env.path)
            last_replan_count = info["replan_count"]

        if render:
            env.render()
        done = terminated or truncated

    return {
        "controller": controller.__class__.__name__,
        "success": bool(info.get("is_goal_reached", False)),
        "collided": bool(info.get("is_collision", False)),
        "steps": steps,
        "reward": episode_reward,
        "mean_abs_cte": float(np.mean(cte_history)) if cte_history else 0.0,
        "replan_count": info.get("replan_count", 0),
        "recovery_count": info.get("recovery_count", 0),
        "wall_time_s": time.time() - start_time,
    }


def summarize(name: str, episodes: list) -> dict:
    successes = [e["success"] for e in episodes]
    rewards = [e["reward"] for e in episodes]
    ctes = [e["mean_abs_cte"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    replans = [e.get("replan_count", 0) for e in episodes]
    recoveries = [e.get("recovery_count", 0) for e in episodes]
    return {
        "name": name,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_abs_cte": float(np.mean(ctes)),
        "mean_steps": float(np.mean(steps)),
        "mean_replans": float(np.mean(replans)),
        "mean_recoveries": float(np.mean(recoveries)),
    }


def print_comparison_table(summaries: list, fog_of_war: bool, enable_recovery: bool = False):
    print("\n" + "=" * 92)
    header = f"{'Controller':<22}{'Success%':>10}{'MeanReward':>14}{'MeanCTE(m)':>14}{'MeanSteps':>14}"
    if fog_of_war:
        header += f"{'MeanReplans':>14}"
    if enable_recovery:
        header += f"{'MeanRecoveries':>16}"
    print(header)
    print("-" * 92)
    for s in summaries:
        row = (
            f"{s['name']:<22}{s['success_rate']*100:>9.1f}%"
            f"{s['mean_reward']:>14.2f}{s['mean_abs_cte']:>14.3f}{s['mean_steps']:>14.1f}"
        )
        if fog_of_war:
            row += f"{s['mean_replans']:>14.1f}"
        if enable_recovery:
            row += f"{s['mean_recoveries']:>16.1f}"
        print(row)
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL local path-tracking controller")
    parser.add_argument("--model", type=str, required=True, help="Path to trained SB3 model (.zip)")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--map", type=str, default="maps/yaml/map_1.yaml", help="Scenario map YAML")
    parser.add_argument("--config", type=str, default="config/RL_config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--compare", action="store_true", help="Also run PID and Pure Pursuit on the same path")
    parser.add_argument("--visualize", action="store_true", help="Render the first RL episode with pygame")
    parser.add_argument(
        "--fog-of-war",
        action="store_true",
        help="Override config: agent only knows obstacles seen via LIDAR; "
        "path is re-planned online as it explores (see path_tracking_env.py)",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Override config: enable the reverse/three-point-turn safety net "
        "for dead ends too narrow to U-turn in with steering alone (see "
        "PathTrackingEnv.enable_recovery). Off by default to match training.",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model not found at {args.model}")
        return 1

    config = ConfigLoader(args.config)
    env_cfg = config.get("environment", {})
    fog_of_war = args.fog_of_war or env_cfg.get("fog_of_war", False)
    enable_recovery = args.recovery or env_cfg.get("enable_recovery", False)

    print("=" * 70)
    print(f"EVALUATING: {args.model}  on map: {args.map}")
    print(f"Fog-of-war online replanning: {'ENABLED' if fog_of_war else 'disabled'}")
    print(f"Reverse recovery safety net: {'ENABLED' if enable_recovery else 'disabled'}")
    print("=" * 70)

    map_env = load_map(args.map)
    vehicle_cfg = config.get_vehicle_config()

    if fog_of_war:
        # Không pre-plan: mỗi make_fresh_env()/reset() sẽ tự tính path ban
        # đầu trên internal_map RỖNG, rồi tự replan khi khám phá thêm.
        path = None
        print("Path is computed online per-episode from an initially empty internal map.")
    else:
        path = plan_path(map_env, grid_resolution=env_cfg.get("grid_resolution", 1.0))
        print(f"A* path (full map, fixed): {len(path.points)} waypoints, length={path.length:.1f}m")

    if args.algorithm == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo
    model = Algo.load(args.model)

    def make_fresh_env(render_mode=None):
        return PathTrackingEnv(
            map_env=map_env,
            path=path,
            vehicle=Vehicle(vehicle_cfg),
            max_steps=env_cfg.get("max_steps", 1000),
            num_lidar_rays=env_cfg.get("num_lidar_rays", 16),
            lidar_range=env_cfg.get("lidar_range", 20.0),
            goal_threshold=env_cfg.get("goal_threshold", 3.0),
            cte_fail_threshold=env_cfg.get("cte_fail_threshold", 8.0),
            render_mode=render_mode,
            fog_of_war=fog_of_war,
            replan_lookahead_wp=env_cfg.get("replan_lookahead_wp", 5),
            replan_grid_resolution=env_cfg.get("replan_grid_resolution", 1.0),
            replan_obstacle_radius=env_cfg.get("replan_obstacle_radius", 0.5),
            enable_recovery=enable_recovery,
            recovery_trigger_distance=env_cfg.get("recovery_trigger_distance", 1.0),
            recovery_reverse_steps=env_cfg.get("recovery_reverse_steps", 15),
        )

    # --- RL ---
    rl_env = make_fresh_env(render_mode="human" if args.visualize else None)
    rl_episodes = []
    for ep in range(args.episodes):
        render_this = args.visualize and ep == 0
        stats = run_rl_episode(model, rl_env, render=render_this)
        rl_episodes.append(stats)
        print(
            f"  [RL] Ep {ep + 1}: reward={stats['reward']:.2f} success={stats['success']} "
            f"steps={stats['steps']} replans={stats['replan_count']} recoveries={stats['recovery_count']}"
        )
    rl_env.close()

    summaries = [summarize("RL (local tracker)", rl_episodes)]

    # --- Classical baselines, cùng path, cùng termination logic ---
    if args.compare:
        pid = PathFollowingPID(vehicle=Vehicle(vehicle_cfg))
        pp = PurePursuitController(vehicle=Vehicle(vehicle_cfg))
        app = AdaptivePurePursuitController(vehicle=Vehicle(vehicle_cfg))

        for name, controller in [("PID", pid), ("PurePursuit", pp), ("AdaptivePurePursuit", app)]:
            env = make_fresh_env()
            episodes = []
            for ep in range(args.episodes):
                stats = run_classical_episode(controller, env)
                episodes.append(stats)
                print(
                    f"  [{name}] Ep {ep + 1}: reward={stats['reward']:.2f} success={stats['success']} "
                    f"steps={stats['steps']} replans={stats['replan_count']} recoveries={stats['recovery_count']}"
                )
            env.close()
            summaries.append(summarize(name, episodes))

    print_comparison_table(summaries, fog_of_war=fog_of_war, enable_recovery=enable_recovery)
    return 0


if __name__ == "__main__":
    exit(main())