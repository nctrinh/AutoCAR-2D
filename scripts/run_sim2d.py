import sys
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pygame.pkgdata"
)


def setup_pythonpath():
    """
    Ensure project root is in PYTHONPATH
    """
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser("Simulator Runner")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max simulation steps"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override target FPS"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    project_root = setup_pythonpath()

    print(f"Project root: {project_root}")

    from src.utils.config_loader import ConfigLoader
    from src.simulation.simulator import Simulator

    config = ConfigLoader(args.config)
    print(f"Using config: {config}")

    sim = Simulator(config=config)

    # Plan path
    if not sim.plan_path():
        print("Path planning failed")
        sys.exit(1)

    max_steps = args.max_steps or sim.sim_params.get("max_steps", 1000)
    fps = args.fps or sim.sim_params.get("fps", 60)

    print(f"Running simulation: max_steps={max_steps}, fps={fps}")

    sim.run(
        max_steps=max_steps,
        target_fps=fps
    )


if __name__ == "__main__":
    main()