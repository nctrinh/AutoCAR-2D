import yaml
from pathlib import Path
from src.core.vehicle import VehicleConfig

class ConfigLoader:
    def __init__(self, file_name: str = 'default.yaml'):
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / 'config'
        self.config_path = config_dir / file_name

        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = self._default_config()
    
    def get(self, key: str, default=None):
        parts = key.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
        else:
            value = default
        return value

    def get_vehicle_config(self):
        v = self.config.get('vehicle', {})
        return VehicleConfig(
            length=v.get("length", 4.0),
            width=v.get("width", 2.0),
            wheelbase=v.get("wheelbase", 2.5),
            max_velocity=v.get("max_velocity", 10.0),
            max_acceleration=v.get("max_acceleration", 3.0),
            max_deceleration=v.get("max_deceleration", -5.0),
            max_steering_angle=v.get("max_steering_angle", 0.6),
        )
    
    def _default_config(self):
        return {
            "simulation": {
                "fps": 40,
                "screen_width": 800,
                "screen_height": 600,
                "dt": 0.1,
            },
            "vehicle": {
                "length": 4.0,
                "width": 2.0,
                "wheelbase": 2.5,
                "max_velocity": 10.0,
                "max_acceleration": 3.0,
                "max_deceleration": -5.0,
                "max_steering_angle": 0.6,
            },
            "map": {
                "width": 100,
                "height": 100,
                "grid_size": 0.5,
                "obstacle_inflation": 0.5,
            },
            "planner": {
                "algorithm": "astar",
                "goal_threshold": 1.0,
                "max_iterations": 10000,
                "step_size": 0.5,
            },
            "controller": {
                "type": "pid",
                "kp": 1.5,
                "ki": 0.05,
                "kd": 0.6,
                "lookahead_distance": 5.0,
                "lookahead_gain": 0.5,
            },
            "training": {
                "algorithm": "ppo",
                "total_timesteps": 100000,
                "learning_rate": 0.003,
                "batch_size": 64,
                "gamma": 0.99,
                "n_steps": 2048,
                "ent_coef": 0.01,
            },
            "visualization": {
                "show_trajectories": True,
                "show_sensors": False,
                "show_path": True,
                "vehicle_color": [0, 100, 255],
                "obstacle_color": [100, 100, 100],
                "path_color": [255, 0, 0],
                "goal_color": [0, 255, 0],
            },
        }
    
    def __repr__(self):
        return f"<ConfigLoader path={self.config_path.name} keys={list(self.config.keys())}>"