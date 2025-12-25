import pygame
import numpy as np
from typing import Optional, Tuple
from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle
from src.core.map import (
    Map2D,
    CircleObstacle,
    RectangleObstacle,
    PolygonObstacle,
)
from src.planning.base_planner import BasePlanner, Path as PlannedPath
from src.control.pid_controller import PathFollowingPID, PIDController
from src.control.pure_pursuit import PurePursuitController, AdaptivePurePursuitController
from src.simulation.renderer import Renderer
from src.utils.config_loader import ConfigLoader


class SimulationState(Enum):
    """Simulation states."""
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Simulator:
    """
    Main simulation environment.
    
    Integrates:
    - Map and obstacles
    - Vehicle dynamics
    - Path planning
    - Control
    - Visualization
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize simulator.
        
        Args:
            config: Configuration loader (uses default if None)
        """
        self.config = config or ConfigLoader()
        
        # Get config parameters
        self.sim_params = self.config.get_simulation_params()
        self.map_params = self.config.get_map_params()
        self.vehicle_params = self.config.get_vehicle_config()
        self.planner_params = self.config.get_planner_params()

        
        # Initialize components
        self.map_env: Optional[Map2D] = self._load_map()
        self.vehicle: Optional[Vehicle] = self._load_vehicle()
        self.planner: Optional[BasePlanner] = self._load_planner()
        self.controller = self._load_controller()
        self.path: Optional[PlannedPath] = None
        
        # Simulation state
        self.state = SimulationState.IDLE
        self.step = 0
        self.simulation_time = 0.0
        self.dt = self.sim_params.get('dt', 0.1)
        
        # Renderer
        self.renderer = Renderer(
            screen_width=self.sim_params.get('screen_width', 1200),
            screen_height=self.sim_params.get('screen_height', 800),
            world_width=self.map_params.get('width', 100),
            world_height=self.map_params.get('height', 100)
        )
        
        # Stats
        self.collision_detected = False
        self.goal_reached = False
        self.total_distance = 0.0
        self.last_position: Optional[Tuple[float, float]] = None
    
    def _load_map(self) -> Map2D:
        """Load map environment from config."""
        world_width = self.map_params.get("width", 100)
        world_height = self.map_params.get("height", 100)
        safety_margin = self.map_params.get("safety_margin", 1.0)

        map_env = Map2D(world_width, world_height, safety_margin)

        start = self.map_params.get("start", None)
        goal = self.map_params.get("goal", None)

        if start is not None:
            if len(start) != 2:
                raise ValueError("map.start must be [x, y]")
            map_env.set_start(start[0], start[1])

        if goal is not None:
            if len(goal) != 2:
                raise ValueError("map.goal must be [x, y]")
            map_env.set_goal(goal[0], goal[1])

        obstacles_cfg = self.map_params.get("obstacles", [])

        for obs_cfg in obstacles_cfg:
            obs_type = obs_cfg.get("type")

            if obs_type == "circle":
                obstacle = CircleObstacle(
                    x=obs_cfg["x"],
                    y=obs_cfg["y"],
                    radius=obs_cfg["radius"],
                )

            elif obs_type == "rectangle":
                obstacle = RectangleObstacle(
                    x=obs_cfg["x"],
                    y=obs_cfg["y"],
                    width=obs_cfg["width"],
                    height=obs_cfg["height"],
                    angle=obs_cfg.get("angle", 0.0),
                )

            elif obs_type == "polygon":
                obstacle = PolygonObstacle(
                    vertices=np.array(obs_cfg["vertices"])
                )

            else:
                raise ValueError(f"Unknown obstacle type: {obs_type}")

            map_env.add_obstacle(obstacle)

        print(
            f"Map loaded: size=({map_env.width}, {map_env.height}), "
            f"obstacles={len(map_env.obstacles)}, "
            f"start={map_env.start}, goal={map_env.goal}, "
            f"safety margin={map_env.safety_margin}"
        )

        return map_env
    
    def _load_vehicle(self) -> Vehicle:
        """Set vehicle."""
        vehicle = Vehicle(self.vehicle_params)
        self.last_position = vehicle.get_position()
        print(f"Vehicle set: {vehicle}")
        return vehicle
    
    def _load_planner(self) -> BasePlanner:
        """Set path planner based on planner config."""
        planner_cfg = self.planner_params
        planner_name = planner_cfg.get("algorithm", "astar")

        if planner_name == "astar":
            from src.planning.a_star import AStarPlanner

            astar_cfg = planner_cfg.get("astar", {})

            planner = AStarPlanner(
                map_env=self.map_env,
                grid_resolution=astar_cfg.get("grid_resolution", 0.5),
                heuristic_weight=astar_cfg.get("heuristic_weight", 1.0),
                max_iterations=astar_cfg.get("max_iterations", 10000)
            )

        elif planner_name == "rrt":
            from src.planning.rrt import RRTPlanner

            rrt_cfg = planner_cfg.get("rrt", {})

            planner = RRTPlanner(
                map_env=self.map_env,
                max_iterations=rrt_cfg.get("max_iterations", 10000),
                step_size=rrt_cfg.get("step_size", 0.5),
                goal_sample_rate=rrt_cfg.get("goal_sample_rate", 0.1),
                goal_threshold=rrt_cfg.get("goal_threshold", 2.0)
            )

        elif planner_name == "rrt_star":
            from src.planning.rrt import RRTStarPlanner

            rrt_star_cfg = planner_cfg.get("rrt_star", {})

            planner = RRTStarPlanner(
                map_env=self.map_env,
                max_iterations=rrt_star_cfg.get("max_iterations", 10000),
                step_size=rrt_star_cfg.get("step_size", 0.5),
                goal_sample_rate=rrt_star_cfg.get("goal_sample_rate", 0.1),
                goal_threshold=rrt_star_cfg.get("goal_threshold", 2.0),
                rewire_radius=rrt_star_cfg.get("rewire_radius", 5.0),
            )

        else:
            raise ValueError(f"Unknown planner algorithm: {planner_name}")

        print(f"[Planner] Using {planner_name}: {planner}")
        return planner
    
    def _load_controller(self):
        """Set controller (PID / Pure Pursuit / Adaptive Pure Pursuit)."""

        ctrl_cfg = self.config.get("controller", {})
        ctrl_type = ctrl_cfg.get("type", "pid").lower()

        controller = None

        if ctrl_type == "pid":
            pid_cfg = ctrl_cfg.get("pid", {})

            pid = PIDController(
                kp_lateral=pid_cfg.get("kp_lateral", 1.0),
                ki_lateral=pid_cfg.get("ki_lateral", 0.0),
                kd_lateral=pid_cfg.get("kd_lateral", 0.0),
                kp_longitudinal=pid_cfg.get("kp_longitudinal", 0.5),
                ki_longitudinal=pid_cfg.get("ki_longitudinal", 0.0),
                kd_longitudinal=pid_cfg.get("kd_longitudinal", 0.1),
                max_steering=pid_cfg.get("max_steering", 0.6),
                max_acceleration=pid_cfg.get("max_acceleration", 3.0),
            )

            controller = PathFollowingPID(
                vehicle=self.vehicle,
                pid_controller=pid,
                lookahead_distance=pid_cfg.get("lookahead_distance", 5.0),
                target_speed=pid_cfg.get("target_speed", 5.0),
                goal_threshold=pid_cfg.get("goal_threshold", 1.0),
                dt=pid_cfg.get("dt", 0.1),
            )

        elif ctrl_type == "pure_pursuit":
            pp_cfg = ctrl_cfg.get("pure_pursuit", {})

            controller = PurePursuitController(
                vehicle=self.vehicle,
                lookahead_distance=pp_cfg.get("lookahead_distance", 5.0),
                lookahead_gain=pp_cfg.get("lookahead_gain", 0.5),
                min_lookahead=pp_cfg.get("min_lookahead", 2.0),
                max_lookahead=pp_cfg.get("max_lookahead", 10.0),
                target_speed=pp_cfg.get("target_speed", 5.0),
                speed_kp=pp_cfg.get("speed_kp", 0.5),
                goal_threshold=pp_cfg.get("goal_threshold", 1.0),
                dt=pp_cfg.get("dt", 0.1),
            )

        elif ctrl_type == "adaptive_pure_pursuit":

            app_cfg = ctrl_cfg.get("adaptive_pure_pursuit", {})

            controller = AdaptivePurePursuitController(
                vehicle=self.vehicle,
                lookahead_distance=app_cfg.get("lookahead_distance", 5.0),
                lookahead_gain=app_cfg.get("lookahead_gain", 0.5),
                min_lookahead=app_cfg.get("min_lookahead", 2.0),
                max_lookahead=app_cfg.get("max_lookahead", 10.0),
                target_speed=app_cfg.get("target_speed", 5.0),
                speed_kp=app_cfg.get("speed_kp", 0.5),
                goal_threshold=app_cfg.get("goal_threshold", 1.0),
                dt=app_cfg.get("dt", 0.1),
                max_curvature_speed=app_cfg.get("max_curvature_speed", 3.0),
                curvature_lookahead=app_cfg.get("curvature_lookahead", 10.0),
            )

        else:
            raise ValueError(f"Unknown controller type: {ctrl_type}")

        print(f"Controller set: {controller.__class__.__name__}")
        return controller
    
    def plan_path(self, start: Optional[Tuple[float, float]] = None,
                  goal: Optional[Tuple[float, float]] = None) -> bool:
        """
        Plan path from start to goal.
        
        Args:
            start: Start position (uses map start if None)
            goal: Goal position (uses map goal if None)
            
        Returns:
            True if planning succeeded
        """
        if self.planner is None:
            print("Error: No planner set!")
            return False
        
        if self.map_env is None:
            print("Error: No map loaded!")
            return False
        
        # Use map start/goal if not provided
        start = start or self.map_env.start
        goal = goal or self.map_env.goal
        
        if start is None or goal is None:
            print("Error: Start or goal not specified!")
            return False
        
        print(f"\nPlanning path from {start} to {goal}...")
        self.state = SimulationState.PLANNING
        
        # Plan
        self.path = self.planner.plan(start, goal)
        
        if self.path is None:
            print("Planning failed!")
            self.state = SimulationState.FAILED
            return False
        
        print(f"Path found! Length: {self.path.length:.2f}m")
        
        # Set path to controller
        if self.controller is not None:
            self.controller.set_path(self.path)
        
        return True
    
    def reset(self):
        """Reset simulation to initial state."""
        if self.map_env and self.map_env.start:
            start = self.map_env.start
            if self.vehicle:
                self.vehicle.reset(x=start[0], y=start[1], theta=0)
        
        self.step = 0
        self.simulation_time = 0.0
        self.collision_detected = False
        self.goal_reached = False
        self.total_distance = 0.0
        self.renderer.trajectory.clear()
        
        if self.controller:
            if hasattr(self.controller, 'reset'):
                self.controller.reset()
            if hasattr(self.controller, 'current_target_idx'):
                self.controller.current_target_idx = 0
        
        self.state = SimulationState.IDLE
        print("Simulation reset")
    
    def step_simulation(self) -> bool:
        """
        Execute one simulation step.
        
        Returns:
            True if simulation should continue
        """
        if self.vehicle is None or self.controller is None:
            return False
        
        if self.state != SimulationState.RUNNING:
            return False
        
        # Get control commands
        acceleration, steering = self.controller.control()
        
        # Update vehicle
        self.vehicle.update(acceleration, steering)
        
        # Update stats
        current_pos = self.vehicle.get_position()
        if self.last_position:
            dx = current_pos[0] - self.last_position[0]
            dy = current_pos[1] - self.last_position[1]
            self.total_distance += (dx**2 + dy**2)**0.5
        self.last_position = current_pos
        
        # Add to trajectory
        self.renderer.add_trajectory_point(*current_pos)
        
        # Check collision
        if self.map_env.is_collision(current_pos[0], current_pos[1], 0):
            self.collision_detected = True
            self.state = SimulationState.FAILED
            print(f"\nCollision detected at step {self.step}!")
            return False
        
        # Check goal
        if self.map_env.goal:
            dist_to_goal = self.vehicle.distance_to(
                self.map_env.goal[0], self.map_env.goal[1]
            )
            
            if dist_to_goal < 2.0:
                self.goal_reached = True
                self.state = SimulationState.COMPLETED
                print(f"\nGoal reached at step {self.step}!")
                print(f"Total distance traveled: {self.total_distance:.2f}m")
                return False
        
        self.step += 1
        self.simulation_time += self.dt
        
        return True
    
    def run(self, max_steps: int = 10000, target_fps: int = 60):
        """
        Run simulation with visualization.
        
        Args:
            max_steps: Maximum simulation steps
            target_fps: Target frame rate
        """
        if self.vehicle is None:
            print("Error: No vehicle set!")
            return
        
        if self.path is None:
            print("Warning: No path planned. Planning now...")
            if not self.plan_path():
                return
        
        print("\n" + "="*70)
        print("STARTING SIMULATION")
        print("="*70)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset")
        print("  ESC - Quit")
        print("="*70 + "\n")
        
        clock = pygame.time.Clock()
        self.state = SimulationState.RUNNING
        running = True
        
        while running and self.step < max_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.state == SimulationState.RUNNING:
                            self.state = SimulationState.PAUSED
                            print("Paused")
                        elif self.state == SimulationState.PAUSED:
                            self.state = SimulationState.RUNNING
                            print("Resumed")
                    elif event.key == pygame.K_r:
                        self.reset()
                        if self.plan_path():
                            self.state = SimulationState.RUNNING
                    elif event.key == pygame.K_g:
                        self.renderer.show_grid = not self.renderer.show_grid
                    elif event.key == pygame.K_p:
                        self.renderer.show_path = not self.renderer.show_path
                    elif event.key == pygame.K_t:
                        self.renderer.show_trajectory = not self.renderer.show_trajectory
                    elif event.key == pygame.K_i:
                        self.renderer.show_info = not self.renderer.show_info
            
            # Update simulation
            if self.state == SimulationState.RUNNING:
                if not self.step_simulation():
                    # Simulation ended (goal reached or collision)
                    pass
            
            # Render
            self.render()
            
            # Control frame rate
            clock.tick(target_fps)
        
        # Print final stats
        self.print_stats()
        
        # Keep window open until closed
        print("\nSimulation ended. Close window or press ESC to quit.")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
            clock.tick(10)
        
        self.renderer.close()
    
    def render(self):
        """Render current simulation state."""
        self.renderer.clear()
        self.renderer.draw_grid()
        
        if self.map_env:
            self.renderer.draw_map(self.map_env)
            if self.map_env.start and self.map_env.goal:
                self.renderer.draw_start_goal(self.map_env.start, self.map_env.goal)
        
        if self.path:
            self.renderer.draw_path(self.path)
        
        self.renderer.draw_trajectory()
        
        if self.vehicle:
            self.renderer.draw_vehicle(self.vehicle)
            
            # Draw lookahead point if using Pure Pursuit
            if isinstance(self.controller, PurePursuitController):
                lookahead = self.controller.get_lookahead_point(self.vehicle)
                if lookahead:
                    self.renderer.draw_point(
                        lookahead[0], lookahead[1],
                        color=(255, 165, 0), radius=6
                    )
        
        # Additional info
        additional_info = {
            'State': self.state.value,
            'Distance': f'{self.total_distance:.1f}m'
        }
        
        if self.vehicle:
            self.renderer.draw_info_panel(
                self.vehicle, self.step,
                pygame.time.Clock().get_fps(),
                additional_info
            )
        
        self.renderer.draw_legend()
        self.renderer.draw_controls_help()
        self.renderer.update()
    
    def print_stats(self):
        """Print simulation statistics."""
        print("\n" + "="*70)
        print("SIMULATION STATISTICS")
        print("="*70)
        print(f"State: {self.state.value}")
        print(f"Steps: {self.step}")
        print(f"Simulation time: {self.simulation_time:.2f}s")
        print(f"Total distance: {self.total_distance:.2f}m")
        
        if self.path:
            print(f"Path length: {self.path.length:.2f}m")
            efficiency = (self.path.length / self.total_distance * 100) if self.total_distance > 0 else 0
            print(f"Path efficiency: {efficiency:.1f}%")
        
        print(f"Goal reached: {self.goal_reached}")
        print(f"Collision: {self.collision_detected}")
        
        if self.planner:
            print(f"\nPlanner: {self.planner.__class__.__name__}")
            print(f"Planning time: {self.planner.planning_time:.3f}s")
        
        if self.controller:
            print(f"Controller: {self.controller.__class__.__name__}")
        
        print("="*70)


# Example usage
if __name__ == "__main__":
    
    sim = Simulator()
    
    sim.plan_path()
    
    sim.run(max_steps=2000, target_fps=60)