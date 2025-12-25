import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.core.vehicle import Vehicle
from src.core.map import Map2D, CircleObstacle, RectangleObstacle
from src.planning.a_star import AStarPlanner
from src.planning.rrt import RRTPlanner, RRTStarPlanner
from src.control.pid_controller import PathFollowingPID
from src.control.pure_pursuit import PurePursuitController


def create_test_map():
    """Create a test map with obstacles."""
    map_env = Map2D(width=100, height=100)
    
    # Add obstacles
    map_env.add_obstacle(CircleObstacle(x=30, y=30, radius=8))
    map_env.add_obstacle(RectangleObstacle(x=60, y=50, width=15, height=30))
    map_env.add_obstacle(CircleObstacle(x=70, y=20, radius=6))
    
    map_env.set_start(10, 10)
    map_env.set_goal(90, 90)
    
    return map_env


def test_planners():
    """Test all planning algorithms."""
    print("=" * 70)
    print("TESTING PLANNERS")
    print("=" * 70)
    
    map_env = create_test_map()
    start = map_env.start
    goal = map_env.goal
    
    results = {}
    
    # Test A*
    print("\n1. Testing A* Planner...")
    print("-" * 70)
    astar = AStarPlanner(map_env, grid_resolution=0.5)
    path_astar = astar.plan(start, goal)
    
    if path_astar:
        results['A*'] = {
            'length': path_astar.length,
            'waypoints': len(path_astar.points),
            'time': astar.planning_time,
            'iterations': astar.iterations
        }
        print(f"✓ A* successful!")
        print(f"  Path length: {path_astar.length:.2f}m")
        print(f"  Waypoints: {len(path_astar.points)}")
        print(f"  Time: {astar.planning_time:.3f}s")
    else:
        print("✗ A* failed!")
    
    # Test RRT
    print("\n2. Testing RRT Planner...")
    print("-" * 70)
    rrt = RRTPlanner(map_env)
    path_rrt = rrt.plan(start, goal)
    
    if path_rrt:
        results['RRT'] = {
            'length': path_rrt.length,
            'waypoints': len(path_rrt.points),
            'time': rrt.planning_time,
            'iterations': rrt.iterations
        }
        print(f"✓ RRT successful!")
        print(f"  Path length: {path_rrt.length:.2f}m")
        print(f"  Waypoints: {len(path_rrt.points)}")
        print(f"  Time: {rrt.planning_time:.3f}s")
    else:
        print("✗ RRT failed!")
    
    # Test RRT*
    print("\n3. Testing RRT* Planner...")
    print("-" * 70)
    rrt_star = RRTStarPlanner(map_env)
    path_rrt_star = rrt_star.plan(start, goal)
    
    if path_rrt_star:
        results['RRT*'] = {
            'length': path_rrt_star.length,
            'waypoints': len(path_rrt_star.points),
            'time': rrt_star.planning_time,
            'iterations': rrt_star.iterations
        }
        print(f"✓ RRT* successful!")
        print(f"  Path length: {path_rrt_star.length:.2f}m")
        print(f"  Waypoints: {len(path_rrt_star.points)}")
        print(f"  Time: {rrt_star.planning_time:.3f}s")
    else:
        print("✗ RRT* failed!")
    
    # Comparison
    if results:
        print("\n" + "=" * 70)
        print("PLANNER COMPARISON")
        print("=" * 70)
        print(f"{'Algorithm':<10} {'Length (m)':<12} {'Time (s)':<10} {'Waypoints':<12}")
        print("-" * 70)
        for name, data in results.items():
            print(f"{name:<10} {data['length']:<12.2f} {data['time']:<10.3f} {data['waypoints']:<12}")
    
    return path_astar, path_rrt, path_rrt_star


def test_controllers(path):
    """Test controllers on a given path."""
    print("\n" + "=" * 70)
    print("TESTING CONTROLLERS")
    print("=" * 70)
    
    if path is None:
        print("No path provided, skipping controller tests")
        return
    
    # Test PID Controller
    print("\n1. Testing PID Controller...")
    print("-" * 70)
    
    vehicle_pid = Vehicle()
    vehicle_pid.reset(x=path.points[0].x, y=path.points[0].y, theta=0)
    dt = 0.1
    controller_pid = PathFollowingPID(
        vehicle_pid, dt=dt
    )
    controller_pid.set_path(path)
    
    # Simulate
    max_steps = 1000
    trajectory_pid = []
    
    for step in range(max_steps):
        acceleration, steering = controller_pid.control()
        vehicle_pid.update(acceleration, steering)
        trajectory_pid.append(vehicle_pid.get_position())
        
        if controller_pid.is_goal_reached():
            print(f"✓ PID reached goal in {step} steps ({step*dt:.1f}s)")
            break
    else:
        print(f"✗ PID did not reach goal in {max_steps} steps")
    
    # Test Pure Pursuit Controller
    print("\n2. Testing Pure Pursuit Controller...")
    print("-" * 70)
    
    vehicle_pp = Vehicle()
    vehicle_pp.reset(x=path.points[0].x, y=path.points[0].y, theta=0)
    threshold=2.0
    
    controller_pp = PurePursuitController(
        vehicle_pp,
        goal_threshold=threshold
    )
    controller_pp.set_path(path)
    
    # Simulate
    trajectory_pp = []
    
    for step in range(max_steps):
        acceleration, steering = controller_pp.control()
        vehicle_pp.update(acceleration, steering)
        trajectory_pp.append(vehicle_pp.get_position())
        
        if controller_pp.is_goal_reached():
            print(f"✓ Pure Pursuit reached goal in {step} steps ({step*dt:.1f}s)")
            break
    else:
        print(f"✗ Pure Pursuit did not reach goal in {max_steps} steps")
    
    # Compare controllers
    print("\n" + "=" * 70)
    print("CONTROLLER COMPARISON")
    print("=" * 70)
    
    # Calculate path following errors
    def calculate_tracking_error(trajectory, path):
        """Calculate average distance from trajectory to path."""
        errors = []
        for pos in trajectory:
            min_dist = float('inf')
            for point in path.points:
                dist = np.sqrt((pos[0] - point.x)**2 + (pos[1] - point.y)**2)
                min_dist = min(min_dist, dist)
            errors.append(min_dist)
        return np.mean(errors), np.max(errors)
    
    if trajectory_pid:
        avg_error_pid, max_error_pid = calculate_tracking_error(trajectory_pid, path)
        print(f"PID Controller:")
        print(f"  Avg tracking error: {avg_error_pid:.3f}m")
        print(f"  Max tracking error: {max_error_pid:.3f}m")
        print(f"  Final position: ({vehicle_pid.get_position()[0]:.2f}, {vehicle_pid.get_position()[1]:.2f})")
    
    if trajectory_pp:
        avg_error_pp, max_error_pp = calculate_tracking_error(trajectory_pp, path)
        print(f"\nPure Pursuit Controller:")
        print(f"  Avg tracking error: {avg_error_pp:.3f}m")
        print(f"  Max tracking error: {max_error_pp:.3f}m")
        print(f"  Final position: ({vehicle_pp.get_position()[0]:.2f}, {vehicle_pp.get_position()[1]:.2f})")


def test_integration():
    """Full integration test: planning + control."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: PLANNING + CONTROL")
    print("=" * 70)
    
    # Create map
    map_env = create_test_map()
    
    # Plan with A*
    planner = AStarPlanner(map_env, grid_resolution=0.5)
    path = planner.plan(map_env.start, map_env.goal)
    
    if not path:
        print("✗ Planning failed, cannot test integration")
        return
    
    print(f"✓ Path planned: {path.length:.2f}m with {len(path.points)} waypoints")
    
    # Follow with Pure Pursuit
    vehicle = Vehicle()
    vehicle.reset(x=path.points[0].x, y=path.points[0].y, theta=0)
    
    controller = PurePursuitController(vehicle=vehicle, goal_threshold=2.0)
    controller.set_path(path)
    
    max_steps = 1000
    collision_detected = False
    
    for step in range(max_steps):
        # Control
        acceleration, steering = controller.control()
        
        # Update
        vehicle.update(acceleration, steering)
        
        # Check collision
        pos = vehicle.get_position()
        if map_env.is_collision(pos[0], pos[1]):
            print(f"✗ Collision detected at step {step}!")
            collision_detected = True
            break
        
        # Check goal
        if controller.is_goal_reached():
            print(f"✓ Goal reached successfully in {step} steps ({step*0.1:.1f}s)!")
            print(f"  Final position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"  Target position: ({map_env.goal[0]:.2f}, {map_env.goal[1]:.2f})")
            print(f"  Distance to goal: {vehicle.distance_to(map_env.goal[0], map_env.goal[1]):.2f}m")
            break
    else:
        print(f"✗ Did not reach goal in {max_steps} steps")
    
    if not collision_detected:
        print("✓ No collisions detected during navigation")


def main():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 70)
    print("PHASE 2 TESTS: PLANNING & CONTROL")
    print("=" * 70 + "\n")
    
    try:
        # Test planners
        path_astar, path_rrt, path_rrt_star = test_planners()
        
        # Test controllers (use A* path if available)
        if path_astar:
            test_controllers(path_astar)
        
        # Integration test
        test_integration()
        
        print("\n" + "=" * 70)
        print("ALL PHASE 2 TESTS COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())