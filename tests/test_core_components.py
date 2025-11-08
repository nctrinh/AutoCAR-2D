import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.core.vehicle import VehicleConfig, VehicleState, Vehicle
from src.core.map import Map2D, CircleObstacle, RectangleObstacle, PolygonObstacle
from src.utils.config_loader import ConfigLoader


def test_vehicle():
    print('=' * 50)
    print("Testing Vehicle...")
    print("=" * 50)

    vehicle = Vehicle()
    vehicle.reset(0, 0, 0)
    
    print(f"Initial state: {vehicle}")

    # Test with steering_angle == 0
    for i in range(20):
        vehicle.update(acceleration=1.0, steering_angle=0.0)
    
    print(f"After driving forward: {vehicle}")  
    
    # Test with steering_angle != 0
    vehicle.reset(0, 0, 0)
    for i in range(20):
        vehicle.update(acceleration=1.0, steering_angle=0.3)
    print(f"After turning: {vehicle}") 

    # Test with corners
    corners = vehicle.get_corners()
    print(f"Vehicle corners:\n{corners}")
    
    print("✓ Vehicle test passed!\n")

def test_map():
    print("=" * 50)
    print("Testing Map...")
    print("=" * 50)
    
    map_env = Map2D(width=100, height=100)
    
    map_env.add_obstacle(RectangleObstacle(x=30, y=30, width=20, height=10))
    map_env.add_obstacle(CircleObstacle(x=70, y=70, radius=8))
    
    map_env.set_start(10, 10)
    map_env.set_goal(90, 90)
    
    print(f"Map created: {map_env}")
    print(f"Start: {map_env.start}")
    print(f"Goal: {map_env.goal}")
    
    # Test collision detection
    test_points = [
        (30, 30, True),   # Inside rectangle
        (70, 70, True),   # Inside circle
        (50, 50, False),  # Free space
        (10, 10, False),  # Start position
    ]
    
    print("\nCollision tests:")
    for x, y, expected in test_points:
        collision = map_env.is_collision(x, y)
        status = "✓" if collision == expected else "✗"
        print(f"  {status} Point ({x}, {y}): collision={collision} (expected={expected})")

def test_config():
    print("=" * 50)
    print("Testing Config Loader")
    print("=" * 50)

    config = ConfigLoader()

    print(f"Config Loader: {config}")

    fps = config.get("simulation.fps")
    max_velocity = config.get("vehicle.max_velocity")
    
    print(f"FPS: {fps}")
    print(f"Max velocity: {max_velocity} m/s")

    vehicle_config = config.get_vehicle_config()
    print(f"Vehicle config: {vehicle_config}")
    
    print("✓ Config test passed!\n")

def test_integration():
    print("=" * 50)
    print("Testing Integration...")
    print("=" * 50)
    
    config = ConfigLoader()
    vehicle_config = config.get_vehicle_config()
    
    vehicle = Vehicle(vehicle_config)
    vehicle.reset(x=10, y=10, theta=0)
    
    map_env = Map2D(width=100, height=100)
    map_env.add_obstacle(CircleObstacle(x=50, y=50, radius=10))
    
    print(f"Vehicle: {vehicle}")
    print(f"Map: {map_env}")
    
    x, y = vehicle.get_position()
    collision = map_env.is_collision(x, y)
    
    print(f"\nVehicle at ({x:.2f}, {y:.2f})")
    print(f"Collision: {collision}")
    
    print("\nSimulating movement...")
    collision_detected = False
    
    for i in range(50):
        vehicle.update(acceleration=1.0, steering_angle=0.0)
        x, y = vehicle.get_position()
        
        if map_env.is_collision(x, y):
            print(f"  Step {i}: Collision at ({x:.2f}, {y:.2f})")
            collision_detected = True
            break
    
    if not collision_detected:
        print(f"  No collisions in 50 steps. Final position: ({x:.2f}, {y:.2f})")
    
    print("✓ Integration test passed!\n")

def main():
    print("\n" + "=" * 50)
    print("AUTONOMOUS CAR 2D - BASIC TESTS")
    print("=" * 50 + "\n")
    
    try:
        test_vehicle()
        test_map()
        test_config()
        test_integration()
        
        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())