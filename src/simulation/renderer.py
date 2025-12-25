import pygame
import numpy as np
from typing import Tuple, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle
from src.core.map import Map2D, CircleObstacle, PolygonObstacle, RectangleObstacle
from src.planning.base_planner import Path as PlannedPath

class Color:
    """Color definitions (RGB)."""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 100, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)

class Renderer:
    """
    Pygame renderer for autonomous car simulation.
    
    Handles visualization of:
    - Map and obstacles
    - Vehicle
    - Planned path
    - Trajectory history
    - Text overlays (stats, info)
    """

    def __init__(self, 
                 screen_width: int = 1200,
                 screen_height: int = 800,
                 world_width: float = 100.0,
                 world_height: float = 100.0,
                 caption: str = "Autonomous Car 2D"):
        
        pygame.init()
        pygame.font.init()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = world_width
        self.world_height = world_height
        
        # Create window
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(caption)
        
        # Fonts
        self.font_small = pygame.font.SysFont('Arial', 14)
        self.font_medium = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Coordinate scaling
        self.scale_x = screen_width / world_width
        self.scale_y = screen_height / world_height
        
        # Visualization options
        self.show_grid = True
        self.show_path = True
        self.show_trajectory = True
        self.show_info = True
        self.show_sensors = False

        self.trajectory : List[Tuple[float, float]] = []
        self.max_trajectory_length = 500

        self.offset_x = 0
        self.offset_y = 0

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int((x + self.offset_x) * self.scale_x)
        screen_y = int(self.screen_height - (y + self.offset_y) * self.scale_y)
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        x = screen_x / self.scale_x - self.offset_x
        y = (self.screen_height - screen_y) / self.scale_y - self.offset_y
        return (x, y)
    
    def clear(self, color: Tuple[int, int, int] = Color.WHITE):
        self.screen.fill(color)

    def draw_grid(self, grid_size: float = 10.0, color: Tuple[int, int, int] = Color.LIGHT_GRAY):
        if not self.show_grid:
            return

        x = 0
        while x < self.world_width:
            start = self.world_to_screen(x, 0)
            end = self.world_to_screen(x, self.world_height)
            pygame.draw.line(self.screen, color, start, end, 1)
            x += grid_size
        
        y = 0
        while y <= self.world_height:
            start = self.world_to_screen(0, y)
            end = self.world_to_screen(self.world_width, y)
            pygame.draw.line(self.screen, color, start, end, 1)
            y += grid_size
        
    def draw_map(self, map_env: Map2D):
        """Draw map obstacles."""
        for obstacle in map_env.obstacles:
            if isinstance(obstacle, CircleObstacle):
                self._draw_circle_obstacle(obstacle)
            elif isinstance(obstacle, RectangleObstacle):
                self._draw_rectangle_obstacle(obstacle)
            elif isinstance(obstacle, PolygonObstacle):
                self._draw_polygon_obstacle(obstacle)

    def _draw_circle_obstacle(self, obstacle: CircleObstacle):
        """Draw circular obstacle."""
        center = self.world_to_screen(obstacle.x, obstacle.y)
        radius = int(obstacle.radius * self.scale_x)
        pygame.draw.circle(self.screen, Color.DARK_GRAY, center, radius)
        pygame.draw.circle(self.screen, Color.BLACK, center, radius, 2)
    
    def _draw_rectangle_obstacle(self, obstacle: RectangleObstacle):
        """Draw rectangular obstacle."""
        corners = obstacle.get_corners()
        screen_points = [self.world_to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(self.screen, Color.DARK_GRAY, screen_points)
        pygame.draw.polygon(self.screen, Color.BLACK, screen_points, 2)
    
    def _draw_polygon_obstacle(self, obstacle: PolygonObstacle):
        """Draw polygon obstacle."""
        screen_points = [self.world_to_screen(x, y) for x, y in obstacle.vertices]
        pygame.draw.polygon(self.screen, Color.DARK_GRAY, screen_points)
        pygame.draw.polygon(self.screen, Color.BLACK, screen_points, 2)

    def draw_vehicle(self, vehicle: Vehicle, color: Tuple[int, int, int] = Color.BLUE):
        """Draw vehicle as a rectangle with heading indicator."""
        corners = vehicle.get_corners()
        screen_corners = [self.world_to_screen(x, y) for x, y in corners]

        pygame.draw.polygon(self.screen, color, screen_corners)
        pygame.draw.polygon(self.screen, Color.BLACK, screen_corners, 2)

        pos = vehicle.get_position()
        theta = vehicle.state.theta
        arrow_length = vehicle.config.length * 0.7
        
        arrow_end_x = pos[0] + arrow_length * np.cos(theta)
        arrow_end_y = pos[1] + arrow_length * np.sin(theta)
        
        start = self.world_to_screen(pos[0], pos[1])
        end = self.world_to_screen(arrow_end_x, arrow_end_y)
        
        pygame.draw.line(self.screen, Color.YELLOW, start, end, 3)
        pygame.draw.circle(self.screen, Color.YELLOW, end, 5)

    def draw_path(self, path: PlannedPath, color: Tuple[int, int, int] = Color.RED):
        """Draw planned path."""
        if not self.show_path or path is None or len(path.points) < 2:
            return
        
        points = [self.world_to_screen(p.x, p.y) for p in path.points]
        pygame.draw.lines(self.screen, color, False, points, 2)
        
        # Draw waypoints
        for point in points[::5]:  # Draw every 5th waypoint
            pygame.draw.circle(self.screen, color, point, 3)


    def draw_trajectory(self, trajectory: Optional[List[Tuple[float, float]]] = None,
                       color: Tuple[int, int, int] = Color.CYAN):
        """Draw vehicle trajectory history."""
        if not self.show_trajectory:
            return
        
        traj = trajectory if trajectory is not None else self.trajectory
        
        if len(traj) < 2:
            return
        
        points = [self.world_to_screen(x, y) for x, y in traj]
        pygame.draw.lines(self.screen, color, False, points, 2)

    def add_trajectory_point(self, x: float, y: float):
        """Add point to trajectory history."""
        self.trajectory.append((x, y))
        
        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def draw_point(self, x: float, y: float, 
                   color: Tuple[int, int, int] = Color.GREEN,
                   radius: int = 5,
                   label: Optional[str] = None):
        """Draw a point with optional label."""
        screen_pos = self.world_to_screen(x, y)
        pygame.draw.circle(self.screen, color, screen_pos, radius)
        pygame.draw.circle(self.screen, Color.BLACK, screen_pos, radius, 1)
        
        if label:
            text = self.font_small.render(label, True, Color.BLACK)
            self.screen.blit(text, (screen_pos[0] + 8, screen_pos[1] - 8))
    
    def draw_start_goal(self, start: Tuple[float, float], 
                       goal: Tuple[float, float]):
        """Draw start and goal positions."""
        self.draw_point(start[0], start[1], Color.GREEN, 8, "START")
        self.draw_point(goal[0], goal[1], Color.RED, 8, "GOAL")
    
    def draw_text(self, text: str, x: int, y: int,
                 color: Tuple[int, int, int] = Color.BLACK,
                 font: Optional[pygame.font.Font] = None):
        """Draw text at screen coordinates."""
        font = font or self.font_medium
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def draw_info_panel(self, vehicle: Vehicle, 
                       step: int = 0,
                       fps: float = 60.0,
                       additional_info: Optional[dict] = None):
        """Draw information panel with stats."""
        if not self.show_info:
            return
        
        # Semi-transparent background
        panel_width = 250
        panel_height = 200
        panel = pygame.Surface((panel_width, panel_height))
        panel.set_alpha(200)
        panel.fill(Color.LIGHT_GRAY)
        self.screen.blit(panel, (10, 10))
        
        # Draw text
        y_offset = 20
        line_height = 25
        
        # Title
        self.draw_text("Vehicle Info", 20, y_offset, Color.BLACK, self.font_large)
        y_offset += line_height + 10
        
        # Vehicle stats
        pos = vehicle.get_position()
        info_lines = [
            f"Position: ({pos[0]:.2f}, {pos[1]:.2f})m",
            f"Heading: {np.degrees(vehicle.state.theta):.1f}°",
            f"Speed: {vehicle.state.velocity:.2f} m/s",
            f"Steering: {np.degrees(vehicle.state.steering_angle):.1f}°",
            f"Step: {step}",
            f"FPS: {fps:.1f}"
        ]
        
        for line in info_lines:
            self.draw_text(line, 20, y_offset, Color.BLACK, self.font_small)
            y_offset += line_height
        
        # Additional info
        if additional_info:
            y_offset += 10
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                self.draw_text(text, 20, y_offset, Color.BLACK, self.font_small)
                y_offset += line_height
    
    def draw_legend(self):
        """Draw legend explaining colors."""
        legend_items = [
            ("Vehicle", Color.BLUE),
            ("Path", Color.RED),
            ("Trajectory", Color.CYAN),
            ("Start", Color.GREEN),
            ("Goal", Color.RED),
            ("Obstacles", Color.DARK_GRAY)
        ]
        
        x_offset = self.screen_width - 150
        y_offset = 10
        
        # Background
        panel = pygame.Surface((140, len(legend_items) * 25 + 20))
        panel.set_alpha(200)
        panel.fill(Color.LIGHT_GRAY)
        self.screen.blit(panel, (x_offset, y_offset))
        
        y_offset += 10
        for label, color in legend_items:
            # Color box
            pygame.draw.rect(self.screen, color, 
                           (x_offset + 10, y_offset, 20, 15))
            pygame.draw.rect(self.screen, Color.BLACK, 
                           (x_offset + 10, y_offset, 20, 15), 1)
            
            # Label
            self.draw_text(label, x_offset + 40, y_offset, 
                          Color.BLACK, self.font_small)
            y_offset += 25
    
    def draw_controls_help(self):
        """Draw keyboard controls help."""
        help_text = [
            "Controls:",
            "G - Toggle Grid",
            "P - Toggle Path",
            "T - Toggle Trajectory",
            "I - Toggle Info",
            "ESC - Quit"
        ]
        
        x_offset = self.screen_width - 180
        y_offset = self.screen_height - len(help_text) * 20 - 20
        
        # Background
        panel = pygame.Surface((170, len(help_text) * 20 + 10))
        panel.set_alpha(180)
        panel.fill(Color.LIGHT_GRAY)
        self.screen.blit(panel, (x_offset, y_offset))
        
        y_offset += 5
        for line in help_text:
            self.draw_text(line, x_offset + 10, y_offset, 
                          Color.BLACK, self.font_small)
            y_offset += 20
    
    def update(self):
        """Update display."""
        pygame.display.flip()
    
    def close(self):
        """Close renderer and quit pygame."""
        pygame.quit()
