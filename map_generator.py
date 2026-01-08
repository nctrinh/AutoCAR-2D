import pygame
import yaml
import os
import math
import glob

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (10, 0, 0)
GREEN = (0, 100, 0)
BLUE = (0, 0, 100)
GRAY = (200, 200, 200)
PREVIEW_COLOR = (100, 100, 255, 128)
POLY_LINE_COLOR = (50, 50, 200)

VISUAL_GRID_DIVISIONS = 10
SNAP_STEP = 5

# Folder output
YAML_DIR = 'maps/yaml'
IMG_DIR = 'maps/images'

def get_next_map_index():
    """Find the next map index based on files in the yaml folder"""
    if not os.path.exists(YAML_DIR):
        return 1
    files = glob.glob(os.path.join(YAML_DIR, 'map_*.yaml'))
    if not files:
        return 1
    
    max_idx = 0
    for f in files:
        try:
            base = os.path.basename(f)
            idx = int(base.replace('map_', '').replace('.yaml', ''))
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1

def main():
    try:
        width = float(input("Enter map width (default: 100): ") or 100)
        height = float(input("Enter map height (default: 100): ") or 100)
        safety_margin = float(input("Enter safety_margin (default: 4.0): ") or 4.0)
        start_x = float(input("Enter start x (default: 10): ") or 10)
        start_y = float(input("Enter start y (default: 10): ") or 10)
        goal_x = float(input("Enter goal x (default: 90): ") or 90)
        goal_y = float(input("Enter goal y (default: 90): ") or 90)
    except ValueError:
        print("Input error. Using default values.")
        width, height = 100.0, 100.0
        safety_margin = 4.0
        start_x, start_y = 10.0, 10.0
        goal_x, goal_y = 90.0, 90.0

    start = [start_x, start_y]
    goal = [goal_x, goal_y]

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    scale_x = SCREEN_WIDTH / width
    scale_y = SCREEN_HEIGHT / height

    obstacles = []
    
    # Mode: 'RECT', 'CIRCLE', 'POLY'
    current_mode = 'RECT' 
    
    drawing_start = None 
    poly_points = []

    def map_to_screen(mx, my):
        sx = mx * scale_x
        sy = SCREEN_HEIGHT - (my * scale_y)
        return int(sx), int(sy)
    
    def screen_to_map(sx, sy):
        mx = sx / scale_x
        my = (SCREEN_HEIGHT - sy) / scale_y
        return mx, my

    def get_rect_data(p1, p2):
        min_x = min(p1[0], p2[0])
        max_x = max(p1[0], p2[0])
        min_y = min(p1[1], p2[1])
        max_y = max(p1[1], p2[1])
        return {
            'type': 'rectangle',
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'width': max_x - min_x, 'height': max_y - min_y,
            'x': min_x + (max_x - min_x) / 2,
            'y': min_y + (max_y - min_y) / 2,
            'angle': 0.0
        }

    def save_data_and_image():
        idx = get_next_map_index()
        filename_base = f"map_{idx}"
        
        os.makedirs(YAML_DIR, exist_ok=True)
        os.makedirs(IMG_DIR, exist_ok=True)

        final_obs = []
        for obs in obstacles:
            if obs['type'] == 'rectangle':
                final_obs.append({
                    'type': 'rectangle',
                    'x': float(obs['x']), 'y': float(obs['y']),
                    'width': float(obs['width']), 'height': float(obs['height']),
                    'angle': 0.0
                })
            elif obs['type'] == 'circle':
                final_obs.append({
                    'type': 'circle',
                    'x': float(obs['x']), 'y': float(obs['y']),
                    'radius': float(obs['radius'])
                })
            elif obs['type'] == 'polygon':
                final_obs.append({
                    'type': 'polygon',
                    'vertices': [[float(p[0]), float(p[1])] for p in obs['vertices']]
                })

        map_data = {
            'map': {
                'width': width, 'height': height,
                'safety_margin': safety_margin,
                'start': start, 'goal': goal,
                'templates': 'custom',
                'obstacles': final_obs
            }
        }

        yaml_path = os.path.join(YAML_DIR, f"{filename_base}.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(map_data, f, default_flow_style=None, sort_keys=False)
        
        img_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        img_surface.fill(WHITE)
        pygame.draw.rect(img_surface, BLACK, (0,0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)
        
        for obs in obstacles:
            if obs['type'] == 'rectangle':
                tl_x = obs['min_x'] * scale_x
                tl_y = SCREEN_HEIGHT - (obs['max_y'] * scale_y)
                w_s = obs['width'] * scale_x
                h_s = obs['height'] * scale_y
                pygame.draw.rect(img_surface, BLUE, (tl_x, tl_y, w_s, h_s))
                pygame.draw.rect(img_surface, BLACK, (tl_x, tl_y, w_s, h_s), 2)
                
            elif obs['type'] == 'circle':
                cx_s, cy_s = map_to_screen(obs['x'], obs['y'])
                r_s = obs['radius'] * scale_x # Giả sử tỉ lệ x,y đều như nhau
                pygame.draw.circle(img_surface, BLUE, (cx_s, cy_s), int(r_s))
                pygame.draw.circle(img_surface, BLACK, (cx_s, cy_s), int(r_s), 2)
                
            elif obs['type'] == 'polygon':
                pts_screen = [map_to_screen(p[0], p[1]) for p in obs['vertices']]
                if len(pts_screen) >= 3:
                    pygame.draw.polygon(img_surface, BLUE, pts_screen)
                    pygame.draw.polygon(img_surface, BLACK, pts_screen, 2)
        
        s_scr = map_to_screen(start[0], start[1])
        g_scr = map_to_screen(goal[0], goal[1])
        pygame.draw.circle(img_surface, GREEN, s_scr, 8)
        pygame.draw.circle(img_surface, RED, g_scr, 8)

        img_path = os.path.join(IMG_DIR, f"{filename_base}.png")
        pygame.image.save(img_surface, img_path)

        print(f"\nMap saved successfully!")
        print(f"   YAML: {yaml_path}")
        print(f"   IMG: {img_path}")

    running = True
    while running:
        mode_str = f"MODE: {current_mode}"
        info_str = f"| R: Rect | C: Circle | P: Poly | Z: Undo | SPACE/R-Click: End Poly | ENTER: Save"
        pygame.display.set_caption(f"{mode_str} {info_str}")

        screen.fill(WHITE)

        for i in range(VISUAL_GRID_DIVISIONS + 1):
            map_val = i * (width / VISUAL_GRID_DIVISIONS)
            sx = map_val * scale_x
            pygame.draw.line(screen, GRAY, (sx, 0), (sx, SCREEN_HEIGHT), 1)
            sy = SCREEN_HEIGHT - (map_val * scale_y)
            pygame.draw.line(screen, GRAY, (0, sy), (SCREEN_WIDTH, sy), 1)

        pygame.draw.circle(screen, GREEN, map_to_screen(start[0], start[1]), 6)
        pygame.draw.circle(screen, RED, map_to_screen(goal[0], goal[1]), 6)

        for obs in obstacles:
            if obs['type'] == 'rectangle':
                tl_x = obs['min_x'] * scale_x
                tl_y = SCREEN_HEIGHT - (obs['max_y'] * scale_y)
                w_s = obs['width'] * scale_x
                h_s = obs['height'] * scale_y
                pygame.draw.rect(screen, BLUE, (tl_x, tl_y, w_s, h_s))
                pygame.draw.rect(screen, BLACK, (tl_x, tl_y, w_s, h_s), 2)
            
            elif obs['type'] == 'circle':
                c_scr = map_to_screen(obs['x'], obs['y'])
                r_scr = obs['radius'] * scale_x
                pygame.draw.circle(screen, BLUE, c_scr, int(r_scr))
                pygame.draw.circle(screen, BLACK, c_scr, int(r_scr), 2)

            elif obs['type'] == 'polygon':
                pts_s = [map_to_screen(p[0], p[1]) for p in obs['vertices']]
                if len(pts_s) >= 3:
                    pygame.draw.polygon(screen, BLUE, pts_s)
                    pygame.draw.polygon(screen, BLACK, pts_s, 2)

        mouse_x, mouse_y = pygame.mouse.get_pos()
        raw_mx = mouse_x / scale_x
        raw_my = (SCREEN_HEIGHT - mouse_y) / scale_y
        
        snap_mx = round(raw_mx / SNAP_STEP) * SNAP_STEP
        snap_my = round(raw_my / SNAP_STEP) * SNAP_STEP
        snap_mx = max(0, min(width, snap_mx))
        snap_my = max(0, min(height, snap_my))
        
        curr_map_pos = [snap_mx, snap_my]
        curr_scr_pos = map_to_screen(snap_mx, snap_my)

        if current_mode == 'RECT' and drawing_start:
            temp_rect = get_rect_data(drawing_start, curr_map_pos)
            tl_x = temp_rect['min_x'] * scale_x
            tl_y = SCREEN_HEIGHT - (temp_rect['max_y'] * scale_y)
            w_s = temp_rect['width'] * scale_x
            h_s = temp_rect['height'] * scale_y
            
            s = pygame.Surface((w_s, h_s), pygame.SRCALPHA)
            s.fill(PREVIEW_COLOR)
            screen.blit(s, (tl_x, tl_y))
            pygame.draw.rect(screen, BLUE, (tl_x, tl_y, w_s, h_s), 1)
            pygame.draw.circle(screen, RED, map_to_screen(drawing_start[0], drawing_start[1]), 4)

        elif current_mode == 'CIRCLE' and drawing_start:
            center_scr = map_to_screen(drawing_start[0], drawing_start[1])
            dist = math.sqrt((curr_map_pos[0]-drawing_start[0])**2 + (curr_map_pos[1]-drawing_start[1])**2)
            r_scr = dist * scale_x
            
            if r_scr > 1:
                pygame.draw.circle(screen, BLUE, center_scr, int(r_scr), 1)
                pygame.draw.line(screen, RED, center_scr, curr_scr_pos, 1)
            pygame.draw.circle(screen, RED, center_scr, 4)

        elif current_mode == 'POLY' and len(poly_points) > 0:
            pts_s = [map_to_screen(p[0], p[1]) for p in poly_points]
            if len(pts_s) > 1:
                pygame.draw.lines(screen, POLY_LINE_COLOR, False, pts_s, 2)
            pygame.draw.line(screen, POLY_LINE_COLOR, pts_s[-1], curr_scr_pos, 1)
            for p in pts_s:
                pygame.draw.circle(screen, RED, p, 3)

        pygame.draw.circle(screen, BLACK, curr_scr_pos, 3)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    current_mode = 'RECT'
                    drawing_start = None
                    poly_points = []
                    print("Switched to RECT mode")
                
                elif event.key == pygame.K_c:
                    current_mode = 'CIRCLE'
                    drawing_start = None
                    poly_points = []
                    print("Switched to CIRCLE mode")

                elif event.key == pygame.K_p:
                    current_mode = 'POLY'
                    drawing_start = None
                    poly_points = []
                    print("Switched to POLYGON mode")

                elif event.key == pygame.K_z:
                    if current_mode == 'POLY' and len(poly_points) > 0:
                        poly_points.pop()
                        print("Removed last polygon vertex")
                    elif drawing_start is not None:
                        drawing_start = None
                        print("Cancelled current shape")
                    elif len(obstacles) > 0:
                        removed = obstacles.pop()
                        print(f"Removed last obstacle: {removed['type']}")
                    else:
                        print("Nothing to undo")

                elif event.key == pygame.K_SPACE:
                    if current_mode == 'POLY' and len(poly_points) >= 3:
                        obstacles.append({
                            'type': 'polygon',
                            'vertices': list(poly_points)
                        })
                        print(f"Created Polygon with {len(poly_points)} vertices")
                        poly_points = []

                elif event.key == pygame.K_RETURN:
                    save_data_and_image()
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if current_mode == 'RECT':
                        if drawing_start is None:
                            drawing_start = curr_map_pos
                        else:
                            if curr_map_pos != drawing_start:
                                new_rect = get_rect_data(drawing_start, curr_map_pos)
                                if new_rect['width'] > 0 and new_rect['height'] > 0:
                                    obstacles.append(new_rect)
                                    print("Added Rectangle")
                                drawing_start = None

                    elif current_mode == 'CIRCLE':
                        if drawing_start is None:
                            drawing_start = curr_map_pos
                        else:
                            radius = math.sqrt((curr_map_pos[0]-drawing_start[0])**2 + (curr_map_pos[1]-drawing_start[1])**2)
                            if radius > 0:
                                obstacles.append({
                                    'type': 'circle',
                                    'x': drawing_start[0],
                                    'y': drawing_start[1],
                                    'radius': radius
                                })
                                print(f"Added Circle R={radius:.2f}")
                                drawing_start = None
                    
                    elif current_mode == 'POLY':
                        poly_points.append(curr_map_pos)
                
                elif event.button == 3:
                    if current_mode == 'POLY' and len(poly_points) >= 3:
                         obstacles.append({
                            'type': 'polygon',
                            'vertices': list(poly_points)
                        })
                         print(f"Created Polygon with {len(poly_points)} vertices (Right-click)")
                         poly_points = []

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()