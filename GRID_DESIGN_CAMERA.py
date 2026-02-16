import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpTasks
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python import vision
import numpy as np
import math
import threading
import random

# =======================================================================================
# 1. CONFIGURATION "PURE NEON TRON"
# =======================================================================================
WIDTH, HEIGHT = 1280, 720
FPS = 60

# --- Color Palette (Cyan/Magenta TRON style) ---
COLOR_BG = (0, 5, 10)         # Very dark bluish black
NEON_CYAN = (0, 255, 255)     # Pure Cyan (Principal)
NEON_MAGENTA = (255, 0, 255)  # Magenta (Accent)
NEON_BLUE_DEEP = (0, 80, 255) # Electric Blue (Grid)
NEON_WHITE = (220, 255, 255)  # Bright core

# --- Parallax Calibration ---
UNIT_SCALE = 100
EYE_DEPTH = 10.0      # Virtual eye-screen distance
ROOM_DEPTH = 30.0     # Depth of the grid room

# --- Sensitivities & Smoothing ---
SENSITIVITY_X = 16.0
SENSITIVITY_Y = 16.0
SMOOTHING = 0.3       # Ideal compromise: smooth but responsive

# =======================================================================================
# 2. GRAPHICS ENGINE (INTENSE BLOOM RENDERING)
# =======================================================================================
def draw_intense_neon_line(surface, p1, p2, color, thickness=2):
    """
    Draws a line with an intense halo (Bloom effect).
    """
    if thickness <= 0: return

    # 1. Outer Glow (Wide and dark, simulates diffusion)
    glow_col = (max(0, color[0]//4), max(0, color[1]//4), max(0, color[2]//4))
    pygame.draw.line(surface, glow_col, p1, p2, thickness + 10)

    # 2. Inner Halo (Medium)
    mid_col = (color[0]//2, color[1]//2, color[2]//2)
    pygame.draw.line(surface, mid_col, p1, p2, thickness + 4)

    # 3. Color Core (Thin and saturated)
    pygame.draw.line(surface, color, p1, p2, thickness)

    # 4. White Hotspot (At the center for the dazzling effect)
    if thickness >= 2:
        pygame.draw.line(surface, NEON_WHITE, p1, p2, 1)

# =======================================================================================
# 3. TRACKING ENGINE
# =======================================================================================
''' Old HeadTracking Code, not adapted with current version
class HeadTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(2) # Change to 1, 2, etc. if camera 0 fails
        
        # Replacing mp.solutions with mp.tasks
        base_options = mpTasks.BaseOptions(model_asset_path="face_landmarker.task")

        options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = vision.RunningMode.VIDEO,
            num_faces = 1
        )

        face_mesh = vision.FaceLandmarker.create_from_options(options)

        #self.mp_face_mesh = mp.solutions.face_mesh
        #self.face_mesh = self.mp_face_mesh.FaceMesh(
        #    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        #
        self.mp_face_mesh = face_mesh
        self.head_x, self.head_y = 0, 0
        self.detected = False
        self.hud_surface = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            # Camera Visual Effect: "Hacker" Blue/Winter Colormap
            hud_frame = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
            hud_frame = cv2.addWeighted(hud_frame, 0.7, np.zeros_like(hud_frame), 0.3, 0)

            if results.multi_face_landmarks:
                self.detected = True
                mesh = results.multi_face_landmarks[0]
                pt = mesh.landmark[168] # Glabella (center of forehead)

                # Normalized coordinates (-1.0 to 1.0)
                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2

                # Drawing Neon mask on the camera HUD
                mp.solutions.drawing_utils.draw_landmarks(
                    image=hud_frame,
                    landmark_list=mesh,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=NEON_CYAN, thickness=1, circle_radius=0)
                )
            else:
                self.detected = False
                cv2.putText(hud_frame, "NO SIGNAL", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert to Pygame surface
            hud_frame = cv2.resize(hud_frame, (320, 240))
            hud_frame = np.rot90(hud_frame)
            hud_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            self.hud_surface = pygame.surfarray.make_surface(hud_frame)

    def stop(self):
        self.running = False
        self.cap.release()
'''
class HeadTracking:
    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Camera failed to open")

        base_options = mpTasks.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.timestamp = 0

        self.head_x = 0
        self.head_y = 0
        self.detected = False
        self.hud_surface = None
        self.running = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        
        while self.running:

            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.face_landmarker.detect_for_video(
                mp_image,
                self.timestamp
            )

            self.timestamp += 1

            hud_frame = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)

            if result.face_landmarks:
                self.detected = True
                mesh = result.face_landmarks[0]
                pt = mesh[168]

                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2

                #Draw simple dots instead of old FACEMESH_TESSELATION
                for lm in mesh:
                    px = int(lm.x * frame.shape[1])
                    py = int(lm.y * frame.shape[0])
                    cv2.circle(hud_frame, (px, py), 1, (0, 255, 255), -1)

            else:
                self.detected = False
                cv2.putText(hud_frame, "NO SIGNAL", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            hud_frame = cv2.resize(hud_frame, (320, 240))
            hud_frame = np.rot90(hud_frame)
            hud_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)

            self.hud_surface = pygame.surfarray.make_surface(hud_frame)

def stop(self):
    self.running = False
    self.cap.release()

# =======================================================================================
# 4. 3D ENGINE (OFF-AXIS PROJECTION)
# =======================================================================================
class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

def project_off_axis(p, head_x, head_y):
    """Calculates the 2D screen position based on the 3D point and head position."""
    total_depth = EYE_DEPTH + p.z
    if total_depth <= 0.1: return None

    ratio = EYE_DEPTH / total_depth
    screen_x_virtual = head_x + (p.x - head_x) * ratio
    screen_y_virtual = head_y + (p.y - head_y) * ratio

    pixel_x = int(WIDTH/2 + screen_x_virtual * UNIT_SCALE)
    pixel_y = int(HEIGHT/2 + screen_y_virtual * UNIT_SCALE)
    return (pixel_x, pixel_y)

class SceneObject:
    """Represents a rotating 3D object (Cube or Diamond) in the scene."""
    def __init__(self, obj_type, x, y, z, size, color):
        self.points, self.edges = [], []
        self.x, self.y, self.z = x, y, z
        self.color = color
        self.rot_x, self.rot_y = random.uniform(0, 6), random.uniform(0, 6)

        if obj_type == "cube":
            s = size
            nodes = [(x, y, z) for x in (-s, s) for y in (-s, s) for z in (-s, s)]
            self.points = [Point3D(*n) for n in nodes]
            self.edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        elif obj_type == "diamond":
            s = size
            self.points = [Point3D(0, s, 0), Point3D(0, -s, 0), Point3D(s, 0, 0), Point3D(-s, 0, 0), Point3D(0, 0, s), Point3D(0, 0, -s)]
            self.edges = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (4, 3), (3, 5), (5, 2)]

    def draw(self, surface, hx, hy):
        self.rot_x += 0.02; self.rot_y += 0.03
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)

        proj_pts = []
        for p in self.points:
            # 3D Rotation (Simplified X then Y)
            y1 = p.y * cx - p.z * sx; z1 = p.y * sx + p.z * cx
            x2 = p.x * cy - z1 * sy; z2 = p.x * sy + z1 * cy
            world_p = Point3D(x2 + self.x, y1 + self.y, z2 + self.z)
            proj_pts.append(project_off_axis(world_p, hx, hy))

        for i, j in self.edges:
            p1, p2 = proj_pts[i], proj_pts[j]
            if p1 and p2:
                # Diminish brightness for objects far in the background
                dist = self.z
                brightness = max(0.5, 1.0 - (dist / ROOM_DEPTH * 0.8))
                draw_col = (int(self.color[0] * brightness), int(self.color[1] * brightness), int(self.color[2] * brightness))
                draw_intense_neon_line(surface, p1, p2, draw_col, 2)

# =======================================================================================
# 5. ENVIRONMENT: THE FULL NEON ROOM
# =======================================================================================
def draw_full_neon_room(surface, hx, hy):
    """
    Draws a fully enclosed grid room: Floor, Ceiling, Left, Right, AND BACK WALL.
    """
    w_room = 9.0; h_room = 5.0
    col_grid = NEON_BLUE_DEEP
    grid_spacing = 2.0

    # --- 1. Longitudinal Lines (Z-axis, lines receding into the screen) ---
    # Floor & Ceiling
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        p_floor_start = project_off_axis(Point3D(x, -h_room, 0), hx, hy)
        p_floor_end = project_off_axis(Point3D(x, -h_room, ROOM_DEPTH), hx, hy)
        if p_floor_start and p_floor_end: draw_intense_neon_line(surface, p_floor_start, p_floor_end, col_grid, 1)

        p_ceil_start = project_off_axis(Point3D(x, h_room, 0), hx, hy)
        p_ceil_end = project_off_axis(Point3D(x, h_room, ROOM_DEPTH), hx, hy)
        if p_ceil_start and p_ceil_end: draw_intense_neon_line(surface, p_ceil_start, p_ceil_end, col_grid, 1)

    # Left & Right Walls
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        p_left_start = project_off_axis(Point3D(-w_room, y, 0), hx, hy)
        p_left_end = project_off_axis(Point3D(-w_room, y, ROOM_DEPTH), hx, hy)
        if p_left_start and p_left_end: draw_intense_neon_line(surface, p_left_start, p_left_end, col_grid, 1)

        p_right_start = project_off_axis(Point3D(w_room, y, 0), hx, hy)
        p_right_end = project_off_axis(Point3D(w_room, y, ROOM_DEPTH), hx, hy)
        if p_right_start and p_right_end: draw_intense_neon_line(surface, p_right_start, p_right_end, col_grid, 1)

    # --- 2. Transversal Lines (Depth slices) ---
    for z in np.arange(0, ROOM_DEPTH + 0.1, grid_spacing):
        # Calculate the 4 corners of the slice at depth Z
        tl = project_off_axis(Point3D(-w_room, h_room, z), hx, hy)
        tr = project_off_axis(Point3D(w_room, h_room, z), hx, hy)
        br = project_off_axis(Point3D(w_room, -h_room, z), hx, hy)
        bl = project_off_axis(Point3D(-w_room, -h_room, z), hx, hy)

        if tl and tr and br and bl:
            # Diminish intensity as it goes further back
            intensity = 1.0 - (z / (ROOM_DEPTH + 5))
            draw_c = (int(col_grid[0] * intensity), int(col_grid[1] * intensity), int(col_grid[2] * intensity))

            draw_intense_neon_line(surface, tl, tr, draw_c, 1) # Top
            draw_intense_neon_line(surface, tr, br, draw_c, 1) # Right
            draw_intense_neon_line(surface, br, bl, draw_c, 1) # Bottom
            draw_intense_neon_line(surface, bl, tl, draw_c, 1) # Left

    # --- 3. THE BACK WALL (The 4th wall at max depth) ---
    # Vertical lines of the back wall
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        p_top = project_off_axis(Point3D(x, h_room, ROOM_DEPTH), hx, hy)
        p_bot = project_off_axis(Point3D(x, -h_room, ROOM_DEPTH), hx, hy)
        if p_top and p_bot: draw_intense_neon_line(surface, p_top, p_bot, col_grid, 1)

    # Horizontal lines of the back wall
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        p_left = project_off_axis(Point3D(-w_room, y, ROOM_DEPTH), hx, hy)
        p_right = project_off_axis(Point3D(w_room, y, ROOM_DEPTH), hx, hy)
        if p_left and p_right: draw_intense_neon_line(surface, p_left, p_right, col_grid, 1)

# =======================================================================================
# 6. MAIN LOOP
# =======================================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PURE NEON PARALLAX")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18, bold=True)

    tracker = HeadTracking()

    # Cyan & Magenta Scene Objects
    objects = [
        SceneObject("diamond", 0, 0, 6, 1.2, NEON_CYAN),
        SceneObject("cube", -4, 2, 10, 0.8, NEON_MAGENTA),
        SceneObject("cube", 4, -2, 8, 0.8, NEON_MAGENTA),
        SceneObject("diamond", -3, -3, 4, 0.6, NEON_CYAN),
        SceneObject("cube", 3, 3, 12, 0.7, NEON_CYAN),
    ]

    smooth_hx, smooth_hy = 0, 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_e):
                running = False

        # --- LOGIC ---
        target_hx = tracker.head_x * SENSITIVITY_X
        target_hy = tracker.head_y * SENSITIVITY_Y

        # Smoothing calculation
        smooth_hx += (target_hx - smooth_hx) * SMOOTHING
        smooth_hy += (target_hy - smooth_hy) * SMOOTHING

        # --- DRAWING ---
        screen.fill(COLOR_BG)

        # 1. The Full Room (Grid Everywhere)
        draw_full_neon_room(screen, smooth_hx, smooth_hy)

        # 2. Objects (Sorted by depth for basic visibility control)
        objects.sort(key=lambda o: o.z, reverse=True)
        for obj in objects:
            obj.draw(screen, smooth_hx, smooth_hy)

        # 3. Camera HUD (Simple and Clean)
        if tracker.hud_surface:
            screen.blit(tracker.hud_surface, (20, HEIGHT - 260))
            pygame.draw.rect(screen, NEON_CYAN, (18, HEIGHT - 262, 324, 244), 2) # Simple border

        # 4. Status Indicator (Center of the screen for head position)
        if not tracker.detected:
            warn = font.render("NO FACE DETECTED", True, (255, 50, 50))
            screen.blit(warn, (WIDTH//2 - 80, HEIGHT - 50))
        else:
            # Draw a crosshair indicating the head position
            cx, cy = int(WIDTH/2 + smooth_hx*5), int(HEIGHT/2 + smooth_hy*5)
            pygame.draw.circle(screen, NEON_CYAN, (cx, cy), 3)
            pygame.draw.circle(screen, NEON_MAGENTA, (cx, cy), 8, 1)

        pygame.display.flip()

    tracker.stop()
    pygame.quit()

if __name__ == "__main__":
    main()