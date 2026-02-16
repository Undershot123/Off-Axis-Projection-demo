import pygame
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# =================================================================
# 1. CONFIGURATION (Constants)
# =================================================================
WIDTH, HEIGHT = 1000, 700
COLOR_BG = (10, 10, 20)
NEON_BLUE = (0, 150, 255) # Grid Color
EYE_DEPTH = 10.0          # Virtual eye-screen distance (z_0)
UNIT_SCALE = 100          # Scale for converting 3D coordinates
FPS = 60
SENSITIVITY = 8           # Amplification of head movement
SMOOTHING = 0.2           # Movement smoothing (0.0 for none, 1.0 for max)
ROOM_DEPTH = 25.0         # Depth of the grid room

# =================================================================
# 2. TRACKING ENGINE (Threading for fluid performance)
# =================================================================
class HeadTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # Camera 0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
        self.head_x, self.head_y = 0.0, 0.0
        self.detected = False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                self.detected = True
                pt = results.multi_face_landmarks[0].landmark[168] # Center point of the face (Glabella)

                # Normalized coordinates (-1.0 to 1.0)
                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2
            else:
                self.detected = False

            time.sleep(1/60) # Limits camera processing rate

    def stop(self):
        self.running = False
        self.cap.release()

# =================================================================
# 3. 3D PROJECTION AND DRAWING FUNCTIONS
# =================================================================
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

def draw_full_grid(surface, hx, hy):
    """Draws the complete 3D grid (Floor, Ceiling, Walls)."""
    w_room = 6.0; h_room = 3.5; grid_spacing = 2.0

    # --- LONGITUDINAL LINES (Lines receding into the screen) ---

    # Floor and Ceiling
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        # Floor
        p1 = project_off_axis(Point3D(x, -h_room, 0), hx, hy)
        p2 = project_off_axis(Point3D(x, -h_room, ROOM_DEPTH), hx, hy)
        if p1 and p2: pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        # Ceiling
        p3 = project_off_axis(Point3D(x, h_room, 0), hx, hy)
        p4 = project_off_axis(Point3D(x, h_room, ROOM_DEPTH), hx, hy)
        if p3 and p4: pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # Left and Right Walls
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        # Left Wall
        p1 = project_off_axis(Point3D(-w_room, y, 0), hx, hy)
        p2 = project_off_axis(Point3D(-w_room, y, ROOM_DEPTH), hx, hy)
        if p1 and p2: pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        # Right Wall
        p3 = project_off_axis(Point3D(w_room, y, 0), hx, hy)
        p4 = project_off_axis(Point3D(w_room, y, ROOM_DEPTH), hx, hy)
        if p3 and p4: pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # --- TRANSVERSAL LINES (Depth slices) ---

    for z in np.arange(0.0, ROOM_DEPTH + 0.1, grid_spacing):
        # 4 corners of the "box" at this depth Z
        tl = project_off_axis(Point3D(-w_room, h_room, z), hx, hy) # Top Left
        tr = project_off_axis(Point3D(w_room, h_room, z), hx, hy)  # Top Right
        br = project_off_axis(Point3D(w_room, -h_room, z), hx, hy) # Bottom Right
        bl = project_off_axis(Point3D(-w_room, -h_room, z), hx, hy) # Bottom Left

        if tl and tr and br and bl:
            pygame.draw.line(surface, NEON_BLUE, tl, tr, 1) # Top
            pygame.draw.line(surface, NEON_BLUE, bl, br, 1) # Bottom
            pygame.draw.line(surface, NEON_BLUE, tl, bl, 1) # Left
            pygame.draw.line(surface, NEON_BLUE, tr, br, 1) # Right

# =================================================================
# 4. MAIN LOOP
# =================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D FULL GRID PARALLAX")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 24)

    tracker = HeadTracking()

    smooth_hx, smooth_hy = 0.0, 0.0

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_e):
                running = False

        # --- LOGIC ---
        # 1. Read head position
        target_hx = tracker.head_x * SENSITIVITY
        target_hy = tracker.head_y * SENSITIVITY

        # 2. Apply smoothing (fluid movement)
        smooth_hx += (target_hx - smooth_hx) * SMOOTHING
        smooth_hy += (target_hy - smooth_hy) * SMOOTHING

        # --- DRAWING ---
        screen.fill(COLOR_BG)

        # Draw the complete 3D grid
        draw_full_grid(screen, smooth_hx, smooth_hy)

        # Display detection status
        if not tracker.detected:
            text = font.render("NOT DETECTED. Check camera (0 or 1).", True, (255, 50, 50))
            screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 50))

        pygame.display.flip()

    tracker.stop()
    pygame.quit()

if __name__ == "__main__":
    main()