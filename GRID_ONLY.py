import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mptest
from mediapipe.tasks.python import vision
import numpy as np
import threading
import time

# =================================================================
# 1. CONFIGURATION
# =================================================================
WIDTH, HEIGHT = 1000, 700
COLOR_BG = (10, 10, 20)
NEON_BLUE = (0, 150, 255)

EYE_DEPTH = 10.0
UNIT_SCALE = 100

FPS = 60
SENSITIVITY = 8
SMOOTHING = 0.2

ROOM_DEPTH = 25.0
MIN_DEPTH = 5.0
MAX_DEPTH = 80.0

MODEL_PATH = "face_landmarker.task"

# Slider UI
SLIDER_WIDTH = 400
SLIDER_HEIGHT = 8
SLIDER_Y_OFFSET = 40
KNOB_RADIUS = 10


# =================================================================
# 2. HEAD TRACKING (MediaPipe Tasks API)
# =================================================================
class HeadTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("ERROR: Camera failed to open")
            return

        base_options = mptest.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.head_x = 0.0
        self.head_y = 0.0
        self.detected = False
        self.running = True
        self.timestamp = 0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.landmarker.detect_for_video(
                mp_image,
                self.timestamp
            )

            self.timestamp += 1

            if result.face_landmarks:
                self.detected = True
                pt = result.face_landmarks[0][168]  # nose tip
                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2
            else:
                self.detected = False

            time.sleep(1 / 60)

    def stop(self):
        self.running = False
        self.cap.release()


# =================================================================
# 3. 3D PROJECTION
# =================================================================
class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def project_off_axis(p, head_x, head_y):
    total_depth = EYE_DEPTH + p.z
    if total_depth <= 0.1:
        return None

    ratio = EYE_DEPTH / total_depth

    screen_x_virtual = head_x + (p.x - head_x) * ratio
    screen_y_virtual = head_y + (p.y - head_y) * ratio

    pixel_x = int(WIDTH / 2 + screen_x_virtual * UNIT_SCALE)
    pixel_y = int(HEIGHT / 2 + screen_y_virtual * UNIT_SCALE)

    return (pixel_x, pixel_y)


def draw_full_grid(surface, hx, hy, depth):
    w_room = 6.0
    h_room = 3.5
    grid_spacing = 2.0

    # Longitudinal lines
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(x, -h_room, 0), hx, hy)
        p2 = project_off_axis(Point3D(x, -h_room, depth), hx, hy)
        p3 = project_off_axis(Point3D(x, h_room, 0), hx, hy)
        p4 = project_off_axis(Point3D(x, h_room, depth), hx, hy)

        if p1 and p2:
            pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # Horizontal lines
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(-w_room, y, 0), hx, hy)
        p2 = project_off_axis(Point3D(-w_room, y, depth), hx, hy)
        p3 = project_off_axis(Point3D(w_room, y, 0), hx, hy)
        p4 = project_off_axis(Point3D(w_room, y, depth), hx, hy)

        if p1 and p2:
            pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # Depth slices
    for z in np.arange(0.0, depth + 0.1, grid_spacing):
        tl = project_off_axis(Point3D(-w_room, h_room, z), hx, hy)
        tr = project_off_axis(Point3D(w_room, h_room, z), hx, hy)
        br = project_off_axis(Point3D(w_room, -h_room, z), hx, hy)
        bl = project_off_axis(Point3D(-w_room, -h_room, z), hx, hy)

        if tl and tr and br and bl:
            pygame.draw.line(surface, NEON_BLUE, tl, tr, 1)
            pygame.draw.line(surface, NEON_BLUE, bl, br, 1)
            pygame.draw.line(surface, NEON_BLUE, tl, bl, 1)
            pygame.draw.line(surface, NEON_BLUE, tr, br, 1)


# =================================================================
# 4. SLIDER UI
# =================================================================
def draw_slider(surface, depth):
    slider_x = WIDTH // 2 - SLIDER_WIDTH // 2
    slider_y = HEIGHT - SLIDER_Y_OFFSET

    t = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    t = max(0.0, min(1.0, t))

    knob_x = slider_x + int(t * SLIDER_WIDTH)
    knob_y = slider_y

    pygame.draw.rect(surface, (80, 80, 80),
                     (slider_x, slider_y - SLIDER_HEIGHT // 2,
                      SLIDER_WIDTH, SLIDER_HEIGHT))

    pygame.draw.rect(surface, (0, 180, 255),
                     (slider_x, slider_y - SLIDER_HEIGHT // 2,
                      int(t * SLIDER_WIDTH), SLIDER_HEIGHT))

    pygame.draw.circle(surface, (255, 255, 255),
                       (knob_x, knob_y), KNOB_RADIUS)

    return slider_x, knob_x


# =================================================================
# 5. MAIN LOOP
# =================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D Full Grid Parallax")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 22)

    tracker = HeadTracking()

    smooth_hx, smooth_hy = 0.0, 0.0
    room_depth = ROOM_DEPTH
    dragging_slider = False

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                slider_x, knob_x = draw_slider(screen, room_depth)
                if abs(mx - knob_x) <= KNOB_RADIUS:
                    dragging_slider = True

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False

            if event.type == pygame.MOUSEMOTION and dragging_slider:
                mx, _ = pygame.mouse.get_pos()
                slider_x = WIDTH // 2 - SLIDER_WIDTH // 2
                t = (mx - slider_x) / SLIDER_WIDTH
                t = max(0.0, min(1.0, t))
                room_depth = MIN_DEPTH + t * (MAX_DEPTH - MIN_DEPTH)

        # Smooth tracking
        target_hx = tracker.head_x * SENSITIVITY
        target_hy = tracker.head_y * SENSITIVITY

        smooth_hx += (target_hx - smooth_hx) * SMOOTHING
        smooth_hy += (target_hy - smooth_hy) * SMOOTHING

        # Draw
        screen.fill(COLOR_BG)
        draw_full_grid(screen, smooth_hx, smooth_hy, room_depth)

        # Draw slider
        draw_slider(screen, room_depth)

        # Depth display text
        depth_text = font.render(f"Depth: {room_depth:.1f}", True, (200, 200, 200))
        screen.blit(depth_text, (WIDTH // 2 - depth_text.get_width() // 2, HEIGHT - 80))

        if not tracker.detected:
            text = font.render("NOT DETECTED. Check camera.", True, (255, 50, 50))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 110))

        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
