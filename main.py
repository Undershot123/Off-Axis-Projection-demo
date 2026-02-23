import pygame
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mptest
from mediapipe.tasks.python import vision
import numpy as np
import threading
import time
import win32gui
import win32con

# =================================================================
# 1. CONFIGURATION
# =================================================================
WIDTH, HEIGHT = 1000, 700  # Initial size, overridden at runtime by maximized dimensions
COLOR_BG = (10, 10, 20)
NEON_BLUE = (0, 150, 255)

# Vertical FOV control
VERTICAL_FOV = 150.0
MIN_FOV = 150.0
MAX_FOV = 170.0

FPS = 60
SENSITIVITY = 8
SMOOTHING = 0.2

ROOM_DEPTH = 50.0
MIN_DEPTH = 25.0
MAX_DEPTH = 100.0

# Box boundaries (must match w_room / h_room in draw_full_grid)
BOX_W = 6.0
BOX_H = 3.5

MODEL_PATH = "face_landmarker.task"

# Window panel UI
PANEL_W       = 320
PANEL_PADDING = 10
PANEL_ROW_H   = 28
CAPTURE_FPS   = 15

# Slider UI — SLIDER_WIDTH is set at runtime relative to screen width
SLIDER_WIDTH = 400
SLIDER_HEIGHT = 8
SLIDER_Y_OFFSET = 90   # distance from bottom for the bottom slider
SLIDER_GAP = 55        # vertical gap between stacked sliders
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
                pt = result.face_landmarks[0][168]  # nose bridge / center-ish
                self.head_x = (pt.x - 0.5) * 2
                self.head_y = (pt.y - 0.5) * 2
            else:
                self.detected = False

            time.sleep(1 / 60)

    def stop(self):
        self.running = False
        self.cap.release()


# =================================================================
# 3. WINDOW CAPTURE
# =================================================================
class WindowCapture:
    """
    Captures a Win32 window by HWND at ~CAPTURE_FPS using mss.
    Thread-safe: background thread writes under a lock; get_frame() returns a copy.
    """
    def __init__(self, hwnd, target_fps=CAPTURE_FPS):
        self.hwnd        = hwnd
        self._target_fps = target_fps
        self._frame      = None
        self._lock       = threading.Lock()
        self.running     = True
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import mss
        with mss.mss() as sct:
            while self.running:
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                    w = right - left
                    h = bottom - top
                    if w > 0 and h > 0:
                        monitor = {"left": left, "top": top, "width": w, "height": h}
                        img = sct.grab(monitor)
                        frame_bgr = np.array(img)[:, :, :3]
                        with self._lock:
                            self._frame = frame_bgr
                except Exception:
                    pass
                time.sleep(1 / self._target_fps)

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self.running = False


# =================================================================
# 4. 3D PROJECTION
# =================================================================
class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def project_off_axis(p, head_x, head_y, fov):
    # Convert vertical FOV to focal length (in world units relative to screen height)
    f = (HEIGHT / 2) / np.tan(np.radians(fov / 2))

    total_depth = p.z + 0.0001
    if total_depth <= 0:
        return None

    # True off-axis projection:
    # Eye sits at (head_x, head_y) at distance f behind the screen plane.
    # Ray from eye through world point p intersects the screen plane at:
    #   screen = head + (p - head) * f / (f + p.z)
    # This creates the "window" illusion: objects on the screen plane (z=0)
    # are fixed; objects behind it shift with parallax as the head moves.
    ratio = f / (f + total_depth)

    screen_x = head_x + (p.x - head_x) * ratio
    screen_y = head_y + (p.y - head_y) * ratio

    pixel_x = int(WIDTH / 2 + screen_x * (WIDTH / 2) / BOX_W)
    pixel_y = int(HEIGHT / 2 + screen_y * (HEIGHT / 2) / BOX_H)

    return (pixel_x, pixel_y)


def draw_full_grid(surface, hx, hy, depth, fov):
    w_room = BOX_W
    h_room = BOX_H
    grid_spacing = 2.0

    # Longitudinal lines (along Z)
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(x, -h_room, 0), hx, hy, fov)
        p2 = project_off_axis(Point3D(x, -h_room, depth), hx, hy, fov)
        p3 = project_off_axis(Point3D(x, h_room, 0), hx, hy, fov)
        p4 = project_off_axis(Point3D(x, h_room, depth), hx, hy, fov)

        if p1 and p2:
            pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # Horizontal lines (along Z)
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(-w_room, y, 0), hx, hy, fov)
        p2 = project_off_axis(Point3D(-w_room, y, depth), hx, hy, fov)
        p3 = project_off_axis(Point3D(w_room, y, 0), hx, hy, fov)
        p4 = project_off_axis(Point3D(w_room, y, depth), hx, hy, fov)

        if p1 and p2:
            pygame.draw.line(surface, NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, NEON_BLUE, p3, p4, 1)

    # Depth slices (rectangles)
    for z in np.arange(0.0, depth + 0.1, grid_spacing):
        tl = project_off_axis(Point3D(-w_room, h_room, z), hx, hy, fov)
        tr = project_off_axis(Point3D(w_room, h_room, z), hx, hy, fov)
        br = project_off_axis(Point3D(w_room, -h_room, z), hx, hy, fov)
        bl = project_off_axis(Point3D(-w_room, -h_room, z), hx, hy, fov)

        if tl and tr and br and bl:
            pygame.draw.line(surface, NEON_BLUE, tl, tr, 1)
            pygame.draw.line(surface, NEON_BLUE, bl, br, 1)
            pygame.draw.line(surface, NEON_BLUE, tl, bl, 1)
            pygame.draw.line(surface, NEON_BLUE, tr, br, 1)


# =================================================================
# 5. FACE DEFINITIONS & WINDOW ENUMERATION
# =================================================================
FACE_NAMES = ["back", "top", "bottom", "left", "right"]


def _make_face_corners(depth):
    """Returns dict of face name -> [tl, tr, br, bl] Point3D for the current depth."""
    d = depth
    return {
        "back":   [Point3D(-BOX_W,  BOX_H, d), Point3D( BOX_W,  BOX_H, d),
                   Point3D( BOX_W, -BOX_H, d), Point3D(-BOX_W, -BOX_H, d)],
        # "top"/"bottom" labels match user expectation (Y-axis is inverted on screen)
        "top":    [Point3D(-BOX_W, -BOX_H, 0), Point3D( BOX_W, -BOX_H, 0),
                   Point3D( BOX_W, -BOX_H, d), Point3D(-BOX_W, -BOX_H, d)],
        "bottom": [Point3D(-BOX_W,  BOX_H, 0), Point3D( BOX_W,  BOX_H, 0),
                   Point3D( BOX_W,  BOX_H, d), Point3D(-BOX_W,  BOX_H, d)],
        "left":   [Point3D(-BOX_W,  BOX_H, 0), Point3D(-BOX_W, -BOX_H, 0),
                   Point3D(-BOX_W, -BOX_H, d), Point3D(-BOX_W,  BOX_H, d)],
        "right":  [Point3D( BOX_W,  BOX_H, 0), Point3D( BOX_W, -BOX_H, 0),
                   Point3D( BOX_W, -BOX_H, d), Point3D( BOX_W,  BOX_H, d)],
    }


def _enum_windows_callback(hwnd, result_list):
    if not win32gui.IsWindowVisible(hwnd):
        return
    title = win32gui.GetWindowText(hwnd)
    if not title:
        return
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        if (r - l) < 10 or (b - t) < 10:
            return
    except Exception:
        return
    if title == "3D Full Grid Parallax":
        return
    result_list.append((hwnd, title))


def enumerate_windows():
    results = []
    win32gui.EnumWindows(_enum_windows_callback, results)
    return results


# Per-face source corner mappings for warpPerspective.
# Lambdas take (w, h) of the native frame so no resize is needed.
# All floor/ceiling/back faces use a vertical flip to correct Y-axis inversion.
# Left/right walls use a 90° rotation to un-rotate the sideways mapping.
_FACE_SRC_PTS = {
    "back":   lambda w, h: np.float32([[0,h],[w,h],[w,0],[0,0]]),  # vertical flip
    "top":    lambda w, h: np.float32([[0,h],[w,h],[w,0],[0,0]]),  # vertical flip
    "bottom": lambda w, h: np.float32([[0,h],[w,h],[w,0],[0,0]]),  # vertical flip
    "left":   lambda w, h: np.float32([[0,0],[0,h],[w,h],[w,0]]),  # 90° rotation
    "right":  lambda w, h: np.float32([[0,0],[0,h],[w,h],[w,0]]),  # 90° rotation
}


# =================================================================
# 6. FACE MAPPER
# =================================================================
class FaceMapper:
    def __init__(self):
        self.captures    = {name: None for name in FACE_NAMES}
        self.assignments = {name: None for name in FACE_NAMES}

    def assign(self, face_name, hwnd, title):
        old = self.captures.get(face_name)
        if old is not None:
            old.stop()
        self.assignments[face_name] = (hwnd, title)
        self.captures[face_name]    = WindowCapture(hwnd)

    def unassign(self, face_name):
        old = self.captures.get(face_name)
        if old is not None:
            old.stop()
        self.captures[face_name]    = None
        self.assignments[face_name] = None

    def stop_all(self):
        for name in FACE_NAMES:
            self.unassign(name)

    def _warp_frame_to_quad(self, frame_bgr, screen_pts, src_pts_fn=None):
        h_f, w_f  = frame_bgr.shape[:2]  # use native capture resolution — no resize
        src_pts   = src_pts_fn(w_f, h_f) if src_pts_fn is not None \
                    else np.float32([[0, 0], [w_f, 0], [w_f, h_f], [0, h_f]])
        dst_pts   = np.float32(screen_pts)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            frame_bgr, M, (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        # Punch out exact-black border pixels so faces don't overwrite each other
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2RGBA)
        mask = (warped[:, :, 0] == 0) & (warped[:, :, 1] == 0) & (warped[:, :, 2] == 0)
        warped_rgba[mask, 3] = 0
        # frombuffer expects a contiguous (H, W, 4) RGBA array — no transpose needed
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(warped_rgba).tobytes(),
            (WIDTH, HEIGHT), "RGBA"
        )
        return surf.convert_alpha()

    def draw_faces(self, surface, hx, hy, depth, fov):
        corners = _make_face_corners(depth)
        for name in FACE_NAMES:
            if self.captures[name] is None:
                continue
            frame = self.captures[name].get_frame()
            if frame is None:
                continue
            screen_pts = [project_off_axis(p, hx, hy, fov) for p in corners[name]]
            if any(pt is None for pt in screen_pts):
                continue
            surf = self._warp_frame_to_quad(frame, screen_pts, _FACE_SRC_PTS[name])
            surface.blit(surf, (0, 0))



# =================================================================
# 7. WINDOW PANEL UI
# =================================================================
class WindowPanelUI:
    REFRESH_INTERVAL = 3.0

    def __init__(self, face_mapper):
        self.face_mapper   = face_mapper
        self.visible       = False
        self.window_list   = []
        self._last_refresh = 0.0
        self.face_pending  = None
        self.hovered_face  = None
        self.scroll_offset  = 0
        self._rect          = pygame.Rect(0, 0, PANEL_W, 0)
        self.show_wireframe = True

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._refresh()

    def _refresh(self):
        self.window_list   = enumerate_windows()
        self._last_refresh = time.time()

    def _visible_rows(self):
        return max(1, (HEIGHT - 60) // PANEL_ROW_H - 1)  # -1 for Clear row

    def _face_col_rect(self, i):
        return pygame.Rect(
            WIDTH - PANEL_W + PANEL_PADDING,
            36 + i * PANEL_ROW_H,
            90, PANEL_ROW_H - 2
        )

    def _clear_row_rect(self):
        return pygame.Rect(
            WIDTH - PANEL_W + 90 + PANEL_PADDING * 2,
            36,
            PANEL_W - 90 - PANEL_PADDING * 3,
            PANEL_ROW_H - 2
        )

    def _win_col_rect(self, j):
        return pygame.Rect(
            WIDTH - PANEL_W + 90 + PANEL_PADDING * 2,
            36 + PANEL_ROW_H + j * PANEL_ROW_H,   # shifted down by one Clear row
            PANEL_W - 90 - PANEL_PADDING * 3,
            PANEL_ROW_H - 2
        )

    def _wireframe_toggle_rect(self):
        return pygame.Rect(
            WIDTH - PANEL_W + PANEL_PADDING,
            HEIGHT - PANEL_ROW_H * 2 - PANEL_PADDING - 100,
            PANEL_W - PANEL_PADDING * 2,
            PANEL_ROW_H - 2
        )

    def handle_event(self, event):
        if not self.visible:
            self.hovered_face = None
            return None

        if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
            self._refresh()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self._wireframe_toggle_rect().collidepoint(mx, my):
                self.show_wireframe = not self.show_wireframe
                return self.hovered_face
            for i, name in enumerate(FACE_NAMES):
                if self._face_col_rect(i).collidepoint(mx, my):
                    self.face_pending = name
                    return self.hovered_face
            if self.face_pending is not None:
                if self._clear_row_rect().collidepoint(mx, my):
                    self.face_mapper.unassign(self.face_pending)
                    self.face_pending = None
                    self.hovered_face = None
                    return None
                vis = self.window_list[self.scroll_offset:
                                       self.scroll_offset + self._visible_rows()]
                for j, (hwnd, title) in enumerate(vis):
                    if self._win_col_rect(j).collidepoint(mx, my):
                        self.face_mapper.assign(self.face_pending, hwnd, title)
                        self.face_pending = None
                        self.hovered_face = None
                        return None

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self.hovered_face = None
            if self.face_pending is not None:
                vis = self.window_list[self.scroll_offset:
                                       self.scroll_offset + self._visible_rows()]
                for j in range(len(vis)):
                    if self._win_col_rect(j).collidepoint(mx, my):
                        self.hovered_face = self.face_pending
                        break
            else:
                for i, name in enumerate(FACE_NAMES):
                    if self._face_col_rect(i).collidepoint(mx, my):
                        self.hovered_face = name
                        break

        if event.type == pygame.MOUSEWHEEL:
            if self._rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset = max(
                    0,
                    min(max(0, len(self.window_list) - self._visible_rows()),
                        self.scroll_offset - event.y)
                )

        return self.hovered_face

    def draw(self, surface, font):
        if not self.visible:
            return

        self._rect = pygame.Rect(WIDTH - PANEL_W, 0, PANEL_W, HEIGHT)
        bg = pygame.Surface((PANEL_W, HEIGHT), pygame.SRCALPHA)
        bg.fill((10, 10, 30, 210))
        surface.blit(bg, (WIDTH - PANEL_W, 0))
        pygame.draw.rect(surface, (0, 150, 255), self._rect, width=1)

        header = font.render("ASSIGN WINDOWS  [Tab]", True, (0, 200, 255))
        surface.blit(header, (WIDTH - PANEL_W + PANEL_PADDING, 8))

        for i, name in enumerate(FACE_NAMES):
            assigned   = self.face_mapper.assignments.get(name)
            is_pending = (name == self.face_pending)
            is_hover   = (name == self.hovered_face and self.face_pending is None)

            if is_pending:
                color = (255, 200, 0)
            elif assigned:
                color = (0, 230, 100)
            elif is_hover:
                color = (0, 200, 255)
            else:
                color = (160, 160, 160)

            y = 36 + i * PANEL_ROW_H
            surface.blit(font.render(name.upper(), True, color),
                         (WIDTH - PANEL_W + PANEL_PADDING, y + 4))
            if assigned:
                short = assigned[1][:11] + ".." if len(assigned[1]) > 11 else assigned[1]
                surface.blit(font.render(short, True, (80, 80, 80)),
                             (WIDTH - PANEL_W + PANEL_PADDING, y + PANEL_ROW_H // 2 + 4))

        win_x     = WIDTH - PANEL_W + 90 + PANEL_PADDING * 2
        max_chars = (PANEL_W - 90 - PANEL_PADDING * 3) // 10

        # Clear row (always visible above the scrollable list)
        cr = self._clear_row_rect()
        is_clear_hover = cr.collidepoint(pygame.mouse.get_pos())
        clear_col = (255, 100, 100) if (is_clear_hover and self.face_pending) else (80, 80, 80)
        surface.blit(font.render("[ Clear ]", True, clear_col), (cr.x, cr.y + 4))

        vis       = self.window_list[self.scroll_offset:
                                     self.scroll_offset + self._visible_rows()]
        for j, (hwnd, title) in enumerate(vis):
            y     = 36 + PANEL_ROW_H + j * PANEL_ROW_H   # offset for Clear row
            short = title[:max_chars] + ".." if len(title) > max_chars else title
            hover = self._win_col_rect(j).collidepoint(pygame.mouse.get_pos())
            col   = (255, 255, 100) if (hover and self.face_pending) else (200, 200, 200)
            surface.blit(font.render(short, True, col), (win_x, y + 4))

        # ── Tools section ──
        tools_label_y = HEIGHT - PANEL_ROW_H * 3 - PANEL_PADDING * 2 - 100
        surface.blit(
            font.render("── Tools ──", True, (0, 150, 255)),
            (WIDTH - PANEL_W + PANEL_PADDING, tools_label_y)
        )
        wr = self._wireframe_toggle_rect()
        wf_hover   = wr.collidepoint(pygame.mouse.get_pos())
        label_text = "[x] Wireframe" if self.show_wireframe else "[ ] Wireframe"
        label_col  = (0, 230, 100) if self.show_wireframe else (160, 160, 160)
        if wf_hover:
            label_col = (255, 255, 100)
        surface.blit(font.render(label_text, True, label_col), (wr.x, wr.y + 4))

        if len(self.window_list) > self._visible_rows():
            hint = font.render(
                f"scroll {self.scroll_offset + 1}/{len(self.window_list)}",
                True, (80, 80, 80)
            )
            surface.blit(hint, (WIDTH - PANEL_W + PANEL_PADDING, HEIGHT - 24))


# =================================================================
# 8. SLIDER UI (STACKED: FOV + DEPTH)
# =================================================================
def _clamp01(t):
    return max(0.0, min(1.0, t))


def draw_slider(surface, value, vmin, vmax, slider_y):
    slider_x = WIDTH // 2 - SLIDER_WIDTH // 2

    t = (value - vmin) / float(vmax - vmin)
    t = _clamp01(t)

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

    return slider_x, knob_x, slider_y


def point_on_knob(mx, my, knob_x, knob_y):
    dx = mx - knob_x
    dy = my - knob_y
    return (dx*dx + dy*dy) <= (KNOB_RADIUS * KNOB_RADIUS)


def value_from_mouse_x(mx, vmin, vmax):
    slider_x = WIDTH // 2 - SLIDER_WIDTH // 2
    t = (mx - slider_x) / float(SLIDER_WIDTH)
    t = _clamp01(t)
    return vmin + t * (vmax - vmin)


# =================================================================
# 5. RESET BUTTON
# =================================================================
RESET_BTN_W = 120
RESET_BTN_H = 32

def draw_reset_button(surface, font):
    btn_x = WIDTH // 2 - RESET_BTN_W // 2
    btn_y = HEIGHT - SLIDER_Y_OFFSET - SLIDER_GAP * 2 - RESET_BTN_H - 10
    rect = pygame.Rect(btn_x, btn_y, RESET_BTN_W, RESET_BTN_H)
    pygame.draw.rect(surface, (30, 30, 60), rect, border_radius=6)
    pygame.draw.rect(surface, (0, 150, 255), rect, width=2, border_radius=6)
    label = font.render("Reset", True, (200, 200, 200))
    surface.blit(label, (btn_x + RESET_BTN_W // 2 - label.get_width() // 2,
                         btn_y + RESET_BTN_H // 2 - label.get_height() // 2))
    return rect


# =================================================================
# 6. MAIN LOOP
# =================================================================
def main():
    global WIDTH, HEIGHT, SLIDER_WIDTH
    pygame.init()
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    SLIDER_WIDTH = int(WIDTH * 0.4)  # ~40% of screen width, same ratio as 400/1000
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("3D Full Grid Parallax")
    hwnd = pygame.display.get_wm_info()["window"]
    ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 22)

    tracker    = HeadTracking()
    face_mapper = FaceMapper()
    panel_ui    = WindowPanelUI(face_mapper)

    smooth_hx, smooth_hy = 0.0, 0.0
    room_depth = ROOM_DEPTH
    vertical_fov = VERTICAL_FOV

    # Calibration offset — set on reset so current face position becomes new center
    calib_x, calib_y = 0.0, 0.0

    dragging_depth = False
    dragging_fov = False
    reset_btn_rect = pygame.Rect(0, 0, RESET_BTN_W, RESET_BTN_H)

    running = True
    while running:
        clock.tick(FPS)

        # Slider Y positions (stacked)
        depth_slider_y = HEIGHT - SLIDER_Y_OFFSET
        fov_slider_y = depth_slider_y - SLIDER_GAP

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    running = False
                if event.key == pygame.K_TAB:
                    panel_ui.toggle()

            panel_ui.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Draw once to get knob positions for hit testing
                _, depth_knob_x, depth_knob_y = draw_slider(screen, room_depth, MIN_DEPTH, MAX_DEPTH, depth_slider_y)
                _, fov_knob_x, fov_knob_y = draw_slider(screen, vertical_fov, MIN_FOV, MAX_FOV, fov_slider_y)

                if point_on_knob(mx, my, depth_knob_x, depth_knob_y):
                    dragging_depth = True
                    dragging_fov = False
                elif point_on_knob(mx, my, fov_knob_x, fov_knob_y):
                    dragging_fov = True
                    dragging_depth = False
                elif reset_btn_rect.collidepoint(mx, my):
                    # Store current face position as the new center
                    calib_x = tracker.head_x
                    calib_y = tracker.head_y
                    smooth_hx, smooth_hy = 0.0, 0.0

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_depth = False
                dragging_fov = False

            if event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()

                if dragging_depth:
                    room_depth = value_from_mouse_x(mx, MIN_DEPTH, MAX_DEPTH)

                if dragging_fov:
                    vertical_fov = value_from_mouse_x(mx, MIN_FOV, MAX_FOV)

        # Smooth tracking — subtract calibration offset so reset position = center
        target_hx = (tracker.head_x - calib_x) * BOX_W * SENSITIVITY * 0.1
        target_hy = (tracker.head_y - calib_y) * BOX_H * SENSITIVITY * 0.1

        smooth_hx += (target_hx - smooth_hx) * SMOOTHING
        smooth_hy += (target_hy - smooth_hy) * SMOOTHING

        # Clamp camera inside the box
        smooth_hx = max(-BOX_W, min(BOX_W, smooth_hx))
        smooth_hy = max(-BOX_H, min(BOX_H, smooth_hy))

        # Draw scene
        screen.fill(COLOR_BG)
        face_mapper.draw_faces(screen, smooth_hx, smooth_hy, room_depth, vertical_fov)
        if panel_ui.show_wireframe:
            draw_full_grid(screen, smooth_hx, smooth_hy, room_depth, vertical_fov)
        panel_ui.draw(screen, font)

        # Draw sliders (FOV above Depth)
        draw_slider(screen, vertical_fov, MIN_FOV, MAX_FOV, fov_slider_y)
        draw_slider(screen, room_depth, MIN_DEPTH, MAX_DEPTH, depth_slider_y)

        # Draw reset button and store its rect for hit testing
        reset_btn_rect = draw_reset_button(screen, font)

        # Labels
        fov_text = font.render(f"Vertical FOV: {vertical_fov:.1f}", True, (200, 200, 200))
        screen.blit(fov_text, (WIDTH // 2 - fov_text.get_width() // 2, fov_slider_y - 30))

        depth_text = font.render(f"Depth: {room_depth:.1f}", True, (200, 200, 200))
        screen.blit(depth_text, (WIDTH // 2 - depth_text.get_width() // 2, depth_slider_y - 30))

        if not tracker.detected:
            text = font.render("NOT DETECTED. Check camera.", True, (255, 50, 50))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 220))

        pygame.display.flip()

    face_mapper.stop_all()
    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
