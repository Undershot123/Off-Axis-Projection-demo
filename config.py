# config.py — all application constants.
#
# WIDTH, HEIGHT, and SLIDER_WIDTH are overridden at runtime by main.py:
#   config.WIDTH  = info.current_w
#   config.HEIGHT = info.current_h
#   config.SLIDER_WIDTH = int(config.WIDTH * 0.4)
#
# Every other module must do:
#   import config          (never "from config import WIDTH")
#   config.WIDTH / 2       (always reads the live runtime value)

# --- Display (runtime-mutable) ---
WIDTH  = 1000
HEIGHT = 700

# --- Colors ---
COLOR_BG  = (10, 10, 20)
NEON_BLUE = (0, 150, 255)

# --- Vertical FOV ---
VERTICAL_FOV = 165.0
MIN_FOV      = 150.0
MAX_FOV      = 170.0

# --- Performance ---
FPS         = 60
SENSITIVITY = 20
SMOOTHING   = 0.2

# --- World depth ---
ROOM_DEPTH = 50.0
MIN_DEPTH  = 25.0
MAX_DEPTH  = 100.0

# --- World geometry ---
BOX_W = 6.0
BOX_H = 3.5

# --- MediaPipe ---
MODEL_PATH = "face_landmarker.task"

# --- Window panel UI ---
PANEL_W       = 320
PANEL_PADDING = 10
PANEL_ROW_H   = 28
CAPTURE_FPS   = 15

# --- Slider UI (SLIDER_WIDTH is runtime-mutable) ---
SLIDER_WIDTH   = 400   # overridden at startup
SLIDER_HEIGHT  = 8
SLIDER_Y_OFFSET = 90   # distance from bottom for the bottom slider
SLIDER_GAP     = 55    # vertical gap between stacked sliders
KNOB_RADIUS    = 10

# --- Reset button ---
RESET_BTN_W = 120
RESET_BTN_H = 32
