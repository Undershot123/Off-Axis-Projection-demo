import ctypes
import pygame

import config
from tracking  import HeadTracking
from faces     import FaceMapper, FACE_NAMES
from projection import draw_full_grid
from ui        import (WindowPanelUI, draw_slider, point_on_knob,
                       value_from_mouse_x, draw_reset_button)


def main():
    pygame.init()

    # DPI awareness + measure screen before creating the window
    ctypes.windll.user32.SetProcessDPIAware()
    info = pygame.display.Info()
    config.WIDTH        = info.current_w
    config.HEIGHT       = info.current_h
    config.SLIDER_WIDTH = int(config.WIDTH * 0.4)

    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("3D Full Grid Parallax")

    hwnd = pygame.display.get_wm_info()["window"]
    ctypes.windll.user32.ShowWindow(hwnd, 3)   # SW_MAXIMIZE

    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Consolas", 22)

    tracker     = HeadTracking()
    face_mapper = FaceMapper()
    panel_ui    = WindowPanelUI(face_mapper)

    smooth_hx, smooth_hy = 0.0, 0.0
    room_depth   = config.ROOM_DEPTH
    vertical_fov = config.VERTICAL_FOV
    calib_x, calib_y = 0.0, 0.0

    dragging_depth = False
    dragging_fov   = False

    # Knob positions cached from the draw phase and used in the event phase
    depth_knob_x = depth_knob_y = 0
    fov_knob_x   = fov_knob_y   = 0
    reset_btn_rect = pygame.Rect(0, 0, config.RESET_BTN_W, config.RESET_BTN_H)

    running = True
    while running:
        clock.tick(config.FPS)

        depth_slider_y = config.HEIGHT - config.SLIDER_Y_OFFSET
        fov_slider_y   = depth_slider_y - config.SLIDER_GAP

        # ----------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    running = False
                if event.key == pygame.K_TAB:
                    panel_ui.toggle()

            panel_ui.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if point_on_knob(mx, my, depth_knob_x, depth_knob_y):
                    dragging_depth = True
                    dragging_fov   = False
                elif panel_ui.show_perspective:
                    if point_on_knob(mx, my, fov_knob_x, fov_knob_y):
                        dragging_fov   = True
                        dragging_depth = False
                    elif reset_btn_rect.collidepoint(mx, my):
                        calib_x, calib_y = tracker.head_x, tracker.head_y
                        smooth_hx, smooth_hy = 0.0, 0.0
                elif reset_btn_rect.collidepoint(mx, my):
                    calib_x, calib_y = tracker.head_x, tracker.head_y
                    smooth_hx, smooth_hy = 0.0, 0.0

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_depth = False
                dragging_fov   = False

            if event.type == pygame.MOUSEMOTION:
                mx, _ = event.pos
                if dragging_depth:
                    room_depth   = value_from_mouse_x(mx, config.MIN_DEPTH, config.MAX_DEPTH)
                if dragging_fov:
                    vertical_fov = value_from_mouse_x(mx, config.MIN_FOV, config.MAX_FOV)

            if event.type == pygame.VIDEORESIZE:
                config.WIDTH        = event.w
                config.HEIGHT       = event.h
                config.SLIDER_WIDTH = int(config.WIDTH * 0.4)
                screen = pygame.display.set_mode(
                    (config.WIDTH, config.HEIGHT), pygame.RESIZABLE)

        # ----------------------------------------------------------------
        # Tracking
        # ----------------------------------------------------------------
        target_hx = (tracker.head_x - calib_x) * config.BOX_W * config.SENSITIVITY * 0.1
        target_hy = (tracker.head_y - calib_y) * config.BOX_H * config.SENSITIVITY * 0.1
        smooth_hx += (target_hx - smooth_hx) * config.SMOOTHING
        smooth_hy += (target_hy - smooth_hy) * config.SMOOTHING
        smooth_hx  = max(-config.BOX_W, min(config.BOX_W, smooth_hx))
        smooth_hy  = max(-config.BOX_H, min(config.BOX_H, smooth_hy))

        # ----------------------------------------------------------------
        # Draw
        # ----------------------------------------------------------------
        screen.fill(config.COLOR_BG)

        face_mapper.draw_faces(screen, smooth_hx, -smooth_hy, room_depth, vertical_fov)

        if panel_ui.show_wireframe:
            draw_full_grid(screen, smooth_hx, -smooth_hy, room_depth, vertical_fov)

        panel_ui.draw(screen, font)

        if panel_ui.show_perspective:
            _, fov_knob_x,   fov_knob_y   = draw_slider(
                screen, vertical_fov, config.MIN_FOV, config.MAX_FOV, fov_slider_y)
            reset_btn_rect = draw_reset_button(screen, font)

            fov_label = font.render(f"Vertical FOV: {vertical_fov:.1f}", True, (200, 200, 200))
            screen.blit(fov_label,
                        (config.WIDTH // 2 - fov_label.get_width() // 2, fov_slider_y - 30))

        _, depth_knob_x, depth_knob_y = draw_slider(
            screen, room_depth, config.MIN_DEPTH, config.MAX_DEPTH, depth_slider_y)

        depth_label = font.render(f"Depth: {room_depth:.1f}", True, (200, 200, 200))
        screen.blit(depth_label,
                    (config.WIDTH // 2 - depth_label.get_width() // 2, depth_slider_y - 30))

        if not tracker.detected:
            warn = font.render("NOT DETECTED. Check camera.", True, (255, 50, 50))
            screen.blit(warn, (config.WIDTH // 2 - warn.get_width() // 2,
                               config.HEIGHT - 220))

        pygame.display.flip()

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    face_mapper.stop_all()
    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
