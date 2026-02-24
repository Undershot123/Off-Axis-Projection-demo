# ui.py — WindowPanelUI, sliders, and reset button.

import time
import pygame

import config
from faces import FACE_NAMES
from capture import enumerate_windows


# =============================================================================
# Slider helpers
# =============================================================================

def _clamp01(t):
    return max(0.0, min(1.0, t))


def draw_slider(surface, value, vmin, vmax, slider_y):
    """Draw a horizontal slider and return (slider_x, knob_x, knob_y)."""
    slider_x = config.WIDTH // 2 - config.SLIDER_WIDTH // 2

    t = _clamp01((value - vmin) / float(vmax - vmin))
    knob_x = slider_x + int(t * config.SLIDER_WIDTH)
    knob_y = slider_y

    pygame.draw.rect(surface, (80, 80, 80),
                     (slider_x, slider_y - config.SLIDER_HEIGHT // 2,
                      config.SLIDER_WIDTH, config.SLIDER_HEIGHT))
    pygame.draw.rect(surface, (0, 180, 255),
                     (slider_x, slider_y - config.SLIDER_HEIGHT // 2,
                      int(t * config.SLIDER_WIDTH), config.SLIDER_HEIGHT))
    pygame.draw.circle(surface, (255, 255, 255), (knob_x, knob_y), config.KNOB_RADIUS)

    return slider_x, knob_x, knob_y


def point_on_knob(mx, my, knob_x, knob_y):
    dx, dy = mx - knob_x, my - knob_y
    return (dx * dx + dy * dy) <= (config.KNOB_RADIUS * config.KNOB_RADIUS)


def value_from_mouse_x(mx, vmin, vmax):
    slider_x = config.WIDTH // 2 - config.SLIDER_WIDTH // 2
    t = _clamp01((mx - slider_x) / float(config.SLIDER_WIDTH))
    return vmin + t * (vmax - vmin)


# =============================================================================
# Reset button
# =============================================================================

def draw_reset_button(surface, font):
    btn_x = config.WIDTH  // 2 - config.RESET_BTN_W // 2
    btn_y = (config.HEIGHT - config.SLIDER_Y_OFFSET
             - config.SLIDER_GAP * 2 - config.RESET_BTN_H - 10)
    rect = pygame.Rect(btn_x, btn_y, config.RESET_BTN_W, config.RESET_BTN_H)
    pygame.draw.rect(surface, (30, 30, 60), rect, border_radius=6)
    pygame.draw.rect(surface, (0, 150, 255), rect, width=2, border_radius=6)
    label = font.render("Reset", True, (200, 200, 200))
    surface.blit(label, (btn_x + config.RESET_BTN_W // 2 - label.get_width()  // 2,
                         btn_y + config.RESET_BTN_H // 2 - label.get_height() // 2))
    return rect


# =============================================================================
# Window panel UI
# =============================================================================

class WindowPanelUI:
    REFRESH_INTERVAL = 3.0

    def __init__(self, face_mapper):
        self.face_mapper    = face_mapper
        self.visible        = False
        self.window_list    = []
        self._last_refresh  = 0.0
        self.face_pending   = None
        self.hovered_face   = None
        self.scroll_offset  = 0
        self._rect          = pygame.Rect(0, 0, config.PANEL_W, 0)
        self.show_wireframe   = True
        self.show_perspective = True

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._refresh()

    def _refresh(self):
        self.window_list   = enumerate_windows()
        self._last_refresh = time.time()

    def _visible_rows(self):
        return max(1, (config.HEIGHT - 60) // config.PANEL_ROW_H - 1)

    def _face_col_rect(self, i):
        return pygame.Rect(
            config.WIDTH - config.PANEL_W + config.PANEL_PADDING,
            36 + i * config.PANEL_ROW_H,
            90, config.PANEL_ROW_H - 2
        )

    def _clear_row_rect(self):
        return pygame.Rect(
            config.WIDTH - config.PANEL_W + 90 + config.PANEL_PADDING * 2,
            36,
            config.PANEL_W - 90 - config.PANEL_PADDING * 3,
            config.PANEL_ROW_H - 2
        )

    def _win_col_rect(self, j):
        return pygame.Rect(
            config.WIDTH - config.PANEL_W + 90 + config.PANEL_PADDING * 2,
            36 + config.PANEL_ROW_H + j * config.PANEL_ROW_H,
            config.PANEL_W - 90 - config.PANEL_PADDING * 3,
            config.PANEL_ROW_H - 2
        )

    def _perspective_toggle_rect(self):
        return pygame.Rect(
            config.WIDTH - config.PANEL_W + config.PANEL_PADDING,
            config.HEIGHT - config.PANEL_ROW_H * 3 - config.PANEL_PADDING - 100,
            config.PANEL_W - config.PANEL_PADDING * 2,
            config.PANEL_ROW_H - 2
        )

    def _wireframe_toggle_rect(self):
        return pygame.Rect(
            config.WIDTH - config.PANEL_W + config.PANEL_PADDING,
            config.HEIGHT - config.PANEL_ROW_H * 2 - config.PANEL_PADDING - 100,
            config.PANEL_W - config.PANEL_PADDING * 2,
            config.PANEL_ROW_H - 2
        )

    def handle_event(self, event):
        if not self.visible:
            self.hovered_face = None
            return None

        if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
            self._refresh()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self._perspective_toggle_rect().collidepoint(mx, my):
                self.show_perspective = not self.show_perspective
                return self.hovered_face
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

        self._rect = pygame.Rect(config.WIDTH - config.PANEL_W, 0,
                                 config.PANEL_W, config.HEIGHT)
        bg = pygame.Surface((config.PANEL_W, config.HEIGHT), pygame.SRCALPHA)
        bg.fill((10, 10, 30, 210))
        surface.blit(bg, (config.WIDTH - config.PANEL_W, 0))
        pygame.draw.rect(surface, (0, 150, 255), self._rect, width=1)

        header = font.render("ASSIGN WINDOWS  [Tab]", True, (0, 200, 255))
        surface.blit(header, (config.WIDTH - config.PANEL_W + config.PANEL_PADDING, 8))

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

            y = 36 + i * config.PANEL_ROW_H
            surface.blit(font.render(name.upper(), True, color),
                         (config.WIDTH - config.PANEL_W + config.PANEL_PADDING, y + 4))
            if assigned:
                short = assigned[1][:11] + ".." if len(assigned[1]) > 11 else assigned[1]
                surface.blit(font.render(short, True, (80, 80, 80)),
                             (config.WIDTH - config.PANEL_W + config.PANEL_PADDING,
                              y + config.PANEL_ROW_H // 2 + 4))

        win_x     = config.WIDTH - config.PANEL_W + 90 + config.PANEL_PADDING * 2
        max_chars = (config.PANEL_W - 90 - config.PANEL_PADDING * 3) // 10

        cr = self._clear_row_rect()
        is_clear_hover = cr.collidepoint(pygame.mouse.get_pos())
        clear_col = (255, 100, 100) if (is_clear_hover and self.face_pending) else (80, 80, 80)
        surface.blit(font.render("[ Clear ]", True, clear_col), (cr.x, cr.y + 4))

        vis = self.window_list[self.scroll_offset:
                               self.scroll_offset + self._visible_rows()]
        for j, (hwnd, title) in enumerate(vis):
            y     = 36 + config.PANEL_ROW_H + j * config.PANEL_ROW_H
            short = title[:max_chars] + ".." if len(title) > max_chars else title
            hover = self._win_col_rect(j).collidepoint(pygame.mouse.get_pos())
            col   = (255, 255, 100) if (hover and self.face_pending) else (200, 200, 200)
            surface.blit(font.render(short, True, col), (win_x, y + 4))

        # Tools section
        tools_label_y = (config.HEIGHT - config.PANEL_ROW_H * 4
                         - config.PANEL_PADDING * 2 - 100)
        surface.blit(
            font.render("── Tools ──", True, (0, 150, 255)),
            (config.WIDTH - config.PANEL_W + config.PANEL_PADDING, tools_label_y)
        )

        pr      = self._perspective_toggle_rect()
        p_hover = pr.collidepoint(pygame.mouse.get_pos())
        p_text  = "[x] Perspective" if self.show_perspective else "[ ] Perspective"
        p_col   = (0, 230, 100) if self.show_perspective else (160, 160, 160)
        if p_hover:
            p_col = (255, 255, 100)
        surface.blit(font.render(p_text, True, p_col), (pr.x, pr.y + 4))

        wr       = self._wireframe_toggle_rect()
        wf_hover = wr.collidepoint(pygame.mouse.get_pos())
        wf_text  = "[x] Wireframe" if self.show_wireframe else "[ ] Wireframe"
        wf_col   = (0, 230, 100) if self.show_wireframe else (160, 160, 160)
        if wf_hover:
            wf_col = (255, 255, 100)
        surface.blit(font.render(wf_text, True, wf_col), (wr.x, wr.y + 4))

        if len(self.window_list) > self._visible_rows():
            hint = font.render(
                f"scroll {self.scroll_offset + 1}/{len(self.window_list)}",
                True, (80, 80, 80)
            )
            surface.blit(hint, (config.WIDTH - config.PANEL_W + config.PANEL_PADDING,
                                config.HEIGHT - 24))
