# faces.py — face definitions, source UV mappings, and FaceMapper.

import numpy as np
import cv2
import pygame

import config
from projection import Point3D, _make_face_corners, project_off_axis
from capture import WindowCapture


FACE_NAMES = ["back", "top", "bottom", "left", "right"]

# Per-face source corner mappings for warpPerspective.
# Lambdas take (w, h) of the native frame so no resize is needed.
# back/top/bottom: vertical flip to correct Y-axis inversion.
# left/right: 90° rotation + vertical flip.
_FACE_SRC_PTS = {
    "back":   lambda w, h: np.float32([[0, h], [w, h], [w, 0], [0, 0]]),
    "top":    lambda w, h: np.float32([[0, h], [w, h], [w, 0], [0, 0]]),
    "bottom": lambda w, h: np.float32([[0, h], [w, h], [w, 0], [0, 0]]),
    "left":   lambda w, h: np.float32([[0, h], [0, 0], [w, 0], [w, h]]),
    "right":  lambda w, h: np.float32([[w, h], [w, 0], [0, 0], [0, h]]),
}


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
        h_f, w_f = frame_bgr.shape[:2]
        src_pts  = src_pts_fn(w_f, h_f) if src_pts_fn is not None \
                   else np.float32([[0, 0], [w_f, 0], [w_f, h_f], [0, h_f]])
        dst_pts  = np.float32(screen_pts)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            frame_bgr, M, (config.WIDTH, config.HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        # Punch out exact-black border pixels so faces don't overwrite each other
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2RGBA)
        mask = (warped[:, :, 0] == 0) & (warped[:, :, 1] == 0) & (warped[:, :, 2] == 0)
        warped_rgba[mask, 3] = 0
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(warped_rgba).tobytes(),
            (config.WIDTH, config.HEIGHT), "RGBA"
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
