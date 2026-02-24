# projection.py — 3D geometry types and off-axis projection math.

import numpy as np
import pygame

import config


class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _make_face_corners(depth):
    """Returns dict: face name -> [tl, tr, br, bl] as Point3D world coords."""
    d = depth
    return {
        "back":   [Point3D(-config.BOX_W,  config.BOX_H, d),
                   Point3D( config.BOX_W,  config.BOX_H, d),
                   Point3D( config.BOX_W, -config.BOX_H, d),
                   Point3D(-config.BOX_W, -config.BOX_H, d)],
        # "top"/"bottom" labels match user expectation (Y-axis inverted on screen)
        "top":    [Point3D(-config.BOX_W, -config.BOX_H, 0),
                   Point3D( config.BOX_W, -config.BOX_H, 0),
                   Point3D( config.BOX_W, -config.BOX_H, d),
                   Point3D(-config.BOX_W, -config.BOX_H, d)],
        "bottom": [Point3D(-config.BOX_W,  config.BOX_H, 0),
                   Point3D( config.BOX_W,  config.BOX_H, 0),
                   Point3D( config.BOX_W,  config.BOX_H, d),
                   Point3D(-config.BOX_W,  config.BOX_H, d)],
        "left":   [Point3D(-config.BOX_W,  config.BOX_H, 0),
                   Point3D(-config.BOX_W, -config.BOX_H, 0),
                   Point3D(-config.BOX_W, -config.BOX_H, d),
                   Point3D(-config.BOX_W,  config.BOX_H, d)],
        "right":  [Point3D( config.BOX_W,  config.BOX_H, 0),
                   Point3D( config.BOX_W, -config.BOX_H, 0),
                   Point3D( config.BOX_W, -config.BOX_H, d),
                   Point3D( config.BOX_W,  config.BOX_H, d)],
    }


def project_off_axis(p, head_x, head_y, fov):
    """
    True off-axis projection (CPU path).
    Eye sits at (head_x, head_y) at focal distance f behind the screen plane (z=0).
    Returns (pixel_x, pixel_y) or None if the point is behind the camera.
    """
    f = (config.HEIGHT / 2) / np.tan(np.radians(fov / 2))

    total_depth = p.z + 0.0001
    if total_depth <= 0:
        return None

    ratio    = f / (f + total_depth)
    screen_x = head_x + (p.x - head_x) * ratio
    screen_y = head_y + (p.y - head_y) * ratio

    pixel_x = int(config.WIDTH  / 2 + screen_x * (config.WIDTH  / 2) / config.BOX_W)
    pixel_y = int(config.HEIGHT / 2 + screen_y * (config.HEIGHT / 2) / config.BOX_H)
    return (pixel_x, pixel_y)


def draw_full_grid(surface, hx, hy, depth, fov):
    """Draw the neon-blue 3D box wireframe onto a pygame surface."""
    w_room = config.BOX_W
    h_room = config.BOX_H
    grid_spacing = 2.0

    # Longitudinal lines (along Z)
    for x in np.arange(-w_room, w_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(x, -h_room, 0),     hx, hy, fov)
        p2 = project_off_axis(Point3D(x, -h_room, depth), hx, hy, fov)
        p3 = project_off_axis(Point3D(x,  h_room, 0),     hx, hy, fov)
        p4 = project_off_axis(Point3D(x,  h_room, depth), hx, hy, fov)
        if p1 and p2:
            pygame.draw.line(surface, config.NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, config.NEON_BLUE, p3, p4, 1)

    # Horizontal lines (along Z)
    for y in np.arange(-h_room, h_room + 0.1, grid_spacing):
        p1 = project_off_axis(Point3D(-w_room, y, 0),     hx, hy, fov)
        p2 = project_off_axis(Point3D(-w_room, y, depth), hx, hy, fov)
        p3 = project_off_axis(Point3D( w_room, y, 0),     hx, hy, fov)
        p4 = project_off_axis(Point3D( w_room, y, depth), hx, hy, fov)
        if p1 and p2:
            pygame.draw.line(surface, config.NEON_BLUE, p1, p2, 1)
        if p3 and p4:
            pygame.draw.line(surface, config.NEON_BLUE, p3, p4, 1)

    # Depth slices (rectangles at constant Z)
    for z in np.arange(0.0, depth + 0.1, grid_spacing):
        tl = project_off_axis(Point3D(-w_room,  h_room, z), hx, hy, fov)
        tr = project_off_axis(Point3D( w_room,  h_room, z), hx, hy, fov)
        br = project_off_axis(Point3D( w_room, -h_room, z), hx, hy, fov)
        bl = project_off_axis(Point3D(-w_room, -h_room, z), hx, hy, fov)
        if tl and tr and br and bl:
            pygame.draw.line(surface, config.NEON_BLUE, tl, tr, 1)
            pygame.draw.line(surface, config.NEON_BLUE, bl, br, 1)
            pygame.draw.line(surface, config.NEON_BLUE, tl, bl, 1)
            pygame.draw.line(surface, config.NEON_BLUE, tr, br, 1)
