import pygame
from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBlendFunc,
    GL_DEPTH_TEST,
    GL_CULL_FACE,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)


def compute_view_rect(screen_w, screen_h, W, H):
    cam_aspect = W / H
    screen_aspect = screen_w / screen_h
    if screen_aspect > cam_aspect:
        view_h = screen_h
        view_w = int(view_h * cam_aspect)
        view_x = (screen_w - view_w) // 2
        view_y = 0
    else:
        view_w = screen_w
        view_h = int(view_w / cam_aspect)
        view_x = 0
        view_y = (screen_h - view_h) // 2
    return view_x, view_y, view_w, view_h


def apply_display_mode(fullscreen, last_view_size, W, H):
    """Switch display between fullscreen and windowed based on current mode.
    Returns: (screen_w, screen_h, view_x, view_y, view_w, view_h, new_last_view)
    """
    if fullscreen:
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        pygame.display.set_mode((screen_w, screen_h), DOUBLEBUF | OPENGL | FULLSCREEN)
        vx, vy, vw, vh = compute_view_rect(screen_w, screen_h, W, H)
        new_last_view = (vw, vh)
    else:
        win_w, win_h = last_view_size
        pygame.display.set_mode((int(win_w), int(win_h)), DOUBLEBUF | OPENGL)
        screen_w, screen_h = int(win_w), int(win_h)
        vx, vy, vw, vh = compute_view_rect(screen_w, screen_h, W, H)
        new_last_view = (vw, vh)

    # Reset GL state
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return screen_w, screen_h, vx, vy, vw, vh, new_last_view

