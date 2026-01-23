# module/sticker3d/renderer.py
import numpy as np
import cv2
import pygame
import ctypes
from pygame.locals import *
from OpenGL.GL import *

import config

def load_opengl_texture(image_path):
    texture_surface = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if texture_surface is None:
        raise FileNotFoundError(f"Texture file not found: {image_path}")
    if texture_surface.shape[2] == 3:
        texture_surface = cv2.cvtColor(texture_surface, cv2.COLOR_BGR2BGRA)
    
    # RGBA 순서로 변환
    texture_surface = cv2.cvtColor(texture_surface, cv2.COLOR_BGRA2RGBA)

    texture_data = texture_surface.tobytes()
    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_surface.shape[1], texture_surface.shape[0],
                 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return texid

def ortho(l, r, b, t, n, f):
    return np.array([
        [2/(r-l), 0, 0, -(r+l)/(r-l)],
        [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, -2/(f-n), -(f+n)/(f-n)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

class Sticker3DRenderer:
    def __init__(self, camera_width: int, camera_height: int, shader_program):
        self.cam_w, self.cam_h = camera_width, camera_height
        self.shader = shader_program

        uv_per_vertex = np.genfromtxt(config.UV_CSV, delimiter=',')
        self.uv_per_vertex = uv_per_vertex[:468].astype(np.float32)

        tri_indices = np.genfromtxt(config.TRI_CSV, delimiter=',', dtype=np.int32)
        self.tri_indices = tri_indices.astype(np.int32)
        self.makeup_tex = load_opengl_texture(config.PATTERN_RGBA)

        self.vbo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # 위치(2) + UV(2) = 4 floats
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

    def draw_face_triangles(self, pts_all_px):
        if not config.UV_STICKER_ENABLE or pts_all_px is None:
            return

        pts_screen = pts_all_px[:468, :2].copy()
        pts_screen[:, 1] = self.cam_h - pts_screen[:, 1]

        uv_coords = self.uv_per_vertex
        idx = self.tri_indices.flatten()
        vertex_pos = pts_screen[idx]
        vertex_uv = uv_coords[idx]
        
        buffer_data = np.hstack((vertex_pos, vertex_uv)).astype(np.float32).flatten()

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, buffer_data.nbytes, buffer_data, GL_STREAM_DRAW)

        projection = ortho(0, self.cam_w, 0, self.cam_h, -1, 1)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uOrtho"), 1, GL_TRUE, projection)
        glUniform1i(glGetUniformLocation(self.shader, "uTexture"), 0)
        glUniform1f(glGetUniformLocation(self.shader, "uAlpha"), config.ALPHA_GAIN)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.makeup_tex)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glDrawArrays(GL_TRIANGLES, 0, len(idx))

        glDepthMask(GL_TRUE)
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
