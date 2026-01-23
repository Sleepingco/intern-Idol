import numpy as np
from OpenGL.GL import (
    glGenTextures,
    glBindTexture,
    glTexParameteri,
    glActiveTexture,
    glTexImage2D,
    glUseProgram,
    glGetUniformLocation,
    glUniform1i,
    glDisable,
    glEnable,
    glDrawElements,
    GL_TEXTURE_2D,
    GL_LINEAR,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE,
    GL_TEXTURE0,
    GL_TEXTURE7,
    GL_TEXTURE8,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GL_R8,
    GL_RED,
    GL_DEPTH_TEST,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
)


def setup_bg_composite_textures():
    """Create textures for GPU background compositing (bg image + person mask)."""
    # Background image texture
    tex_bgimg = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_bgimg)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Person mask texture (R8)
    tex_bgmask = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_bgmask)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    return tex_bgimg, tex_bgmask


def render_bg_composite(
    prog_bg,
    vao_bg,
    tex_fg,
    tex_bgimg,
    tex_bgmask,
    frame_rgb,
    bg_frame_rgb,
    mask_u8,
    W,
    H,
):
    """Composite background on the GPU using FS_BG shader and draw full-screen quad."""
    # Upload FG (camera) to TEX0
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_fg)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    # Upload BG image to TEX8
    glActiveTexture(GL_TEXTURE8)
    glBindTexture(GL_TEXTURE_2D, tex_bgimg)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, bg_frame_rgb)

    # Upload mask to TEX7 (R8)
    glActiveTexture(GL_TEXTURE7)
    glBindTexture(GL_TEXTURE_2D, tex_bgmask)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0, GL_RED, GL_UNSIGNED_BYTE, mask_u8)

    glUseProgram(prog_bg)
    glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 1)
    glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
    glUniform1i(glGetUniformLocation(prog_bg, "uTexBG"), 8)
    glUniform1i(glGetUniformLocation(prog_bg, "uMask"), 7)

    # vao_bg and draw call handled by caller to keep coupling low
    glDisable(GL_DEPTH_TEST)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glEnable(GL_DEPTH_TEST)

