# rendering.py - 렌더링 관련 함수들
import numpy as np
import ctypes
from OpenGL.GL import *
from config import MAX_BONES
from utils import align_y_to_vec
import config


def _extract_position_matrix(M_hat_shared: np.ndarray) -> np.ndarray:
    """M_hat_shared에서 위치(translation)만 유지한 4x4 행렬 생성"""
    M_position_only = np.eye(4, dtype=np.float32)
    M_position_only[:3, 3] = M_hat_shared[:3, 3]
    return M_position_only


def setup_background_vao():
    """배경 렌더링용 VAO 설정"""
    quad = np.array([-1, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 0, -1, 1, 0, 0], np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    vao_bg = glGenVertexArrays(1)
    glBindVertexArray(vao_bg)
    vbo_bg, ebo_bg = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bg)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_bg)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    glBindVertexArray(0)
    return vao_bg


def setup_textures():
    """배경, 턱, 세그멘테이션 텍스처 설정"""
    tex_bg = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_bg)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    tex_jaw = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_jaw)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    tex_seg = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_seg)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glActiveTexture(GL_TEXTURE0)

    return tex_bg, tex_jaw, tex_seg


def setup_obj_vao(obj_verts):
    """OBJ 메쉬용 VAO 설정"""
    vao_hat = glGenVertexArrays(1)
    glBindVertexArray(vao_hat)
    vbo_hat = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_hat)
    glBufferData(GL_ARRAY_BUFFER, obj_verts.nbytes, obj_verts, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    glBindVertexArray(0)
    return vao_hat


def setup_mesh_vao(mesh_verts):
    """일반 메쉬용 VAO 설정"""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, mesh_verts.nbytes, mesh_verts, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return vao


def upload_texture_2d(
    tex_id,
    data,
    width,
    height,
    format_gl=GL_RGB,
    internal_format=GL_RGB,
    data_type=GL_UNSIGNED_BYTE,
):
    """2D 텍스처 업로드"""
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D, 0, internal_format, width, height, 0, format_gl, data_type, data
    )


def upload_jaw_mask(tex_jaw, jaw_mask_u8, W, H):
    """턱 마스크 텍스처 업로드"""
    glActiveTexture(GL_TEXTURE4)
    glBindTexture(GL_TEXTURE_2D, tex_jaw)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    if jaw_mask_u8 is None:
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            W,
            H,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            np.zeros((H, W), np.uint8),
        )
    else:
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            W,
            H,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            np.flipud(jaw_mask_u8),
        )
    glActiveTexture(GL_TEXTURE0)


def upload_seg_mask(tex_seg, seg_mask_u8, W, H):
    """세그멘테이션 마스크 텍스처 업로드"""
    glActiveTexture(GL_TEXTURE5)
    glBindTexture(GL_TEXTURE_2D, tex_seg)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    if seg_mask_u8 is None:
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            W,
            H,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            np.zeros((H, W), np.uint8),
        )
    else:
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            W,
            H,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            np.flipud(seg_mask_u8),
        )
    glActiveTexture(GL_TEXTURE0)


def render_background(prog_bg, vao_bg, tex_bg, frame_rgb, W, H):
    """배경 렌더링"""
    glBindTexture(GL_TEXTURE_2D, tex_bg)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    glUseProgram(prog_bg)
    glBindVertexArray(vao_bg)
    glDisable(GL_DEPTH_TEST)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glEnable(GL_DEPTH_TEST)


def setup_skin_program_uniforms(prog_skin, W, H):
    """스킨 프로그램 유니폼 설정"""
    glUseProgram(prog_skin)
    glUniform1i(glGetUniformLocation(prog_skin, "uBaseTex"), 5)
    glUniform1i(glGetUniformLocation(prog_skin, "uMetallicRoughnessTex"), 6)
    glUniform1i(glGetUniformLocation(prog_skin, "uJawTex"), 4)
    glUniform2f(glGetUniformLocation(prog_skin, "uScreen"), float(W), float(H))
    glUniform2f(glGetUniformLocation(prog_skin, "uViewport"), 0.0, 0.0)
    glUniform1f(glGetUniformLocation(prog_skin, "uSegThr"), 0.85)
    glUniform1f(glGetUniformLocation(prog_skin, "uAlphaCut"), 0.5)
    glUniform1i(glGetUniformLocation(prog_skin, "uPremulti"), 0)
    glUniform3f(glGetUniformLocation(prog_skin, "uLightDir"), 0.4, 0.8, 0.6)
    glUseProgram(0)


def setup_mesh_program_uniforms(prog_mesh, W, H, seg_thr_hat):
    """메쉬 프로그램 유니폼 설정"""
    glUseProgram(prog_mesh)
    # Ensure uScreen matches current viewport for mask sampling
    try:
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        glUniform2f(glGetUniformLocation(prog_mesh, "uScreen"), float(vw), float(vh))
    except Exception:
        pass
    glUniform1i(glGetUniformLocation(prog_mesh, "texBase"), 0)
    glUniform1i(glGetUniformLocation(prog_mesh, "texAlpha"), 1)
    glUniform1i(glGetUniformLocation(prog_mesh, "texNormal"), 2)
    glUniform1i(glGetUniformLocation(prog_mesh, "texRoughness"), 3)
    glUniform1i(glGetUniformLocation(prog_mesh, "uJawTex"), 4)
    glUniform1i(glGetUniformLocation(prog_mesh, "uSegTex"), 5)
    glUniform2f(glGetUniformLocation(prog_mesh, "uScreen"), float(W), float(H))
    glUniform2f(glGetUniformLocation(prog_mesh, "uViewport"), 0.0, 0.0)
    glUniform3f(glGetUniformLocation(prog_mesh, "uLightDir"), 0.4, 0.8, 0.6)
    glUniform1i(glGetUniformLocation(prog_mesh, "uMode"), 0)
    glUniform1f(glGetUniformLocation(prog_mesh, "uAlphaCut"), 0.1)
    glUniform1i(glGetUniformLocation(prog_mesh, "uPremulti"), 0)
    glUniform1f(glGetUniformLocation(prog_mesh, "uSegThr"), float(seg_thr_hat))
    glUniform1i(glGetUniformLocation(prog_mesh, "uHardCutout"), 0)
    glUniform1f(glGetUniformLocation(prog_mesh, "uAlphaMipBias"), 0.75)
    glUseProgram(0)


def setup_occlusion_program_uniforms(prog_occ, W, H, seg_thr_hat):
    """오클루전 프로그램 유니폼 설정"""
    glUseProgram(prog_occ)
    glUniform1i(glGetUniformLocation(prog_occ, "uSegTex"), 5)
    glUniform2f(glGetUniformLocation(prog_occ, "uScreen"), float(W), float(H))
    glUniform2f(glGetUniformLocation(prog_occ, "uViewport"), 0.0, 0.0)
    glUniform1f(glGetUniformLocation(prog_occ, "uSegThr"), float(seg_thr_hat))
    glUseProgram(0)


def render_strap_batch(
    prog_skin, batch, M_final, P, JAW_OCCLUSION_ENABLED, first_pose_ready
):
    """턱끈 배치 렌더링"""
    glUseProgram(prog_skin)
    # Ensure uScreen/uViewport match current viewport (handles letterboxing)
    try:
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        glUniform2f(glGetUniformLocation(prog_skin, "uScreen"), float(vw), float(vh))
        glUniform2f(glGetUniformLocation(prog_skin, "uViewport"), float(vx), float(vy))
    except Exception:
        pass
    glUniformMatrix4fv(glGetUniformLocation(prog_skin, "uViewProj"), 1, GL_FALSE, P.T)
    glUniform1i(glGetUniformLocation(prog_skin, "uUseGate"), 0)
    glUniform1i(
        glGetUniformLocation(prog_skin, "uUseJaw"), 1 if JAW_OCCLUSION_ENABLED else 0
    )

    glUniform4f(
        glGetUniformLocation(prog_skin, "uColor"),
        float(batch["color"][0]),
        float(batch["color"][1]),
        float(batch["color"][2]),
        float(batch["color"][3]),
    )

    if batch["tex"] is not None:
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, batch["tex"])
        glUniform1i(glGetUniformLocation(prog_skin, "uHasTex"), 1)
    else:
        glUniform1i(glGetUniformLocation(prog_skin, "uHasTex"), 0)

    glUniform1f(
        glGetUniformLocation(prog_skin, "uMetallicFactor"),
        batch.get("metallicFactor", 0.0),
    )
    glUniform1f(
        glGetUniformLocation(prog_skin, "uRoughnessFactor"),
        batch.get("roughnessFactor", 1.0),
    )

    if batch.get("metallic_roughness_tex") is not None:
        glActiveTexture(GL_TEXTURE6)
        glBindTexture(GL_TEXTURE_2D, batch["metallic_roughness_tex"])
        glUniform1i(glGetUniformLocation(prog_skin, "uMetallicRoughnessTex"), 6)
        glUniform1i(glGetUniformLocation(prog_skin, "uHasMetallicRoughnessTex"), 1)
    else:
        glUniform1i(glGetUniformLocation(prog_skin, "uHasMetallicRoughnessTex"), 0)

    # 스킨 애니메이션 처리
    if (
        batch["has_skin"] == 1
        and batch.get("skin_meta") is not None
        and batch.get("joint_palette") is not None
        and first_pose_ready
    ):

        sm = batch["skin_meta"]
        pal = batch["joint_palette"]
        seq = batch["seq"]
        P_world = batch["P_world"]

        sm_inv = batch["skin_meta"]["inv_bind"]
        G_bind_model_all = batch["globals_bind_all_model"]
        G_all_world = np.einsum("ij,njk->nik", M_final, G_bind_model_all).astype(
            np.float32
        )

        for si, j in enumerate(seq):
            dirv = P_world[si + 1] - P_world[si]
            G = align_y_to_vec(dirv)
            G[:3, 3] = P_world[si]
            G_all_world[j] = G.astype(np.float32)

        mats = []
        nb = min(len(pal), MAX_BONES)
        for j in pal[:nb]:
            Mbone = (G_all_world[j] @ sm_inv[j]).astype(np.float32)
            mats.append(Mbone)
        for _ in range(MAX_BONES - nb):
            mats.append(np.eye(4, dtype=np.float32))
        mats_np = np.stack(mats, axis=0)

        I = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(prog_skin, "uModel"), 1, GL_FALSE, I.T)
        glUniform1i(glGetUniformLocation(prog_skin, "uHasSkin"), 1)
        glUniformMatrix4fv(
            glGetUniformLocation(prog_skin, "uBones[0]"),
            MAX_BONES,
            GL_FALSE,
            mats_np.transpose(0, 2, 1),
        )
    else:
        glUniform1i(glGetUniformLocation(prog_skin, "uHasSkin"), 0)
        glUniformMatrix4fv(
            glGetUniformLocation(prog_skin, "uModel"), 1, GL_FALSE, M_final.T
        )

    glBindVertexArray(batch["vao"])
    glDrawElements(GL_TRIANGLES, batch["count"], batch["index_type"], None)


def render_occlusion_depth_pass(
    prog_occ,
    vao_green,
    green_count,
    vao_blue,
    blue_count,
    M_hat_shared,
    green_tx,
    green_ty,
    green_tz,
    green_pitch_deg,
    green_scale,
    blue_pitch_deg,
    blue_size,
    blue_tx,
        blue_ty,
        blue_tz,
        P,
    USE_SEG_MASK,
    seg_thr_hat,
):
    """가려짐 깊이 패스 렌더링"""
    from utils import rot_x4

    glUseProgram(prog_occ)
    VP = (P @ np.eye(4, dtype=np.float32)).astype(np.float32)
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uViewProj"), 1, GL_FALSE, VP.T)

    glActiveTexture(GL_TEXTURE5)
    glUniform1i(glGetUniformLocation(prog_occ, "uSegTex"), 5)
    glUniform2f(
        glGetUniformLocation(prog_occ, "uScreen"), float(640), float(480)
    )  # W, H 필요시 파라미터로
    glUniform1f(glGetUniformLocation(prog_occ, "uSegThr"), float(seg_thr_hat))
    glUniform1i(glGetUniformLocation(prog_occ, "uUseSeg"), 1 if USE_SEG_MASK else 0)
    # Override uScreen/uViewport with current viewport (fix for hardcoded size)
    try:
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        glUseProgram(prog_occ)
        glUniform2f(glGetUniformLocation(prog_occ, "uScreen"), float(vw), float(vh))
        glUniform2f(glGetUniformLocation(prog_occ, "uViewport"), float(vx), float(vy))
    except Exception:
        pass

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-1.0, -1.0)

    # 초록 디스크(깊이 전패스)
    Tm_green = np.eye(4, dtype=np.float32)
    Tm_green[0, 3] = green_tx
    Tm_green[1, 3] = green_ty
    Tm_green[2, 3] = green_tz
    Rx_green = rot_x4(green_pitch_deg)
    Sg_green = np.diag([green_scale, 1.0, green_scale, 1.0]).astype(np.float32)
    M_green = (M_hat_shared @ Tm_green @ Rx_green @ Sg_green).astype(np.float32)
    glUniform1i(glGetUniformLocation(prog_occ, "uUseGate"), 0)
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uModel"), 1, GL_FALSE, M_green.T)
    glBindVertexArray(vao_green)
    glDrawArrays(GL_TRIANGLES, 0, green_count)

    # 파란 평면
    Tm_blue = np.eye(4, dtype=np.float32)
    Tm_blue[0, 3] = blue_tx
    Tm_blue[1, 3] = blue_ty
    Tm_blue[2, 3] = blue_tz
    Rx_blue = rot_x4(blue_pitch_deg)
    Sg_blue = np.diag([blue_size, 1.0, blue_size, 1.0]).astype(np.float32)
    M_position_only = _extract_position_matrix(M_hat_shared)
    M_blue = (M_position_only @ Tm_blue @ Rx_blue @ Sg_blue).astype(np.float32)

    # 게이트 평면
    n_local = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    n_world = (M_hat_shared @ Tm_green @ Rx_green @ n_local)[:3]
    n_world = n_world / (np.linalg.norm(n_world) + 1e-8)
    origin_world = (
        M_hat_shared @ Tm_green @ np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    )[:3]
    D_world = -float(np.dot(n_world, origin_world))

    glUniform1i(glGetUniformLocation(prog_occ, "uUseGate"), 1)
    glUniform3f(
        glGetUniformLocation(prog_occ, "uGateN0"),
        float(n_world[0]),
        float(n_world[1]),
        float(n_world[2]),
    )
    glUniform1f(glGetUniformLocation(prog_occ, "uGateD0"), D_world)
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uModel"), 1, GL_FALSE, M_blue.T)
    glBindVertexArray(vao_blue)
    glDrawArrays(GL_TRIANGLES, 0, blue_count)

    glDisable(GL_POLYGON_OFFSET_FILL)
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)


def render_hat_mesh(prog_mesh, vao_hat, hat_count, M_hat_shared, hat_tex, P):
    """갓 메쉬 렌더링"""
    glUseProgram(prog_mesh)
    VP = (P @ np.eye(4, dtype=np.float32)).astype(np.float32)
    # Sync uScreen/uViewport with current viewport for correct mask sampling
    try:
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        glUseProgram(prog_mesh)  # 모자 그릴 때 사용하는 프로그램으로 교체
        glUniform2f(glGetUniformLocation(prog_mesh, "uScreen"), float(vw), float(vh))
        glUniform2f(glGetUniformLocation(prog_mesh, "uViewport"), float(vx), float(vy))
    except Exception:
        pass

    # 텍스처 바인딩
    glEnable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, hat_tex.get("base", 0))
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, hat_tex.get("alpha", 0))
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, hat_tex.get("normal", 0))
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, hat_tex.get("roughness", 0))
    glActiveTexture(GL_TEXTURE0)

    glUniformMatrix4fv(glGetUniformLocation(prog_mesh, "uViewProj"), 1, GL_FALSE, VP.T)
    glUniformMatrix4fv(
        glGetUniformLocation(prog_mesh, "uModel"), 1, GL_FALSE, M_hat_shared.T
    )
    glUniform1i(glGetUniformLocation(prog_mesh, "uUseSeg"), 0)
    glUniform1i(glGetUniformLocation(prog_mesh, "uUseGate"), 0)
    glUniform1i(glGetUniformLocation(prog_mesh, "uUseJaw"), 0)
    glUniform1i(glGetUniformLocation(prog_mesh, "uMode"), 0)
    glUniform1f(glGetUniformLocation(prog_mesh, "uAlphaCut"), 0.1)
    glUniform1i(glGetUniformLocation(prog_mesh, "uPremulti"), 0)

    # 개선된 블렌딩 및 컬링 설정
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    # 알파 테스트 활성화로 경계 품질 향상
    glEnable(GL_ALPHA_TEST)
    glAlphaFunc(GL_GREATER, 0.1)

    glBindVertexArray(vao_hat)
    glDrawArrays(GL_TRIANGLES, 0, hat_count)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_CULL_FACE)  # 알파 테스트 비활성화 (렌더링 완료 후)
    glDisable(GL_ALPHA_TEST)


def render_visualization_areas(
    prog_occ,
    vao_green,
    green_count,
    vao_blue,
    blue_count,
    M_hat_shared,
    green_tx,
    green_ty,
    green_tz,
    green_pitch_deg,
    green_scale,
    blue_pitch_deg,
    blue_size,
    blue_tx,
        blue_ty,
        blue_tz,
        blue_color,
    blue_alpha,
    P,
    SHOW_GREEN_COLOR,
    SHOW_BLUE_COLOR,
):
    """초록/파란 영역 시각화 렌더링"""
    from utils import rot_x4

    if not (SHOW_GREEN_COLOR or SHOW_BLUE_COLOR):
        return

    glUseProgram(prog_occ)
    VP = (P @ np.eye(4, dtype=np.float32)).astype(np.float32)
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uViewProj"), 1, GL_FALSE, VP.T)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    if SHOW_GREEN_COLOR:
        Tm_green = np.eye(4, dtype=np.float32)
        Tm_green[0, 3] = green_tx
        Tm_green[1, 3] = green_ty
        Tm_green[2, 3] = green_tz
        Rx_green = rot_x4(green_pitch_deg)
        Sg_green = np.diag([green_scale, 1.0, green_scale, 1.0]).astype(np.float32)
        M_green_vis = (M_hat_shared @ Tm_green @ Rx_green @ Sg_green).astype(np.float32)
        glUniformMatrix4fv(
            glGetUniformLocation(prog_occ, "uModel"), 1, GL_FALSE, M_green_vis.T
        )
        glUniform3f(glGetUniformLocation(prog_occ, "uColor"), 0.0, 1.0, 0.0)
        glUniform1f(glGetUniformLocation(prog_occ, "uAlpha"), 0.3)
        glUniform1i(glGetUniformLocation(prog_occ, "uUseGate"), 0)
        glUniform1i(glGetUniformLocation(prog_occ, "uUseSeg"), 0)
        glBindVertexArray(vao_green)
        glDrawArrays(GL_TRIANGLES, 0, green_count)

    if SHOW_BLUE_COLOR:
        Tm_blue = np.eye(4, dtype=np.float32)
        Tm_blue[0, 3] = blue_tx
        Tm_blue[1, 3] = blue_ty
        Tm_blue[2, 3] = blue_tz
        Rx_blue = rot_x4(blue_pitch_deg)
        Sg_blue = np.diag([blue_size, 1.0, blue_size, 1.0]).astype(np.float32)
        M_position_only = _extract_position_matrix(M_hat_shared)
        M_blue_vis = (M_position_only @ Tm_blue @ Rx_blue @ Sg_blue).astype(np.float32)

        Tm_green_gate = np.eye(4, dtype=np.float32)
        Tm_green_gate[0, 3] = green_tx
        Tm_green_gate[1, 3] = green_ty
        Tm_green_gate[2, 3] = green_tz
        Rx_green = rot_x4(green_pitch_deg)
        n_local = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
        n_world = (M_hat_shared @ Tm_green_gate @ Rx_green @ n_local)[:3]
        n_world = n_world / (np.linalg.norm(n_world) + 1e-8)
        origin_world = (
            M_hat_shared @ Tm_green_gate @ np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        )[:3]
        D_world = -float(np.dot(n_world, origin_world))

        glUniformMatrix4fv(
            glGetUniformLocation(prog_occ, "uModel"), 1, GL_FALSE, M_blue_vis.T
        )
        glUniform3f(
            glGetUniformLocation(prog_occ, "uColor"),
            blue_color[0],
            blue_color[1],
            blue_color[2],
        )
        glUniform1f(glGetUniformLocation(prog_occ, "uAlpha"), blue_alpha)
        glUniform1i(glGetUniformLocation(prog_occ, "uUseGate"), 1)
        glUniform3f(
            glGetUniformLocation(prog_occ, "uGateN0"),
            float(n_world[0]),
            float(n_world[1]),
            float(n_world[2]),
        )
        glUniform1f(glGetUniformLocation(prog_occ, "uGateD0"), float(D_world))
        glUniform1i(glGetUniformLocation(prog_occ, "uUseSeg"), 0)
    glBindVertexArray(vao_blue)
    glDrawArrays(GL_TRIANGLES, 0, blue_count)

    glEnable(GL_DEPTH_TEST)

# --- 축(axes) VAO 생성: X(빨강), Y(초록), Z(파랑) ---
def setup_axes_vaos(axis_len=1.0):
    """
    세 축을 GL_LINES로 그릴 VAO를 생성해 반환합니다.
    반환: dict {'x': vao_x, 'y': vao_y, 'z': vao_z}
    """
    def _mk_vao(line_verts):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        arr = np.array(line_verts, dtype=np.float32).reshape(-1, 3)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)
        return vao

    L = float(axis_len)
    vao_x = _mk_vao([[0,0,0],[L,0,0]])
    vao_y = _mk_vao([[0,0,0],[0,L,0]])
    vao_z = _mk_vao([[0,0,0],[0,0,L]])
    return {'x': vao_x, 'y': vao_y, 'z': vao_z}

# --- 축(axes) 렌더링: prog_occ(오클루전 셰이더) 재사용 ---
def render_axes(prog_occ, axes_vaos, M_model, P, line_width=2.0):
    """
    prog_occ를 이용해 한 오브젝트의 기준축을 그립니다.
    - M_model: 해당 오브젝트의 모델행렬(= 기준축이자 원점)
    - P: 투영행렬
    """
    if not axes_vaos:
        return
    glUseProgram(prog_occ)

    # 레터박스/뷰포트 보정치 유니폼 반영
    try:
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        glUniform2f(glGetUniformLocation(prog_occ, "uScreen"), float(vw), float(vh))
        glUniform2f(glGetUniformLocation(prog_occ, "uViewport"), float(vx), float(vy))
    except Exception:
        pass

    # 오클루전 셰이더 유니폼 세팅 (게이트/세그 안씀)
    glUniform1i(glGetUniformLocation(prog_occ, "uUseGate"), 0)
    glUniform1i(glGetUniformLocation(prog_occ, "uUseSeg"), 0)
    glUniform1f(glGetUniformLocation(prog_occ, "uAlpha"), float(getattr(config, "AXES_ALPHA", 1.0)))

    # 변환행렬
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uViewProj"), 1, GL_FALSE, P.T)
    glUniformMatrix4fv(glGetUniformLocation(prog_occ, "uModel"),   1, GL_FALSE, M_model.T)

    # 선 굵기
    try:
        glLineWidth(float(line_width))
    except Exception:
        pass  # 일부 플랫폼에서 고정

    # X축 (빨강)
    glUniform3f(glGetUniformLocation(prog_occ, "uColor"), 1.0, 0.0, 0.0)
    glBindVertexArray(axes_vaos['x'])
    glDrawArrays(GL_LINES, 0, 2)

    # Y축 (초록)
    glUniform3f(glGetUniformLocation(prog_occ, "uColor"), 0.0, 1.0, 0.0)
    glBindVertexArray(axes_vaos['y'])
    glDrawArrays(GL_LINES, 0, 2)

    # Z축 (파랑)
    glUniform3f(glGetUniformLocation(prog_occ, "uColor"), 0.0, 0.6, 1.0)
    glBindVertexArray(axes_vaos['z'])
    glDrawArrays(GL_LINES, 0, 2)

    glBindVertexArray(0)
    glUseProgram(0)





