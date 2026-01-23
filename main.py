# -*- coding: utf-8 -*-`r`n# main.py - 메인 실행 파일
import os

_CUDA_DLL_DIRS = [
    r"C:\\Windows\\System32",
    r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
    r"C:\\Program Files\\NVIDIA\\CUDNN\\v9.10\\bin\\12.9",
]
if hasattr(os, "add_dll_directory"):
    for _d in _CUDA_DLL_DIRS:
        if os.path.isdir(_d):
            try:    
                os.add_dll_directory(_d)
            except Exception:
                pass
del _CUDA_DLL_DIRS
import time
import cv2
import pygame
import numpy as np
import mediapipe as mp
import imageio.v2 as imageio
from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN
from OpenGL.GL import *
from app.camera import build_projection_from_intrinsics, normalize_rotation3x3
from app.display import compute_view_rect, apply_display_mode
from app.background import fit_background_image
import app.debug as app_debug


def _build_projection_from_intrinsics(W, H):
    fx = fy = cx = cy = None
    if getattr(config, "USE_INTRINSICS", False):
        try:
            import json, os

            jpath = getattr(config, "INTRINSICS_JSON", "")
            if jpath and os.path.isfile(jpath):
                with open(jpath, "r", encoding="utf-8") as f:
                    j = json.load(f)
                fx = float(j["fx"])
                fy = float(j["fy"])
                cx = float(j.get("cx", W / 2.0))
                cy = float(j.get("cy", H / 2.0))
        except Exception as e:
            print("[WARN] intrinsics.json 로드 실패:", e)
    # Fallback: 보정 없는 안전 기본값 (HFOV에 덜 민감, 웬만한 카메라에서 안정적)
    if fx is None:
        fx = fy = float(W) * 0.4  # 더 넓은 FOV로 오브젝트가 작게 보이도록 조정
        cx = float(W) / 2.0
        cy = float(H) / 2.0

    from utils import perspective_from_intrinsics

    P = perspective_from_intrinsics(fx, fy, cx, cy, config.NEAR_Z, config.FAR_Z, W, H)
    return P, (fx, fy, cx, cy)  # <- 변경: intrinsics도 같이 반환


def _fit_background_image(img_bgr, W, H, mode="cover"):
    h, w = img_bgr.shape[:2]
    if mode == "stretch":
        return cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    scale_w = W / float(w)
    scale_h = H / float(h)
    if mode == "contain":
        s = min(scale_w, scale_h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = (W - nw) // 2
        y0 = (H - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas
    # cover (default): fill and center-crop
    s = max(scale_w, scale_h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (nw - W) // 2
    y0 = (nh - H) // 2
    return resized[y0 : y0 + H, x0 : x0 + W]


def _compute_view_rect(screen_w, screen_h, W, H):
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


def _apply_display_mode(fullscreen, last_view_size, W, H):
    """fullscreen True: 모니터 전체, False: '현재 출력 해상도' 창 모드.
    반환: (screen_w, screen_h, view_x, view_y, view_w, view_h, new_last_view)
    """
    if fullscreen:
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        pygame.display.set_mode((screen_w, screen_h), DOUBLEBUF | OPENGL | FULLSCREEN)
        vx, vy, vw, vh = compute_view_rect(screen_w, screen_h, W, H)
        new_last_view = (vw, vh)  # 다음 창 모드 크기로 기억
    else:
        win_w, win_h = last_view_size
        pygame.display.set_mode((int(win_w), int(win_h)), DOUBLEBUF | OPENGL)
        screen_w, screen_h = int(win_w), int(win_h)
        vx, vy, vw, vh = compute_view_rect(screen_w, screen_h, W, H)
        new_last_view = (vw, vh)

    # GL 상태만 다시 세팅 (프로그램/유니폼은 밖에서 세팅)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return screen_w, screen_h, vx, vy, vw, vh, new_last_view


# 모듈 임포트
import config
from utils import (
    rebuild_O,
    quat_t_to_mat4,
    eye_open_ratio,
)
from mesh import build_disk_mesh, build_plane_mesh
from loaders import load_obj_mesh, load_textures_for_obj, load_glb_batches
from shaders import create_all_programs
from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniform1i, glBindVertexArray
from rendering import (
    setup_background_vao,
    setup_textures,
    setup_obj_vao,
    setup_mesh_vao,
    upload_jaw_mask,
    upload_seg_mask,
    render_background,
    setup_skin_program_uniforms,
    setup_mesh_program_uniforms,
    setup_occlusion_program_uniforms,
    render_strap_batch,
    render_occlusion_depth_pass,
    render_hat_mesh,
    render_visualization_areas,
    setup_axes_vaos,
    render_axes,
)
from bg_composite import setup_bg_composite_textures, render_bg_composite
from owner_tracker import OwnerTracker
from physics import update_rope_physics
from face_processing import (
    initialize_face_detector,
    initialize_segmentation,
    process_face_pose,
    create_jaw_mask,
    process_jaw_mask,
    create_collision_data,
    process_segmentation_mask,
    get_all_face_landmarks_from_result,
)
from input_handler import (
    handle_keyboard_input,
)
from face_effects import Effects2D
from pose_processing import (
    initialize_pose_detector,
    build_owner_pose_roi_mask,
)
from sticker3d.renderer import Sticker3DRenderer

from config import (
    SKIN_TONE_ENABLE,
    SEG_EVERY_N,
    LEFT_UP_IDX,
    LEFT_LO_IDX,
    LEFT_CORNERS,
    RIGHT_UP_IDX,
    RIGHT_LO_IDX,
    RIGHT_CORNERS,
    EYE_OPEN_SMOOTH,
    EYE_OPEN_THR_CLOSE,
    EYE_OPEN_THR_OPEN,
)


 


 


 


def _two_orthonormal(v):
    """v(3,)에 수직인 2개 직교기저 U(3x2) 반환"""
    v = v / (np.linalg.norm(v) + 1e-8)
    a = (
        np.array([1.0, 0.0, 0.0], np.float32)
        if abs(v[0]) < 0.9
        else np.array([0.0, 1.0, 0.0], np.float32)
    )
    u1 = np.cross(v, a)
    u1 /= np.linalg.norm(u1) + 1e-8
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2) + 1e-8
    return np.stack([u1, u2], axis=1)  # (3x2)


def main():
    landmarker = initialize_face_detector(config.MODEL_PATH)
    mp_selfie = initialize_segmentation()
    # Pose detector for owner-only background gating
    mp_pose = initialize_pose_detector()

    cap = cv2.VideoCapture(config.CAM_ID)
    if not cap.isOpened():
        raise RuntimeError("카메라 열기 실패")

    W, H = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"카메라 해상도 {W} X {H}")

    pygame.init()

    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    print(f"전체 화면 해상도 {screen_w} X {screen_h}")
    # 시작 모드
    fullscreen_mode = bool(getattr(config, "START_FULLSCREEN", True))
    pygame.display.set_mode((screen_w, screen_h), DOUBLEBUF | OPENGL | FULLSCREEN)
    pygame.display.set_caption("AR Face Sticker")

    # 초기 뷰 계산
    view_x, view_y, view_w, view_h = compute_view_rect(screen_w, screen_h, W, H)
    # '현재 출력 해상도' 기억(창 모드 전환 시 사용)
    last_view_size = (view_w, view_h)

    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    prog_bg, prog_skin, prog_mesh, prog_occ, prog_sticker = create_all_programs()
    vao_bg = setup_background_vao()
    tex_bg, tex_jaw, tex_seg = setup_textures()
    # Textures for GPU background compositing
    tex_bgimg, tex_bgmask = setup_bg_composite_textures()
    batches = load_glb_batches(config.GLB_PATH)
    hat_verts = load_obj_mesh(config.OBJ_PATH)
    vao_hat = setup_obj_vao(hat_verts)
    hat_count = hat_verts.shape[0]
    hat_tex = load_textures_for_obj(config.OBJ_PATH)
    green_disk_verts = build_disk_mesh(
        config.green_radius, config.green_half_h, config.green_segments
    )
    blue_plane_verts = build_plane_mesh()

    vao_green = setup_mesh_vao(green_disk_verts)
    vao_blue = setup_mesh_vao(blue_plane_verts)
    axes_vaos = setup_axes_vaos(getattr(config, "AXES_LEN", 2.0))

    green_count = green_disk_verts.shape[0]
    blue_count = blue_plane_verts.shape[0]
    P, CAM_INTR = build_projection_from_intrinsics(W, H)
    setup_skin_program_uniforms(prog_skin, W, H)
    setup_mesh_program_uniforms(prog_mesh, W, H, config.seg_thr_hat)
    setup_occlusion_program_uniforms(prog_occ, W, H, config.seg_thr_hat)

    config_vars = {
        "SHOW_STRAP": config.SHOW_STRAP,
        "SHOW_HAT": config.SHOW_HAT,
        "JAW_OCCLUSION_ENABLED": config.JAW_OCCLUSION_ENABLED,
        "JAW_DEBUG_SHOW": config.JAW_DEBUG_SHOW,
        "MIRROR_MODE": config.MIRROR_MODE,
        "GRAVITY_ON": config.GRAVITY_ON,
        "g_strength": config.g_strength,
        "USE_SEG_MASK": config.USE_SEG_MASK,
        "SHOW_GREEN_COLOR": config.SHOW_GREEN_COLOR,
        "SHOW_BLUE_COLOR": config.SHOW_BLUE_COLOR,
        "OCCLUSION_ENABLED": config.OCCLUSION_ENABLED,
        "O": rebuild_O(),
        "SHOW_AXES_HAT": config.SHOW_AXES_HAT,
        "SHOW_AXES_STRAP": config.SHOW_AXES_STRAP,
        "FULLSCREEN": True,
        "DISPLAY_DIRTY": False,
        "SHOW_SEG_DEBUG": False,  # F9 for segmentation debug view
        "_last_d_press_time": 0.0,  # For key debounce
        "CAM_INTR": None,  # (fx,fy,cx,cy)
        "AXES_ANCHOR_CAM": None,  # 최근 앵커 3D(카메라좌표) for smoothing
        "AXES_M_HAT": None,  # 모자/갓 축용 최종 모델행렬 캐시
        "ANCHOR_LOCAL_Z": None,  # 이전 방식의 z 앵커 (호환성용)
        "LM9_LOCAL_POS": None,  # 9번 랜드마크의 얼굴 로컬 좌표
    }
    config_vars["CAM_INTR"] = CAM_INTR

    prev_q = prev_t = None
    clock = pygame.time.Clock()
    running = True
    first_pose_ready = False
    prev_face_detected = False  # 이전 프레임에서 얼굴이 감지되었는지 추적

    effects = Effects2D()
    renderer = Sticker3DRenderer(W, H, prog_sticker)
    prev_skin_mask = None
    frame_idx = 0
    eye_state = {"earL_ma": None, "earR_ma": None, "openL": True, "openR": True}

    # --- Owner tracker (choose closest one person and near-gate) ---
    owner_tracker = OwnerTracker(W, H)
    last_near_log = 0.0  # 측정 로그 간격 제어

    # Optional: background image replacement using multiclass ONNX
    bg_enabled = bool(getattr(config, "USE_BG_IMAGE", False))
    bg_path = getattr(config, "BG_IMAGE_PATH", "")
    bg_fit = str(getattr(config, "BG_FIT_MODE", "cover")).lower()
    bg_img_bgr = None
    gif_reader = None
    gif_frames = []
    gif_frame_dur = []  # seconds per frame
    gif_total_dur = 0.0
    # Cache for person mask to avoid running ONNX every frame
    prev_person_mask = None
    # Cache for pose ROI mask
    pose_roi_mask_cached = None
    # Buffers for GPU background compositing per-frame
    gpu_bg_frame_rgb = None
    gpu_person_mask = None
    if bg_enabled and bg_path:
        try:
            if bg_path.lower().endswith(".gif") and os.path.isfile(bg_path):
                preload = bool(getattr(config, "BG_GIF_PRELOAD", True))
                gif_reader = imageio.get_reader(bg_path)
                meta = {}
                try:
                    meta = gif_reader.get_meta_data()
                except Exception:
                    meta = {}
                default_ms = float(meta.get("duration", 100.0))  # ms
                default_sec = max(0.01, default_ms / 1000.0)
                if preload:
                    for frame in gif_reader:
                        # frame is RGB
                        f_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                        f_bgr = fit_background_image(f_bgr, W, H, bg_fit)
                        gif_frames.append(f_bgr)
                        gif_frame_dur.append(default_sec)
                    gif_total_dur = max(
                        default_sec, default_sec * max(1, len(gif_frames))
                    )
                    try:
                        gif_reader.close()
                        gif_reader = None
                    except Exception:
                        pass
                else:
                    # lazy mode: keep reader, compute nframes if possible
                    try:
                        n = gif_reader.get_length()
                    except Exception:
                        n = 0
                    gif_total_dur = default_sec * max(1, int(n))
                print(
                    f"[BG] GIF loaded. preload={preload}, frames={len(gif_frames) or 'lazy'}, dur={default_sec:.3f}s"
                )
            elif os.path.isfile(bg_path):
                _bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
                if _bg is not None:
                    bg_img_bgr = fit_background_image(_bg, W, H, bg_fit)
                else:
                    print(f"[BG] Failed to load image: {bg_path}")
            else:
                print(f"[BG] Invalid path: {bg_path}")
        except Exception as e:
            print(f"[BG] Error loading background: {e}")

    while running:
        dt_ms = clock.tick(60)
        dt = min(max(dt_ms / 1000.0, 1.0 / 120.0), 1.0 / 30.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        handle_keyboard_input(pygame.key.get_pressed(), config_vars)

        # ESC key: request graceful exit
        if config_vars.pop("EXIT", False):
            running = False
            continue

        # F8로 모드 전환 요청 시 처리
        if config_vars.pop("DISPLAY_DIRTY", False):
            if config_vars["FULLSCREEN"]:
                last_view_size = (view_w, view_h)

            screen_w, screen_h, view_x, view_y, view_w, view_h, last_view_size = (
                apply_display_mode(config_vars["FULLSCREEN"], last_view_size, W, H)
            )

            # ★ 모드 전환 후, 프로그램별 유니폼/뷰포트 관련 값 재설정
            setup_skin_program_uniforms(prog_skin, W, H)
            setup_mesh_program_uniforms(prog_mesh, W, H, config.seg_thr_hat)
            setup_occlusion_program_uniforms(prog_occ, W, H, config.seg_thr_hat)

            print(
                f"[DISPLAY] mode={'FULL' if config_vars['FULLSCREEN'] else 'WINDOW'} "
                f"screen=({screen_w}x{screen_h}) view=({view_w}x{view_h}+{view_x},{view_y})"
            )

        ok, frame = cap.read()
        if not ok:
            continue
        if config_vars["MIRROR_MODE"]:
            frame = cv2.flip(frame, 1)

        original_frame = frame.copy()

        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
        faces_pts = get_all_face_landmarks_from_result(res, W, H)
        category_mask = None

        # --- Select owner face ---
        # Select owner and apply near gating
        owner_pts = owner_tracker.select_owner(faces_pts)
        owner_pts = owner_tracker.near_gate(owner_pts)

        # Near gating: only allow owner if close enough to camera
        if owner_pts is not None:
            xs, ys = owner_pts[:, 0], owner_pts[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            face_h = max(1.0, y1 - y0)
            min_h_px = float(getattr(config, "OWNER_NEAR_MIN_HEIGHT_PX", 120))
            min_h_ratio = float(
                getattr(config, "OWNER_NEAR_MIN_RATIO", 0.22)
            )  # of frame height
            h_thr = max(min_h_px, min_h_ratio * H)
            # 측정 로그: 1초에 한 번 출력 (임시)
            try:
                tnow = time.monotonic()
                if tnow - last_near_log > 1.0:
                    # print(
                    #     f"[NEAR_MEASURE] face_h={face_h:.1f}px, H={H}, ratio={face_h/H:.3f}, thr={h_thr:.1f}px"
                    # )
                    last_near_log = tnow
            except Exception:
                pass
            if face_h < h_thr:
                owner_pts = None

        # pts_all은 이후 공통 처리에서 참조되므로 항상 정의
        pts_all = owner_pts if owner_pts is not None else None

        # === Effects for owner only ===
        if owner_pts is not None:
            # 주인공 랜드마크만 사용
            pts_all = owner_pts
            earL_raw = eye_open_ratio(pts_all, LEFT_UP_IDX, LEFT_LO_IDX, LEFT_CORNERS)
            earR_raw = eye_open_ratio(
                pts_all, RIGHT_UP_IDX, RIGHT_LO_IDX, RIGHT_CORNERS
            )
            if eye_state["earL_ma"] is None:
                eye_state["earL_ma"], eye_state["earR_ma"] = earL_raw, earR_raw
            else:
                eye_state["earL_ma"] = (
                    eye_state["earL_ma"] * (1.0 - EYE_OPEN_SMOOTH)
                    + earL_raw * EYE_OPEN_SMOOTH
                )
                eye_state["earR_ma"] = (
                    eye_state["earR_ma"] * (1.0 - EYE_OPEN_SMOOTH)
                    + earR_raw * EYE_OPEN_SMOOTH
                )
            if eye_state["openL"]:
                if eye_state["earL_ma"] < EYE_OPEN_THR_CLOSE:
                    eye_state["openL"] = False
            else:
                if eye_state["earL_ma"] > EYE_OPEN_THR_OPEN:
                    eye_state["openL"] = True
            if eye_state["openR"]:
                if eye_state["earR_ma"] < EYE_OPEN_THR_CLOSE:
                    eye_state["openR"] = False
            else:
                if eye_state["earR_ma"] > EYE_OPEN_THR_OPEN:
                    eye_state["openR"] = True

            if SKIN_TONE_ENABLE:
                eye_union_mask = effects.build_eye_union_mask(H, W, pts_all)
                if (prev_skin_mask is None) or (
                    SEG_EVERY_N > 0 and (frame_idx % SEG_EVERY_N) == 1
                ):
                    # get_skin_mask_onnx가 필터링까지 모두 처리
                    skin_mask, category_mask = effects.get_skin_mask_onnx(
                        original_frame, pts_all
                    )

                    # 눈 영역은 피부색 변경에서 제외 (최종 단계)
                    skin_mask = cv2.bitwise_and(
                        skin_mask, cv2.bitwise_not(eye_union_mask)
                    )
                    prev_skin_mask = skin_mask
                else:
                    skin_mask = prev_skin_mask

                # 디버그 뷰가 아닐 때만 피부색 적용
                if not config_vars.get("SHOW_SEG_DEBUG", False):
                    frame = effects.apply_skin_tone(frame, skin_mask)

            pts468 = pts_all[:468]
            frame = effects.apply_smoky(frame, pts468)
            frame = effects.apply_lip_color(frame, pts_all)
            frame = effects.apply_sclera_color(
                frame,
                pts_all,
                left_open=eye_state["openL"],
                right_open=eye_state["openR"],
            )
            frame = effects.apply_eye_effect(
                frame,
                pts_all,
                left_open=eye_state["openL"],
                right_open=eye_state["openR"],
            )

        # If enabled, replace background using multiclass ONNX person mask (image or GIF)
        if bg_enabled and (
            bg_img_bgr is not None or gif_reader is not None or gif_frames
        ):
            try:
                # Select background frame
                bg_frame = bg_img_bgr
                if gif_frames:
                    # Preloaded frames with uniform duration
                    t = time.monotonic()
                    default_sec = gif_frame_dur[0] if gif_frame_dur else 0.1
                    total = max(default_sec, default_sec * len(gif_frames))
                    idx = int(((t % total) / default_sec)) % len(gif_frames)
                    bg_frame = gif_frames[idx]
                elif gif_reader is not None:
                    # Lazy mode: approximate current frame by time and read
                    meta = {}
                    try:
                        meta = gif_reader.get_meta_data()
                    except Exception:
                        meta = {}
                    default_ms = float(meta.get("duration", 100.0))
                    default_sec = max(0.01, default_ms / 1000.0)
                    try:
                        n = gif_reader.get_length()
                    except Exception:
                        n = 1
                    t = time.monotonic()
                    idx = int(((t % (default_sec * max(1, n))) / default_sec)) % max(
                        1, n
                    )
                    try:
                        frame_rgb = gif_reader.get_data(idx)
                        f_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
                        bg_frame = fit_background_image(f_bgr, W, H, bg_fit)
                    except Exception:
                        pass

                if bg_frame is not None:
                    # Always update BG frame; composite only when we have a valid/near owner
                    gpu_bg_frame_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
                    if owner_pts is not None:
                        # Recompute person mask every N frames for performance
                        bg_class = int(getattr(config, "BG_CLASS_INDEX", 0))
                        BG_SEG_EVERY_N = int(getattr(config, "BG_SEG_EVERY_N", 2))
                        need_seg = (
                            prev_person_mask is None
                            or (
                                BG_SEG_EVERY_N > 1 and (frame_idx % BG_SEG_EVERY_N) == 1
                            )
                            or BG_SEG_EVERY_N <= 1
                        )
                        if need_seg:
                            # Optionally refresh Pose ROI at a lower cadence
                            try:
                                POSE_EVERY_N = int(getattr(config, "POSE_EVERY_N", 3))
                            except Exception:
                                POSE_EVERY_N = 3
                            if (POSE_EVERY_N <= 1) or ((frame_idx % POSE_EVERY_N) == 1):
                                try:
                                    pose_roi_mask_cached = (
                                        build_owner_pose_roi_mask(
                                            mp_pose, image_rgb, owner_pts, W, H
                                        )
                                        if bool(
                                            getattr(config, "BG_POSE_GATE_ENABLE", True)
                                        )
                                        else None
                                    )
                                except Exception:
                                    pose_roi_mask_cached = None

                            # Choose background mask source
                            mask_source = str(
                                getattr(config, "BG_MASK_SOURCE", "onnx")
                            ).lower()
                            if mask_source == "selfie":
                                # Use MediaPipe SelfieSegmentation person mask (soft 0..255)
                                m = process_segmentation_mask(mp_selfie, image_rgb)
                                if m is None:
                                    m = np.zeros((H, W), np.uint8)

                                # Optional gamma/threshold shaping
                                try:
                                    gamma = float(getattr(config, "BG_MASK_GAMMA", 1.0))
                                except Exception:
                                    gamma = 1.0
                                try:
                                    thr_bin = int(getattr(config, "BG_MASK_BIN_THR", 0))
                                except Exception:
                                    thr_bin = 0
                                if gamma != 1.0 or thr_bin > 0:
                                    mf = m.astype(np.float32) / 255.0
                                    if gamma != 1.0:
                                        mf = np.power(np.clip(mf, 0.0, 1.0), gamma)
                                    if thr_bin > 0:
                                        t = np.clip(thr_bin / 255.0, 0.0, 1.0)
                                        mf = (mf >= t).astype(np.float32)
                                    m = np.clip((mf * 255.0).astype(np.uint8), 0, 255)

                                # Morphological smoothing
                                try:
                                    close_it = max(
                                        0, int(getattr(config, "BG_MASK_CLOSE", 1))
                                    )
                                except Exception:
                                    close_it = 1
                                try:
                                    open_it = max(
                                        0, int(getattr(config, "BG_MASK_OPEN", 0))
                                    )
                                except Exception:
                                    open_it = 0
                                k3 = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (3, 3)
                                )
                                if close_it > 0:
                                    m = cv2.morphologyEx(
                                        m, cv2.MORPH_CLOSE, k3, iterations=close_it
                                    )
                                if open_it > 0:
                                    m = cv2.morphologyEx(
                                        m, cv2.MORPH_OPEN, k3, iterations=open_it
                                    )

                                # Pose ROI gate (optional)
                                if pose_roi_mask_cached is not None and bool(
                                    getattr(config, "BG_POSE_GATE_ENABLE", True)
                                ):
                                    m = cv2.bitwise_and(m, pose_roi_mask_cached)

                                # Face gating (simple bbox mode for Selfie path)
                                if owner_pts is not None and bool(
                                    getattr(config, "BG_GATE_TO_FACE", True)
                                ):
                                    ys = owner_pts[:, 1]
                                    xs = owner_pts[:, 0]
                                    x0 = int(max(0, xs.min()))
                                    x1 = int(min(W - 1, xs.max()))
                                    y0 = int(max(0, ys.min()))
                                    y1 = int(min(H - 1, ys.max()))
                                    m_top = int(
                                        getattr(config, "BG_GATE_MARGIN_TOP_PX", 100)
                                    )
                                    m_bottom = int(
                                        getattr(config, "BG_GATE_MARGIN_BOTTOM_PX", 400)
                                    )
                                    m_left = int(
                                        getattr(config, "BG_GATE_MARGIN_LEFT_PX", 150)
                                    )
                                    m_right = int(
                                        getattr(config, "BG_GATE_MARGIN_RIGHT_PX", 150)
                                    )
                                    gx0 = max(0, x0 - m_left)
                                    gx1 = min(W - 1, x1 + m_right)
                                    gy0 = max(0, y0 - m_top)
                                    gy1 = min(H - 1, y1 + m_bottom)
                                    gate = np.zeros_like(m, dtype=np.uint8)
                                    if gx1 > gx0 and gy1 > gy0:
                                        gate[gy0 : gy1 + 1, gx0 : gx1 + 1] = 255
                                        m = cv2.bitwise_and(m, gate)

                                # Optional guided filtering to align edges
                                if bool(getattr(config, "BG_MASK_GUIDED", False)):
                                    try:
                                        import cv2.ximgproc as xip

                                        guide = cv2.cvtColor(
                                            original_frame, cv2.COLOR_BGR2GRAY
                                        )
                                        src = m.astype(np.float32) / 255.0
                                        gf = xip.guidedFilter(
                                            guide=guide, src=src, radius=8, eps=1e-4
                                        )
                                        m = np.clip(
                                            (gf * 255.0).astype(np.uint8), 0, 255
                                        )
                                    except Exception:
                                        pass

                                person_mask = m
                            else:
                                # Default: ONNX multiclass person mask
                                person_mask = effects.get_person_mask_onnx(
                                    original_frame,
                                    bg_class_index=bg_class,
                                    pts_all=owner_pts,
                                    pose_roi_mask=pose_roi_mask_cached,
                                )
                            # Temporal smoothing in mask domain (soft mask compatible)
                            alpha = float(
                                getattr(config, "BG_MASK_TEMPORAL_ALPHA", 0.0)
                            )
                            if prev_person_mask is not None and 0.0 < alpha < 1.0:
                                person_mask = (
                                    alpha * prev_person_mask.astype(np.float32)
                                    + (1.0 - alpha) * person_mask.astype(np.float32)
                                ).astype(np.uint8)
                            prev_person_mask = person_mask
                        else:
                            person_mask = prev_person_mask

                        # Prepare inputs for GPU compositing in the draw pass
                        gpu_person_mask = person_mask.astype(np.uint8)
                    else:
                        gpu_person_mask = None
            except Exception:
                # fail-safe: keep original frame
                pass

        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(view_x, view_y, view_w, view_h)

        seg_mask_u8 = (
            process_segmentation_mask(mp_selfie, image_rgb)
            if config_vars["USE_SEG_MASK"]
            else None
        )
        upload_seg_mask(tex_seg, seg_mask_u8, W, H)

        M_smooth, prev_q, prev_t = process_face_pose(res, W, H, prev_q, prev_t)

        # --- 얼굴 포즈 준비 완료 체크 ---
        if M_smooth is None:
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # If BG enabled: draw composite if mask, else draw BG-only; otherwise passthrough camera
            if bg_enabled and gpu_bg_frame_rgb is not None:
                if gpu_person_mask is not None:
                    glUseProgram(prog_bg)
                    glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 1)
                    glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
                    glUniform1i(glGetUniformLocation(prog_bg, "uTexBG"), 8)
                    glUniform1i(glGetUniformLocation(prog_bg, "uMask"), 7)
                    glBindVertexArray(vao_bg)
                    render_bg_composite(
                        prog_bg,
                        vao_bg,
                        tex_bg,
                        tex_bgimg,
                        tex_bgmask,
                        frame_rgb,
                        gpu_bg_frame_rgb,
                        gpu_person_mask,
                        W,
                        H,
                    )
                else:
                    glUseProgram(prog_bg)
                    glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 0)
                    glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
                    render_background(prog_bg, vao_bg, tex_bg, gpu_bg_frame_rgb, W, H)
            else:
                glUseProgram(prog_bg)
                glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 0)
                glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
                render_background(prog_bg, vao_bg, tex_bg, frame_rgb, W, H)
            pygame.display.flip()
            continue

        # --- 1) 회전(스케일 제거) ---
        R_s = M_smooth[:3, :3]
        R = normalize_rotation3x3(R_s)

        # --- 2) 선택 랜드마크 픽셀좌표(u,v) ---
        lm_idx = int(getattr(config, "ANCHOR_LM_IDX", 9))  # 인중/미간 등
        if pts_all is not None and 0 <= lm_idx < len(pts_all):
            u, v = float(pts_all[lm_idx][0]), float(pts_all[lm_idx][1])
        else:
            u, v = None, None

        # ---- 얼굴 포즈 행렬 분해 ----
        R_s = M_smooth[:3, :3]
        R = normalize_rotation3x3(R_s)  # 3x3
        t = M_smooth[:3, 3]  # 3,

        # ---- 현재 프레임 앵커 픽셀(u,v) ----
        lm_idx = int(getattr(config, "ANCHOR_LM_IDX", 9))
        u = v = None
        if pts_all is not None and 0 <= lm_idx < len(pts_all):
            u, v = float(pts_all[lm_idx][0]), float(pts_all[lm_idx][1])

        # ---- 9번 랜드마크 직접 방식으로 축 설정 ----
        if u is not None and pts_all is not None:
            # 9번 랜드마크의 픽셀 좌표를 얼굴 포즈 행렬의 위치로 직접 오버라이드
            # 얼굴 중심 대신 9번 랜드마크 위치 사용
            fx, fy, cx, cy = config_vars["CAM_INTR"]

            # 픽셀 좌표를 정규화 (미러 모드 보정)
            x_norm = (u - cx) / fx
            if config_vars["MIRROR_MODE"]:
                x_norm = -x_norm  # 미러 모드에서 X축 반전
            y_norm = (v - cy) / fy

            # 얼굴 중심의 Z 깊이를 사용하되, XY는 9번 랜드마크 사용
            face_z = t[2]
            landmark_cam = np.array(
                [x_norm * face_z, y_norm * face_z, face_z], dtype=np.float32
            )

            # 축 위치는 그대로 유지 (Z 오프셋 제거)

            # 최소 스무딩
            s = float(getattr(config, "ANCHOR_SMOOTH", 0.1))
            p_prev = config_vars.get("AXES_ANCHOR_CAM")
            if p_prev is not None:
                landmark_cam = s * p_prev + (1.0 - s) * landmark_cam
            config_vars["AXES_ANCHOR_CAM"] = landmark_cam.copy()

            # 축 모델행렬: 9번 랜드마크 위치 + 얼굴 회전
            M_axes_cam = np.eye(4, dtype=np.float32)
            M_axes_cam[:3, :3] = R  # 얼굴 회전 그대로
            M_axes_cam[:3, 3] = landmark_cam  # 9번 랜드마크의 3D 위치
            M_axes = (M_axes_cam @ np.diag([1, 1, -1, 1]).astype(np.float32)).astype(
                np.float32
            )
            config_vars["AXES_M_HAT"] = M_axes

            # 디버깅 (첫 프레임만)
            if config_vars.get("LM9_LOCAL_POS") is None:
                config_vars["LM9_LOCAL_POS"] = True  # 마킹용
                print(f"[DEBUG] 직접 방식 9번 랜드마크:")
                print(f"  픽셀 좌표: ({u:.1f}, {v:.1f})")
                print(f"  계산된 카메라 좌표: {landmark_cam}")
                print(f"  얼굴 중심(t): {t}")
                print(f"  Z 오프셋: {config.AXIS_Z_OFFSET}")

            # Z 오프셋 변경 디버깅
            if (
                hasattr(config_vars, "_prev_z_offset")
                and config_vars["_prev_z_offset"] != config.AXIS_Z_OFFSET
            ):
                print(
                    f"[DEBUG] Z 오프셋 변경됨: {config_vars['_prev_z_offset']} -> {config.AXIS_Z_OFFSET}"
                )
            config_vars["_prev_z_offset"] = config.AXIS_Z_OFFSET

        # 갓 오브젝트가 9번 랜드마크 기준 축을 사용하도록 설정
        M_axes = config_vars.get("AXES_M_HAT")
        if M_axes is not None:
            # 갓 크기 조정을 위한 스케일 행렬
            hat_scale_matrix = np.diag(
                [
                    config.HAT_SCALE_MULTIPLIER,
                    config.HAT_SCALE_MULTIPLIER,
                    config.HAT_SCALE_MULTIPLIER,
                    1.0,
                ]
            ).astype(np.float32)
            from utils import rot_x4, rot_y4, rot_z4

            Rx_obj = rot_x4(config.OBJECT_PITCH_DEG)
            Ry_obj = rot_y4(config.OBJECT_YAW_DEG)
            Rz_obj = rot_z4(config.OBJECT_ROLL_DEG)
            rotation_matrix = (Rz_obj @ Ry_obj @ Rx_obj).astype(np.float32)
            # 오브젝트 위치 오프셋을 위한 이동 행렬
            hat_offset_matrix = np.eye(4, dtype=np.float32)
            hat_offset_matrix[0, 3] = config.OBJECT_OFFSET_X
            hat_offset_matrix[1, 3] = config.OBJECT_OFFSET_Y
            hat_offset_matrix[2, 3] = config.OBJECT_OFFSET_Z + float(
                getattr(config, "AXIS_Z_OFFSET", 0.0)
            )
            # 9번 랜드마크 기준 축에 오브젝트 변환, 크기 조정, 위치 오프셋 적용
            M_hat_shared = (
                M_axes
                @ rotation_matrix
                @ hat_scale_matrix
                @ hat_offset_matrix
                @ config_vars["O"]
            ).astype(np.float32)
        else:
            # 축이 설정되지 않았으면 기존 방식 사용
            hat_scale_matrix = np.diag(
                [
                    config.HAT_SCALE_MULTIPLIER,
                    config.HAT_SCALE_MULTIPLIER,
                    config.HAT_SCALE_MULTIPLIER,
                    1.0,
                ]
            ).astype(np.float32)
            from utils import rot_x4, rot_y4, rot_z4

            Rx_obj = rot_x4(config.OBJECT_PITCH_DEG)
            Ry_obj = rot_y4(config.OBJECT_YAW_DEG)
            Rz_obj = rot_z4(config.OBJECT_ROLL_DEG)
            rotation_matrix = (Rz_obj @ Ry_obj @ Rx_obj).astype(np.float32)
            hat_offset_matrix = np.eye(4, dtype=np.float32)
            hat_offset_matrix[0, 3] = config.OBJECT_OFFSET_X
            hat_offset_matrix[1, 3] = config.OBJECT_OFFSET_Y
            hat_offset_matrix[2, 3] = config.OBJECT_OFFSET_Z + float(
                getattr(config, "AXIS_Z_OFFSET", 0.0)
            )
            M_hat_shared = (
                M_smooth
                @ np.diag([1, 1, -1, 1]).astype(np.float32)
                @ hat_offset_matrix
                @ hat_scale_matrix
                @ config_vars["O"]
            ).astype(np.float32)

        # 얼굴 감지 상태 체크
        current_face_detected = M_smooth is not None
        face_just_detected = current_face_detected and not prev_face_detected
        prev_face_detected = current_face_detected

        if M_smooth is None:
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            render_background(prog_bg, vao_bg, tex_bg, frame_rgb, W, H)
            pygame.display.flip()
            continue

        jaw_mask_u8 = create_jaw_mask(res, W, H)
        jaw_mask_u8 = process_jaw_mask(jaw_mask_u8, res)
        collide_mask, inside_dist, gx, gy = create_collision_data(jaw_mask_u8)
        upload_jaw_mask(tex_jaw, jaw_mask_u8, W, H)

        # --- 2D 효과 적용 후 최종 프레임 결정 ---
        if config_vars.get("SHOW_SEG_DEBUG") and category_mask is not None:
            # 클래스별 색상 맵 (BGR)
            # 0:배경, 1:머리카락, 2:몸피부, 3:얼굴피부, 4:옷, 5:기타
            color_map = np.array(
                [
                    [0, 0, 0],  # Black
                    [0, 0, 255],  # Red
                    [0, 255, 0],  # Green
                    [255, 0, 0],  # Blue
                    [0, 255, 255],  # Yellow
                    [255, 0, 255],  # Magenta
                ],
                dtype=np.uint8,
            )

            # 카테고리 마스크를 컬러 이미지로 변환
            debug_seg_frame = color_map[category_mask]

            # 원본 영상과 블렌딩하여 확인 (RGB로 변환)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.addWeighted(frame_rgb, 0.4, debug_seg_frame, 0.6, 0)
            frame_rgb = np.ascontiguousarray(frame_rgb)
        else:
            frame_rgb = (
                app_debug.create_debug_frame(frame, jaw_mask_u8)
                if config_vars["JAW_DEBUG_SHOW"]
                else np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )

        # 9번 랜드마크 위치에 빨간 점 표시 (디버깅용)
        if pts_all is not None and config_vars.get("SHOW_AXES_HAT"):
            lm_idx = int(getattr(config, "ANCHOR_LM_IDX", 9))
            if 0 <= lm_idx < len(pts_all):
                x, y = int(pts_all[lm_idx][0]), int(pts_all[lm_idx][1])
                cv2.circle(frame_rgb, (x, y), 8, (255, 0, 0), -1)  # 빨간 원
                cv2.circle(frame_rgb, (x, y), 10, (255, 255, 255), 2)  # 흰 테두리
                # 랜드마크 번호 표시
                cv2.putText(
                    frame_rgb,
                    f"LM{lm_idx}",
                    (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        # Background draw: composite if mask; else BG-only if available; else camera passthrough
        if bg_enabled and gpu_bg_frame_rgb is not None:
            if gpu_person_mask is not None:
                glUseProgram(prog_bg)
                glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 1)
                glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
                glUniform1i(glGetUniformLocation(prog_bg, "uTexBG"), 8)
                glUniform1i(glGetUniformLocation(prog_bg, "uMask"), 7)
                glBindVertexArray(vao_bg)
                render_bg_composite(
                    prog_bg,
                    vao_bg,
                    tex_bg,
                    tex_bgimg,
                    tex_bgmask,
                    frame_rgb,
                    gpu_bg_frame_rgb,
                    gpu_person_mask,
                    W,
                    H,
                )
            else:
                glUseProgram(prog_bg)
                glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 0)
                glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
                render_background(prog_bg, vao_bg, tex_bg, gpu_bg_frame_rgb, W, H)
        else:
            glUseProgram(prog_bg)
            glUniform1i(glGetUniformLocation(prog_bg, "uUseComposite"), 0)
            glUniform1i(glGetUniformLocation(prog_bg, "uTexFG"), 0)
            render_background(prog_bg, vao_bg, tex_bg, frame_rgb, W, H)

        # 스티커를 배경 다음, 3D 객체 이전에 렌더링
        if config.UV_STICKER_ENABLE and pts_all is not None:
            renderer.draw_face_triangles(pts_all)

        if config_vars["SHOW_STRAP"]:
            # 끈 오브젝트도 9번 랜드마크 기준 축을 사용
            M_axes = config_vars.get("AXES_M_HAT")
            if M_axes is not None:
                # 끈 크기 조정을 위한 스케일 행렬
                strap_scale_matrix = np.diag(
                    [
                        config.STRAP_SCALE_MULTIPLIER,
                        config.STRAP_SCALE_MULTIPLIER,
                        config.STRAP_SCALE_MULTIPLIER,
                        1.0,
                    ]
                ).astype(np.float32)
                # 오브젝트 위치 오프셋을 위한 이동 행렬
                strap_offset_matrix = np.eye(4, dtype=np.float32)
                strap_offset_matrix[0, 3] = config.OBJECT_OFFSET_X
                strap_offset_matrix[1, 3] = config.OBJECT_OFFSET_Y
                strap_offset_matrix[2, 3] = config.OBJECT_OFFSET_Z + float(
                    getattr(config, "AXIS_Z_OFFSET", 0.0)
                )
                M_final = (
                    M_axes @ strap_offset_matrix @ strap_scale_matrix @ config_vars["O"]
                ).astype(np.float32)
            else:
                # 축이 설정되지 않았으면 기존 방식 사용
                strap_scale_matrix = np.diag(
                    [
                        config.STRAP_SCALE_MULTIPLIER,
                        config.STRAP_SCALE_MULTIPLIER,
                        config.STRAP_SCALE_MULTIPLIER,
                        1.0,
                    ]
                ).astype(np.float32)
                strap_offset_matrix = np.eye(4, dtype=np.float32)
                strap_offset_matrix[0, 3] = config.OBJECT_OFFSET_X
                strap_offset_matrix[1, 3] = config.OBJECT_OFFSET_Y
                strap_offset_matrix[2, 3] = config.OBJECT_OFFSET_Z + float(
                    getattr(config, "AXIS_Z_OFFSET", 0.0)
                )
                M_final = (
                    M_smooth
                    @ np.diag([1, 1, -1, 1]).astype(np.float32)
                    @ strap_offset_matrix
                    @ strap_scale_matrix
                    @ config_vars["O"]
                ).astype(np.float32)
            for batch in batches:
                if update_rope_physics(
                    batch,
                    M_final,
                    dt,
                    config_vars["GRAVITY_ON"],
                    collide_mask,
                    inside_dist,
                    gx,
                    gy,
                    P,
                    W,
                    H,
                    config.ANCHOR_SMOOTHING,
                    config.MAX_ANCHOR_DISTANCE,
                    config.STABILIZATION_FRAMES,
                    config.MAX_ROPE_VELOCITY,
                    face_just_detected,
                ):
                    first_pose_ready = True
                render_strap_batch(
                    prog_skin,
                    batch,
                    M_final,
                    P,
                    config_vars["JAW_OCCLUSION_ENABLED"],
                    first_pose_ready,
                )

        # 축 시각화
        if config_vars.get("SHOW_AXES_STRAP"):
            render_axes(
                prog_occ, axes_vaos, M_final, P, getattr(config, "AXES_LINE_WIDTH", 2.0)
            )

        if config_vars["SHOW_HAT"]:
            if config_vars["OCCLUSION_ENABLED"]:
                render_occlusion_depth_pass(
                    prog_occ,
                    vao_green,
                    green_count,
                    vao_blue,
                    blue_count,
                    M_hat_shared,
                    config.green_tx,
                    config.green_ty,
                    config.green_tz,
                    config.green_pitch_deg,
                    config.green_scale,
                    config.blue_pitch_deg,
                    config.blue_size,
                    config.blue_tx,
                    config.blue_ty,
                    config.blue_tz,
                    P,
                    config_vars["USE_SEG_MASK"],
                    config.seg_thr_hat,
                )
            render_hat_mesh(prog_mesh, vao_hat, hat_count, M_hat_shared, hat_tex, P)
            render_visualization_areas(
                prog_occ,
                vao_green,
                green_count,
                vao_blue,
                blue_count,
                M_hat_shared,
                config.green_tx,
                config.green_ty,
                config.green_tz,
                config.green_pitch_deg,
                config.green_scale,
                config.blue_pitch_deg,
                config.blue_size,
                config.blue_tx,
                config.blue_ty,
                config.blue_tz,
                config.blue_color,
                config.blue_alpha,
                P,
                config_vars["SHOW_GREEN_COLOR"],
                config_vars["SHOW_BLUE_COLOR"],
            )

        # 축 시각화 - 9번 랜드마크 위치의 축 사용
        if config_vars.get("SHOW_AXES_HAT"):
            M_axes = config_vars.get("AXES_M_HAT")
            if M_axes is not None:
                glDisable(GL_DEPTH_TEST)  # x-ray로 항상 보이게(원하면 제거)
                render_axes(
                    prog_occ,
                    axes_vaos,
                    M_axes,
                    P,
                    getattr(config, "AXES_LINE_WIDTH", 2.0),
                )
                glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
