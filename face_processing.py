# face_processing.py - 얼굴 추적 및 처리
import math
import numpy as np
import cv2
import mediapipe as mp
import config
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from config import FACE_OVAL_IDX, JAW_GROW_PX, JAW_BLUR_SIGMA
from utils import (
    mat4_to_quat_t,
    slerp_quat,
    quat_t_to_mat4,
)


def initialize_face_detector(model_path):
    """얼굴 감지기 초기화"""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=int(getattr(config, "OWNER_NUM_FACES", 5)),
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return landmarker


def initialize_segmentation():
    """인물 세그멘테이션 초기화"""
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)


def process_face_pose(res, W, H, prev_q=None, prev_t=None):
    """얼굴 포즈 처리 및 스무딩"""
    if not res.facial_transformation_matrixes:
        return None, prev_q, prev_t

    M = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4).astype(np.float32)

    q, t = mat4_to_quat_t(M)
    if prev_q is None:
        prev_q, prev_t = q, t

    # 스무딩
    q_s = slerp_quat(prev_q, q, 0.5)
    t_s = prev_t * 0.5 + t * 0.5
    prev_q, prev_t = q_s, t_s

    M_smooth = quat_t_to_mat4(q_s, t_s)
    return M_smooth, prev_q, prev_t


def create_jaw_mask(res, W, H):
    """턱 마스크 생성"""
    if not res.facial_transformation_matrixes or not res.face_landmarks:
        return None

    lms = res.face_landmarks[0]
    pts = []
    for idx in FACE_OVAL_IDX:
        lm = lms[idx]
        x = int(np.clip(lm.x, 0, 1) * W)
        y = int(np.clip(lm.y, 0, 1) * H)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        jaw_mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(jaw_mask, hull, 255)
        return jaw_mask
    return None


def process_jaw_mask(jaw_mask_u8, res):
    """턱 마스크 후처리 (확장 및 블러)"""
    if jaw_mask_u8 is None:
        return None

    # 얼굴 회전각에 따른 확장 조정
    if res.facial_transformation_matrixes:
        M = (
            np.array(res.facial_transformation_matrixes[0])
            .reshape(4, 4)
            .astype(np.float32)
        )
        R = M[:3, :3]
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    else:
        yaw = 0.0

    base = int(JAW_GROW_PX)
    extra_x = int(min(14, max(0, abs(yaw) * 0.25)))
    kx = 2 * (base + extra_x) + 1
    ky = 2 * base + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    jaw_mask_u8 = cv2.dilate(jaw_mask_u8, k, iterations=1)

    if JAW_BLUR_SIGMA > 0.0:
        jaw_mask_u8 = cv2.GaussianBlur(jaw_mask_u8, (0, 0), JAW_BLUR_SIGMA)

    return jaw_mask_u8


def create_collision_data(jaw_mask_u8):
    """충돌 감지용 데이터 생성"""
    if jaw_mask_u8 is None:
        return None, None, None, None

    _inside = (jaw_mask_u8 > 127).astype(np.uint8) * 255
    inside_dist = cv2.distanceTransform(_inside, cv2.DIST_L2, 3).astype(np.float32)
    gx = cv2.Sobel(inside_dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inside_dist, cv2.CV_32F, 0, 1, ksize=3)
    collide_mask = _inside

    return collide_mask, inside_dist, gx, gy


def process_segmentation_mask(mp_selfie, image_rgb):
    """인물 세그멘테이션 마스크 처리"""
    seg_res = mp_selfie.process(image_rgb)
    if seg_res.segmentation_mask is not None:
        seg_mask_u8 = (np.clip(seg_res.segmentation_mask, 0.0, 1.0) * 255).astype(
            np.uint8
        )
        return seg_mask_u8


def get_face_landmarks_from_result(res, W, H):
    """MediaPipe 결과에서 랜드마크 좌표(px)를 추출"""
    if not res or not res.face_landmarks:
        return None

    face_landmarks = res.face_landmarks[0]  # 첫 번째 얼굴만 사용
    pts = np.array([(lm.x * W, lm.y * H) for lm in face_landmarks], dtype=np.float32)
    return pts


def get_all_face_landmarks_from_result(res, W, H):
    """모든 얼굴의 랜드마크 목록을 반환. 없으면 빈 리스트.
    반환: [np.ndarray(478x2, float32), ...]
    """
    faces = []
    if not res or not res.face_landmarks:
        return faces
    for fl in res.face_landmarks:
        pts = np.array([(lm.x * W, lm.y * H) for lm in fl], dtype=np.float32)
        faces.append(pts)
    return faces
