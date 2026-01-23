# utils.py - 유틸리티 함수들
import math
import numpy as np
import cv2
from config import (
    o_params,
    hat_base_params,
)


def rot_x4(deg):
    """X축 회전 4x4 행렬 생성"""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=np.float32)
    M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    return M

def rot_y4(deg):
    """Y축 회전 4x4 행렬 생성"""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=np.float32)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    return M

def rot_z4(deg):
    """Z축 회전 4x4 행렬 생성"""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4, dtype=np.float32)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def make_O(
    scale=1.0,
    tx=0.0,
    ty=0.0,
    tz=0.0,
    rx_deg=0.0,
    ry_deg=0.0,
    rz_deg=0.0,
    mirror_fix=False,
):
    """오브젝트 변환 행렬 생성"""
    sx = sy = sz = scale
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        M = np.eye(4, dtype=np.float32)
        M[1, 1] = c
        M[1, 2] = -s
        M[2, 1] = s
        M[2, 2] = c
        return M

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        M = np.eye(4, dtype=np.float32)
        M[0, 0] = c
        M[0, 2] = s
        M[2, 0] = -s
        M[2, 2] = c
        return M

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        M = np.eye(4, dtype=np.float32)
        M[0, 0] = c
        M[0, 1] = -s
        M[1, 0] = s
        M[1, 1] = c
        return M

    S = np.diag([sx, sy, sz, 1]).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    O = T @ (Rz(rz) @ Ry(ry) @ Rx(rx)) @ S
    if mirror_fix:
        O = np.diag([-1, 1, 1, 1]).astype(np.float32) @ O
    return O


def rebuild_O():
    """현재 설정으로 턱끈 변환 행렬 재구성"""
    return make_O(
        scale=o_params["scale"],
        tx=o_params["tx"],
        ty=o_params["ty"],
        tz=o_params["tz"],
        rx_deg=o_params["rx"],
        ry_deg=o_params["ry"],
        rz_deg=o_params["rz"],
        mirror_fix=o_params["mirror"],
    )


def rebuild_hat_base_O():
    """현재 설정으로 갓 기본 변환 행렬 재구성"""
    return make_O(
        scale=hat_base_params["scale"],
        tx=hat_base_params["tx"],
        ty=hat_base_params["ty"],
        tz=hat_base_params["tz"],
        rx_deg=hat_base_params["rx"],
        ry_deg=hat_base_params["ry"],
        rz_deg=hat_base_params["rz"],
        mirror_fix=hat_base_params["mirror"],
    )


def perspective_from_hfov(hfov_deg, aspect, near, far):
    """수평 시야각으로부터 원근 투영 행렬 생성"""
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
    f = 1.0 / math.tan(vfov / 2.0)
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P




def world_to_screen_px(P_world_3, P, W, H):
    """월드 좌표를 스크린 픽셀 좌표로 변환"""
    x, y, z = float(P_world_3[0]), float(P_world_3[1]), float(P_world_3[2])
    clip = P @ np.array([x, y, z, 1.0], np.float32)
    w = float(clip[3])
    if abs(w) < 1e-8:
        return None
    ndc_x = float(clip[0] / w)
    ndc_y = float(clip[1] / w)
    sx = (ndc_x * 0.5 + 0.5) * float(W)
    sy = (1.0 - (ndc_y * 0.5 + 0.5)) * float(H)
    return sx, sy, z


def screen_px_to_world_delta(dx_px, dy_px, z_world, P, W, H):
    """스크린 픽셀 델타를 월드 좌표 델타로 변환"""
    depth = -float(z_world)  # 카메라 전방은 음수 → 양수로
    if depth <= 1e-6:
        return np.zeros(3, np.float32)
    kx = (2.0 * depth) / (float(P[0, 0]) * float(W))
    ky = (2.0 * depth) / (float(P[1, 1]) * float(H))
    return np.array([kx * float(dx_px), -ky * float(dy_px), 0.0], np.float32)


def slerp_quat(q0, q1, t):
    """쿼터니언 구면 선형 보간"""
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = math.acos(np.clip(dot, -1, 1))
    s0 = math.sin(th0)
    return (math.sin((1 - t) * th0) / s0) * q0 + (math.sin(t * th0) / s0) * q1


def mat4_to_quat_t(M):
    """4x4 행렬을 쿼터니언과 평행이동으로 분해"""
    R = M[:3, :3]
    t = M[:3, 3].copy()
    qw = math.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 1e-8)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    return np.array([qw, qx, qy, qz], np.float32), t.astype(np.float32)


def quat_t_to_mat4(q, t):
    """쿼터니언과 평행이동을 4x4 행렬로 합성"""
    qw, qx, qy, qz = q
    R = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        np.float32,
    )
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def align_y_to_vec(v):
    """벡터 v를 Y축으로 하는 변환 행렬 생성"""
    d = np.array(v, np.float32)
    n = np.linalg.norm(d) + 1e-8
    y = d / n
    up = np.array([0, 0, 1], np.float32)
    if abs(float(np.dot(y, up))) > 0.999:
        up = np.array([1, 0, 0], np.float32)
    x = np.cross(up, y)
    x /= np.linalg.norm(x) + 1e-8
    z = np.cross(y, x)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = x
    M[:3, 1] = y
    M[:3, 2] = z
    return M


def gauss_blur(img, sigma: float):
    if sigma <= 0.0:
        return img
    k = int(max(1, round(sigma * 3))) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)

def eye_open_ratio(pts, up_idx, lo_idx, corner_pair):
    up = np.mean(pts[up_idx], axis=0)
    lo = np.mean(pts[lo_idx], axis=0)
    vertical = float(np.linalg.norm(up - lo))
    c0, c1 = corner_pair
    horiz = float(np.linalg.norm(pts[c0] - pts[c1]))
    if horiz < 1e-6:
        return 0.0
    return vertical / horiz

def perspective_from_intrinsics(fx, fy, cx, cy, near, far, W, H):
    """
    OpenCV 카메라내부파라미터로부터 OpenGL 투영행렬 구성.
    이 행렬로 gl_Position = P * [Xc,Yc,Zc,1] (카메라좌표) 시,
    화면 픽셀 중심(cx,cy) 및 fx/fy 스케일을 정확히 반영.
    """
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2.0 * fx / float(W)
    P[1, 1] = 2.0 * fy / float(H)
    P[0, 2] = 1.0 - 2.0 * cx / float(W)
    P[1, 2] = 2.0 * cy / float(H) - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -(2.0 * far * near) / (far - near)
    P[3, 2] = -1.0
    return P
