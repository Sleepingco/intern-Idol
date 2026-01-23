import os
import json
import numpy as np

import config
from utils import perspective_from_intrinsics


def build_projection_from_intrinsics(W, H):
    fx = fy = cx = cy = None
    if getattr(config, "USE_INTRINSICS", False):
        try:
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

    # Fallback: 대략적인 기본값
    if fx is None:
        fx = fy = float(W)
        cx = float(W) / 2.0
        cy = float(H) / 2.0

    P = perspective_from_intrinsics(
        fx, fy, cx, cy, config.NEAR_Z, config.FAR_Z, W, H
    )
    return P, (fx, fy, cx, cy)


def normalize_rotation3x3(R):
    Rn = R.copy()
    for i in range(3):
        n = np.linalg.norm(Rn[:, i]) + 1e-8
        Rn[:, i] /= n
    return Rn


def two_orthonormal(v):
    """Given v(3,), return two orthonormal vectors as (3x2) matrix."""
    v = v / (np.linalg.norm(v) + 1e-8)
    a = np.array([1.0, 0.0, 0.0], np.float32) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0], np.float32)
    u1 = np.cross(v, a)
    u1 /= (np.linalg.norm(u1) + 1e-8)
    u2 = np.cross(v, u1)
    u2 /= (np.linalg.norm(u2) + 1e-8)
    return np.stack([u1, u2], axis=1)

