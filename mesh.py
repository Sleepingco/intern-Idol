# mesh.py - 메쉬 생성 및 처리
import math
import numpy as np


def build_disk_mesh(r=1.0, h=0.02, seg=64):
    """원판 메쉬 생성"""
    verts = []
    # 상단면
    for i in range(seg):
        a0 = 2.0 * math.pi * (i / seg)
        a1 = 2.0 * math.pi * ((i + 1) / seg)
        x0, z0 = r * math.cos(a0), r * math.sin(a0)
        x1, z1 = r * math.cos(a1), r * math.sin(a1)
        verts += [[0.0, +h, 0.0], [x0, +h, z0], [x1, +h, z1]]
    # 하단면
    for i in range(seg):
        a0 = 2.0 * math.pi * (i / seg)
        a1 = 2.0 * math.pi * ((i + 1) / seg)
        x0, z0 = r * math.cos(a0), r * math.sin(a0)
        x1, z1 = r * math.cos(a1), r * math.sin(a1)
        verts += [[0.0, -h, 0.0], [x1, -h, z1], [x0, -h, z0]]
    # 측면
    for i in range(seg):
        a0 = 2.0 * math.pi * (i / seg)
        a1 = 2.0 * math.pi * ((i + 1) / seg)
        x0, z0 = r * math.cos(a0), r * math.sin(a0)
        x1, z1 = r * math.cos(a1), r * math.sin(a1)
        verts += [
            [x0, +h, z0],
            [x0, -h, z0],
            [x1, -h, z1],
            [x0, +h, z0],
            [x1, -h, z1],
            [x1, +h, z1],
        ]
    return np.array(verts, dtype=np.float32)


def build_plane_mesh():
    """평면 메쉬 생성"""
    return np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
