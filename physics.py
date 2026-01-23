# physics.py - 물리 시뮬레이션 모듈
import math
import numpy as np
from utils import world_to_screen_px, screen_px_to_world_delta
from config import g_strength


def update_rope_physics(
    batch,
    M_final,
    dt,
    gravity_on,
    collide_mask=None,
    inside_dist=None,
    gx=None,
    gy=None,
    P=None,
    W=None,
    H=None,
    anchor_smoothing=0.15,  # 앵커 평활화 계수 (0=평활화 없음, 1=최대 평활화)
    max_anchor_distance=50.0,  # 프레임 간 앵커 최대 이동 거리
    stabilization_frames=30,  # 안정화 프레임 수
    max_rope_velocity=100.0,  # 최대 로프 속도 제한
    face_just_detected=False,  # 얼굴이 새로 감지되었는지 여부
):
    """로프 물리 업데이트"""
    if not batch.get("has_skin") or batch.get("skin_meta") is None:
        return False

    seq = batch["seq"]
    L = batch["Lseg"]
    P_ext_model = batch["P_ext_model"]

    # 첫 초기화
    if batch["P_world"] is None:
        hp = np.concatenate(
            [P_ext_model, np.ones((P_ext_model.shape[0], 1), np.float32)],
            axis=1,
        )
        P_world_init = (M_final @ hp.T).T[:, :3]
        batch["P_world"] = P_world_init.copy()
        batch["P_prev_world"] = P_world_init.copy()
        # 이전 앵커 위치 저장 및 안정화 카운터 초기화
        batch["prev_anchorA"] = None
        batch["prev_anchorB"] = None
        batch["stabilization_counter"] = 0
        batch["is_stabilizing"] = True
        return True

    P_world = batch["P_world"]
    P_prev = batch["P_prev_world"]
    N_all = P_world.shape[0]
    
    # 안정화 상태 확인 및 업데이트
    if "stabilization_counter" not in batch:
        batch["stabilization_counter"] = 0
        batch["is_stabilizing"] = True
    
    # 얼굴이 새로 감지되었으면 안정화 재시작
    if face_just_detected:
        batch["stabilization_counter"] = 0
        batch["is_stabilizing"] = True
    
    if batch["is_stabilizing"]:
        batch["stabilization_counter"] += 1
        if batch["stabilization_counter"] >= stabilization_frames:
            batch["is_stabilizing"] = False

    # 앵커 포인트 계산
    anchorA_raw = (
        M_final
        @ np.array(
            [P_ext_model[0, 0], P_ext_model[0, 1], P_ext_model[0, 2], 1.0],
            np.float32,
        )
    )[:3]
    anchorB_raw = (
        M_final
        @ np.array(
            [
                P_ext_model[-1, 0],
                P_ext_model[-1, 1],
                P_ext_model[-1, 2],
                1.0,
            ],
            np.float32,
        )
    )[:3]
    
    # 앵커 평활화 및 거리 제한 (안정화 중일 때는 더 강하게 적용)
    if batch["prev_anchorA"] is None:
        anchorA_world = anchorA_raw
        anchorB_world = anchorB_raw
    else:
        # 안정화 중일 때는 더 강한 제어 적용
        current_smoothing = anchor_smoothing * (3.0 if batch["is_stabilizing"] else 1.0)
        current_max_distance = max_anchor_distance * (0.3 if batch["is_stabilizing"] else 1.0)
        
        # 이전 앵커 위치와의 거리 계산
        distA = float(np.linalg.norm(anchorA_raw - batch["prev_anchorA"]))
        distB = float(np.linalg.norm(anchorB_raw - batch["prev_anchorB"]))
        
        # 거리 제한 적용
        if distA > current_max_distance:
            dirA = (anchorA_raw - batch["prev_anchorA"]) / distA
            anchorA_limited = batch["prev_anchorA"] + dirA * current_max_distance
        else:
            anchorA_limited = anchorA_raw
            
        if distB > current_max_distance:
            dirB = (anchorB_raw - batch["prev_anchorB"]) / distB
            anchorB_limited = batch["prev_anchorB"] + dirB * current_max_distance
        else:
            anchorB_limited = anchorB_raw
        
        # 평활화 적용 (안정화 중일 때는 더 부드럽게)
        smooth_factor = min(current_smoothing, 0.8)  # 최대 0.8로 제한
        anchorA_world = (
            batch["prev_anchorA"] * smooth_factor 
            + anchorA_limited * (1.0 - smooth_factor)
        ).astype(np.float32)
        anchorB_world = (
            batch["prev_anchorB"] * smooth_factor 
            + anchorB_limited * (1.0 - smooth_factor)
        ).astype(np.float32)
    
    # 현재 앵커 위치 저장
    batch["prev_anchorA"] = anchorA_world.copy()
    batch["prev_anchorB"] = anchorB_world.copy()

    # 베를레 적분 (안정화 중일 때는 속도 제한 적용)
    Vv = (P_world - P_prev) * (1.0 - batch["rope_damp"])
    
    # 안정화 중일 때 속도 제한
    if batch["is_stabilizing"]:
        for i in range(N_all):
            v_magnitude = float(np.linalg.norm(Vv[i]))
            if v_magnitude > max_rope_velocity * 0.3:  # 안정화 중에는 30%로 제한
                Vv[i] = Vv[i] / v_magnitude * max_rope_velocity * 0.3
    else:
        # 일반 상태에서도 극한 속도 제한
        for i in range(N_all):
            v_magnitude = float(np.linalg.norm(Vv[i]))
            if v_magnitude > max_rope_velocity:
                Vv[i] = Vv[i] / v_magnitude * max_rope_velocity
    
    # 안정화 중일 때는 중력을 줄임
    gravity_factor = 0.2 if batch["is_stabilizing"] else 1.0
    acc = np.array([0.0, g_strength * gravity_factor if gravity_on else 0.0, 0.0], np.float32)
    P_new = P_world + Vv + acc * (dt * dt)

    # 앵커 고정
    P_new[0] = anchorA_world
    P_new[-1] = anchorB_world

    # 거리 제약 (안정화 중일 때는 더 부드럽게)
    constraint_iters = batch["rope_iters"] if not batch["is_stabilizing"] else max(3, batch["rope_iters"] // 3)
    constraint_strength = 0.3 if batch["is_stabilizing"] else 1.0
    
    for _ in range(constraint_iters):
        for i in range(N_all - 1):
            d = P_new[i + 1] - P_new[i]
            dist = float(np.linalg.norm(d)) + 1e-8
            diff = (dist - float(L[i])) / dist * constraint_strength
            if i == 0:
                P_new[i + 1] -= d * diff
            elif i == N_all - 2:
                P_new[i] += d * diff
            else:
                corr = 0.5 * d * diff
                P_new[i] += corr
                P_new[i + 1] -= corr
        P_new[0] = anchorA_world
        P_new[-1] = anchorB_world

    # 충돌 처리 (안정화 중에는 비활성화)
    if (
        not batch["is_stabilizing"]  # 안정화 중이 아닐 때만 충돌 처리
        and collide_mask is not None
        and inside_dist is not None
        and gx is not None
        and gy is not None
        and P is not None
        and W is not None
        and H is not None
    ):
        apply_collision_constraints(
            P_new, N_all, collide_mask, inside_dist, gx, gy, P, W, H
        )

    batch["P_prev_world"] = P_world
    batch["P_world"] = P_new
    return True


def apply_collision_constraints(
    P_new, N_all, collide_mask, inside_dist, gx, gy, P, W, H
):
    """충돌 제약 적용"""
    CONTACT_EPS_PX = 1.0
    Hm1 = H - 1
    Wm1 = W - 1
    collided_idx = []

    for i_pt in range(1, N_all - 1):
        proj = world_to_screen_px(P_new[i_pt], P, W, H)
        if proj is None:
            continue
        sx, sy, z_world = proj
        ix = int(np.clip(round(sx), 0, Wm1))
        iy = int(np.clip(round(sy), 0, Hm1))

        if collide_mask[iy, ix] == 255:
            dpx = float(inside_dist[iy, ix])
            if dpx > 0.2:
                nx = -float(gx[iy, ix])
                ny = -float(gy[iy, ix])
                nlen = math.hypot(nx, ny) + 1e-8
                nx /= nlen
                ny /= nlen
                push_px = dpx + CONTACT_EPS_PX
                dW = screen_px_to_world_delta(
                    push_px * nx, push_px * ny, z_world, P, W, H
                )
                P_new[i_pt] = (P_new[i_pt] + dW).astype(np.float32)
                collided_idx.append(i_pt)

    return collided_idx
