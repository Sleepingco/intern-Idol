import numpy as np
import cv2
import mediapipe as mp
import config


def initialize_pose_detector():
    """Initialize MediaPipe Pose (single-person)."""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_pose_landmarks(mp_pose, image_rgb, W, H):
    """Return pose landmarks in pixel coords as (N,2) float32, or None."""
    if mp_pose is None:
        return None
    res = mp_pose.process(image_rgb)
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks.landmark
    pts = []
    for lm in lms:
        x = float(np.clip(lm.x, 0.0, 1.0)) * float(W)
        y = float(np.clip(lm.y, 0.0, 1.0)) * float(H)
        pts.append([x, y])
    if not pts:
        return None
    return np.array(pts, dtype=np.float32)


def _bbox_from_points(pts):
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def build_pose_roi_mask(landmarks_px, W, H):
    """Build a binary ROI mask (uint8) for the detected pose with configurable margins.
    Uses bbox of pose landmarks expanded by margins.
    """
    if landmarks_px is None or landmarks_px.size == 0:
        return None
    x0, y0, x1, y1 = _bbox_from_points(landmarks_px)

    # Asymmetrical margins to include torso/arms more below the face
    m_top = int(getattr(config, "POSE_GATE_MARGIN_TOP_PX", 60))
    m_bottom = int(getattr(config, "POSE_GATE_MARGIN_BOTTOM_PX", 420))
    m_left = int(getattr(config, "POSE_GATE_MARGIN_LEFT_PX", 140))
    m_right = int(getattr(config, "POSE_GATE_MARGIN_RIGHT_PX", 140))

    ix0 = max(0, int(round(x0 - m_left)))
    ix1 = min(int(W) - 1, int(round(x1 + m_right)))
    iy0 = max(0, int(round(y0 - m_top)))
    iy1 = min(int(H) - 1, int(round(y1 + m_bottom)))
    if ix1 <= ix0 or iy1 <= iy0:
        return None

    mask = np.zeros((int(H), int(W)), np.uint8)
    mask[iy0:iy1 + 1, ix0:ix1 + 1] = 255

    dil = int(getattr(config, "POSE_GATE_DILATE_PX", 0))
    if dil > 0:
        ksize = (max(1, dil) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, k, iterations=1)

    sigma = float(getattr(config, "POSE_GATE_SOFT_SIGMA", 0.0))
    if sigma > 0.0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return mask


def build_owner_pose_roi_mask(mp_pose, image_rgb, owner_pts, W, H):
    """Detect pose and return ROI mask if it plausibly matches the owner face.
    If mismatch, return None to let face-gate fallback handle it.
    """
    if mp_pose is None or owner_pts is None:
        return None
    pose_pts = detect_pose_landmarks(mp_pose, image_rgb, W, H)
    if pose_pts is None:
        return None

    # Check proximity: pose center close to owner's face center
    cx_pose, cy_pose = float(pose_pts[:, 0].mean()), float(pose_pts[:, 1].mean())
    cx_face, cy_face = float(owner_pts[:, 0].mean()), float(owner_pts[:, 1].mean())
    dx, dy = (cx_pose - cx_face), (cy_pose - cy_face)
    d2 = dx * dx + dy * dy
    # Accept radius proportional to face width
    fx0, fy0, fx1, fy1 = _bbox_from_points(owner_pts)
    face_w = max(1.0, fx1 - fx0)
    thr = float(getattr(config, "POSE_OWNER_MAX_DIST_RATIO", 1.4)) * face_w
    if d2 > (thr * thr):
        return None
    return build_pose_roi_mask(pose_pts, W, H)

