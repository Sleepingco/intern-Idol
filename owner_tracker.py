import time
import numpy as np
import config


class OwnerTracker:
    """Tracks a single 'owner' face across frames and enforces near gating.

    - select_owner(faces_pts): choose and maintain the current owner using
      proximity to previous center with hysteresis on competitor size.
    - near_gate(owner_pts): return None if the owner is not close enough to camera.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.W = int(frame_w)
        self.H = int(frame_h)
        self.owner_center = None  # (cx, cy)
        self.owner_area = 0.0
        self.owner_miss = 0
        self.switch_sustain = 0
        self.owner_idx = -1
        self.last_near_log = 0.0

    def reset(self):
        self.owner_center = None
        self.owner_area = 0.0
        self.owner_miss = 0
        self.switch_sustain = 0
        self.owner_idx = -1

    def select_owner(self, faces_pts):
        """faces_pts: list of (478x2) np.ndarray; returns owner_pts or None."""
        if not faces_pts:
            timeout = int(getattr(config, "OWNER_TIMEOUT_FRAMES", 30))
            self.owner_miss += 1
            if self.owner_miss >= timeout:
                self.reset()
            return None

        # Compute centers and areas
        infos = []  # (idx, (cx,cy), area)
        for i, pts in enumerate(faces_pts):
            xs, ys = pts[:, 0], pts[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
            area = w * h
            cx, cy = float(xs.mean()), float(ys.mean())
            infos.append((i, (cx, cy), area))

        switch_margin = float(getattr(config, "OWNER_SWITCH_MARGIN", 1.15))
        switch_frames = int(getattr(config, "OWNER_SWITCH_FRAMES", 5))

        # Largest by area (proxy for closeness)
        imax = max(infos, key=lambda t: t[2])[0]

        if self.owner_center is None:
            self.owner_idx = imax
            self.owner_center = infos[self.owner_idx][1]
            self.owner_area = infos[self.owner_idx][2]
            self.owner_miss = 0
            self.switch_sustain = 0
        else:
            # Choose the face nearest to previous owner center
            def d2(p, q):
                return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

            inear = min(infos, key=lambda t: d2(t[1], self.owner_center))[0]
            oc = infos[inear]
            self.owner_idx = inear
            self.owner_center = oc[1]
            owner_area_cur = oc[2]

            # Hysteresis: switch only if a competitor is significantly larger for several frames
            comp_area = infos[imax][2]
            if imax != inear and comp_area > owner_area_cur * switch_margin:
                self.switch_sustain += 1
            else:
                self.switch_sustain = 0
            if self.switch_sustain >= switch_frames:
                self.owner_idx = imax
                self.owner_center = infos[imax][1]
                self.owner_area = infos[imax][2]
                self.switch_sustain = 0
            else:
                self.owner_area = owner_area_cur
            self.owner_miss = 0

        return faces_pts[self.owner_idx]

    def near_gate(self, owner_pts):
        """Return owner_pts if close enough to camera, else None. Logs measurement once per second."""
        if owner_pts is None:
            return None
        xs, ys = owner_pts[:, 0], owner_pts[:, 1]
        y0, y1 = float(ys.min()), float(ys.max())
        face_h = max(1.0, y1 - y0)
        min_h_px = float(getattr(config, "OWNER_NEAR_MIN_HEIGHT_PX", 120))
        min_h_ratio = float(getattr(config, "OWNER_NEAR_MIN_RATIO", 0.22))
        h_thr = max(min_h_px, min_h_ratio * self.H)

        # Throttled measurement logging
        try:
            tnow = time.monotonic()
            if tnow - self.last_near_log > 1.0:
                # print(
                #     f"[NEAR_MEASURE] face_h={face_h:.1f}px, H={self.H}, ratio={face_h/self.H:.3f}, thr={h_thr:.1f}px"
                # )
                self.last_near_log = tnow
        except Exception:
            pass

        if face_h < h_thr:
            return None
        return owner_pts
