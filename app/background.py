import cv2
import numpy as np


def fit_background_image(img_bgr, W, H, mode="cover"):
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
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas
    # cover (default): fill and center-crop
    s = max(scale_w, scale_h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (nw - W) // 2
    y0 = (nh - H) // 2
    return resized[y0:y0+H, x0:x0+W]

