import cv2
import numpy as np
import config


def create_debug_frame(frame, jaw_mask_u8):
    if jaw_mask_u8 is None:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:] = config.JAW_COLOR
    m = (jaw_mask_u8.astype(np.float32) / 255.0) * float(config.JAW_ALPHA)
    frame_rgb = np.ascontiguousarray(
        (overlay * (1.0 - m[..., None]) + color_layer * m[..., None]).astype(np.uint8)
    )
    return frame_rgb

