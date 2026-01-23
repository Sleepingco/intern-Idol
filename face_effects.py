# module/effects2d/effects.py
import numpy as np
import cv2
import onnxruntime as ort
import config

from utils import gauss_blur
from config import (
    EYE_ENABLE, IRIS_SCALE, EYE_ALPHA, EYE_EDGE_BLUR,
    EYE_H_IN, EYE_S_IN, EYE_V_IN, EYE_H_OUT, EYE_S_OUT, EYE_V_OUT,
    PUPIL_SIZE, PUPIL_THIN, PUPIL_ALPHA,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    SMOKY_ENABLE, SMOKY_COLOR, SMOKY_SIGMA,
    SCLERA_COLOR_ENABLE, SCLERA_COLOR, SCLERA_ALPHA, SCLERA_FEATHER,
    SKIN_TONE_ENABLE, SKIN_TONE_TARGET, SKIN_TONE_STRENGTH, SKIN_TONE_BRIGHTNESS, SKIN_TONE_FEATHER,
    SELFIE_MODEL_PATH, MODEL_INPUT_SIZE,
    SKIN_CLASS_IDS, SKIN_PROB_THR, USE_SKIN_PROB_SUM,
    LIP_COLOR_ENABLE, LIP_COLOR, LIP_ALPHA, LIP_FEATHER, LIP_BRIGHTNESS, LIP_OUTER_IDX, LIP_INNER_IDX
)


class Effects2D:
    def __init__(self):
        # Pick the best available ONNXRuntime provider (GPU if possible)
        try:
            avail = set(ort.get_available_providers())
            preferred = [
                'CUDAExecutionProvider',
                'DmlExecutionProvider',
                'OpenVINOExecutionProvider',
                'CoreMLExecutionProvider',
                'CPUExecutionProvider',
            ]
            providers = [p for p in preferred if p in avail] or ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(
                SELFIE_MODEL_PATH,
                providers=providers,
            )
        except Exception:
            # Fallback to CPU
            self.ort_session = ort.InferenceSession(
                SELFIE_MODEL_PATH,
                providers=['CPUExecutionProvider'],
            )

    # --- Letterbox preprocess helpers (preserve aspect ratio) ---
    def _preprocess_letterbox(self, frame_bgr, input_size):
        """Convert BGR->RGB, resize with aspect preserved into a padded canvas.
        Returns (input_tensor[1,H,W,3], meta dict for reverse mapping).
        """
        h, w = frame_bgr.shape[:2]
        model_w, model_h = int(input_size[0]), int(input_size[1])
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # scale to fit into model size while preserving aspect ratio
        scale = min(model_w / float(w), model_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

        # letterbox pad to center
        canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        x0 = (model_w - new_w) // 2
        y0 = (model_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # normalize to [0,1] and add batch dim
        input_tensor = (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :, :]
        meta = {
            'x0': x0,
            'y0': y0,
            'new_w': new_w,
            'new_h': new_h,
            'orig_w': w,
            'orig_h': h,
            'model_w': model_w,
            'model_h': model_h,
        }
        return input_tensor, meta

    def _restore_mask_from_letterbox(self, meta, mask_model_space, is_label=True):
        """Crop padded region and resize mask back to original frame size.
        mask_model_space: 2D array (model_h, model_w) in model space.
        is_label: INTER_NEAREST for label maps; otherwise linear.
        """
        x0 = meta['x0']; y0 = meta['y0']
        nw = meta['new_w']; nh = meta['new_h']
        ow = meta['orig_w']; oh = meta['orig_h']
        cropped = mask_model_space[y0:y0 + nh, x0:x0 + nw]
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
        if cropped.size == 0:
            return cv2.resize(mask_model_space, (ow, oh), interpolation=interp)
        return cv2.resize(cropped, (ow, oh), interpolation=interp)

    def _to_probabilities(self, logits_or_probs: np.ndarray) -> np.ndarray:
        """Ensure (H,W,C) tensor is probabilities along C.
        If already in [0,1] and sums≈1, return as-is; else apply softmax.
        """
        arr = logits_or_probs.astype(np.float32)
        # quick check: values in [0,1] and sum close to 1
        mn, mx = float(arr.min()), float(arr.max())
        sums = arr.sum(axis=-1, keepdims=True)
        if mn >= 0.0 and mx <= 1.0:
            # tolerate small drift
            if np.all(np.isfinite(sums)) and np.median(np.abs(sums - 1.0)) < 1e-2:
                return arr
        # softmax
        x = arr - arr.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        denom = np.clip(ex.sum(axis=-1, keepdims=True), 1e-8, None)
        return ex / denom

    # === get_skin_mask_onnx: letterbox + 한 사람 필터링 로직 통합 ===
    def get_skin_mask_onnx(self, frame_bgr, pts_all, input_size=MODEL_INPUT_SIZE):
        input_tensor, meta = self._preprocess_letterbox(frame_bgr, input_size)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        outputs = self.ort_session.run([output_name], {input_name: input_tensor})
        segmentation_output = outputs[0]
        seg_map = np.squeeze(segmentation_output, axis=0)

        category_mask_model = np.argmax(seg_map, axis=-1).astype(np.uint8)
        
        if USE_SKIN_PROB_SUM:
            probs = self._to_probabilities(seg_map)
            p_skin = probs[..., SKIN_CLASS_IDS].sum(axis=-1)
            full_skin_mask_model = (p_skin >= float(SKIN_PROB_THR)).astype(np.uint8) * 255
        else:
            skin_condition = np.isin(category_mask_model, np.array(SKIN_CLASS_IDS, dtype=np.int64))
            full_skin_mask_model = np.where(skin_condition, 255, 0).astype(np.uint8)

        full_skin_mask = self._restore_mask_from_letterbox(meta, full_skin_mask_model, is_label=True)
        category_mask = self._restore_mask_from_letterbox(meta, category_mask_model, is_label=True)
        
        # --- 한 사람 필터링 로직 (main.py에서 이동) ---
        skin_mask = np.zeros_like(full_skin_mask)
        if pts_all is not None:
            # 1. 얼굴피부(클래스 3) 마스크와 몸피부(클래스 2) 마스크를 분리
            face_skin_mask = np.where(category_mask == 3, 255, 0).astype(np.uint8)
            body_skin_mask = np.where(category_mask == 2, 255, 0).astype(np.uint8)

            # 2. 주인공 얼굴 찾기
            f_num, f_labels, f_stats, f_centroids = cv2.connectedComponentsWithStats(face_skin_mask, connectivity=8)
            my_face_label = -1
            if f_num > 1:
                face_center = np.mean(pts_all, axis=0)
                for i in range(1, f_num):
                    x, y, w, h, _ = f_stats[i]
                    if x <= face_center[0] < x + w and y <= face_center[1] < y + h:
                        my_face_label = i
                        break
            
            if my_face_label != -1:
                # 3. 주인공 얼굴을 최종 마스크에 추가
                skin_mask[f_labels == my_face_label] = 255
                my_face_centroid = f_centroids[my_face_label]
                my_face_width = f_stats[my_face_label][2]
                
                # 4. 주인공 얼굴과 가까운 몸피부 영역들 추가
                b_num, b_labels, b_stats, b_centroids = cv2.connectedComponentsWithStats(body_skin_mask, connectivity=8)
                if b_num > 1:
                    threshold_dist = my_face_width * 3.0
                    for i in range(1, b_num):
                        dist = np.linalg.norm(b_centroids[i] - my_face_centroid)
                        if dist < threshold_dist:
                            skin_mask[b_labels == i] = 255
            else:
                # Fallback: 주인공 얼굴 못찾으면 가장 큰 피부 영역 사용
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_skin_mask, connectivity=8)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    skin_mask[labels == largest_label] = 255
        
        return skin_mask, category_mask

    def get_person_mask_onnx(self, frame_bgr, input_size=MODEL_INPUT_SIZE, bg_class_index: int = 0, pts_all=None, pose_roi_mask=None):
        """Returns a person mask using the multiclass ONNX model.
        - If BG_MASK_SOFT is True: returns a soft mask (0..255) from probabilities (1 - p_bg),
          restored with bilinear filtering for smoother edges.
        - Else: returns a binary mask (255=person, 0=background) via argmax label, restored with nearest.
        - If BG_GATE_TO_FACE is True and pts_all provided: keep only the connected component that contains
          the current face center so that only the person with hat/pattern is segmented.
        """
        input_tensor, meta = self._preprocess_letterbox(frame_bgr, input_size)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        outputs = self.ort_session.run([output_name], {input_name: input_tensor})
        segmentation_output = outputs[0]  # (1, H, W, C)
        segmentation_map = np.squeeze(segmentation_output, axis=0)
        category_mask = np.argmax(segmentation_map, axis=-1)

        use_soft = bool(getattr(config, "BG_MASK_SOFT", True))
        if use_soft:
            # Build soft mask from probabilities: person = 1 - p_bg
            probs = self._to_probabilities(segmentation_map)
            p_bg = probs[..., int(bg_class_index)]
            p_person = 1.0 - p_bg
            soft_mask_model = np.clip((p_person * 255.0).astype(np.uint8), 0, 255)
            person_mask = self._restore_mask_from_letterbox(meta, soft_mask_model, is_label=False)
            # Optional gamma/threshold shaping to reduce bleed-through
            gamma = float(getattr(config, "BG_MASK_GAMMA", 1.0))
            thr = int(getattr(config, "BG_MASK_BIN_THR", 0))
            if gamma != 1.0 or thr > 0:
                m = person_mask.astype(np.float32) / 255.0
                if gamma != 1.0:
                    # gamma>1: harder edges (more background rejection)
                    # gamma<1: softer edges (more foreground preservation)
                    m = np.power(np.clip(m, 0.0, 1.0), gamma)
                if thr > 0:
                    t = np.clip(thr / 255.0, 0.0, 1.0)
                    m = (m >= t).astype(np.float32)
                person_mask = np.clip((m * 255.0).astype(np.uint8), 0, 255)
        else:
            person_condition = (category_mask != int(bg_class_index))
            person_mask_model = np.where(person_condition, 255, 0).astype(np.uint8)
            person_mask = self._restore_mask_from_letterbox(meta, person_mask_model, is_label=True)

        # Morphological smoothing (optional)
        close_it = max(0, int(getattr(config, "BG_MASK_CLOSE", 1)))
        open_it = max(0, int(getattr(config, "BG_MASK_OPEN", 0)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if close_it > 0:
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, k, iterations=close_it)
        if open_it > 0:
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, k, iterations=open_it)

        # Optional guided filtering to align edges to image gradients
        if bool(getattr(config, "BG_MASK_GUIDED", False)):
            try:
                import cv2.ximgproc as xip
                guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                src = (person_mask.astype(np.float32) / 255.0)
                gf = xip.guidedFilter(guide=guide, src=src, radius=8, eps=1e-4)
                person_mask = np.clip((gf * 255.0).astype(np.uint8), 0, 255)
            except Exception:
                pass

        # Gate to current face (only segment the person wearing hat/pattern)
        if pts_all is not None and bool(getattr(config, "BG_GATE_TO_FACE", True)):
            gate_mode = str(getattr(config, "BG_GATE_MODE", "cc")).lower()  # 'cc' or 'bbox'
            ys, xs = pts_all[:, 1], pts_all[:, 0]
            cx = float(np.mean(xs)); cy = float(np.mean(ys))
            x0 = int(max(0, xs.min())); x1 = int(min(person_mask.shape[1]-1, xs.max()))
            y0 = int(max(0, ys.min())); y1 = int(min(person_mask.shape[0]-1, ys.max()))
            face_w = max(1.0, float(x1 - x0))
            face_h = max(1.0, float(y1 - y0))

            if gate_mode == "bbox":
                # Asymmetrical margins for a taller ROI (to include torso)
                m_top = int(getattr(config, "BG_GATE_MARGIN_TOP_PX", 100))
                m_bottom = int(getattr(config, "BG_GATE_MARGIN_BOTTOM_PX", 400))
                m_left = int(getattr(config, "BG_GATE_MARGIN_LEFT_PX", 150))
                m_right = int(getattr(config, "BG_GATE_MARGIN_RIGHT_PX", 150))

                # x0, y0 are top-left of face bbox; x1, y1 are bottom-right
                gx0 = max(0, x0 - m_left)
                gx1 = min(person_mask.shape[1] - 1, x1 + m_right)
                gy0 = max(0, y0 - m_top)
                gy1 = min(person_mask.shape[0] - 1, y1 + m_bottom)
                
                gate = np.zeros_like(person_mask, dtype=np.uint8)
                gate[gy0:gy1 + 1, gx0:gx1 + 1] = 255
                person_mask = cv2.bitwise_and(person_mask, gate)
            else:
                # Connected-components with ellipse vote (default)
                thr_bin = int(getattr(config, "BG_GATE_BIN_THR", 96))
                thr_low = int(getattr(config, "BG_GATE_THR_LOW", 48))
                # Use lower threshold for CC discovery to not lose thin connections (arms/cloth)
                bin_mask = (person_mask.astype(np.uint8) >= thr_low).astype(np.uint8) * 255

                # Elliptical ROI around the face center to vote the correct CC
                roi_scale = float(getattr(config, "BG_GATE_ROI_SCALE", 1.4))
                rx = int(max(1, round(face_w * roi_scale * 0.5)))
                ry = int(max(1, round(face_h * roi_scale * 0.5)))
                Y, X = np.ogrid[:person_mask.shape[0], :person_mask.shape[1]]
                roi_ellipse = (((X - cx) / float(rx)) ** 2 + ((Y - cy) / float(ry)) ** 2) <= 1.0
                try:
                    num_labels, labels, stats, cent = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                    chosen = 0
                    if num_labels > 1:
                        # Score by overlap with face-ellipse ROI, tie-break by proximity to face center
                        best_score = -1.0
                        best_dist = 1e12
                        for i in range(1, num_labels):
                            mroi = roi_ellipse & (labels == i)
                            score = float(mroi.sum())
                            x, y, w, h, area = stats[i]
                            mx, my = x + w * 0.5, y + h * 0.5
                            dist = (mx - cx) ** 2 + (my - cy) ** 2
                            if score > best_score or (score == best_score and dist < best_dist):
                                best_score = score
                                best_dist = dist
                                chosen = i
                    if chosen > 0:
                        # --- NEW DILATION-BASED LOGIC ---
                        
                        # 1. Create a mask for just the chosen face component.
                        face_comp_mask = (labels == chosen).astype(np.uint8) * 255

                        # 2. Dilate this face mask. The kernel size is proportional to the face width
                        #    to ensure it can bridge gaps to the torso/clothes.
                        dilate_ratio = float(getattr(config, "BG_GATE_CC_DILATE_RATIO", 0.5))
                        k_size = int(face_w * dilate_ratio) # Kernel is X% of face width
                        if k_size < 3: k_size = 3
                        if k_size % 2 == 0: k_size += 1
                        
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                        dilated_face_mask = cv2.dilate(face_comp_mask, k, iterations=1)

                        # 3. Find all components from the original full mask that are "touched"
                        #    by our dilated face mask.
                        touched_labels = np.unique(labels[dilated_face_mask > 0])
                        
                        # 4. Build the final component mask from all touched components.
                        #    (Excluding label 0, which is the background).
                        if touched_labels.size > 0:
                            final_person_components = np.isin(labels, touched_labels[touched_labels != 0])
                            comp = final_person_components.astype(np.uint8) * 255
                        else:
                            comp = face_comp_mask # Fallback to just the face if nothing is touched
                        
                        # --- END OF NEW LOGIC ---

                        # optional dilation to include near-body (can still be useful)
                        dil = int(getattr(config, "BG_GATE_DILATE", 1)) # Reduced default from 2 to 1
                        if dil > 0:
                            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            comp = cv2.dilate(comp, k, iterations=dil)
                            
                        # Apply component as a gate on the original (stronger-thr) mask
                        strong_bin = (person_mask.astype(np.uint8) >= thr_bin).astype(np.uint8) * 255
                        person_mask = cv2.bitwise_and(strong_bin, comp)
                except Exception:
                    pass

        # Additional gating using Pose ROI mask (if provided)
        if pose_roi_mask is not None and bool(getattr(config, "BG_POSE_GATE_ENABLE", True)):
            try:
                m = pose_roi_mask
                if m.dtype != np.uint8:
                    m = np.clip(m, 0, 255).astype(np.uint8)
                person_mask = cv2.bitwise_and(person_mask, m)
            except Exception:
                pass

        # Final light feather
        sigma = float(getattr(config, "BG_FEATHER_SIGMA", 1.0))
        if sigma > 0.0:
            person_mask = cv2.GaussianBlur(person_mask, (0, 0), sigma)

        return person_mask

    # === 기존 apply_skin_tone ===
    def apply_skin_tone(self, frame_bgr: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
        if skin_mask is None or not SKIN_TONE_ENABLE:
            return frame_bgr
        ys, xs = np.nonzero(skin_mask)
        if ys.size == 0:
            return frame_bgr
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        roi_img = frame_bgr[y0:y1, x0:x1]
        roi_mask = skin_mask[y0:y1, x0:x1]
        if SKIN_TONE_FEATHER > 0:
            roi_mask = cv2.GaussianBlur(roi_mask, (0, 0), SKIN_TONE_FEATHER)
        m = np.clip(roi_mask, 0, 255).astype(np.float32) / 255.0
        if m.max() <= 0:
            return frame_bgr
        ycrcb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y, Cr, Cb = cv2.split(ycrcb)
        tgt = cv2.cvtColor(np.uint8([[SKIN_TONE_TARGET]]), cv2.COLOR_BGR2YCrCb)[0, 0]
        tCr, tCb = float(tgt[1]), float(tgt[2])
        s = float(np.clip(SKIN_TONE_STRENGTH, 0.0, 1.0))
        Cr_t = Cr * (1.0 - s) + tCr * s
        Cb_t = Cb * (1.0 - s) + tCb * s
        Y_t = np.clip(Y * float(SKIN_TONE_BRIGHTNESS), 0, 255)
        toned = cv2.cvtColor(cv2.merge([Y_t, Cr_t, Cb_t]).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
        out_roi = roi_img.astype(np.float32) * (1.0 - m[..., None]) + toned * m[..., None]
        frame_bgr[y0:y1, x0:x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
        return frame_bgr

    # === 기존 apply_smoky ===
    def apply_smoky(self, frame_bgr, face_pts):
        if not SMOKY_ENABLE or SMOKY_SIGMA <= 0:
            return frame_bgr
        out = frame_bgr.astype(np.float32)
        h, w = frame_bgr.shape[:2]
        for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            poly = face_pts[eye_idx].astype(np.int32)
            rect = cv2.boundingRect(poly)
            margin = int(SMOKY_SIGMA * 3)
            roi_x = max(0, rect[0] - margin)
            roi_y = max(0, rect[1] - margin)
            roi_w = min(w - roi_x, rect[2] + 2 * margin)
            roi_h = min(h - roi_y, rect[3] + 2 * margin)
            if roi_w <= 0 or roi_h <= 0:
                continue
            poly_local = poly - np.array([roi_x, roi_y], dtype=np.int32)
            eyeball_mask_roi = np.zeros((roi_h, roi_w), np.uint8)
            cv2.fillPoly(eyeball_mask_roi, [poly_local], 255)
            smoky_area_mask_roi = cv2.GaussianBlur(eyeball_mask_roi, (0, 0), SMOKY_SIGMA)
            final_mask_roi = np.clip(smoky_area_mask_roi.astype(np.float32) - eyeball_mask_roi.astype(np.float32), 0, 255)
            m = (final_mask_roi / 255.0)[..., None]
            roi_slice = out[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            color_arr = np.array(SMOKY_COLOR, np.float32)[None, None, :]
            roi_slice[:] = roi_slice * (1.0 - m) + color_arr * m
        return np.clip(out, 0, 255).astype(np.uint8)

    # === 기존 apply_sclera_color ===
    def apply_sclera_color(self, frame_bgr, pts_all, left_open=True, right_open=True):
        if not SCLERA_COLOR_ENABLE:
            return frame_bgr
        out = frame_bgr.astype(np.float32)
        h, w = frame_bgr.shape[:2]
        EYE_ROIS = [LEFT_EYE_IDX, RIGHT_EYE_IDX]
        IRIS_ROIS = [LEFT_IRIS_IDX, RIGHT_IRIS_IDX]
        open_flags = [left_open, right_open]
        for eye_poly_idx, iris_idx, is_open in zip(EYE_ROIS, IRIS_ROIS, open_flags):
            if not is_open:
                continue
            eye_poly = pts_all[eye_poly_idx].astype(np.int32)
            rect = cv2.boundingRect(eye_poly)
            margin = 2
            roi_x = max(0, rect[0] - margin)
            roi_y = max(0, rect[1] - margin)
            roi_w = min(w - roi_x, rect[2] + 2 * margin)
            roi_h = min(h - roi_y, rect[3] + 2 * margin)
            if roi_w <= 0 or roi_h <= 0:
                continue
            eye_poly_roi = eye_poly - np.array([roi_x, roi_y], dtype=np.int32)
            iris_pts_roi = pts_all[iris_idx].astype(np.float32) - np.array([roi_x, roi_y], dtype=np.float32)
            eyeball_mask = np.zeros((roi_h, roi_w), np.uint8)
            cv2.fillPoly(eyeball_mask, [eye_poly_roi], 255)
            iris_center_roi = np.mean(iris_pts_roi, axis=0)
            iris_radius = int(np.mean(np.linalg.norm(iris_pts_roi - iris_center_roi, axis=1)) * IRIS_SCALE) + 1
            iris_mask = np.zeros((roi_h, roi_w), np.uint8)
            cv2.circle(iris_mask, tuple(iris_center_roi.astype(np.int32)), iris_radius, 255, -1)
            sclera_only_mask = cv2.bitwise_and(eyeball_mask, cv2.bitwise_not(iris_mask))
            if SCLERA_FEATHER > 0:
                sclera_only_mask = gauss_blur(sclera_only_mask, sigma=SCLERA_FEATHER)
            m = (sclera_only_mask.astype(np.float32) / 255.0 * SCLERA_ALPHA)[..., None]
            if m.max() == 0:
                continue
            color_bgr = np.array(SCLERA_COLOR, np.float32)[None, None, :]
            roi_slice = out[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            roi_slice[:] = roi_slice * (1.0 - m) + color_bgr * m
        return np.clip(out, 0, 255).astype(np.uint8)

    # === 기존 apply_eye_effect ===
    def apply_eye_effect(self, frame, pts_all, left_open=True, right_open=True):
        if not EYE_ENABLE or pts_all.shape[0] < 478:
            return frame
        h, w = frame.shape[:2]
        out = frame.astype(np.float32)

        def iris_center_and_radius(idxs):
            c = np.mean(pts_all[idxs], axis=0)
            r = np.mean(np.linalg.norm(pts_all[idxs] - c, axis=1))
            return c, max(1.0, r)

        cL, rL = iris_center_and_radius(LEFT_IRIS_IDX)
        cR, rR = iris_center_and_radius(RIGHT_IRIS_IDX)
        if IRIS_SCALE != 1.0:
            rL *= IRIS_SCALE
            rR *= IRIS_SCALE

        def paint_iris(center, radius):
            x0 = int(max(0, center[0] - radius * 1.2)); y0 = int(max(0, center[1] - radius * 1.2))
            x1 = int(min(w, center[0] + radius * 1.2)); y1 = int(min(h, center[1] + radius * 1.2))
            if x1 <= x0 or y1 <= y0:
                return
            roi = out[y0:y1, x0:x1]
            Hh, Hw = roi.shape[:2]
            ys, xs = np.mgrid[0:Hh, 0:Hw]
            xs = xs + x0; ys = ys + y0
            d = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
            r = radius
            t = np.clip(d / r, 0.0, 1.0)
            H = (EYE_H_IN * (1.0 - t) + EYE_H_OUT * t).astype(np.float32)
            S = (EYE_S_IN * (1.0 - t) + EYE_S_OUT * t).astype(np.float32)
            V = (EYE_V_IN * (1.0 - t) + EYE_V_OUT * t).astype(np.float32)
            hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
            alpha = (1.0 - t).astype(np.float32)
            if EYE_EDGE_BLUR > 0:
                alpha = gauss_blur(alpha, EYE_EDGE_BLUR)
            alpha *= EYE_ALPHA
            alpha = alpha[..., None]
            roi[:] = roi * (1.0 - alpha) + bgr * alpha
            slit = np.zeros((Hh, Hw), dtype=np.uint8)
            ax_major = int(radius * PUPIL_SIZE)
            ax_minor = max(1, int(ax_major * PUPIL_THIN))
            cv2.ellipse(slit, (int(center[0] - x0), int(center[1] - y0)), (ax_minor, ax_major), 0, 0, 360, 255, -1)
            if EYE_EDGE_BLUR > 0:
                slit = gauss_blur(slit, EYE_EDGE_BLUR)
            a = (slit.astype(np.float32) / 255.0) * PUPIL_ALPHA
            roi[:] = roi * (1.0 - a[..., None])

        if left_open:
            paint_iris(cL, rL)
        if right_open:
            paint_iris(cR, rR)
        return np.clip(out, 0, 255).astype(np.uint8)

    # === Lip colorization ===
    def apply_lip_color(self, frame_bgr, pts_all):
        if not LIP_COLOR_ENABLE or pts_all is None or pts_all.shape[0] < 468:
            return frame_bgr
        outer = pts_all[LIP_OUTER_IDX].astype(np.int32)
        inner = pts_all[LIP_INNER_IDX].astype(np.int32)
        if outer.size == 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        rect = cv2.boundingRect(outer)
        margin = int(max(3, (LIP_FEATHER if LIP_FEATHER > 0 else 0) * 3))
        roi_x = max(0, rect[0] - margin)
        roi_y = max(0, rect[1] - margin)
        roi_w = min(w - roi_x, rect[2] + 2 * margin)
        roi_h = min(h - roi_y, rect[3] + 2 * margin)
        if roi_w <= 0 or roi_h <= 0:
            return frame_bgr
        outer_local = outer - np.array([roi_x, roi_y], dtype=np.int32)
        inner_local = inner - np.array([roi_x, roi_y], dtype=np.int32)

        mask = np.zeros((roi_h, roi_w), np.uint8)
        cv2.fillPoly(mask, [outer_local], 255)
        if inner_local.size > 0:
            cv2.fillPoly(mask, [inner_local], 0)
        if LIP_FEATHER > 0:
            mask = gauss_blur(mask, sigma=LIP_FEATHER)
        m = (mask.astype(np.float32) / 255.0 * float(LIP_ALPHA))[..., None]
        if m.max() <= 0:
            return frame_bgr

        roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].astype(np.float32)
        hsv = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        tgt_hsv = cv2.cvtColor(np.uint8([[LIP_COLOR]]), cv2.COLOR_BGR2HSV)[0, 0]
        H_t = np.full_like(H, float(tgt_hsv[0]))
        S_t = np.full_like(S, float(tgt_hsv[1]))
        V_t = np.clip(V * float(LIP_BRIGHTNESS), 0, 255)
        hsv_t = cv2.merge([H_t, S_t, V_t]).astype(np.uint8)
        bgr_t = cv2.cvtColor(hsv_t, cv2.COLOR_HSV2BGR).astype(np.float32)
        roi_out = roi * (1.0 - m) + bgr_t * m
        out = frame_bgr.astype(np.float32)
        out[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_out
        return np.clip(out, 0, 255).astype(np.uint8)

    # === 기존 build_eye_union_mask ===
    def build_eye_union_mask(self, h, w, pts_all):
        eye_union = np.zeros((h, w), np.uint8)
        if pts_all is None or pts_all.shape[0] < 478:
            return eye_union
        for eye_idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            poly = pts_all[eye_idx].astype(np.int32)
            if poly.size > 0:
                cv2.fillPoly(eye_union, [poly], 255)
        for iris_idx in (LEFT_IRIS_IDX, RIGHT_IRIS_IDX):
            iris_pts = pts_all[iris_idx].astype(np.int32)
            if iris_pts.size > 0:
                cv2.fillConvexPoly(eye_union, cv2.convexHull(iris_pts), 255)
        eye_union = cv2.dilate(eye_union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        return eye_union
