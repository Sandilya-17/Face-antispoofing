"""
Region-local feature extraction for face anti-spoofing (v3).

v1 (backed up as region_features_v1_haarcascade_backup.py) used a Haar-
cascade eye detector plus fixed anthropometric proportions to guess
nose/mouth boxes. This version (v3) uses MediaPipe FaceMesh (468-point
face landmarks) for eye/nose/mouth localization, which is far more
accurate than proportional guessing, especially for the "easy" (single-
region, localized) spoof attacks this experiment targets.

Public interface (extract_region_features, region_feature_dim) is
unchanged from v1 so no other script needs to change.
"""
import os
import numpy as np
import cv2
import mediapipe as mp

from feature_extraction import (
    multiscale_lbp_features,
    color_space_stats,
    hog_features,
)

PATCH_SIZE = 64

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
)

# MediaPipe FaceMesh landmark index groups (468-point topology)
_LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155, 246]
_RIGHT_EYE = [263, 362, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382, 466]
_NOSE = [1, 2, 98, 327, 168, 197, 5, 4, 45, 275, 220, 440]
_MOUTH = [61, 291, 39, 181, 0, 17, 84, 314, 405, 321, 375, 291, 78, 308]


def _clip_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(int(x0), w - 1))
    y0 = max(0, min(int(y0), h - 1))
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    return x0, y0, x1, y1


def _landmark_box(landmarks, idxs, w, h, pad_frac=0.35):
    """Bounding box around a set of landmark indices, with padding."""
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = (x1 - x0) * pad_frac + 5
    pad_y = (y1 - y0) * pad_frac + 5
    return _clip_box(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)


def _detect_landmark_boxes(img_rgb):
    """Run FaceMesh and return (eye_box, nose_box, mouth_box) or None if
    no face is detected."""
    h, w = img_rgb.shape[:2]
    result = _mp_face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark

    eye_idxs = _LEFT_EYE + _RIGHT_EYE
    eye_box = _landmark_box(landmarks, eye_idxs, w, h, pad_frac=0.25)
    nose_box = _landmark_box(landmarks, _NOSE, w, h, pad_frac=0.30)
    mouth_box = _landmark_box(landmarks, _MOUTH, w, h, pad_frac=0.30)
    return eye_box, nose_box, mouth_box


def _fallback_boxes(gray):
    """Proportional fallback (same as v1) for the rare case FaceMesh
    fails to find a face -- keeps extract_region_features from ever
    returning None."""
    h, w = gray.shape[:2]
    eye_box = _clip_box(0.10 * w, 0.20 * h, 0.90 * w, 0.50 * h, w, h)
    nose_box = _clip_box(0.30 * w, 0.35 * h, 0.70 * w, 0.70 * h, w, h)
    mouth_box = _clip_box(0.20 * w, 0.60 * h, 0.80 * w, 0.90 * h, w, h)
    return eye_box, nose_box, mouth_box


def _crop_and_resize(img_bgr, box, size=PATCH_SIZE):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def _region_descriptor(patch_bgr):
    """Same three descriptor families as the global pipeline, on one patch."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    lbp = multiscale_lbp_features(gray)
    color = color_space_stats(patch_bgr)
    hg = hog_features(gray)
    return np.concatenate([lbp, color, hg])


def extract_region_features(path, img_size=256):
    """
    Returns a single concatenated region-local feature vector
    (eyes + nose + mouth), located via MediaPipe FaceMesh landmarks.
    Falls back to proportional boxes only if FaceMesh fails to detect
    a face, so this practically never returns None.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = _detect_landmark_boxes(img_rgb)
    if boxes is None:
        boxes = _fallback_boxes(gray)
    eye_box, nose_box, mouth_box = boxes

    feats = []
    for box in (eye_box, nose_box, mouth_box):
        patch = _crop_and_resize(img, box)
        if patch is None:
            return None
        feats.append(_region_descriptor(patch))

    return np.concatenate(feats)


def region_feature_dim():
    dummy = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    d = len(_region_descriptor(dummy))
    return d, d * 3  # (per-region dim, total dim for 3 regions)


if __name__ == "__main__":
    per_region, total = region_feature_dim()
    print(f"Per-region feature dim: {per_region}, total (3 regions): {total}")
