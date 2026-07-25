"""
Region-local feature extraction for face anti-spoofing (v2 - Haar cascade
based, no external model downloads required).

WHY THIS FILE EXISTS
---------------------
The original ablation (report Sec.5.4) found that "easy" spoof attacks --
small, localized warps around the eyes/nose/mouth -- are detected *worse*
than "mid"/"hard" attacks by the global LBP+HOG+color pipeline, because
global descriptors average signal over the whole 128x128 image and dilute
a manipulation confined to a small region.

This module tests the direct hypothesis that follows from that finding:
extracting the SAME descriptor families (LBP, HOG, color stats) but computed
LOCALLY on cropped eye/nose/mouth patches -- rather than globally on the
whole face -- should recover signal the global pipeline misses, specifically
on "easy" attacks.

IMPLEMENTATION NOTE (v2)
--------------------------
An earlier version of this module used MediaPipe's legacy `solutions.face_mesh`
API. As of mediapipe>=0.10.10 that API was removed in favor of the Tasks API,
which requires downloading a ~10 MB model bundle at runtime -- an extra
dependency and failure point. This version instead uses OpenCV's Haar
cascade face/eye detectors, which ship inside opencv-python (already a
project dependency, zero extra installs, no network calls at runtime), and
verified working on a real face photo before being handed off. Nose and
mouth regions are located via well-established anthropometric proportions
of the face bounding box (eyes ~ upper-middle third, nose ~ middle third,
mouth ~ lower third), rather than precise landmarks -- a standard, robust
fallback used in classical face-analysis pipelines when full landmark
models aren't available or are overkill for patch-level texture analysis.

METHOD
------
1. Detect the face bounding box (Haar frontal-face cascade).
2. Detect both eyes within the upper half of the face box (Haar eye
   cascade); if eye detection fails, fall back to a fixed proportional
   region (standard face-anatomy ratio) so the pipeline never silently
   drops an image.
3. Derive nose and mouth boxes from face-box proportions.
4. Run the exact same descriptor functions from feature_extraction.py on
   each of the 3 region crops (reused, not reimplemented, so region and
   global features are directly comparable / fusable).
5. Concatenate into one region-local feature vector per image.
"""

import cv2
import numpy as np

from feature_extraction import (
    multiscale_lbp_features,
    color_space_stats,
    hog_features,
)

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

PATCH_SIZE = 64  # each region crop is resized to this before feature extraction


def _clip_box(x0, y0, x1, y1, w, h):
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))


def _detect_face_box(gray):
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # take the largest detected face (most confident / most likely the subject)
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return fx, fy, fw, fh


def _eye_region_box(gray, face_box):
    """Detect both eyes inside the upper half of the face box; union their
    extent into one box. Falls back to a fixed proportional region
    (standard face-anatomy: eyes sit ~25-45% down the face) if detection
    fails, so this never returns None."""
    fx, fy, fw, fh = face_box
    upper_half = gray[fy:fy + fh // 2, fx:fx + fw]
    eyes = _eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    if len(eyes) >= 1:
        xs0 = [fx + ex for (ex, ey, ew, eh) in eyes]
        ys0 = [fy + ey for (ex, ey, ew, eh) in eyes]
        xs1 = [fx + ex + ew for (ex, ey, ew, eh) in eyes]
        ys1 = [fy + ey + eh for (ex, ey, ew, eh) in eyes]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        pad_x, pad_y = (x1 - x0) * 0.25, (y1 - y0) * 0.4
        return x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    # fallback: proportional region
    return fx, fy + 0.20 * fh, fx + fw, fy + 0.50 * fh


def _nose_region_box(face_box):
    fx, fy, fw, fh = face_box
    return fx + 0.20 * fw, fy + 0.35 * fh, fx + 0.80 * fw, fy + 0.68 * fh


def _mouth_region_box(face_box):
    fx, fy, fw, fh = face_box
    return fx + 0.15 * fw, fy + 0.62 * fh, fx + 0.85 * fw, fy + 0.92 * fh


def _crop_and_resize(img_bgr, box, size=PATCH_SIZE):
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = _clip_box(*box, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img_bgr[y0:y1, x0:x1]
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
    (eyes + nose + mouth). Practically never returns None on curated,
    pre-cropped face datasets like this project's, because of the
    whole-frame fallback below.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_box = _detect_face_box(gray)
    if face_box is None:
        # fallback: assume the (already face-cropped) image IS the face,
        # common for curated PAD datasets like this one -- use the whole
        # frame as the face box rather than dropping the image.
        h, w = gray.shape[:2]
        face_box = (0, 0, w, h)

    eye_box = _eye_region_box(gray, face_box)
    nose_box = _nose_region_box(face_box)
    mouth_box = _mouth_region_box(face_box)

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
