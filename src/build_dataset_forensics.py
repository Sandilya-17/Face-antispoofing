"""
Builds the splice-forensics (ELA + SRM noise-residual) feature dataset,
global + per-region (eyes/nose/mouth via MediaPipe FaceMesh, reusing the
same landmark-box logic as region_features.py).

Run AFTER build_dataset.py (needs the same raw images; independent of its
output otherwise).

Output: results/forensics_features.npz containing X, y, difficulty, filenames
    X has FEATURE_DIM_WITH_REGIONS = 88 dims when a face is detected by
    MediaPipe, else FEATURE_DIM_GLOBAL_ONLY = 22 dims zero-padded to 88
    (region columns zero-filled, matches region_features.py's fallback
    convention -- check `region_detected` array to see which rows had a
    face detected).
"""

import glob
import os
import time

import numpy as np

from feature_extraction import load_and_preprocess
from splice_forensics_features import (
    splice_forensics_features,
    FEATURE_DIM_WITH_REGIONS,
)
from region_features import _detect_landmark_boxes  # reuse existing MediaPipe logic

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_and_fake_face")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "forensics_features.npz")


def difficulty_from_name(fname):
    base = os.path.basename(fname).lower()
    for level in ("easy", "mid", "hard"):
        if base.startswith(level):
            return level
    return "n/a"


def _boxes_to_region_dict(img_rgb):
    detected = _detect_landmark_boxes(img_rgb)
    if detected is None:
        return None
    eye_box, nose_box, mouth_box = detected
    return {"eyes": eye_box, "nose": nose_box, "mouth": mouth_box}


def main():
    real_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_real", "*.jpg")))
    fake_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_fake", "*.jpg")))
    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    X, y, difficulty, filenames, region_detected = [], [], [], [], []
    t0 = time.time()
    for i, (path, label) in enumerate(all_files):
        try:
            img_bgr = load_and_preprocess(path)
            img_rgb = img_bgr[:, :, ::-1]
            region_boxes = _boxes_to_region_dict(img_rgb)
            feats = splice_forensics_features(img_bgr, region_boxes)
            if feats.shape[0] < FEATURE_DIM_WITH_REGIONS:
                feats = np.pad(feats, (0, FEATURE_DIM_WITH_REGIONS - feats.shape[0]))
        except Exception as e:
            print(f"skip {path}: {e}")
            continue

        X.append(feats)
        y.append(label)
        difficulty.append(difficulty_from_name(path) if label == 1 else "real")
        filenames.append(os.path.basename(path))
        region_detected.append(region_boxes is not None)

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(all_files)}  ({time.time()-t0:.1f}s)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    difficulty = np.array(difficulty)
    filenames = np.array(filenames)
    region_detected = np.array(region_detected)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH, X=X, y=y, difficulty=difficulty, filenames=filenames,
        region_detected=region_detected,
    )
    print(f"Saved {X.shape} forensics feature matrix to {OUT_PATH} "
          f"in {time.time()-t0:.1f}s total "
          f"({region_detected.mean()*100:.1f}% had region boxes detected)")


if __name__ == "__main__":
    main()
