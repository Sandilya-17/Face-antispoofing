"""
Builds the region-local feature dataset (eyes/nose/mouth patches) to
complement results/features.npz (the global descriptor set).

Output: results/region_features.npz containing X, y, difficulty, filenames.

Run AFTER build_dataset.py. Uses the same file discovery + difficulty
labeling logic so rows align 1:1 with results/features.npz by filename
(train_hybrid_fusion.py re-verifies alignment by filename anyway, so order
mismatches are safe, just would be slower).
"""

import glob
import os
import time

import numpy as np

from region_features import extract_region_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_and_fake_face")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "region_features.npz")


def difficulty_from_name(fname):
    base = os.path.basename(fname).lower()
    for level in ("easy", "mid", "hard"):
        if base.startswith(level):
            return level
    return "n/a"


def main():
    real_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_real", "*.jpg")))
    fake_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_fake", "*.jpg")))
    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    X, y, difficulty, filenames, has_face = [], [], [], [], []
    n_failed = 0
    t0 = time.time()

    for i, (path, label) in enumerate(all_files):
        try:
            feats = extract_region_features(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if feats is None:
            n_failed += 1
            continue
        X.append(feats)
        y.append(label)
        difficulty.append(difficulty_from_name(path) if label == 1 else "real")
        filenames.append(os.path.basename(path))
        has_face.append(True)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i+1}/{len(all_files)}  ({elapsed:.1f}s, {n_failed} skipped so far)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    difficulty = np.array(difficulty)
    filenames = np.array(filenames)
    has_face = np.array(has_face)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH, X=X, y=y, difficulty=difficulty, filenames=filenames, has_face=has_face
    )
    print(f"Saved {X.shape} region-feature matrix to {OUT_PATH} in {time.time()-t0:.1f}s")
    print(f"Skipped (unreadable): {n_failed}/{len(all_files)}")


if __name__ == "__main__":
    main()
