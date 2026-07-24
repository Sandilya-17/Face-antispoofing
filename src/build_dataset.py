"""
Builds the feature dataset from the Real-and-Fake-Face-Detection image
corpus (Yonsei CVIP / Kaggle "ciplab/real-and-fake-face-detection" style
dataset: 1081 real faces, 960 GAN/print spoofed faces labelled by attack
difficulty easy/mid/hard in the filename).

Output: results/features.npz containing X, y, difficulty, filenames
"""

import glob
import os
import time

import numpy as np

from feature_extraction import extract_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_and_fake_face")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "features.npz")


def difficulty_from_name(fname):
    base = os.path.basename(fname).lower()
    for level in ("easy", "mid", "hard"):
        if base.startswith(level):
            return level
    return "n/a"  # real images


def main():
    real_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_real", "*.jpg")))
    fake_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_fake", "*.jpg")))

    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]  # 0=real,1=fake

    X, y, difficulty, filenames = [], [], [], []
    t0 = time.time()
    for i, (path, label) in enumerate(all_files):
        try:
            feats = extract_features(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        X.append(feats)
        y.append(label)
        difficulty.append(difficulty_from_name(path) if label == 1 else "real")
        filenames.append(os.path.basename(path))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i+1}/{len(all_files)}  ({elapsed:.1f}s)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    difficulty = np.array(difficulty)
    filenames = np.array(filenames)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH, X=X, y=y, difficulty=difficulty, filenames=filenames
    )
    print(f"Saved {X.shape} feature matrix to {OUT_PATH} in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
