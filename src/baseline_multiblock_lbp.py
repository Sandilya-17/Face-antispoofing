"""
Reproduces, as a directly-run comparison point, the multi-block LBP
approach of Guenay Yilmaz, Turhal & Nabiyev (2023) [20] -- "Face
presentation attack detection performances of facial regions with
multi-block LBP features" -- on the identical train/val/test split used
throughout this project.

Why this script exists
-----------------------
The paper (report/paper.pdf, Sec. 2.3 / Table 1) cites [20] and [21] as
the closest prior work but never actually re-runs them as baselines, so
the "this study sits closest to [20, 21]" claim was an assertion, not a
number. This script closes that gap: it divides each 128x128 face image
into a fixed grid of non-overlapping blocks, computes a uniform LBP
histogram per block (the core idea in [20], as opposed to this project's
own global multi-scale LBP or its landmark-driven region-local fusion in
Section 11), concatenates block histograms, and evaluates with the exact
same scale -> PCA -> SVM-RBF grid-search protocol as Section 5 of the
paper, so the comparison is apples-to-apples.

Output: results/multiblock_lbp_baseline.json
"""

import json
import os
import time

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import load_and_preprocess, IMG_SIZE

import cv2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FEATURES_NPZ = os.path.join(RESULTS_DIR, "features.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "multiblock_lbp_baseline.json")

# [20]-style grid: divide the face into GRID_N x GRID_N non-overlapping blocks
GRID_N = 4          # 4x4 = 16 blocks, a standard multi-block LBP configuration
LBP_P, LBP_R = 8, 1  # single-scale uniform LBP per block, as in [20]
RANDOM_STATE = 42


def multiblock_lbp(gray):
    h, w = gray.shape
    bh, bw = h // GRID_N, w // GRID_N
    n_bins = LBP_P + 2
    feats = []
    for i in range(GRID_N):
        for j in range(GRID_N):
            block = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            lbp = local_binary_pattern(block, LBP_P, LBP_R, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def extract_multiblock_features(path):
    img = load_and_preprocess(path, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return multiblock_lbp(gray)


def main():
    t0 = time.time()
    data = np.load(FEATURES_NPZ, allow_pickle=True)
    filenames = data["filenames"]
    y = data["y"]
    difficulty = data["difficulty"] if "difficulty" in data else None

    print(f"Re-extracting multi-block LBP features for {len(filenames)} images "
          f"({GRID_N}x{GRID_N} grid, LBP(P={LBP_P},R={LBP_R})) ...")
    X = np.stack([extract_multiblock_features(f) for f in filenames])
    print(f"Multi-block LBP feature dim: {X.shape[1]}")

    # Identical 70/15/15 split logic as the main pipeline (same seed -> same indices)
    n = len(y)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    param_grid = {"C": [1, 10, 50], "gamma": ["scale", 0.01]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid,
                         scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(np.vstack([X_train_p, X_val_p]), np.concatenate([y_train, y_val]))
    best = grid.best_estimator_

    y_pred = best.predict(X_test_p)
    y_proba = best.predict_proba(X_test_p)[:, 1]

    metrics = {
        "method": f"Multi-block LBP ({GRID_N}x{GRID_N} grid), reproducing [20]-style features",
        "raw_dims": int(X.shape[1]),
        "pca_dims": int(X_train_p.shape[1]),
        "best_params": grid.best_params_,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "n_test": int(len(y_test)),
        "runtime_sec": round(time.time() - t0, 1),
    }

    if difficulty is not None:
        diff_test = difficulty[test_idx]
        breakdown = {}
        for level in ("easy", "mid", "hard"):
            mask = (diff_test == level)
            if mask.sum() > 0:
                breakdown[level] = {
                    "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_test[mask], y_pred[mask])),
                }
        metrics["difficulty_breakdown"] = breakdown

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to {OUT_PATH}")
    print("\nCompare against this project's fused SVM-RBF pipeline "
          "(results/metrics_summary.json) to get a direct, executed "
          "head-to-head against the [20]-style multi-block LBP approach.")


if __name__ == "__main__":
    main()
