"""
Hybrid fusion: global (whole-image) features + region-local (eyes/nose/mouth)
features, trained and evaluated with the exact same protocol as
train_evaluate.py, so results are directly comparable to the original report.

CORE EXPERIMENT
----------------
Compare, on the SAME train/val/test split:
  (A) Global-only     (results/features.npz)              -- the original baseline
  (B) Region-only      (results/region_features.npz)        -- new
  (C) Global + Region  (concatenated)                       -- the proposed method

...and report per-difficulty (easy/mid/hard) accuracy for each, since the
whole point of adding region-local features is to fix the "easy" attack
blind spot documented in report §5.4. If (C) improves "easy" accuracy over
(A) without hurting mid/hard, that is the paper's headline result.
"""

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["difficulty"], d["filenames"]


def align_by_filename(X_g, y_g, diff_g, fn_g, X_r, fn_r, has_face=None):
    """Region features must be reordered/filtered to match the global set's
    filename order exactly (build order should already match, but this is
    a safety check for the fusion step)."""
    idx_r = {f: i for i, f in enumerate(fn_r)}
    keep, X_r_aligned = [], []
    for i, f in enumerate(fn_g):
        j = idx_r.get(f)
        if j is None:
            continue
        if has_face is not None and not has_face[j]:
            continue
        keep.append(i)
        X_r_aligned.append(X_r[j])
    keep = np.array(keep)
    return X_g[keep], y_g[keep], diff_g[keep], np.stack(X_r_aligned)


def fit_eval(X, y, difficulty, label, seed=SEED):
    # IDENTICAL protocol to train_evaluate.py: scaler + PCA + CV are fit on the
    # TRAIN split only. The validation split is never used for fitting anything
    # in the classical pipeline (matches Section 6.1's stated protocol). Earlier
    # versions of this script fit scaler/PCA on train+val, which is why results
    # here previously diverged from the canonical train_evaluate.py numbers.
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, difficulty, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=seed).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        param_grid={"C": [1, 10, 50], "gamma": ["scale", 0.01]},
        scoring="f1", cv=cv, n_jobs=-1,
    )
    grid.fit(X_train_p, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test_p)
    y_prob = best.predict_proba(X_test_p)[:, 1]

    metrics = {
        "label": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "pca_dims": X_train_p.shape[1],
        "raw_dims": X.shape[1],
        "best_params": grid.best_params_,
    }

    # per-difficulty breakdown on the fake subset of the test split
    diff_breakdown = {}
    for level in ("easy", "mid", "hard"):
        mask = (d_test == level)
        if mask.sum() == 0:
            continue
        diff_breakdown[level] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_test[mask], y_pred[mask])),
        }
    metrics["difficulty_breakdown"] = diff_breakdown
    return metrics


def main():
    X_g, y_g, diff_g, fn_g = load(os.path.join(RESULTS_DIR, "features.npz"))

    region_path = os.path.join(RESULTS_DIR, "region_features.npz")
    d_r = np.load(region_path, allow_pickle=True)
    X_r_raw, fn_r = d_r["X"], d_r["filenames"]
    has_face = d_r["has_face"] if "has_face" in d_r else None

    X_g_al, y_al, diff_al, X_r_al = align_by_filename(X_g, y_g, diff_g, fn_g, X_r_raw, fn_r, has_face)
    print(f"Aligned {len(y_al)} / {len(y_g)} images have both global + region features "
          f"({len(y_g) - len(y_al)} dropped for missing face detection)")

    X_fused = np.concatenate([X_g_al, X_r_al], axis=1)

    results = {}
    results["global_only"] = fit_eval(X_g_al, y_al, diff_al, "global_only")
    results["region_only"] = fit_eval(X_r_al, y_al, diff_al, "region_only")
    results["global_plus_region"] = fit_eval(X_fused, y_al, diff_al, "global_plus_region")

    out_path = os.path.join(RESULTS_DIR, "hybrid_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for key, r in results.items():
        print(f"{key:20s}  acc={r['accuracy']:.3f}  auc={r['auc']:.3f}  "
              f"easy={r['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
