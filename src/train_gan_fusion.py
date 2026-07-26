"""
Fuses the GAN-based one-class reconstruction/anomaly features
(results/gan_features.npz, see src/gan_reconstruction.py) with the existing
hand-crafted global color-texture features (results/features.npz) and,
optionally, the region-local hand-crafted features (results/region_features.npz),
under the EXACT SAME train/val/test protocol as train_evaluate.py and
train_hybrid_fusion.py (seed=42, 70/15/15 stratified split, scaler+PCA fit
on train only, 5-fold CV grid search for SVM-RBF), so results are directly
comparable to report/REPORT.md and results/hybrid_fusion_results.json.

CORE EXPERIMENT
---------------
Compare, on the SAME split:
  (A) Global hand-crafted only        (baseline, report Sec. 5.1: 69.1% / 0.736 AUC)
  (B) GAN anomaly features only        (7-d: global MSE/L1/SSIM, discriminator
                                         realness, eye/nose/mouth recon MSE)
  (C) Global + GAN                     (the proposed fused method)
  (D) Global + Region + GAN            (everything combined, if region
                                         features are available)

The headline claims this script is designed to test:
  1. Does the GAN's learned anomaly score add signal beyond hand-crafted
     descriptors (C vs A)?
  2. Does it specifically fix the "easy"-attack blind spot documented in
     report Sec. 5.4 (per-difficulty breakdown for each configuration)?
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


def align_by_filename(X_g, y_g, diff_g, fn_g, X_other, fn_other):
    """Reorder/filter X_other to match fn_g's order exactly (safety check,
    same pattern as train_hybrid_fusion.py's align_by_filename)."""
    idx = {f: i for i, f in enumerate(fn_other)}
    keep, X_aligned = [], []
    for i, f in enumerate(fn_g):
        j = idx.get(f)
        if j is None:
            continue
        keep.append(i)
        X_aligned.append(X_other[j])
    keep = np.array(keep)
    return X_g[keep], y_g[keep], diff_g[keep], np.stack(X_aligned)


def fit_eval(X, y, difficulty, label, seed=SEED):
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, difficulty, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # PCA only helps de-correlate large, redundant feature banks (HOG-heavy
    # global vectors). The 7-d GAN-only vector is already compact and
    # decorrelated by construction, so skip PCA when there is nothing to
    # compress (avoids PCA(0.95) degenerately keeping ~all 7 dims anyway,
    # but keeps the code path uniform for the fused configs).
    if X_train_s.shape[1] > 10:
        pca = PCA(n_components=0.95, random_state=seed).fit(X_train_s)
        X_train_p = pca.transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
    else:
        X_train_p, X_test_p = X_train_s, X_test_s

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
    X_gan_raw, y_gan_raw, diff_gan_raw, fn_gan_raw = load(os.path.join(RESULTS_DIR, "gan_features.npz"))

    X_g_al, y_al, diff_al, X_gan_al = align_by_filename(X_g, y_g, diff_g, fn_g, X_gan_raw, fn_gan_raw)
    print(f"Aligned {len(y_al)} / {len(y_g)} images have both global + GAN features")

    X_fused = np.concatenate([X_g_al, X_gan_al], axis=1)

    results = {}
    results["global_only"] = fit_eval(X_g_al, y_al, diff_al, "global_only")
    results["gan_only"] = fit_eval(X_gan_al, y_al, diff_al, "gan_only")
    results["global_plus_gan"] = fit_eval(X_fused, y_al, diff_al, "global_plus_gan")

    region_path = os.path.join(RESULTS_DIR, "region_features.npz")
    if os.path.exists(region_path):
        try:
            d_r = np.load(region_path, allow_pickle=True)
            X_r_raw, fn_r = d_r["X"], d_r["filenames"]
            has_face = d_r["has_face"] if "has_face" in d_r else None

            idx_r = {f: i for i, f in enumerate(fn_r)}
            keep, X_r_al, X_g_al2, X_gan_al2, y_al2, diff_al2 = [], [], [], [], [], []
            for i, f in enumerate(fn_g):
                j = idx_r.get(f)
                if j is None or (has_face is not None and not has_face[j]):
                    continue
                k = None
                gan_idx = {f2: idx for idx, f2 in enumerate(fn_gan_raw)}
                k = gan_idx.get(f)
                if k is None:
                    continue
                X_r_al.append(X_r_raw[j])
                X_g_al2.append(X_g[i])
                X_gan_al2.append(X_gan_raw[k])
                y_al2.append(y_g[i])
                diff_al2.append(diff_g[i])

            X_r_al = np.stack(X_r_al)
            X_g_al2 = np.stack(X_g_al2)
            X_gan_al2 = np.stack(X_gan_al2)
            y_al2 = np.array(y_al2)
            diff_al2 = np.array(diff_al2)

            X_all = np.concatenate([X_g_al2, X_r_al, X_gan_al2], axis=1)
            print(f"Aligned {len(y_al2)} images have global + region + GAN features")
            results["global_plus_region_plus_gan"] = fit_eval(X_all, y_al2, diff_al2, "global_plus_region_plus_gan")
        except Exception as e:
            print(f"Skipping global+region+GAN config: {e}")

    out_path = os.path.join(RESULTS_DIR, "gan_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for key, r in results.items():
        db = r["difficulty_breakdown"]
        print(f"{key:28s}  acc={r['accuracy']:.3f}  auc={r['auc']:.3f}  "
              f"easy={db.get('easy', {}).get('accuracy', float('nan')):.3f}  "
              f"mid={db.get('mid', {}).get('accuracy', float('nan')):.3f}  "
              f"hard={db.get('hard', {}).get('accuracy', float('nan')):.3f}")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
