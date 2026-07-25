"""
Score-level (decision-level) fusion: train separate SVM-RBF classifiers on
global and region-local features, then combine their PREDICTED PROBABILITIES
(not raw feature vectors) via weighted averaging.

WHY THIS FILE EXISTS
---------------------
train_hybrid_fusion.py tested feature-level fusion (concatenate global +
region raw features, then train one classifier on the combined vector) and
found it HURT easy-attack accuracy relative to global-only (see
results/hybrid_fusion_results.json). A plausible explanation: the
region-local features are noisier (approximate Haar-cascade crops, not
precise landmarks), and raw concatenation lets that noise dilute the
global signal in a high-dimensional, PCA-reduced space with limited data
(2,041 images) to learn which dimensions to trust.

Score-level fusion is a standard alternative: train each modality's own
classifier independently (so region noise can't directly corrupt global
feature weighting), then combine only the final calibrated probabilities.
This is more robust to one weak/noisy modality precisely because the two
classifiers never see each other's raw features.

METHOD
------
1. Same train/val/test split as train_hybrid_fusion.py (identical seed,
   identical align_by_filename logic) for direct comparability.
2. Train SVM-RBF independently on global features (scale->PCA->grid search)
   and independently on region features (scale->PCA->grid search).
3. On the test set, combine each model's predict_proba() output via
   weighted average: p_fused = w * p_global + (1-w) * p_region.
4. Sweep w over a small grid (using the val set to pick w, test set held
   out) rather than fixing w=0.5, since one modality (region) is
   substantially weaker and an even blend may not be optimal.
5. Report the same metric set + difficulty breakdown as train_hybrid_fusion.py
   for direct comparison.
"""

import json
import os

import numpy as np
from sklearn.decomposition import PCA
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


def fit_one_modality(X_train, y_train, X_val, y_val, seed=SEED):
    """Fit scale->PCA->SVM-RBF (grid search) on ONE modality's TRAIN split
    only, matching train_evaluate.py's canonical protocol (Section 6.1: no
    test- or val-set information is used during fitting/model selection).
    X_val/y_val are accepted but intentionally unused here -- the val split
    is reserved for the fusion-weight sweep in main(), never for fitting the
    scaler/PCA/classifier. (Previously this function fit on train+val
    combined, which both leaked val into feature fitting and, along with the
    same issue in train_hybrid_fusion.py, was the source of the cross-script
    numerical drift reported in the paper's Section 11.4.)"""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    pca = PCA(n_components=0.95, random_state=seed).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        param_grid={"C": [1, 10, 50], "gamma": ["scale", 0.01]},
        scoring="f1", cv=cv, n_jobs=-1,
    )
    grid.fit(X_train_p, y_train)
    return scaler, pca, grid.best_estimator_, grid.best_params_


def score(scaler, pca, model, X):
    """Returns calibrated P(real) via predict_proba(), thresholded at 0.5
    downstream for accuracy. NOTE: this decision rule is NOT identical to
    SVC.predict(), which uses the underlying margin/decision-function sign
    rather than a literal 0.5 cut on the Platt-scaled probability -- the two
    disagree on ~5% of predictions even when ranking (AUC) is identical.
    predict_proba()-thresholding is required here because score-level fusion
    needs calibrated probabilities to combine; train_evaluate.py and
    train_hybrid_fusion.py use .predict() instead. This means the
    'global_alone_on_test' baseline reported by THIS script is evaluated
    under a different decision rule than the headline SVM-RBF numbers in
    train_evaluate.py/train_hybrid_fusion.py, and the two should not be
    read as more-precise disagreements -- they are different, both valid,
    decision rules applied to the identical fitted model."""
    return model.predict_proba(pca.transform(scaler.transform(X)))[:, 1]


def difficulty_breakdown(y_true, y_pred, difficulty):
    out = {}
    for level in ("easy", "mid", "hard"):
        mask = (difficulty == level)
        if mask.sum() == 0:
            continue
        out[level] = {"n": int(mask.sum()), "accuracy": float(accuracy_score(y_true[mask], y_pred[mask]))}
    return out


def main():
    X_g, y_g, diff_g, fn_g = load(os.path.join(RESULTS_DIR, "features.npz"))
    d_r = np.load(os.path.join(RESULTS_DIR, "region_features.npz"), allow_pickle=True)
    X_r_raw, fn_r = d_r["X"], d_r["filenames"]
    has_face = d_r["has_face"] if "has_face" in d_r else None

    X_g_al, y_al, diff_al, X_r_al = align_by_filename(X_g, y_g, diff_g, fn_g, X_r_raw, fn_r, has_face)
    print(f"Aligned {len(y_al)} images with both global + region features")

    # IDENTICAL split logic/seed to train_hybrid_fusion.py's fit_eval()
    idx = np.arange(len(y_al))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y_al, test_size=0.30, stratify=y_al, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    diff_test = diff_al[idx_test]

    print("Training global-feature classifier...")
    g_scaler, g_pca, g_model, g_params = fit_one_modality(
        X_g_al[idx_train], y_al[idx_train], X_g_al[idx_val], y_al[idx_val]
    )
    print(f"  global best_params={g_params}")

    print("Training region-feature classifier...")
    r_scaler, r_pca, r_model, r_params = fit_one_modality(
        X_r_al[idx_train], y_al[idx_train], X_r_al[idx_val], y_al[idx_val]
    )
    print(f"  region best_params={r_params}")

    # score on VAL (to pick fusion weight) and TEST (held out, final report)
    g_val_p = score(g_scaler, g_pca, g_model, X_g_al[idx_val])
    r_val_p = score(r_scaler, r_pca, r_model, X_r_al[idx_val])
    g_test_p = score(g_scaler, g_pca, g_model, X_g_al[idx_test])
    r_test_p = score(r_scaler, r_pca, r_model, X_r_al[idx_test])

    # sweep fusion weight w on VAL only, pick best by F1, then apply to TEST
    best_w, best_val_f1 = 1.0, -1
    for w in np.arange(0.0, 1.01, 0.1):
        p = w * g_val_p + (1 - w) * r_val_p
        pred = (p >= 0.5).astype(int)
        f1 = f1_score(y_al[idx_val], pred)
        if f1 > best_val_f1:
            best_val_f1, best_w = f1, w
    print(f"Selected fusion weight (global weight) w={best_w:.1f} via val F1={best_val_f1:.4f}")

    fused_test_p = best_w * g_test_p + (1 - best_w) * r_test_p
    fused_pred = (fused_test_p >= 0.5).astype(int)

    results = {
        "fusion_weight_global": float(best_w),
        "accuracy": accuracy_score(y_test, fused_pred),
        "precision": precision_score(y_test, fused_pred),
        "recall": recall_score(y_test, fused_pred),
        "f1": f1_score(y_test, fused_pred),
        "auc": roc_auc_score(y_test, fused_test_p),
        "difficulty_breakdown": difficulty_breakdown(y_test, fused_pred, diff_test),
        "global_alone_on_test": {
            "accuracy": accuracy_score(y_test, (g_test_p >= 0.5).astype(int)),
            "auc": roc_auc_score(y_test, g_test_p),
            "difficulty_breakdown": difficulty_breakdown(y_test, (g_test_p >= 0.5).astype(int), diff_test),
        },
        "region_alone_on_test": {
            "accuracy": accuracy_score(y_test, (r_test_p >= 0.5).astype(int)),
            "auc": roc_auc_score(y_test, r_test_p),
            "difficulty_breakdown": difficulty_breakdown(y_test, (r_test_p >= 0.5).astype(int), diff_test),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "score_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SCORE-FUSION SUMMARY ===")
    print(f"global_alone        acc={results['global_alone_on_test']['accuracy']:.3f}  "
          f"auc={results['global_alone_on_test']['auc']:.3f}  "
          f"easy={results['global_alone_on_test']['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"region_alone         acc={results['region_alone_on_test']['accuracy']:.3f}  "
          f"auc={results['region_alone_on_test']['auc']:.3f}  "
          f"easy={results['region_alone_on_test']['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"score_fused (w={best_w:.1f})   acc={results['accuracy']:.3f}  "
          f"auc={results['auc']:.3f}  "
          f"easy={results['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()