"""
Stacked meta-classifier fusion of the global hand-crafted classifier and the
GAN one-class anomaly-score classifier.

train_gan_score_fusion.py fuses the two calibrated probabilities with a
single scalar weight w swept on the validation split (p_fused = w*p_g +
(1-w)*p_gan). That is a linear blend along one axis and cannot learn, e.g.,
"trust the GAN score more when the global classifier is unsure" -- it can
only interpolate.

This script instead fits a small logistic-regression meta-classifier on top
of [p_global, p_gan] (out-of-fold on train+val, applied once to test),
letting the fusion learn a genuine 2D decision boundary between the two
modalities rather than a fixed linear interpolation. This is the "stacked
meta-classifier" next step flagged in report/REPORT.md Sec 8, aimed
directly at the fusion-asymmetry finding in Sec 6.2.
"""

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["difficulty"], d["filenames"]


def align_by_filename(X_g, y_g, diff_g, fn_g, X_r, fn_r):
    idx_r = {f: i for i, f in enumerate(fn_r)}
    keep, X_r_aligned = [], []
    for i, f in enumerate(fn_g):
        j = idx_r.get(f)
        if j is None:
            continue
        keep.append(i)
        X_r_aligned.append(X_r[j])
    keep = np.array(keep)
    return X_g[keep], y_g[keep], diff_g[keep], np.stack(X_r_aligned)


def fit_one_modality(X_train, y_train, seed=SEED, use_pca=True):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    pca = None
    if use_pca and X_train_s.shape[1] > 10:
        pca = PCA(n_components=0.95, random_state=seed).fit(X_train_s)
        X_train_p = pca.transform(X_train_s)
    else:
        X_train_p = X_train_s
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        param_grid={"C": [1, 10, 50], "gamma": ["scale", 0.01]},
        scoring="f1", cv=cv, n_jobs=-1,
    )
    grid.fit(X_train_p, y_train)
    return scaler, pca, grid.best_estimator_, grid.best_params_


def score(scaler, pca, model, X):
    Xs = scaler.transform(X)
    if pca is not None:
        Xs = pca.transform(Xs)
    return model.predict_proba(Xs)[:, 1]


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
    X_gan_raw, y_gan_raw, diff_gan_raw, fn_gan_raw = load(os.path.join(RESULTS_DIR, "gan_features.npz"))
    X_g_al, y_al, diff_al, X_gan_al = align_by_filename(X_g, y_g, diff_g, fn_g, X_gan_raw, fn_gan_raw)
    print(f"Aligned {len(y_al)} images with both global + GAN features")

    idx = np.arange(len(y_al))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y_al, test_size=0.30, stratify=y_al, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    diff_test = diff_al[idx_test]

    # Fit base modalities on train only (same as train_gan_score_fusion.py)
    print("Training global-feature classifier...")
    g_scaler, g_pca, g_model, g_params = fit_one_modality(X_g_al[idx_train], y_al[idx_train])
    print("Training GAN-feature classifier...")
    gan_scaler, gan_pca, gan_model, gan_params = fit_one_modality(
        X_gan_al[idx_train], y_al[idx_train], use_pca=False
    )

    g_val_p = score(g_scaler, g_pca, g_model, X_g_al[idx_val])
    gan_val_p = score(gan_scaler, gan_pca, gan_model, X_gan_al[idx_val])
    g_test_p = score(g_scaler, g_pca, g_model, X_g_al[idx_test])
    gan_test_p = score(gan_scaler, gan_pca, gan_model, X_gan_al[idx_test])

    # Meta-classifier: fit on VAL split only (never touches test), 2D input
    # [p_global, p_gan]. Small, well-regularized (C searched by CV) since
    # this is only ~306 points and 2 features.
    meta_X_val = np.column_stack([g_val_p, gan_val_p])
    meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta_grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000),
        param_grid={"C": [0.01, 0.1, 1, 10]},
        scoring="f1", cv=meta_cv,
    )
    meta_grid.fit(meta_X_val, y_val)
    meta = meta_grid.best_estimator_
    print(f"Meta-classifier best C={meta_grid.best_params_['C']}, "
          f"learned coefs (global,gan)={meta.coef_[0]}, intercept={meta.intercept_[0]:.3f}")

    meta_X_test = np.column_stack([g_test_p, gan_test_p])
    fused_test_p = meta.predict_proba(meta_X_test)[:, 1]
    fused_pred = (fused_test_p >= 0.5).astype(int)

    results = {
        "method": "logistic_regression_stacking_on_[p_global,p_gan]",
        "meta_best_C": meta_grid.best_params_["C"],
        "meta_coefs_global_gan": meta.coef_[0].tolist(),
        "meta_intercept": float(meta.intercept_[0]),
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
        "gan_alone_on_test": {
            "accuracy": accuracy_score(y_test, (gan_test_p >= 0.5).astype(int)),
            "auc": roc_auc_score(y_test, gan_test_p),
            "difficulty_breakdown": difficulty_breakdown(y_test, (gan_test_p >= 0.5).astype(int), diff_test),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "gan_stacked_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== GAN STACKED-FUSION SUMMARY ===")
    print(f"global_alone     acc={results['global_alone_on_test']['accuracy']:.3f}  auc={results['global_alone_on_test']['auc']:.3f}")
    print(f"gan_alone        acc={results['gan_alone_on_test']['accuracy']:.3f}  auc={results['gan_alone_on_test']['auc']:.3f}")
    print(f"stacked_fused    acc={results['accuracy']:.3f}  auc={results['auc']:.3f}  "
          f"easy={results['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
