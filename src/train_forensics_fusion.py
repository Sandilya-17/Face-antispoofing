"""
Tests whether splice-forensics (ELA + SRM) features improve on the
classical LBP+HOG+color pipeline -- run with the SAME honesty standard as
train_gan_fusion.py: report forensics-only, global-only, and fused results
side by side, and do NOT cherry-pick which comparison to keep.

Requires results/features.npz (from build_dataset.py) and
results/forensics_features.npz (from build_dataset_forensics.py) to have
been built from the SAME image ordering (both scripts glob the same
directories in the same sorted order, so row i in one matches row i in
the other -- this is asserted below via filenames).
"""

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42


def load_and_align():
    global_data = np.load(os.path.join(RESULTS_DIR, "features.npz"), allow_pickle=True)
    forensics_data = np.load(os.path.join(RESULTS_DIR, "forensics_features.npz"), allow_pickle=True)

    assert list(global_data["filenames"]) == list(forensics_data["filenames"]), (
        "Row order mismatch between features.npz and forensics_features.npz -- "
        "re-run build_dataset.py and build_dataset_forensics.py so both glob "
        "the same files in the same order before fusing."
    )

    return (
        global_data["X"], forensics_data["X"], global_data["y"],
        global_data["difficulty"], global_data["filenames"],
    )


def split_three(X_global, X_forensics, y, diff, files):
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=SEED
    )
    return idx_train, idx_val, idx_test


def fit_eval_svm(X_train, y_train, X_test, y_test, d_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    n_comp = min(0.95, X_train_s.shape[1] - 1) if X_train_s.shape[1] > 1 else 1
    pca = PCA(n_components=n_comp, random_state=SEED).fit(X_train_s)
    X_train_p, X_test_p = pca.transform(X_train_s), pca.transform(X_test_s)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
        {"C": [1, 10, 50], "gamma": ["scale", 0.01]}, cv=cv, scoring="f1", n_jobs=1,
    )
    gs.fit(X_train_p, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test_p)
    y_score = best.predict_proba(X_test_p)[:, 1]

    breakdown = {}
    for level in ("easy", "mid", "hard"):
        mask = d_test == level
        if mask.sum() > 0:
            breakdown[level] = dict(
                n=int(mask.sum()), accuracy=float(accuracy_score(y_test[mask], y_pred[mask]))
            )

    return dict(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        auc=float(roc_auc_score(y_test, y_score)),
        pca_dims=int(X_train_p.shape[1]),
        raw_dims=int(X_train.shape[1]),
        best_params=gs.best_params_,
        difficulty_breakdown=breakdown,
    )


def main():
    X_global, X_forensics, y, diff, files = load_and_align()
    idx_train, idx_val, idx_test = split_three(X_global, X_forensics, y, diff, files)

    y_train, y_test, d_test = y[idx_train], y[idx_test], diff[idx_test]

    variants = {
        "global_only": (X_global[idx_train], X_global[idx_test]),
        "forensics_only": (X_forensics[idx_train], X_forensics[idx_test]),
        "global_plus_forensics": (
            np.concatenate([X_global[idx_train], X_forensics[idx_train]], axis=1),
            np.concatenate([X_global[idx_test], X_forensics[idx_test]], axis=1),
        ),
    }

    results = {}
    for name, (X_tr, X_te) in variants.items():
        print(f"Fitting {name} (train dims={X_tr.shape[1]}) ...")
        results[name] = fit_eval_svm(X_tr, y_train, X_te, y_test, d_test)
        r = results[name]
        print(f"  acc={r['accuracy']:.4f}  auc={r['auc']:.4f}  "
              f"easy={r['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.4f}")

    out_path = os.path.join(RESULTS_DIR, "forensics_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {out_path}")

    print("\n=== SUMMARY (test split, SVM-RBF) ===")
    for name, r in results.items():
        print(f"{name:25s} acc={r['accuracy']:.3f}  auc={r['auc']:.3f}")


if __name__ == "__main__":
    main()
