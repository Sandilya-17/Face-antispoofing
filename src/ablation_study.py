"""
Ablation study: isolates the contribution of each feature family
(multi-scale LBP, HSV/YCbCr color statistics, HOG) to final classification
performance, using the exact same train/val/test split, scaler-fit and
PCA-variance-retention protocol as the main pipeline (train_evaluate.py).

Feature vector layout (see feature_extraction.py):
    [0:54]     multi-scale LBP  (3 scales x 18-bin uniform histograms... see below)
    [54:162]   HSV + YCbCr color statistics/histograms
    [162:1730] HOG

For each of {LBP-only, Color-only, HOG-only, Combined (all)}, we:
  1. Slice the feature matrix to the relevant columns.
  2. Refit StandardScaler + PCA(95% variance) on TRAIN only (fair, no leakage;
     each subset gets its own PCA since dimensionality differs).
  3. Grid-search Logistic Regression and SVM-RBF via 5-fold stratified CV
     on TRAIN (same grids as the main pipeline).
  4. Report held-out TEST accuracy / F1 / AUC for each (feature-set, model)
     combination.

This directly answers "does the fusion of LBP+color+HOG actually help, or
is one family doing all the work?" -- which the feature_group_importance
number in the original report could only gesture at indirectly (Gini
importance from a single RandomForest is not a controlled ablation).
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SEED = 42

FEATURE_GROUPS = {
    "LBP-only": (0, 54),
    "Color-only": (54, 162),
    "HOG-only": (162, 1730),
    "Combined (LBP+Color+HOG)": (0, 1730),
}


def load_data():
    data = np.load(os.path.join(RESULTS_DIR, "features.npz"), allow_pickle=True)
    return data["X"], data["y"]


def split_indices(y):
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, stratify=y[idx], random_state=SEED
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=SEED
    )
    return idx_train, idx_val, idx_test


def build_models():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {"C": [1, 10, 50], "gamma": ["scale", 0.01]},
        ),
    }


def run_ablation():
    X, y = load_data()
    idx_train, idx_val, idx_test = split_indices(y)
    y_train, y_test = y[idx_train], y[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for group_name, (lo, hi) in FEATURE_GROUPS.items():
        X_sub = X[:, lo:hi]
        Xtr, Xte = X_sub[idx_train], X_sub[idx_test]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        pca = PCA(n_components=0.95, random_state=SEED).fit(Xtr_s)
        Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)

        results[group_name] = {"n_dims_raw": hi - lo, "n_dims_pca": int(Xtr_p.shape[1])}

        for model_name, (estimator, grid) in build_models().items():
            gs = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=1)
            gs.fit(Xtr_p, y_train)
            best = gs.best_estimator_
            y_pred = best.predict(Xte_p)
            y_score = best.predict_proba(Xte_p)[:, 1]

            metrics = dict(
                accuracy=accuracy_score(y_test, y_pred),
                f1=f1_score(y_test, y_pred),
                auc=roc_auc_score(y_test, y_score),
                best_params=gs.best_params_,
                cv_f1=gs.best_score_,
            )
            results[group_name][model_name] = metrics
            print(f"[{group_name:28s}] {model_name:20s} "
                  f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} "
                  f"(dims {hi-lo} -> {Xtr_p.shape[1]} after PCA)")

    return results


def plot_ablation(results, path):
    groups = list(results.keys())
    models = ["LogisticRegression", "SVM_RBF"]
    metric = "accuracy"

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        vals = [results[g][m][metric] for g in groups]
        bars = ax.bar(x + i * width, vals, width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                     ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation: Feature-Family Contribution to Test Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = run_ablation()
    plot_ablation(results, os.path.join(FIG_DIR, "ablation_study.png"))
    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/ablation_results.json and figures/ablation_study.png")


if __name__ == "__main__":
    main()
