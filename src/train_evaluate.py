"""
Trains and evaluates several classical classifiers on the extracted
color-texture feature set for face anti-spoofing (real vs. spoof).

Protocol
--------
- Stratified 70/15/15 train/val/test split (fixed random seed for
  reproducibility).
- StandardScaler fit on train only.
- PCA (retain 95% variance) to de-correlate/compress the 1730-d HOG-heavy
  feature vector before SVM/LogReg (helps both speed and generalization).
- Model selection via 5-fold stratified cross-validation on the TRAIN split
  (small grid search per model).
- Final comparison on the held-out TEST split, reported with accuracy,
  precision, recall, F1, ROC-AUC, and confusion matrices.
- Per-attack-difficulty breakdown (easy/mid/hard) for the best model, since
  aggregate accuracy hides how well the model handles harder attacks.
"""

import json
import os
import time

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SEED = 42


def load_data():
    data = np.load(os.path.join(RESULTS_DIR, "features.npz"), allow_pickle=True)
    return data["X"], data["y"], data["difficulty"], data["filenames"]


def split_data(X, y, diff, files):
    X_train, X_temp, y_train, y_temp, d_train, d_temp, f_train, f_temp = train_test_split(
        X, y, diff, files, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, d_val, d_test, f_val, f_test = train_test_split(
        X_temp, y_temp, d_temp, f_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return dict(
        train=(X_train, y_train, d_train, f_train),
        val=(X_val, y_val, d_val, f_val),
        test=(X_test, y_test, d_test, f_test),
    )


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
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced"),
            {"n_estimators": [200, 400], "max_depth": [None, 20]},
        ),
    }


def evaluate(y_true, y_pred, y_score):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_score),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, path):
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Test Set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def difficulty_breakdown(y_test, y_pred, d_test):
    breakdown = {}
    for level in ["easy", "mid", "hard"]:
        mask = d_test == level
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_test[mask], y_pred[mask])
        breakdown[level] = dict(n=int(mask.sum()), recall_as_fake_detected=float(acc))
    return breakdown


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    X, y, diff, files = load_data()
    splits = split_data(X, y, diff, files)
    X_train, y_train, d_train, f_train = splits["train"]
    X_val, y_val, d_val, f_val = splits["val"]
    X_test, y_test, d_test, f_test = splits["test"]

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=SEED).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)
    print(f"PCA: {X_train.shape[1]} -> {X_train_p.shape[1]} dims (95% variance)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models()

    results = {}
    roc_curves = {}
    fitted_models = {}
    cv_summary = {}

    for name, (estimator, grid) in models.items():
        t0 = time.time()
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=1)
        gs.fit(X_train_p, y_train)
        best = gs.best_estimator_
        fitted_models[name] = best
        cv_summary[name] = dict(best_params=gs.best_params_, cv_best_f1=gs.best_score_)

        y_pred = best.predict(X_test_p)
        y_score = best.predict_proba(X_test_p)[:, 1]
        metrics = evaluate(y_test, y_pred, y_score)
        results[name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves[name] = (fpr, tpr, metrics["auc"])

        plot_confusion(
            np.array(metrics["confusion_matrix"]), f"{name} — Confusion Matrix",
            os.path.join(FIG_DIR, f"confusion_{name}.png"),
        )
        print(f"[{name}] best_params={gs.best_params_} cv_f1={gs.best_score_:.4f} "
              f"test_acc={metrics['accuracy']:.4f} test_f1={metrics['f1']:.4f} "
              f"test_auc={metrics['auc']:.4f}  ({time.time()-t0:.1f}s)")

    plot_roc(roc_curves, os.path.join(FIG_DIR, "roc_curves.png"))
    plot_model_comparison(results, os.path.join(FIG_DIR, "model_comparison.png"))

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    y_pred_best = best_model.predict(X_test_p)
    breakdown = difficulty_breakdown(y_test, y_pred_best, d_test)

    print(f"\nBest model: {best_name}")
    print("Per-attack-difficulty test accuracy:", breakdown)

    # Feature importance for RandomForest (interpretability)
    if "RandomForest" in fitted_models:
        rf_on_raw = RandomForestClassifier(
            n_estimators=300, random_state=SEED, class_weight="balanced"
        ).fit(X_train_s, y_train)  # fit on scaled raw feats (not PCA) for interpretability
        importances = rf_on_raw.feature_importances_
        group_bounds = {"LBP (54)": (0, 54), "Color-texture (108)": (54, 162), "HOG (1568)": (162, 1730)}
        group_importance = {
            g: float(importances[a:b].sum()) for g, (a, b) in group_bounds.items()
        }
        print("Feature-group importance (RandomForest, gini):", group_importance)
    else:
        group_importance = {}

    summary = dict(
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        pca_components=int(X_train_p.shape[1]),
        cv_summary=cv_summary,
        test_results=results,
        best_model=best_name,
        difficulty_breakdown=breakdown,
        feature_group_importance=group_importance,
    )
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Persist full inference pipeline (scaler + PCA + best model)
    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(model_dir, "pca.pkl"))
    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    with open(os.path.join(model_dir, "model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nSaved metrics_summary.json, figures to {FIG_DIR}, and model pipeline to {model_dir}")


if __name__ == "__main__":
    main()
