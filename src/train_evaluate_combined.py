"""
Trains the same classifier bank as train_evaluate.py, but on the COMBINED
splice+print dataset (results/features_combined.npz), and reports results
BOTH overall and broken down by attack-type source (splice vs. print).

This is the direct test of the paper's Section 12 follow-up question: does
training on multiple attack types produce a classifier that handles both,
instead of collapsing to chance on whichever type it didn't see (as the
splice-only model does on NUAA)? The per-source breakdown is what answers
that -- don't just look at the overall number.

For a fair comparison against the original splice-only SVM-RBF numbers in
the paper (69.1% acc / 0.736 AUC within-domain, 40.5% acc / 0.507 AUC on
NUAA zero-shot), this script also evaluates the ORIGINAL splice-only model
(results/models/best_model.pkl, if present) on this combined test split, so
you get an apples-to-apples "before vs. after adding print attacks to
training" comparison in the same run.

Prerequisite: run build_dataset_combined.py first.

Usage:
    python src/train_evaluate_combined.py
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
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
COMBINED_MODEL_DIR = os.path.join(RESULTS_DIR, "models_combined")
SEED = 42


def load_data():
    path = os.path.join(RESULTS_DIR, "features_combined.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run build_dataset_combined.py first.")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["source"], data["filenames"]


def split_data(X, y, source, files):
    # Stratify on (label, source) jointly so both attack types stay
    # proportionally represented in every split.
    strat_key = np.array([f"{yi}_{si}" for yi, si in zip(y, source)])
    X_train, X_temp, y_train, y_temp, s_train, s_temp, f_train, f_temp = train_test_split(
        X, y, source, files, test_size=0.30, stratify=strat_key, random_state=SEED
    )
    strat_key_temp = np.array([f"{yi}_{si}" for yi, si in zip(y_temp, s_temp)])
    X_val, X_test, y_val, y_test, s_val, s_test, f_val, f_test = train_test_split(
        X_temp, y_temp, s_temp, f_temp, test_size=0.50,
        stratify=strat_key_temp, random_state=SEED,
    )
    return dict(
        train=(X_train, y_train, s_train, f_train),
        val=(X_val, y_val, s_val, f_val),
        test=(X_test, y_test, s_test, f_test),
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


def source_breakdown(y_true, y_pred, y_score, source):
    breakdown = {}
    for s in sorted(set(source.tolist())):
        mask = source == s
        if mask.sum() == 0 or len(set(y_true[mask].tolist())) < 2:
            continue
        breakdown[s] = dict(
            n=int(mask.sum()),
            accuracy=float(accuracy_score(y_true[mask], y_pred[mask])),
            auc=float(roc_auc_score(y_true[mask], y_score[mask])),
        )
    return breakdown


def plot_source_comparison(breakdown, model_name, path):
    sources = list(breakdown.keys())
    accs = [breakdown[s]["accuracy"] for s in sources]
    aucs = [breakdown[s]["auc"] for s in sources]
    x = np.arange(len(sources))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, aucs, width, label="AUC")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model_name} — Accuracy/AUC by Attack Type (Combined Training)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eval_original_splice_only_model(X_test_raw, y_test, source):
    """Evaluate the ORIGINAL splice-only model (from train_evaluate.py) on
    this combined test set, for a direct before/after comparison. Returns
    None if that model isn't present."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    pca_path = os.path.join(MODEL_DIR, "pca.pkl")
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not (os.path.exists(scaler_path) and os.path.exists(pca_path) and os.path.exists(model_path)):
        print("Original splice-only model not found in results/models/ -- skipping before/after comparison.")
        return None

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    model = joblib.load(model_path)

    X_s = scaler.transform(X_test_raw)
    X_p = pca.transform(X_s)
    y_pred = model.predict(X_p)
    y_score = model.predict_proba(X_p)[:, 1]

    overall = evaluate(y_test, y_pred, y_score)
    by_source = source_breakdown(y_test, y_pred, y_score, source)
    return dict(overall=overall, by_source=by_source)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(COMBINED_MODEL_DIR, exist_ok=True)

    X, y, source, files = load_data()
    splits = split_data(X, y, source, files)
    X_train, y_train, s_train, f_train = splits["train"]
    X_val, y_val, s_val, f_val = splits["val"]
    X_test, y_test, s_test, f_test = splits["test"]

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"Train sources: splice={int((s_train=='splice').sum())} print={int((s_train=='print').sum())}")
    print(f"Test sources:  splice={int((s_test=='splice').sum())} print={int((s_test=='print').sum())}")

    # --- Before/after: original splice-only model evaluated on this combined test set ---
    print("\n--- Evaluating ORIGINAL splice-only model on combined test set (before) ---")
    before = eval_original_splice_only_model(X_test, y_test, s_test)
    if before:
        print(json.dumps(before, indent=2))

    # --- Train new combined model ---
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=SEED).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"\nPCA: {X_train.shape[1]} -> {X_train_p.shape[1]} dims (95% variance)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models()

    results = {}
    roc_curves = {}
    fitted_models = {}
    cv_summary = {}
    by_source_all = {}

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

        breakdown = source_breakdown(y_test, y_pred, y_score, s_test)
        by_source_all[name] = breakdown
        plot_source_comparison(breakdown, name, os.path.join(FIG_DIR, f"combined_by_source_{name}.png"))

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves[name] = (fpr, tpr, metrics["auc"])

        print(f"[{name}] best_params={gs.best_params_} cv_f1={gs.best_score_:.4f} "
              f"overall_acc={metrics['accuracy']:.4f} overall_auc={metrics['auc']:.4f} "
              f"by_source={breakdown}  ({time.time()-t0:.1f}s)")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\nBest combined model: {best_name}")
    print(f"  overall: {results[best_name]}")
    print(f"  by source: {by_source_all[best_name]}")

    summary = dict(
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        train_sources={"splice": int((s_train == "splice").sum()), "print": int((s_train == "print").sum())},
        test_sources={"splice": int((s_test == "splice").sum()), "print": int((s_test == "print").sum())},
        pca_components=int(X_train_p.shape[1]),
        cv_summary=cv_summary,
        test_results=results,
        by_source=by_source_all,
        best_model=best_name,
        before_combined_training=before,
    )
    with open(os.path.join(RESULTS_DIR, "metrics_summary_combined.json"), "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(scaler, os.path.join(COMBINED_MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(COMBINED_MODEL_DIR, "pca.pkl"))
    joblib.dump(best_model, os.path.join(COMBINED_MODEL_DIR, "best_model.pkl"))
    with open(os.path.join(COMBINED_MODEL_DIR, "model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nSaved metrics_summary_combined.json, figures to {FIG_DIR}, "
          f"and model pipeline to {COMBINED_MODEL_DIR}")


if __name__ == "__main__":
    main()
