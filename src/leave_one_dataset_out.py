"""
Leave-one-dataset-out (LODO) generalization protocol.
Trains on all-but-one dataset, tests on the held-out one, for every
dataset in results/multi_features.npz. Reports accuracy, AUC, and
ISO/IEC 30107-3 APCER/BPCER/ACER per fold and averaged.
Prerequisite: run build_multi_dataset.py first.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SEED = 42


def apcer_bpcer_acer(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fake_mask, real_mask = y_true == 1, y_true == 0
    apcer = float(np.mean(y_pred[fake_mask] == 0)) if fake_mask.sum() else float("nan")
    bpcer = float(np.mean(y_pred[real_mask] == 1)) if real_mask.sum() else float("nan")
    return dict(apcer=apcer, bpcer=bpcer, acer=(apcer + bpcer) / 2)


def main():
    data_path = os.path.join(RESULTS_DIR, "multi_features.npz")
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found. Run build_multi_dataset.py first.")

    data = np.load(data_path, allow_pickle=True)
    X, y, dataset = data["X"], data["y"], data["dataset"]

    datasets = sorted(set(dataset.tolist()))
    if len(datasets) < 2:
        raise SystemExit(f"Only found 1 dataset ({datasets}) -- LODO needs >=2.")

    print(f"Datasets in pool: {datasets}")
    fold_results = {}

    for held_out in datasets:
        train_mask = dataset != held_out
        test_mask = dataset == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[skip] '{held_out}': missing a class on one side.")
            continue

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        pca = PCA(n_components=0.95, random_state=SEED).fit(X_train_s)
        X_train_p, X_test_p = pca.transform(X_train_s), pca.transform(X_test_s)

        clf = SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                   class_weight="balanced")
        clf.fit(X_train_p, y_train)

        y_pred = clf.predict(X_test_p)
        y_score = clf.predict_proba(X_test_p)[:, 1]

        metrics = dict(
            held_out_dataset=held_out,
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            accuracy=accuracy_score(y_test, y_pred),
            auc=roc_auc_score(y_test, y_score),
            **apcer_bpcer_acer(y_test, y_pred),
        )
        fold_results[held_out] = metrics
        print(f"[{held_out}] n_test={metrics['n_test']} "
              f"acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
              f"acer={metrics['acer']:.3f}")

    if not fold_results:
        raise SystemExit("No valid LODO folds could be run.")

    agg = dict(
        mean_accuracy=float(np.mean([m["accuracy"] for m in fold_results.values()])),
        mean_auc=float(np.mean([m["auc"] for m in fold_results.values()])),
        mean_apcer=float(np.mean([m["apcer"] for m in fold_results.values()])),
        mean_bpcer=float(np.mean([m["bpcer"] for m in fold_results.values()])),
        mean_acer=float(np.mean([m["acer"] for m in fold_results.values()])),
    )
    print("\nAggregate LODO (mean over folds):", json.dumps(agg, indent=2))

    out = dict(per_dataset=fold_results, aggregate=agg)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "lodo_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    names = list(fold_results.keys())
    acers = [fold_results[n]["acer"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, acers, color="indianred")
    ax.axhline(agg["mean_acer"], color="black", linestyle="--",
               label=f"mean ACER={agg['mean_acer']:.3f}")
    ax.set_ylabel("ACER (lower = better)")
    ax.set_title("Leave-One-Dataset-Out generalization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lodo_acer_by_dataset.png"), dpi=150)
    plt.close(fig)

    print("\nSaved results/lodo_results.json and results/figures/lodo_acer_by_dataset.png")


if __name__ == "__main__":
    main()
