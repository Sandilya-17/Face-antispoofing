"""
Statistical robustness for the model comparison.

A single 70/15/15 split (train_evaluate.py) reports SVM-RBF at 69.1% vs
LogReg at 67.8% test accuracy. That 1.3-point gap is not, by itself,
evidence that SVM-RBF is the "better" model -- it could easily be split
noise on a 307-image test set. Two standard remedies:

1. BOOTSTRAP CONFIDENCE INTERVALS on the fixed test set: resample the
   307 test predictions (with replacement) 5,000x and recompute accuracy/
   F1/AUC each time, giving a 95% CI around each model's point estimate.
   This tells us how much the *reported number* could plausibly wobble
   due to the specific 307 images drawn into the test set.

2. REPEATED RANDOM SPLITS + PAIRED TEST: re-run the full
   split -> scale -> PCA -> fit -> evaluate pipeline 30x with different
   random seeds, collect per-split test accuracy for SVM-RBF and
   LogisticRegression, and run a paired Wilcoxon signed-rank test (and
   a paired t-test as a cross-check) on the per-split accuracy
   differences. This is the standard way to ask "does model A reliably
   beat model B across resampling of the data", rather than trusting one
   split.

Both are cheap here because the feature extraction is already cached in
features.npz -- only the classifier fit/predict step is repeated.
"""

import json
import os

import numpy as np
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 42
N_BOOTSTRAP = 5000
N_REPEATS = 30

# Fixed hyperparameters (best_params found by GridSearchCV in train_evaluate.py),
# held fixed here so that we're testing model-family stability, not re-running a
# grid search inside every one of the 30 repeats (which would be 30x slower and
# would conflate "hyperparameter search variance" with "split variance").
FIXED_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.1
    ),
    "SVM_RBF": lambda: SVC(
        kernel="rbf", probability=True, class_weight="balanced", C=1, gamma="scale"
    ),
}


def load_data():
    data = np.load(os.path.join(RESULTS_DIR, "features.npz"), allow_pickle=True)
    return data["X"], data["y"]


def fit_predict(X, y, seed):
    """One full split -> scale -> PCA -> fit -> test-predict pass."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    pca = PCA(n_components=0.95, random_state=seed).fit(X_train_s)
    X_train_p, X_test_p = pca.transform(X_train_s), pca.transform(X_test_s)

    out = {}
    for name, ctor in FIXED_MODELS.items():
        model = ctor().fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_score = model.predict_proba(X_test_p)[:, 1]
        out[name] = dict(
            y_test=y_test, y_pred=y_pred, y_score=y_score,
            accuracy=accuracy_score(y_test, y_pred),
        )
    return out


def bootstrap_ci(y_test, y_pred, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_test)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ys = y_test[idx], y_pred[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined if resample is single-class
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp))
        aucs.append(roc_auc_score(yt, ys))

    def ci(arr):
        arr = np.array(arr)
        return dict(
            mean=float(arr.mean()),
            ci_lower=float(np.percentile(arr, 2.5)),
            ci_upper=float(np.percentile(arr, 97.5)),
        )

    return dict(accuracy=ci(accs), f1=ci(f1s), auc=ci(aucs), n_bootstrap=len(accs))


def main():
    X, y = load_data()

    # --- Part 1: bootstrap CI on the ORIGINAL fixed test split (seed=42) ---
    print("=" * 70)
    print("PART 1: Bootstrap 95% CIs on the original held-out test split (n=307)")
    print("=" * 70)
    single_split = fit_predict(X, y, SEED)
    bootstrap_results = {}
    for name, res in single_split.items():
        ci = bootstrap_ci(res["y_test"], res["y_pred"], res["y_score"])
        bootstrap_results[name] = ci
        print(f"[{name}] accuracy = {ci['accuracy']['mean']:.3f} "
              f"[{ci['accuracy']['ci_lower']:.3f}, {ci['accuracy']['ci_upper']:.3f}]  "
              f"AUC = {ci['auc']['mean']:.3f} "
              f"[{ci['auc']['ci_lower']:.3f}, {ci['auc']['ci_upper']:.3f}]")

    # --- Part 2: repeated random splits + paired significance test ---
    print()
    print("=" * 70)
    print(f"PART 2: {N_REPEATS} repeated random splits, paired test "
          f"(SVM-RBF vs LogisticRegression)")
    print("=" * 70)
    accs_svm, accs_logreg = [], []
    for i in range(N_REPEATS):
        seed = 1000 + i
        res = fit_predict(X, y, seed)
        accs_svm.append(res["SVM_RBF"]["accuracy"])
        accs_logreg.append(res["LogisticRegression"]["accuracy"])
        print(f"  split {i+1:2d}/{N_REPEATS}  seed={seed}  "
              f"SVM_RBF={accs_svm[-1]:.4f}  LogReg={accs_logreg[-1]:.4f}  "
              f"diff={accs_svm[-1]-accs_logreg[-1]:+.4f}")

    accs_svm = np.array(accs_svm)
    accs_logreg = np.array(accs_logreg)
    diffs = accs_svm - accs_logreg

    t_stat, t_p = stats.ttest_rel(accs_svm, accs_logreg)
    try:
        w_stat, w_p = stats.wilcoxon(accs_svm, accs_logreg)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    repeated_split_summary = dict(
        n_repeats=N_REPEATS,
        svm_rbf_mean_acc=float(accs_svm.mean()),
        svm_rbf_std_acc=float(accs_svm.std(ddof=1)),
        logreg_mean_acc=float(accs_logreg.mean()),
        logreg_std_acc=float(accs_logreg.std(ddof=1)),
        mean_diff=float(diffs.mean()),
        std_diff=float(diffs.std(ddof=1)),
        paired_ttest=dict(statistic=float(t_stat), p_value=float(t_p)),
        wilcoxon_signed_rank=dict(statistic=float(w_stat), p_value=float(w_p)),
    )

    print()
    print(f"SVM-RBF:  mean acc = {accs_svm.mean():.4f} +/- {accs_svm.std(ddof=1):.4f}")
    print(f"LogReg:   mean acc = {accs_logreg.mean():.4f} +/- {accs_logreg.std(ddof=1):.4f}")
    print(f"Mean paired difference (SVM - LogReg) = {diffs.mean():+.4f} +/- {diffs.std(ddof=1):.4f}")
    print(f"Paired t-test:      t={t_stat:.3f}, p={t_p:.4f}")
    print(f"Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")
    alpha = 0.05
    verdict = ("statistically significant" if t_p < alpha else
               "NOT statistically significant")
    print(f"\nAt alpha=0.05, the SVM-RBF vs. LogReg accuracy gap is {verdict} "
          f"across {N_REPEATS} resampled splits.")

    out = dict(
        bootstrap_ci_original_split=bootstrap_results,
        repeated_split_comparison=repeated_split_summary,
        interpretation=(
            f"Across {N_REPEATS} independent random splits, SVM-RBF beat "
            f"LogisticRegression on mean test accuracy by "
            f"{diffs.mean():+.4f} (paired t-test p={t_p:.4f}, "
            f"Wilcoxon p={w_p:.4f}). "
            + ("This gap is unlikely to be due to chance."
               if t_p < alpha else
               "This gap is small relative to split-to-split noise and "
               "should NOT be reported as a reliable ranking between the "
               "two models on this dataset/feature set.")
        ),
    )
    with open(os.path.join(RESULTS_DIR, "statistical_significance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/statistical_significance.json")


if __name__ == "__main__":
    main()
