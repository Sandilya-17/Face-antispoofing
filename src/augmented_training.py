"""
Data-augmentation expansion of the training set, to directly test whether
the "small dataset -> overfitting/generalization risk" limitation
(report/REPORT.md Sec 7 item 5) is at least partially addressable without
collecting a new corpus.

HONESTY NOTE (read before citing this as "fixing" the dataset-size problem)
----------------------------------------------------------------------------
Augmentation is NOT a substitute for more real, diverse data -- it cannot
manufacture new capture setups, sensors, subjects, or lighting conditions,
so it will NOT be expected to close the cross-dataset gap in Sec 6.5 (that
gap is a domain-shift problem, not a variance problem). What augmentation
CAN legitimately do is reduce overfitting/variance on the SAME domain by
giving the classifier more label-preserving views of the ~1,428 training
images it already has -- so the correct, honest claim to make from this
script's output is about the WITHIN-DATASET variance (bootstrap CI width,
train/test gap), not about cross-domain generalization. Report both, and
report a negative outcome exactly as prominently as a positive one.

METHOD
------
For each image in the TRAINING split only (never val/test -- those must
stay untouched for a fair comparison against every other result in this
report), generate N_AUG label-preserving augmented copies using mild,
spoof-artifact-preserving transforms:
  - horizontal flip
  - small rotation (+/-8 degrees)
  - brightness/contrast jitter (+/-15%)
  - slight Gaussian noise
Deliberately mild and NOT using aggressive crop/occlusion, since those risk
destroying the very local splice artifacts (Sec 5.4, Sec 6) this project's
whole methodology is built on detecting.

Extract the same hand-crafted feature vector (feature_extraction.py) from
every augmented copy, concatenate with the original training features, and
retrain the classical pipeline (identical CV/model-selection protocol as
train_evaluate.py) on this enlarged set. Evaluate on the SAME untouched
test split as train_evaluate.py for a direct, apples-to-apples comparison,
and re-run a bootstrap CI (statistical_significance.py's method) on the
augmented model to quantify whether variance actually shrank.

Requires the raw dataset at data/real_and_fake_face (same requirement as
build_dataset.py).
"""

import json
import os
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

from feature_extraction import extract_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_and_fake_face")
OUT_PATH = os.path.join(RESULTS_DIR, "augmented_training_results.json")
SEED = 42
N_AUG = 3  # augmented copies per training image (3x -> ~5,712 effective training images)
N_BOOTSTRAP = 2000


def augment_image(img_bgr, rng):
    """One random mild augmentation. img_bgr: uint8 HxWx3."""
    out = img_bgr.copy()

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    angle = rng.uniform(-8, 8)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-15, 15)      # brightness
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        noise = rng.normal(0, 4, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def difficulty_from_name(fname):
    base = os.path.basename(fname).lower()
    for level in ("easy", "mid", "hard"):
        if base.startswith(level):
            return level
    return "real"


def _path_for_filename(fname, label):
    sub = "training_fake" if label == 1 else "training_real"
    return os.path.join(DATA_DIR, sub, fname)


def fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, diff_test, seed=SEED):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    pca = PCA(n_components=0.95, random_state=seed).fit(Xtr)
    Xtr = pca.transform(Xtr)
    Xte = pca.transform(scaler.transform(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        param_grid={"C": [1, 10, 50], "gamma": ["scale", 0.01]},
        scoring="f1", cv=cv, n_jobs=-1,
    )
    grid.fit(Xtr, y_train)
    model = grid.best_estimator_

    p_test = model.predict_proba(Xte)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    diff_bd = {}
    for level in ("easy", "mid", "hard"):
        mask = (diff_test == level)
        if mask.sum() == 0:
            continue
        diff_bd[level] = float(accuracy_score(y_test[mask], pred_test[mask]))

    metrics = {
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test),
        "recall": recall_score(y_test, pred_test),
        "f1": f1_score(y_test, pred_test),
        "auc": roc_auc_score(y_test, p_test),
        "difficulty_breakdown": diff_bd,
    }

    # Bootstrap CI on this fitted model's test predictions, mirroring
    # statistical_significance.py's protocol, to check whether more
    # effective training data narrowed the CI.
    rng = np.random.RandomState(seed)
    accs, aucs = [], []
    n = len(y_test)
    for _ in range(N_BOOTSTRAP):
        bidx = rng.randint(0, n, n)
        if len(np.unique(y_test[bidx])) < 2:
            continue
        accs.append(accuracy_score(y_test[bidx], pred_test[bidx]))
        aucs.append(roc_auc_score(y_test[bidx], p_test[bidx]))
    metrics["bootstrap_ci"] = {
        "accuracy": {"mean": float(np.mean(accs)), "ci_lower": float(np.percentile(accs, 2.5)),
                     "ci_upper": float(np.percentile(accs, 97.5))},
        "auc": {"mean": float(np.mean(aucs)), "ci_lower": float(np.percentile(aucs, 2.5)),
                "ci_upper": float(np.percentile(aucs, 97.5))},
        "n_bootstrap": len(accs),
    }
    return metrics


def main():
    d = np.load(os.path.join(RESULTS_DIR, "features.npz"), allow_pickle=True)
    X, y, diff, files = d["X"], d["y"], d["difficulty"], d["filenames"]

    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    diff_test = diff[idx_test]

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    print("=== Baseline (no augmentation) ===")
    baseline = fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, diff_test)
    print(f"acc={baseline['accuracy']:.3f}  auc={baseline['auc']:.3f}  "
          f"95% CI acc=[{baseline['bootstrap_ci']['accuracy']['ci_lower']:.3f}, "
          f"{baseline['bootstrap_ci']['accuracy']['ci_upper']:.3f}]")

    print(f"\n=== Generating {N_AUG}x augmented copies of {len(idx_train)} training images ===")
    rng = np.random.RandomState(SEED)
    aug_feats, aug_labels = [], []
    t0 = time.time()
    n_skip = 0
    for k, i in enumerate(idx_train):
        fname, label = str(files[i]), int(y[i])
        path = _path_for_filename(fname, label)
        img = cv2.imread(path)
        if img is None:
            n_skip += 1
            continue
        for _ in range(N_AUG):
            aug_img = augment_image(img, rng)
            tmp_path = "/tmp/_aug_tmp.jpg"
            cv2.imwrite(tmp_path, aug_img)
            try:
                feats = extract_features(tmp_path)
            except Exception:
                continue
            aug_feats.append(feats)
            aug_labels.append(label)
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(idx_train)}  ({time.time()-t0:.0f}s)")

    if n_skip:
        print(f"  ({n_skip} training images could not be read and were skipped)")

    X_aug = np.stack(aug_feats).astype(np.float32)
    y_aug = np.array(aug_labels, dtype=np.int64)
    X_train_expanded = np.concatenate([X_train, X_aug], axis=0)
    y_train_expanded = np.concatenate([y_train, y_aug], axis=0)
    print(f"Expanded training set: {len(y_train)} -> {len(y_train_expanded)} images "
          f"({len(y_train_expanded) / len(y_train):.1f}x)")

    print("\n=== Augmented (expanded training set) ===")
    augmented = fit_and_eval(X_train_expanded, y_train_expanded, X_val, y_val, X_test, y_test, diff_test)
    print(f"acc={augmented['accuracy']:.3f}  auc={augmented['auc']:.3f}  "
          f"95% CI acc=[{augmented['bootstrap_ci']['accuracy']['ci_lower']:.3f}, "
          f"{augmented['bootstrap_ci']['accuracy']['ci_upper']:.3f}]")

    baseline_ci_width = (baseline["bootstrap_ci"]["accuracy"]["ci_upper"]
                          - baseline["bootstrap_ci"]["accuracy"]["ci_lower"])
    augmented_ci_width = (augmented["bootstrap_ci"]["accuracy"]["ci_upper"]
                           - augmented["bootstrap_ci"]["accuracy"]["ci_lower"])

    results = {
        "n_aug_per_image": N_AUG,
        "n_train_original": int(len(y_train)),
        "n_train_expanded": int(len(y_train_expanded)),
        "baseline": baseline,
        "augmented": augmented,
        "ci_width_change": {
            "baseline_acc_ci_width": baseline_ci_width,
            "augmented_acc_ci_width": augmented_ci_width,
            "narrowed": bool(augmented_ci_width < baseline_ci_width),
        },
        "interpretation": (
            "Augmentation changes WITHIN-DATASET variance/accuracy only; it does "
            "NOT test cross-dataset generalization (see cross_dataset_eval.py / "
            "report Sec 6.5 for that, which augmentation is not expected to fix, "
            "since it cannot manufacture new capture setups or sensors)."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {OUT_PATH}")
    print(f"CI width: baseline={baseline_ci_width:.3f}  augmented={augmented_ci_width:.3f}  "
          f"narrowed={results['ci_width_change']['narrowed']}")


if __name__ == "__main__":
    main()
