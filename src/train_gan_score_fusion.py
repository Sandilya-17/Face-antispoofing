"""
Score-level (decision-level) fusion of the global hand-crafted classifier
with the GAN one-class anomaly-score classifier, mirroring the exact
protocol this project already established in train_score_fusion.py for
region features -- for the same reason documented there:

train_gan_fusion.py (feature-level concatenation) found that raw-concatenating
the 7-d GAN anomaly vector into the 1,730-d global feature vector HURTS
overall accuracy/AUC relative to the global-only baseline (see
results/gan_fusion_results.json), even though the GAN features ALONE are
extremely good at flagging fakes (>94% accuracy on the fake subset at every
difficulty level) -- they are simply mis-calibrated on real faces (a known
weakness of one-class reconstruction-based anomaly scores: held-out real
faces the GAN never trained on also incur some reconstruction penalty,
inflating false positives). Feature-level concatenation lets that
miscalibration corrupt the SVM's decision boundary in the shared PCA space,
exactly the same failure mode this project already documented for noisy
region features.

Score-level fusion sidesteps this: each modality gets its OWN classifier
(global SVM-RBF, GAN SVM-RBF), fit independently, and only their calibrated
P(fake) probabilities are combined via a weight swept on the validation
split (never the test split), then applied once to test. This lets the
fusion learn to mostly trust the well-calibrated global classifier while
still borrowing the GAN classifier's near-perfect fake-recall on the
specific images where it's confident -- rather than letting noisy GAN
dimensions silently reweight the whole global feature space.
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

    print("Training global-feature classifier...")
    g_scaler, g_pca, g_model, g_params = fit_one_modality(X_g_al[idx_train], y_al[idx_train])
    print(f"  global best_params={g_params}")

    print("Training GAN-feature classifier...")
    gan_scaler, gan_pca, gan_model, gan_params = fit_one_modality(
        X_gan_al[idx_train], y_al[idx_train], use_pca=False
    )
    print(f"  gan best_params={gan_params}")

    g_val_p = score(g_scaler, g_pca, g_model, X_g_al[idx_val])
    gan_val_p = score(gan_scaler, gan_pca, gan_model, X_gan_al[idx_val])
    g_test_p = score(g_scaler, g_pca, g_model, X_g_al[idx_test])
    gan_test_p = score(gan_scaler, gan_pca, gan_model, X_gan_al[idx_test])

    best_w, best_val_f1 = 1.0, -1
    for w in np.arange(0.0, 1.01, 0.05):
        p = w * g_val_p + (1 - w) * gan_val_p
        pred = (p >= 0.5).astype(int)
        f1 = f1_score(y_al[idx_val], pred)
        if f1 > best_val_f1:
            best_val_f1, best_w = f1, w
    print(f"Selected fusion weight (global weight) w={best_w:.2f} via val F1={best_val_f1:.4f}")

    fused_test_p = best_w * g_test_p + (1 - best_w) * gan_test_p
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
        "gan_alone_on_test": {
            "accuracy": accuracy_score(y_test, (gan_test_p >= 0.5).astype(int)),
            "auc": roc_auc_score(y_test, gan_test_p),
            "difficulty_breakdown": difficulty_breakdown(y_test, (gan_test_p >= 0.5).astype(int), diff_test),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "gan_score_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== GAN SCORE-FUSION SUMMARY ===")
    print(f"global_alone        acc={results['global_alone_on_test']['accuracy']:.3f}  "
          f"auc={results['global_alone_on_test']['auc']:.3f}  "
          f"easy={results['global_alone_on_test']['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"gan_alone           acc={results['gan_alone_on_test']['accuracy']:.3f}  "
          f"auc={results['gan_alone_on_test']['auc']:.3f}  "
          f"easy={results['gan_alone_on_test']['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"score_fused (w={best_w:.2f})   acc={results['accuracy']:.3f}  "
          f"auc={results['auc']:.3f}  "
          f"easy={results['difficulty_breakdown'].get('easy', {}).get('accuracy', float('nan')):.3f}")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
