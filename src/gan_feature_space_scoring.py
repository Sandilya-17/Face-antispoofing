"""
Feature-space (AnoGAN-style) anomaly scoring for the one-class GAN.

WHY THIS SCRIPT EXISTS
-----------------------
Sec 6.2/6.3 of report/REPORT.md diagnose the GAN extension's core problem:
the pixel-space reconstruction score (global_mse/global_l1/global_ssim,
disc_realness in results/gan_features.npz) has AUC ~0.499 in isolation --
essentially uninformative once real test faces are included -- because
pixel-space L1/MSE reconstruction error is a crude anomaly signal that
also penalizes held-out genuine faces the generator never trained on
(different lighting/subject/pose), not just spliced regions.

report/REPORT.md Sec 6.3/8 flags the standard fix from the anomaly-
detection literature (Schlegl et al., "AnoGAN", 2017): score anomalies in
the DISCRIMINATOR'S LEARNED FEATURE SPACE (its penultimate-layer
embedding), not in raw pixel space. The intuition: the discriminator was
trained adversarially to tell real faces from the generator's
reconstructions, so its penultimate embedding encodes a much richer,
learned notion of "what a genuine face looks like" than a raw pixel
difference does, and should be more robust to exact-pixel domain shift
(lighting/pose) between train and held-out genuine faces.

METHOD
------
1. Load the already-trained discriminator (results/models/gan_discriminator.keras)
   -- no GAN retraining needed, this only adds a new scoring head on top.
2. Build a feature-extractor sub-model that outputs the discriminator's
   penultimate (post-Flatten, pre-Dropout, pre-final-Dense) activation for
   any image -- an 8192-d embedding at this architecture's 64x64 input size.
3. Fit a robust Gaussian (EmpiricalCovariance/Mahalanobis) model on the
   embeddings of REAL TRAINING-SPLIT images only (one-class, exactly as
   strict as the original GAN training protocol -- fakes are never seen).
4. Anomaly score for any image = Mahalanobis distance from its embedding to
   the genuine-face centroid in this learned space.
5. Because a raw Mahalanobis distance is not a calibrated P(fake), fit an
   isotonic regression calibrator mapping distance -> P(fake) on the
   VALIDATION split labels (never test) -- this also directly addresses
   report/REPORT.md Sec 6.3 item 3 ("no probability calibration was
   applied to the GAN score before fusion").
6. Evaluate on the same held-out test split as every other result in this
   report, and report accuracy/AUC/difficulty breakdown alongside the
   original pixel-space score for a direct, apples-to-apples comparison.

This does NOT require retraining the GAN (uses the already-saved
generator/discriminator) but DOES require re-running inference over all
images, so it needs the raw dataset at data/real_and_fake_face (same
requirement as gan_reconstruction.py).
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.covariance import EmpiricalCovariance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from gan_reconstruction import (
    IMG_SIZE, MODEL_DIR, FEATURES_PATH, _path_for_filename, get_train_split_filenames,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "gan_feature_space_results.json")
SEED = 42


def load_image_norm(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (img / 127.5) - 1.0  # [-1, 1], matches gan_reconstruction.py normalization


def build_embedding_model(discriminator):
    """Sub-model outputting the discriminator's penultimate (pre-Dropout,
    pre-final-Dense) Flatten activation for a given input batch."""
    flatten_layer = None
    for layer in discriminator.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            flatten_layer = layer
    if flatten_layer is None:
        raise RuntimeError("Could not locate Flatten layer in discriminator")
    return tf.keras.Model(discriminator.input, flatten_layer.output, name="disc_embedding")


def difficulty_breakdown(y_true, y_pred, difficulty):
    out = {}
    for level in ("easy", "mid", "hard"):
        mask = (difficulty == level)
        if mask.sum() == 0:
            continue
        out[level] = {"n": int(mask.sum()), "accuracy": float(accuracy_score(y_true[mask], y_pred[mask]))}
    return out


def main():
    print("Loading trained discriminator (no GAN retraining needed)...")
    discriminator = tf.keras.models.load_model(os.path.join(MODEL_DIR, "gan_discriminator.keras"))
    embed_model = build_embedding_model(discriminator)

    print("Loading feature index / reproducing canonical train/val/test split...")
    d = np.load(FEATURES_PATH, allow_pickle=True)
    y_full, diff_full, files_full = d["y"], d["difficulty"], d["filenames"]
    train_filenames, _, _ = get_train_split_filenames()

    idx = np.arange(len(files_full))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y_full, test_size=0.30, stratify=y_full, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    diff_test = diff_full[idx_test]

    def embed_all(indices):
        embs, keep_labels, keep_idx = [], [], []
        for i in indices:
            fname, label = files_full[i], y_full[i]
            path = _path_for_filename(fname, label)
            img = load_image_norm(path)
            if img is None:
                continue
            e = embed_model(img[None, ...], training=False).numpy()[0]
            embs.append(e)
            keep_labels.append(label)
            keep_idx.append(i)
        return np.stack(embs), np.array(keep_labels), np.array(keep_idx)

    print("Extracting discriminator embeddings for train (real-only), val, test splits...")
    idx_train_real = idx_train[y_full[idx_train] == 0]
    emb_train_real, _, _ = embed_all(idx_train_real)
    emb_val, y_val_kept, idx_val_kept = embed_all(idx_val)
    emb_test, y_test_kept, idx_test_kept = embed_all(idx_test)
    diff_test_kept = diff_full[idx_test_kept]

    print(f"  {emb_train_real.shape[0]} real train embeddings "
          f"({emb_train_real.shape[1]}-d) used to fit the one-class Gaussian")

    print("Fitting one-class Mahalanobis model on genuine training embeddings...")
    cov = EmpiricalCovariance().fit(emb_train_real)

    def mahalanobis_dist(emb):
        return cov.mahalanobis(emb)

    dist_val = mahalanobis_dist(emb_val)
    dist_test = mahalanobis_dist(emb_test)

    print("Calibrating distance -> P(fake) via isotonic regression on the validation split...")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(dist_val, y_val_kept)

    p_test = calibrator.predict(dist_test)
    pred_test = (p_test >= 0.5).astype(int)

    results = {
        "method": "discriminator_penultimate_embedding_mahalanobis_isotonic",
        "embedding_dim": int(emb_train_real.shape[1]),
        "n_train_real_for_oneclass_fit": int(emb_train_real.shape[0]),
        "accuracy": accuracy_score(y_test_kept, pred_test),
        "auc": roc_auc_score(y_test_kept, p_test),
        "difficulty_breakdown": difficulty_breakdown(y_test_kept, pred_test, diff_test_kept),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FEATURE-SPACE (AnoGAN-style) GAN SCORE SUMMARY ===")
    print(f"accuracy={results['accuracy']:.3f}  auc={results['auc']:.3f}")
    print("difficulty breakdown:", results["difficulty_breakdown"])
    print("\nCompare against the pixel-space score in results/gan_features.npz / "
          "report Sec 6.2 (accuracy=0.469, AUC=0.499) -- an AUC clearly above "
          "0.5 here would confirm the feature-space score carries real, "
          "usable information the pixel-space score did not.")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
