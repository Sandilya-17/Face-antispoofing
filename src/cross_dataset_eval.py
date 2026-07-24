"""
Cross-dataset generalization protocol.

WHY THIS FILE EXISTS
---------------------
Single-dataset accuracy answers "can this model separate real/fake images
that look like its own training distribution?" It does NOT answer the
question the face-PAD field actually cares about: does the model generalize
to a different camera sensor, different spoof-medium construction, and
different acquisition conditions? That requires training on one dataset and
testing on another, e.g.:

    Train: Real-and-Fake-Face-Detection corpus (this project's data)
    Test:  OULU-NPU Protocol 1 (unseen sensors) OR CASIA-FASD OR
           Replay-Attack (Idiap)

and reporting the standard PAD generalization metrics -- APCER, BPCER,
ACER (ISO/IEC 30107-3) -- alongside plain accuracy/AUC, since cross-dataset
performance is usually reported in APCER/BPCER/ACER in the PAD literature,
not accuracy.

NOT EXECUTED IN THIS REPO'S RESULTS
-------------------------------------
This script is provided ready-to-run but was NOT executed here, because:
  1. No network access in this environment to download OULU-NPU / CASIA-FASD
     / Replay-Attack / SiW -- these require signing individual data-use
     agreements with the releasing institutions (Idiap, Oulu University,
     CASIA), which is a manual, per-user process that can't be scripted.
  2. This project's own corpus is 2D print/GAN spoofs only (no video,
     no replay, no 3D mask) -- so even after downloading a target set,
     you may need to subset it to comparable attack types for a fair
     reading of the numbers (e.g. exclude video-replay attacks against a
     model that only ever saw single-frame print spoofs).

HOW TO RUN THIS ONCE YOU HAVE ACCESS TO A TARGET DATASET
-----------------------------------------------------------
1. Apply for / download one of: OULU-NPU, CASIA-FASD, Replay-Attack, SiW.
2. Arrange its images into `<target_root>/real/*.jpg` and
   `<target_root>/fake/*.jpg` (or edit `load_target_dataset` below to your
   directory layout -- e.g. per-video-frame extraction for Replay-Attack).
3. python src/cross_dataset_eval.py --target_root /path/to/target_dataset
4. Results (accuracy, AUC, APCER, BPCER, ACER) are written to
   results/cross_dataset_metrics.json.
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np

from feature_extraction import extract_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")


def load_target_dataset(target_root):
    """Expects target_root/real/*.jpg (or .png) and target_root/fake/*.jpg.
    Edit this function if your downloaded dataset uses a different layout
    (e.g. per-frame video extraction, nested protocol folders for OULU-NPU)."""
    real = sorted(
        glob.glob(os.path.join(target_root, "real", "*.jpg"))
        + glob.glob(os.path.join(target_root, "real", "*.png"))
    )
    fake = sorted(
        glob.glob(os.path.join(target_root, "fake", "*.jpg"))
        + glob.glob(os.path.join(target_root, "fake", "*.png"))
    )
    if not real or not fake:
        raise SystemExit(
            f"Expected {target_root}/real/*.jpg and {target_root}/fake/*.jpg "
            f"-- found {len(real)} real, {len(fake)} fake. Edit "
            f"load_target_dataset() to match your downloaded dataset's layout."
        )
    files = real + fake
    labels = np.array([0] * len(real) + [1] * len(fake))
    return files, labels


def apcer_bpcer_acer(y_true, y_pred):
    """ISO/IEC 30107-3 style PAD metrics.
    APCER: Attack Presentation Classification Error Rate
           -- fraction of FAKE (attack) samples wrongly classified as REAL.
    BPCER: Bona-fide Presentation Classification Error Rate
           -- fraction of REAL (bona fide) samples wrongly classified as FAKE.
    ACER:  Average Classification Error Rate = (APCER + BPCER) / 2.
    Convention here: label 1 = fake/attack, label 0 = real/bona fide."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fake_mask = y_true == 1
    real_mask = y_true == 0
    apcer = float(np.mean(y_pred[fake_mask] == 0)) if fake_mask.sum() else float("nan")
    bpcer = float(np.mean(y_pred[real_mask] == 1)) if real_mask.sum() else float("nan")
    acer = (apcer + bpcer) / 2
    return dict(apcer=apcer, bpcer=bpcer, acer=acer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_root", required=True,
                         help="Path to an external PAD dataset laid out as "
                              "<target_root>/{real,fake}/*.jpg")
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

    files, y_true = load_target_dataset(args.target_root)
    print(f"Extracting features for {len(files)} cross-dataset images...")
    X = np.stack([extract_features(f) for f in files]).astype(np.float32)

    X_s = scaler.transform(X)
    X_p = pca.transform(X_s)
    y_pred = model.predict(X_p)
    y_score = model.predict_proba(X_p)[:, 1]

    metrics = dict(
        accuracy=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_score),
        **apcer_bpcer_acer(y_true, y_pred),
        n_target=len(files),
    )
    print(json.dumps(metrics, indent=2))

    out_path = os.path.join(RESULTS_DIR, "cross_dataset_metrics.json")
    with open(out_path, "w") as f:
        json.dump(dict(target_root=args.target_root, **metrics), f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
