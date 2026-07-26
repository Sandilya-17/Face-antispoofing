"""
Builds a COMBINED multi-attack-type feature dataset by merging:
  - the existing splice-attack corpus (Real-and-Fake-Face-Detection,
    already extracted into results/features.npz by build_dataset.py)
  - the NUAA Photograph Imposter Database (print attacks), extracted
    fresh from the same folder layout already used by cross_dataset_eval.py
    (<nuaa_root>/real/*.jpg, <nuaa_root>/fake/*.jpg)

Every sample is tagged with a `source` label ("splice" or "print") so that
downstream training/evaluation can report accuracy broken down by attack
type -- this is what lets you test whether training on BOTH attack types
produces a classifier that transfers better than the splice-only model in
Section 12 of the paper, instead of assuming it.

Prerequisite: run build_dataset.py first (so results/features.npz exists).

Usage:
    python src/build_dataset_combined.py --nuaa_root /path/to/nuaa

Output: results/features_combined.npz containing X, y, source, filenames
(source: "splice" or "print"; y: 0=real, 1=fake, same convention throughout)
"""

import argparse
import glob
import os
import time

import numpy as np

from feature_extraction import extract_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SPLICE_FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "features_combined.npz")


def load_splice_features():
    if not os.path.exists(SPLICE_FEATURES_PATH):
        raise SystemExit(
            f"{SPLICE_FEATURES_PATH} not found -- run build_dataset.py first."
        )
    data = np.load(SPLICE_FEATURES_PATH, allow_pickle=True)
    X, y, filenames = data["X"], data["y"], data["filenames"]
    source = np.array(["splice"] * len(y))
    print(f"Loaded {len(y)} splice-corpus samples from {SPLICE_FEATURES_PATH}")
    return X, y, source, filenames


def extract_nuaa_features(nuaa_root):
    real_files = sorted(
        glob.glob(os.path.join(nuaa_root, "real", "*.jpg"))
        + glob.glob(os.path.join(nuaa_root, "real", "*.png"))
    )
    fake_files = sorted(
        glob.glob(os.path.join(nuaa_root, "fake", "*.jpg"))
        + glob.glob(os.path.join(nuaa_root, "fake", "*.png"))
    )
    if not real_files or not fake_files:
        raise SystemExit(
            f"Expected {nuaa_root}/real/*.jpg and {nuaa_root}/fake/*.jpg -- "
            f"found {len(real_files)} real, {len(fake_files)} fake. This should "
            f"be the same folder you already point cross_dataset_eval.py at."
        )
    print(f"Found {len(real_files)} NUAA real, {len(fake_files)} NUAA fake images")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    X, y, filenames = [], [], []
    t0 = time.time()
    for i, (path, label) in enumerate(all_files):
        try:
            feats = extract_features(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        X.append(feats)
        y.append(label)
        filenames.append(os.path.basename(path))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i+1}/{len(all_files)}  ({elapsed:.1f}s)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    source = np.array(["print"] * len(y))
    filenames = np.array(filenames)
    print(f"Extracted NUAA features: {X.shape} in {time.time()-t0:.1f}s")
    return X, y, source, filenames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuaa_root", required=True,
        help="Path to NUAA folder laid out as <nuaa_root>/{real,fake}/*.jpg "
             "(same layout already used for cross_dataset_eval.py --target_root)",
    )
    args = parser.parse_args()

    X_splice, y_splice, src_splice, f_splice = load_splice_features()
    X_nuaa, y_nuaa, src_nuaa, f_nuaa = extract_nuaa_features(args.nuaa_root)

    if X_splice.shape[1] != X_nuaa.shape[1]:
        raise SystemExit(
            f"Feature dimension mismatch: splice={X_splice.shape[1]}d, "
            f"NUAA={X_nuaa.shape[1]}d. feature_extraction.py must be unchanged "
            f"between build_dataset.py and this script."
        )

    X = np.concatenate([X_splice, X_nuaa], axis=0)
    y = np.concatenate([y_splice, y_nuaa], axis=0)
    source = np.concatenate([src_splice, src_nuaa], axis=0)
    filenames = np.concatenate([f_splice, f_nuaa], axis=0)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(OUT_PATH, X=X, y=y, source=source, filenames=filenames)
    print(
        f"\nSaved combined dataset: {X.shape[0]} samples "
        f"({(source == 'splice').sum()} splice, {(source == 'print').sum()} print) "
        f"-> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
