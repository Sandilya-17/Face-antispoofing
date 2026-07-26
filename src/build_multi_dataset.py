"""
Multi-dataset feature builder for cross-domain generalization work.
Extracts the same feature vector as build_dataset.py but from multiple
dataset roots, tagging each sample with its source dataset so
leave_one_dataset_out.py can hold datasets out cleanly.
Output: results/multi_features.npz
"""

import glob
import os
import time

import numpy as np

from feature_extraction import extract_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "multi_features.npz")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

DATASETS = {
    "ciplab_splice": {
        "real_dir": os.path.join(DATA_ROOT, "real_and_fake_face", "training_real"),
        "fake_dir": os.path.join(DATA_ROOT, "real_and_fake_face", "training_fake"),
        "attack_type": "splice",
    },
    "nuaa_print": {
        "real_dir": os.path.join(DATA_ROOT, "nuaa_final", "real"),
        "fake_dir": os.path.join(DATA_ROOT, "nuaa_final", "fake"),
        "attack_type": "print",
    },
}


def collect_images(*dirs):
    files = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(sorted(glob.glob(os.path.join(d, ext))))
    return files


def main():
    all_records = []

    for ds_name, cfg in DATASETS.items():
        real_files = collect_images(cfg["real_dir"])
        fake_files = collect_images(cfg["fake_dir"])
        if not real_files and not fake_files:
            print(f"[skip] '{ds_name}': no images found at "
                  f"{cfg['real_dir']} / {cfg['fake_dir']}")
            continue
        print(f"[{ds_name}] real={len(real_files)} fake={len(fake_files)} "
              f"attack_type={cfg['attack_type']}")
        all_records += [(f, 0, ds_name, cfg["attack_type"]) for f in real_files]
        all_records += [(f, 1, ds_name, cfg["attack_type"]) for f in fake_files]

    if not all_records:
        raise SystemExit("No datasets found. Edit DATASETS in this file.")

    X, y, dataset, attack_type, filenames = [], [], [], [], []
    t0 = time.time()
    for i, (path, label, ds_name, atype) in enumerate(all_records):
        try:
            feats = extract_features(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        X.append(feats)
        y.append(label)
        dataset.append(ds_name)
        attack_type.append(atype)
        filenames.append(os.path.basename(path))

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(all_records)}  ({time.time()-t0:.1f}s)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    dataset = np.array(dataset)
    attack_type = np.array(attack_type)
    filenames = np.array(filenames)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        OUT_PATH, X=X, y=y, dataset=dataset, attack_type=attack_type,
        filenames=filenames,
    )
    print(f"\nSaved combined {X.shape} feature matrix -> {OUT_PATH}")
    print("Per-dataset counts:")
    for ds_name in np.unique(dataset):
        mask = dataset == ds_name
        print(f"  {ds_name}: real={int((y[mask]==0).sum())} "
              f"fake={int((y[mask]==1).sum())}")


if __name__ == "__main__":
    main()
