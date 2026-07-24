"""
Deep-learning baseline: fine-tuned MobileNetV2 (ImageNet-pretrained).

WHY THIS FILE EXISTS
---------------------
The classical color-texture pipeline (feature_extraction.py + train_evaluate.py)
needs a deep-learning comparison point to be a complete PAD study -- reviewers
in this field will always ask "how does this compare to a CNN?". This script
is that comparison.

NOT EXECUTED IN THIS REPO'S RESULTS
-------------------------------------
This script is provided ready-to-run but was NOT executed to produce numbers
in results/. Two hard constraints in the environment this project was
completed in:
  1. Only extracted 1730-d hand-crafted feature vectors were cached
     (results/features.npz) -- the raw JPEGs are not present, and a CNN
     needs raw pixels, not LBP/HOG/color-histogram summaries.
  2. No network access, so the original image corpus could not be
     re-downloaded, and no GPU was available to fine-tune in reasonable time
     on CPU.

Run this yourself with:
    git clone https://github.com/Sandilya-17/dataset.git /tmp/dataset_repo
    ln -s /tmp/dataset_repo/real_and_fake_face_detection/real_and_fake_face \
        data/real_and_fake_face
    pip install tensorflow   # or torch + torchvision, see PyTorch variant below
    python src/deep_baseline_mobilenetv2.py

and drop the resulting metrics into results/deep_baseline_metrics.json --
report.py / REPORT.md are written to slot them in under Section 6
("Comparison to a Deep Baseline") once present.

PROTOCOL (kept identical to the classical pipeline for a fair comparison)
---------------------------------------------------------------------------
- Same stratified 70/15/15 split, same random seed (42), same file lists
  (so the classical model and MobileNetV2 are compared on the exact same
  test images).
- Images resized to 160x160 (net's expected input) and pixel-normalized
  per Keras' `mobilenet_v2.preprocess_input`.
- MobileNetV2 base pretrained on ImageNet, frozen for the first 5 epochs
  (train only the new classification head), then the top ~30 layers
  unfrozen and fine-tuned at a lower LR for another 10 epochs
  (standard two-phase transfer-learning recipe).
- Binary cross-entropy loss, class weights balanced (960 fake / 1081 real
  is close to balanced, but we set them explicitly for parity with the
  classical models' class_weight="balanced").
- Light augmentation (horizontal flip, +/-10% brightness/contrast jitter)
  on TRAIN only -- print/replay artifacts (moire, print-dot patterns) can
  be washed out by aggressive augmentation, so we keep it conservative.
- Reports the same metric set as train_evaluate.py: accuracy, precision,
  recall, F1, ROC-AUC, confusion matrix, and the same per-attack-difficulty
  (easy/mid/hard) breakdown -- for a like-for-like table in the report.
"""

import glob
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_and_fake_face")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
IMG_SIZE = 160
SEED = 42
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 10
FINETUNE_LR = 1e-5
HEAD_LR = 1e-3
UNFREEZE_LAST_N = 30


def difficulty_from_name(fname):
    base = os.path.basename(fname).lower()
    for level in ("easy", "mid", "hard"):
        if base.startswith(level):
            return level
    return "real"


def build_file_lists():
    """Mirrors build_dataset.py's file discovery + labeling exactly, so the
    split below lines up 1:1 with the classical pipeline's split."""
    real_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_real", "*.jpg")))
    fake_files = sorted(glob.glob(os.path.join(DATA_DIR, "training_fake", "*.jpg")))
    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    difficulty = ["real"] * len(real_files) + [difficulty_from_name(f) for f in fake_files]
    return np.array(files), np.array(labels), np.array(difficulty)


def main():
    # Imported lazily so this file can be inspected/tested without requiring
    # tensorflow to be installed in every environment that touches this repo.
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"Dataset not found at {DATA_DIR}. See data/README_DATA.md to "
            f"fetch it, then re-run this script."
        )

    files, labels, difficulty = build_file_lists()
    idx = np.arange(len(files))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, stratify=labels[idx], random_state=SEED
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=labels[idx_temp], random_state=SEED
    )

    def make_ds(indices, training):
        f = files[indices]
        y = labels[indices]

        def load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            if training:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.1)
                img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((f, y))
        if training:
            ds = ds.shuffle(len(f), seed=SEED)
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(idx_train, training=True)
    val_ds = make_ds(idx_val, training=False)
    test_ds = make_ds(idx_test, training=False)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    class_weight = {
        0: len(labels) / (2 * np.sum(labels == 0)),
        1: len(labels) / (2 * np.sum(labels == 1)),
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(HEAD_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print("Phase 1: training classification head (base frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weight)

    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print(f"Phase 2: fine-tuning last {UNFREEZE_LAST_N} layers")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, class_weight=class_weight)

    y_test = labels[idx_test]
    d_test = difficulty[idx_test]
    y_score = model.predict(test_ds).ravel()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_score),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
    )

    breakdown = {}
    for level in ["easy", "mid", "hard"]:
        mask = d_test == level
        if mask.sum() == 0:
            continue
        breakdown[level] = dict(
            n=int(mask.sum()),
            recall_as_fake_detected=float(accuracy_score(y_test[mask], y_pred[mask])),
        )

    out = dict(
        model="MobileNetV2 (ImageNet-pretrained, fine-tuned)",
        img_size=IMG_SIZE, epochs_head=EPOCHS_HEAD, epochs_finetune=EPOCHS_FINETUNE,
        n_train=len(idx_train), n_val=len(idx_val), n_test=len(idx_test),
        test_results=metrics, difficulty_breakdown=breakdown,
    )
    out_path = os.path.join(RESULTS_DIR, "deep_baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTest accuracy={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
