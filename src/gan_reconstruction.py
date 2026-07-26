"""
GAN-based one-class reconstruction anomaly scoring for face anti-spoofing.

NOVELTY / RESEARCH MOTIVATION
-----------------------------
data/README_DATA.md documents an important, easy-to-miss fact about this
corpus: the "fake" images are NOT GAN-generated. They are expert-generated
Photoshop composites -- human retouchers splicing eyes/nose/mouth/whole-face
regions from DIFFERENT real photographs together. The blending seams left by
this splicing (mismatched skin tone, lighting, micro-texture, and alignment
at the splice boundary) are local statistical discontinuities relative to a
genuine, single-source face.

report/REPORT.md's own feature-importance analysis (Sec 5.5) found that HOG
(edge/shape) features dominate the classical pipeline's decisions (92.3% of
Random-Forest Gini importance) -- consistent with the model picking up on
exactly these edge/gradient discontinuities at splice boundaries, but doing
so only through a global, hand-crafted, non-learned descriptor. Sec 5.4 also
found the classical pipeline is WORST on "easy" (subtle, single-region)
attacks, precisely because a small, localized splice barely perturbs global
image statistics.

This module proposes and implements a complementary, learned anomaly cue:

  Train a small convolutional GAN (encoder-decoder generator + discriminator)
  in a strictly ONE-CLASS fashion -- using ONLY real training-split face
  images, and NEVER any fake/spliced image. The generator therefore learns
  to reconstruct the manifold of genuine, single-source faces. A spliced
  face, containing a region whose texture/lighting statistics come from a
  DIFFERENT source photo, is out-of-distribution for that manifold and
  should reconstruct poorly -- especially in a *localized* way, concentrated
  at the spliced region, which is exactly the failure mode (Sec 5.4) that
  global hand-crafted descriptors miss.

At inference we extract, for every image (real or fake, any split):
  1. global reconstruction error (pixel L1/L2 + SSIM)               -- a
     learned, holistic "naturalness" residual (complements HOG/color/LBP).
  2. per-facial-region reconstruction error (eyes / nose / mouth,
     located via the same MediaPipe FaceMesh landmark boxes used in
     region_features.py) -- a learned, LOCALIZED residual designed
     specifically to catch subtle single-region splices.
  3. the discriminator's learned "realness" score for the original image
     -- an implicit deep naturalness detector trained adversarially.

These GAN-derived scalars are saved to results/gan_features.npz and later
fused (src/train_gan_fusion.py) with the existing hand-crafted feature
vector, under the IDENTICAL train/val/test split protocol (seed=42) used
throughout this project, so results are directly comparable to
report/REPORT.md and results/hybrid_fusion_results.json.

Because the GAN only ever needs REAL images to train, this approach is also
practically attractive: it does not require a labelled, diverse fake corpus
to build the anomaly detector, which is valuable given how narrow and
attack-specific most public spoof datasets are.
"""

import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

# NOTE: src/region_features.py (v3) depends on mediapipe's legacy
# `solutions.face_mesh` API, which was removed in mediapipe>=0.10.10 (see
# the docstring of region_features_v1_haarcascade_backup.py for the same
# issue already documented elsewhere in this project). Rather than add a
# fragile mediapipe-version pin, this module uses the same Haar-cascade +
# anthropometric-proportion fallback strategy as that backup file --
# zero extra downloads, robust, and consistent with this project's own
# precedent for handling that dependency break.
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _detect_face_box(gray):
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return fx, fy, fw, fh


def _eye_region_box(gray, face_box):
    fx, fy, fw, fh = face_box
    upper_half = gray[fy:fy + fh // 2, fx:fx + fw]
    eyes = _eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    if len(eyes) >= 1:
        xs0 = [fx + ex for (ex, ey, ew, eh) in eyes]
        ys0 = [fy + ey for (ex, ey, ew, eh) in eyes]
        xs1 = [fx + ex + ew for (ex, ey, ew, eh) in eyes]
        ys1 = [fy + ey + eh for (ex, ey, ew, eh) in eyes]
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        pad_x, pad_y = (x1 - x0) * 0.25, (y1 - y0) * 0.4
        return x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    return fx, fy + 0.20 * fh, fx + fw, fy + 0.50 * fh


def _nose_region_box(face_box):
    fx, fy, fw, fh = face_box
    return fx + 0.20 * fw, fy + 0.35 * fh, fx + 0.80 * fw, fy + 0.68 * fh


def _mouth_region_box(face_box):
    fx, fy, fw, fh = face_box
    return fx + 0.15 * fw, fy + 0.62 * fh, fx + 0.85 * fw, fy + 0.92 * fh


def _fallback_boxes(gray):
    h, w = gray.shape[:2]
    eye_box = (0.10 * w, 0.20 * h, 0.90 * w, 0.50 * h)
    nose_box = (0.30 * w, 0.35 * h, 0.70 * w, 0.70 * h)
    mouth_box = (0.20 * w, 0.60 * h, 0.80 * w, 0.90 * h)
    return eye_box, nose_box, mouth_box


def _detect_landmark_boxes(gray):
    """Returns (eye_box, nose_box, mouth_box) in absolute pixel coords, or
    None if no face is detected at all (fallback boxes are used instead)."""
    face_box = _detect_face_box(gray)
    if face_box is None:
        return None
    eye_box = _eye_region_box(gray, face_box)
    nose_box = _nose_region_box(face_box)
    mouth_box = _mouth_region_box(face_box)
    return eye_box, nose_box, mouth_box


SEED = 42
IMG_SIZE = 64          # GAN operates on downsampled faces (CPU-friendly)
LANDMARK_SIZE = 256    # resolution at which FaceMesh landmark boxes are computed (matches region_features.py)
LATENT_DIM = 128
EPOCHS = 40
BATCH_SIZE = 32

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..", "data", "real_and_fake_face")
RESULTS_DIR = os.path.join(ROOT, "..", "results")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "gan_features.npz")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

tf.random.set_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _path_for_filename(fname, label):
    sub = "training_real" if label == 0 else "training_fake"
    return os.path.join(DATA_DIR, sub, fname)


def load_image_64(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def get_train_split_filenames():
    """Reproduce the EXACT same 70/15/15 stratified split (seed=42) used by
    train_evaluate.py / train_hybrid_fusion.py, so the GAN only ever sees
    training-split REAL images -- no leakage from val/test into training."""
    d = np.load(FEATURES_PATH, allow_pickle=True)
    y, files = d["y"], d["filenames"]
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    return set(files[idx_train]), y, files


# --------------------------------------------------------------------------
# Model: small conv encoder-decoder generator + conv discriminator
# --------------------------------------------------------------------------

def build_generator():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")(inp)   # 32x32
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(x)     # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)    # 8x8
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(LATENT_DIM, name="latent")(x)

    x = tf.keras.layers.Dense(8 * 8 * 128)(latent)
    x = tf.keras.layers.Reshape((8, 8, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)  # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)   # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)   # 64x64
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    return tf.keras.Model(inp, out, name="generator")


def build_discriminator():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")(inp)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1)(x)  # logits
    return tf.keras.Model(inp, out, name="discriminator")


# --------------------------------------------------------------------------
# Training (adversarial autoencoder: L1 recon + adversarial loss for G,
# standard GAN loss for D, discriminating real faces from reconstructions)
# --------------------------------------------------------------------------

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def train_gan(train_imgs):
    generator = build_generator()
    discriminator = build_discriminator()
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    n = train_imgs.shape[0]
    steps_per_epoch = max(1, n // BATCH_SIZE)
    recon_weight = 100.0

    print(f"Training GAN on {n} real training-split images, "
          f"{EPOCHS} epochs x {steps_per_epoch} steps...")
    t0 = time.time()
    for epoch in range(EPOCHS):
        perm = np.random.permutation(n)
        g_losses, d_losses, r_losses = [], [], []
        for step in range(steps_per_epoch):
            batch_idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            real_batch = train_imgs[batch_idx]

            with tf.GradientTape() as d_tape:
                recon = generator(real_batch, training=True)
                real_logits = discriminator(real_batch, training=True)
                fake_logits = discriminator(recon, training=True)
                d_loss = bce(tf.ones_like(real_logits), real_logits) + \
                    bce(tf.zeros_like(fake_logits), fake_logits)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            with tf.GradientTape() as g_tape:
                recon = generator(real_batch, training=True)
                fake_logits = discriminator(recon, training=True)
                adv_loss = bce(tf.ones_like(fake_logits), fake_logits)
                recon_loss = tf.reduce_mean(tf.abs(recon - real_batch))
                g_loss = adv_loss + recon_weight * recon_loss
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

            g_losses.append(float(adv_loss))
            d_losses.append(float(d_loss))
            r_losses.append(float(recon_loss))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  "
                  f"D={np.mean(d_losses):.3f}  G_adv={np.mean(g_losses):.3f}  "
                  f"L1_recon={np.mean(r_losses):.4f}  ({time.time()-t0:.0f}s elapsed)")

    return generator, discriminator


# --------------------------------------------------------------------------
# Feature extraction: reconstruction residuals + discriminator score
# --------------------------------------------------------------------------

def extract_gan_features(path, generator, discriminator):
    img_lm = cv2.imread(path)
    if img_lm is None:
        return None
    img_lm = cv2.resize(img_lm, (LANDMARK_SIZE, LANDMARK_SIZE), interpolation=cv2.INTER_AREA)
    gray_lm = cv2.cvtColor(img_lm, cv2.COLOR_BGR2GRAY)

    boxes = _detect_landmark_boxes(gray_lm)
    if boxes is None:
        boxes = _fallback_boxes(gray_lm)
    eye_box, nose_box, mouth_box = boxes
    scale = IMG_SIZE / LANDMARK_SIZE
    scaled_boxes = [tuple(int(round(v * scale)) for v in b) for b in (eye_box, nose_box, mouth_box)]

    img64 = cv2.resize(img_lm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img64_rgb = cv2.cvtColor(img64, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img64_rgb / 127.5) - 1.0  # [-1, 1]

    batch = img_norm[None, ...]
    recon = generator(batch, training=False).numpy()[0]
    disc_logit = float(discriminator(batch, training=False).numpy()[0, 0])
    disc_score = 1.0 / (1.0 + np.exp(-disc_logit))  # sigmoid -> "realness" in [0,1]

    orig_01 = (img_norm + 1.0) / 2.0
    recon_01 = (recon + 1.0) / 2.0

    global_mse = float(np.mean((orig_01 - recon_01) ** 2))
    global_l1 = float(np.mean(np.abs(orig_01 - recon_01)))
    try:
        global_ssim = float(ssim(orig_01, recon_01, channel_axis=2, data_range=1.0))
    except TypeError:
        global_ssim = float(ssim(orig_01, recon_01, multichannel=True, data_range=1.0))

    region_mses = []
    for (x0, y0, x1, y1) in scaled_boxes:
        x0, x1 = max(0, x0), min(IMG_SIZE, max(x1, x0 + 1))
        y0, y1 = max(0, y0), min(IMG_SIZE, max(y1, y0 + 1))
        patch_o = orig_01[y0:y1, x0:x1]
        patch_r = recon_01[y0:y1, x0:x1]
        if patch_o.size == 0:
            region_mses.append(global_mse)
        else:
            region_mses.append(float(np.mean((patch_o - patch_r) ** 2)))

    eye_mse, nose_mse, mouth_mse = region_mses
    return np.array([
        global_mse, global_l1, global_ssim, disc_score,
        eye_mse, nose_mse, mouth_mse,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "global_mse", "global_l1", "global_ssim", "disc_realness",
    "eye_mse", "nose_mse", "mouth_mse",
]


def main():
    print("Loading feature index / reproducing canonical train split...")
    train_filenames, y_all, files_all = get_train_split_filenames()

    print("Loading REAL training-split images for one-class GAN training...")
    train_imgs = []
    for fname, label in zip(files_all, y_all):
        if label == 0 and fname in train_filenames:
            path = _path_for_filename(fname, label)
            img = load_image_64(path)
            if img is not None:
                train_imgs.append((img / 127.5) - 1.0)
    train_imgs = np.stack(train_imgs).astype(np.float32)
    print(f"  {train_imgs.shape[0]} real training images loaded at {IMG_SIZE}x{IMG_SIZE}")

    generator, discriminator = train_gan(train_imgs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    generator.save(os.path.join(MODEL_DIR, "gan_generator.keras"))
    discriminator.save(os.path.join(MODEL_DIR, "gan_discriminator.keras"))
    print(f"Saved generator/discriminator to {MODEL_DIR}")

    print("Extracting GAN reconstruction/anomaly features for ALL images...")
    X_gan, y_out, diff_out, fn_out = [], [], [], []
    d = np.load(FEATURES_PATH, allow_pickle=True)
    y_full, diff_full, files_full = d["y"], d["difficulty"], d["filenames"]

    t0 = time.time()
    n_skip = 0
    for i, (fname, label, diff) in enumerate(zip(files_full, y_full, diff_full)):
        path = _path_for_filename(fname, label)
        feats = extract_gan_features(path, generator, discriminator)
        if feats is None:
            n_skip += 1
            continue
        X_gan.append(feats)
        y_out.append(label)
        diff_out.append(diff)
        fn_out.append(fname)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files_full)}  ({time.time()-t0:.0f}s)")

    X_gan = np.stack(X_gan).astype(np.float32)
    y_out = np.array(y_out, dtype=np.int64)
    diff_out = np.array(diff_out)
    fn_out = np.array(fn_out)

    np.savez_compressed(
        OUT_PATH, X=X_gan, y=y_out, difficulty=diff_out, filenames=fn_out,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved {X_gan.shape} GAN feature matrix to {OUT_PATH} "
          f"({n_skip} images skipped, {time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
