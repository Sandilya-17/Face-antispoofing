"""
Feature extraction for face anti-spoofing.

Methodology
-----------
This module implements a color-texture analysis pipeline in the spirit of
Boulkenafet, Komulainen & Hadid (2015/2016), "Face Anti-Spoofing Based on
Color Texture Analysis", and the classical LBP work of Ojala, Pietikainen &
Maenpaa (2002). The core hypothesis in this line of research is that print /
replay / mask spoofing artefacts (moire patterns, printing dot patterns,
loss of high-frequency skin micro-texture, color re-reproduction artefacts)
show up as measurable differences in:

  1. Local micro-texture (captured via multi-radius Local Binary Patterns)
  2. Global shape/edge structure (captured via HOG)
  3. Color reproduction statistics in perceptually/physically motivated
     color spaces (HSV and YCbCr), since spoofing mediums alter chrominance
     statistics relative to real skin reflectance.

Combining these into a single hand-crafted feature vector and feeding it to
a discriminative classifier is a well-established, lightweight, and fully
interpretable baseline for face PAD (Presentation Attack Detection).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

IMG_SIZE = 128  # resize target (speed/quality tradeoff)

# LBP configurations: (P, R) pairs -> multi-scale micro-texture
LBP_CONFIGS = [(8, 1), (16, 2), (24, 3)]


def load_and_preprocess(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img  # BGR, uint8


def lbp_histogram(gray, P, R):
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def multiscale_lbp_features(gray):
    feats = []
    for P, R in LBP_CONFIGS:
        feats.append(lbp_histogram(gray, P, R))
    return np.concatenate(feats)


def color_space_stats(img_bgr):
    """Mean/std/skew-ish stats per channel in HSV and YCbCr color spaces."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    feats = []
    for space in (hsv, ycc):
        for ch in range(3):
            channel = space[:, :, ch].astype(np.float32)
            feats.append(channel.mean())
            feats.append(channel.std())
            # simple histogram (coarse, 16 bins) captures color reproduction shape
            hist, _ = np.histogram(channel, bins=16, range=(0, 255), density=True)
            feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)


def hog_features(gray):
    feat = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_features(path, img_size=IMG_SIZE):
    img = load_and_preprocess(path, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp_feats = multiscale_lbp_features(gray)
    color_feats = color_space_stats(img)
    hog_feats = hog_features(gray)

    return np.concatenate([lbp_feats, color_feats, hog_feats])


def feature_dim(img_size=IMG_SIZE):
    """Utility to compute the total feature dimensionality (for docs/sanity)."""
    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
    lbp = multiscale_lbp_features(gray)
    color = color_space_stats(dummy)
    hg = hog_features(gray)
    return len(lbp), len(color), len(hg), len(lbp) + len(color) + len(hg)


if __name__ == "__main__":
    print("Feature dims (lbp, color, hog, total):", feature_dim())
