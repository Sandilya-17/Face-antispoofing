"""
Splice-forensics feature extraction: Error Level Analysis (ELA) + noise-
residual (SRM-style) statistics.

NOVELTY / RESEARCH MOTIVATION
------------------------------
Section 8 of the paper (and results/metrics_summary.json feature_group_
importance) establishes that HOG carries ~92% of this pipeline's signal,
and Section 8.1 gives the mechanistic reason: this dataset's "fake" class
is expert Photoshop splicing, and HOG is picking up the *edge discontinuity*
at the splice boundary (mismatched skin tone/lighting/alignment where two
photos are blended). HOG was designed as a generic shape/edge descriptor,
not a splice detector -- it happens to catch this signal as a side effect.

This module adds two features purpose-built for splice detection in the
image-forensics literature, which the existing pipeline has never used:

1. Error Level Analysis (ELA). Recompress the image at a fixed JPEG
   quality and take the per-pixel absolute difference from the original.
   Regions with a different original compression history (as when a patch
   is pasted in from a different source photo) re-compress differently and
   show a different error-level signature than the rest of the image, most
   visibly right at the splice edge. Standard, well-established forensic
   technique (Krawetz, "A Picture's Worth: Digital Image Analysis and
   Forensics", 2007).

2. Noise-residual statistics (SRM-lite). Splice regions carry the noise
   fingerprint of their *source* photo (sensor noise, prior compression,
   local sharpening), which usually does not match the host image's noise
   pattern. We approximate a steganalysis rich model (SRM) by passing the
   image through a small bank of high-pass filters and computing regional
   statistics of the residuals -- a different, complementary signal to
   HOG's gradient-orientation histograms.

Both features are computed globally (whole face) AND per-region (eyes/
nose/mouth via the existing MediaPipe FaceMesh boxes in region_features.py)
since Section 7.3 shows subtle "easy" (single-region) splices are exactly
what the existing global descriptors miss.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Error Level Analysis
# ---------------------------------------------------------------------------

ELA_QUALITY = 90  # JPEG recompression quality
ELA_SCALE = 15    # brightness scaling so weak differences are numerically usable


def error_level_analysis(img_bgr, quality=ELA_QUALITY):
    """Returns an ELA residual map (float32, same H,W as input, single channel)."""
    ok, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG re-encode failed during ELA")
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(img_bgr.astype(np.int16), recompressed.astype(np.int16))
    diff = diff.astype(np.float32) * ELA_SCALE
    diff = np.clip(diff, 0, 255)
    ela_gray = diff.mean(axis=2)  # collapse to single channel
    return ela_gray


def ela_region_stats(ela_gray, box=None):
    """Mean/std/max/energy of an ELA map, optionally restricted to a box
    (x1, y1, x2, y2)."""
    patch = ela_gray if box is None else ela_gray[box[1]:box[3], box[0]:box[2]]
    if patch.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [patch.mean(), patch.std(), patch.max(), np.sqrt((patch ** 2).mean())],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# 2. Noise-residual (SRM-lite) statistics
# ---------------------------------------------------------------------------

# A small, standard bank of high-pass residual filters (subset in the spirit
# of the Rich Models used in JPEG/spatial steganalysis -- first- and second-
# order pixel-difference kernels). Not the full SRM (30 filters); a compact
# subset chosen for speed while covering horizontal/vertical/diagonal and
# 2nd-order local structure.
_SRM_KERNELS = [
    np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32),      # horiz 1st-order
    np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),      # vert 1st-order
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),      # diag 1st-order
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,  # horiz 2nd-order
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,  # vert 2nd-order
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 4,  # laplacian-ish
]


def noise_residuals(gray):
    """Applies the SRM-lite filter bank; returns list of residual maps."""
    gray_f = gray.astype(np.float32)
    return [cv2.filter2D(gray_f, -1, k, borderType=cv2.BORDER_REFLECT) for k in _SRM_KERNELS]


def srm_region_stats(residuals, box=None):
    """Mean-abs / std / kurtosis-proxy per residual map, optionally boxed."""
    feats = []
    for r in residuals:
        patch = r if box is None else r[box[1]:box[3], box[0]:box[2]]
        if patch.size == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        abs_patch = np.abs(patch)
        feats.append(float(abs_patch.mean()))
        feats.append(float(patch.std()))
        # simple 4th-moment "peakedness" proxy, cheap alternative to scipy kurtosis
        std = patch.std() + 1e-6
        feats.append(float(((patch / std) ** 4).mean()))
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Combined global + region feature vector
# ---------------------------------------------------------------------------

def splice_forensics_features(img_bgr, region_boxes=None):
    """
    Parameters
    ----------
    img_bgr : np.ndarray, the already-loaded/resized face crop (as produced
        by feature_extraction.load_and_preprocess).
    region_boxes : optional dict {"eyes": (x1,y1,x2,y2), "nose": ..., "mouth": ...}
        Pixel boxes in the SAME coordinate frame as img_bgr. If omitted, only
        the global (whole-image) ELA/SRM stats are computed.

    Returns
    -------
    np.ndarray, float32, shape (D,) where
        D = 4 (global ELA) + 18 (global SRM, 6 filters x 3 stats)
          + [region present] * (4 + 18) per region
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    ela_map = error_level_analysis(img_bgr)
    residual_maps = noise_residuals(gray)

    feats = [ela_region_stats(ela_map), srm_region_stats(residual_maps)]

    if region_boxes:
        for region_name in ("eyes", "nose", "mouth"):
            box = region_boxes.get(region_name)
            feats.append(ela_region_stats(ela_map, box))
            feats.append(srm_region_stats(residual_maps, box))

    return np.concatenate(feats).astype(np.float32)


FEATURE_DIM_GLOBAL_ONLY = 4 + 18            # = 22
FEATURE_DIM_WITH_REGIONS = 22 + 3 * 22      # = 88 (global + eyes + nose + mouth)
