"""
Unit tests for src/feature_extraction.py.

These tests are deliberately dataset-free: they synthesize small in-memory
images so the pipeline's shape/consistency guarantees can be checked in CI
without requiring the (large, license-restricted) CIPLAB/NUAA corpora. They
are correctness/regression checks on the descriptor plumbing, not a
replacement for the reported accuracy/AUC numbers in results/ and
report/REPORT.md, which require the real datasets to reproduce.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_extraction import (  # noqa: E402
    IMG_SIZE,
    LBP_CONFIGS,
    color_space_stats,
    feature_dim,
    hog_features,
    multiscale_lbp_features,
)


def _random_bgr_image(size=IMG_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _random_gray_image(size=IMG_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def test_lbp_histogram_dims_match_configs():
    gray = _random_gray_image()
    feats = multiscale_lbp_features(gray)
    expected_dim = sum(p + 2 for p, _ in LBP_CONFIGS)
    assert feats.shape == (expected_dim,)
    assert feats.dtype == np.float32


def test_lbp_histograms_are_valid_densities():
    gray = _random_gray_image()
    feats = multiscale_lbp_features(gray)
    # Each per-scale histogram is a `density=True` np.histogram output, so
    # values must be non-negative (they integrate to 1, not individually).
    assert np.all(feats >= 0)


def test_color_space_stats_dim_and_range():
    img = _random_bgr_image()
    feats = color_space_stats(img)
    # 2 color spaces x 3 channels x (mean + std + 16-bin hist) = 2*3*18 = 108
    assert feats.shape == (108,)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_hog_features_nonempty_and_finite():
    gray = _random_gray_image()
    feats = hog_features(gray)
    assert feats.ndim == 1
    assert feats.shape[0] > 0
    assert np.all(np.isfinite(feats))


def test_feature_dim_matches_readme_total():
    lbp_dim, color_dim, hog_dim, total = feature_dim()
    assert lbp_dim == 54
    assert color_dim == 108
    # README / report advertise a 1,730-d combined vector.
    assert total == 1730
    assert total == lbp_dim + color_dim + hog_dim


def test_features_are_deterministic_for_same_input():
    gray = _random_gray_image(seed=42)
    img = _random_bgr_image(seed=42)
    first = np.concatenate(
        [multiscale_lbp_features(gray), color_space_stats(img), hog_features(gray)]
    )
    second = np.concatenate(
        [multiscale_lbp_features(gray), color_space_stats(img), hog_features(gray)]
    )
    np.testing.assert_array_equal(first, second)


def test_extract_features_raises_on_missing_file(tmp_path):
    from feature_extraction import extract_features

    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ValueError):
        extract_features(str(missing))
