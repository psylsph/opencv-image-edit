"""Tests for app.pipeline.grain — luminance-aware film grain (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.grain import apply_grain

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bgr(value: int = 128, w: int = 64, h: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (value,) * 3
    return img


def _bgra(w: int = 32, h: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = (100, 150, 200)  # B, G, R
    img[:, :, 3] = 255
    return img


def _split_bw(w: int = 64, h: int = 64) -> np.ndarray:
    """Half black, half white. Useful for the dark-vs-bright grain test."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, :] = 0  # black
    img[:, w // 2 :, :] = 255  # white
    return img


@pytest.fixture
def gray() -> np.ndarray:
    return _bgr(128, 64, 64)


@pytest.fixture
def bgr_image() -> np.ndarray:
    return _bgr(180, 64, 64)


@pytest.fixture
def bgra_image() -> np.ndarray:
    return _bgra(32, 32)


@pytest.fixture
def split_bw() -> np.ndarray:
    return _split_bw(64, 64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grain_intensity_zero_passthrough(gray):
    """intensity=0 must return an unchanged image and a zero debug plane.

    No grain is added at intensity=0, so the debug visualization is a 2-D
    zero array (nothing to show). The shape matches the input's spatial dims.
    """
    grainy, grain_only = apply_grain(gray, intensity=0.0)
    assert np.array_equal(grainy, gray)
    assert grain_only.shape == gray.shape[:2]
    assert grain_only.dtype == np.uint8
    assert np.all(grain_only == 0)


def test_grain_intensity_zero_with_seed(gray):
    """intensity=0 must short-circuit before touching the RNG."""
    grainy, _ = apply_grain(gray, intensity=0.0, seed=42)
    assert np.array_equal(grainy, gray)


def test_grain_output_shape_matches_input(bgr_image):
    grainy, grain_only = apply_grain(bgr_image, intensity=0.5, seed=1)
    assert grainy.shape == bgr_image.shape
    assert grain_only.shape[:2] == bgr_image.shape[:2]


def test_grain_dtype_uint8(bgr_image):
    grainy, grain_only = apply_grain(bgr_image, intensity=0.5, seed=2)
    assert grainy.dtype == np.uint8
    assert grain_only.dtype == np.uint8


def test_grain_handles_bgra(bgra_image):
    """4-channel input must come back 4-channel; alpha must be preserved."""
    grainy, _ = apply_grain(bgra_image, intensity=0.5, seed=3)
    assert grainy.shape == bgra_image.shape
    assert grainy.shape[2] == 4
    # Alpha channel must be unchanged (255 across the board).
    assert np.all(grainy[:, :, 3] == bgra_image[:, :, 3])


def test_grain_handles_grayscale():
    """2D (grayscale) input must come back 2D."""
    img = np.full((40, 50), 100, dtype=np.uint8)
    grainy, grain_only = apply_grain(img, intensity=0.5, seed=4)
    assert grainy.ndim == 2
    assert grainy.shape == img.shape
    assert grain_only.ndim == 2
    assert grain_only.shape == img.shape


def test_grain_only_image_is_same_shape(bgr_image):
    """(grainy, grain_only) both share H, W with the input."""
    grainy, grain_only = apply_grain(bgr_image, intensity=0.5, seed=5)
    assert grainy.shape[:2] == bgr_image.shape[:2]
    assert grain_only.shape[:2] == bgr_image.shape[:2]


def test_grain_intensity_1_actually_adds_noise(bgr_image):
    """intensity=1.0 should produce a measurably different image."""
    grainy, _ = apply_grain(bgr_image, intensity=1.0, seed=6)
    assert not np.array_equal(grainy, bgr_image)
    # Mean abs delta should be non-trivial.
    diff = np.abs(grainy.astype(np.int16) - bgr_image.astype(np.int16))
    assert diff.mean() > 0.1


def test_grain_dark_pixels_get_more_grain(split_bw):
    """A black region should accumulate more grain than a white one."""
    grainy, _ = apply_grain(split_bw, intensity=0.9, seed=7)
    # Measure per-pixel std of the deviation in each half.
    diff = grainy.astype(np.int16) - split_bw.astype(np.int16)
    w = split_bw.shape[1]
    dark_std = diff[:, : w // 2, 0].std()
    bright_std = diff[:, w // 2 :, 0].std()
    assert dark_std > bright_std, (
        f"Expected dark region to have more grain. "
        f"dark_std={dark_std:.3f} bright_std={bright_std:.3f}"
    )


def test_grain_deterministic_with_seed(bgr_image):
    """Seeding must produce identical results on repeat calls."""
    np.random.seed(42)  # not used — purely to be sure seed arg wins
    a, a_dbg = apply_grain(bgr_image, intensity=0.6, seed=42)
    b, b_dbg = apply_grain(bgr_image, intensity=0.6, seed=42)
    assert np.array_equal(a, b)
    assert np.array_equal(a_dbg, b_dbg)


def test_grain_clipping(bgr_image):
    """Output values must stay within [0, 255]."""
    grainy, grain_only = apply_grain(bgr_image, intensity=1.0, seed=8)
    assert grainy.min() >= 0
    assert grainy.max() <= 255
    assert grain_only.min() >= 0
    assert grain_only.max() <= 255


def test_grain_handles_uint8_clamp():
    """Even with extreme intensity, the output must not overflow."""
    img = _bgr(0, 32, 32)  # all black — worst-case for positive noise
    grainy, _ = apply_grain(img, intensity=2.5, seed=9)
    assert grainy.max() <= 255
    assert grainy.min() >= 0
    # All-black image with max intensity should still produce some change.
    assert not np.array_equal(grainy, img)


def test_grain_negative_intensity_passthrough(bgr_image):
    """A negative intensity should be treated like zero (no-op)."""
    grainy, _ = apply_grain(bgr_image, intensity=-0.5, seed=10)
    assert np.array_equal(grainy, bgr_image)
