"""Tests for app.pipeline.comparison — before/after side-by-side and diff overlay (TDD)."""
from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.comparison import (
    DEFAULT_DIVIDER_COLOR,
    DEFAULT_DIVIDER_WIDTH,
    diff_overlay,
    side_by_side,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _solid_bgr(w: int, h: int, value: tuple[int, int, int] = (10, 20, 30)) -> np.ndarray:
    """Solid-color BGR image (deterministic)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = value
    return img


def _solid_bgra(w: int, h: int, value: tuple[int, int, int, int] = (10, 20, 30, 255)) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = value
    return img


# ---------------------------------------------------------------------------
# side_by_side
# ---------------------------------------------------------------------------


def test_side_by_side_width_is_sum():
    before = _solid_bgr(200, 100, (0, 0, 0))
    after = _solid_bgr(300, 100, (255, 255, 255))
    out = side_by_side(before, after)
    assert out.shape[1] == 200 + DEFAULT_DIVIDER_WIDTH + 300


def test_side_by_side_height_matches():
    before = _solid_bgr(200, 100)
    after = _solid_bgr(300, 100)
    out = side_by_side(before, after)
    assert out.shape[0] == 100


def test_side_by_side_raises_on_height_mismatch():
    before = _solid_bgr(200, 100)
    after = _solid_bgr(300, 120)  # different height
    with pytest.raises(ValueError):
        side_by_side(before, after)


def test_side_by_side_default_divider_color():
    """The default divider is green (0, 255, 0) in BGR."""
    assert DEFAULT_DIVIDER_COLOR == (0, 255, 0)
    before = _solid_bgr(50, 40, (0, 0, 0))
    after = _solid_bgr(50, 40, (0, 0, 0))
    out = side_by_side(before, after)
    # Divider strip starts at column 50 and is 3 wide
    divider = out[:, 50:50 + DEFAULT_DIVIDER_WIDTH, :]
    # Every pixel in the divider must be green
    expected = np.array(DEFAULT_DIVIDER_COLOR, dtype=np.uint8)
    assert np.all(divider == expected), f"divider not green: {divider[0, 0]}"


def test_side_by_side_custom_divider_color():
    before = _solid_bgr(50, 40, (0, 0, 0))
    after = _solid_bgr(50, 40, (0, 0, 0))
    out = side_by_side(before, after, divider_color=(255, 0, 0))
    divider = out[:, 50:50 + DEFAULT_DIVIDER_WIDTH, :]
    assert np.all(divider[:, :, 0] == 255)  # B
    assert np.all(divider[:, :, 1] == 0)
    assert np.all(divider[:, :, 2] == 0)


def test_side_by_side_preserves_dtype():
    before = _solid_bgr(40, 30, (10, 20, 30))
    after = _solid_bgr(50, 30, (40, 50, 60))
    out = side_by_side(before, after)
    assert out.dtype == np.uint8


def test_side_by_side_handles_bgra():
    before = _solid_bgra(40, 30, (10, 20, 30, 200))
    after = _solid_bgra(50, 30, (40, 50, 60, 200))
    out = side_by_side(before, after)
    # Output must preserve 4 channels when given BGRA
    assert out.ndim == 3
    assert out.shape[2] == 4
    # And the alpha of the divider region must be fully opaque
    divider = out[:, 40:40 + DEFAULT_DIVIDER_WIDTH, :]
    assert np.all(divider[:, :, 3] == 255)


def test_side_by_side_bgr_with_bgra_falls_back_to_bgr():
    """If one is BGR and the other is BGRA, the output uses 3 channels (drop alpha)."""
    before = _solid_bgr(40, 30, (10, 20, 30))
    after = _solid_bgra(50, 30, (40, 50, 60, 200))
    out = side_by_side(before, after)
    assert out.shape[2] == 3


def test_side_by_side_left_region_matches_before():
    before = _solid_bgr(40, 30, (10, 20, 30))
    after = _solid_bgr(50, 30, (40, 50, 60))
    out = side_by_side(before, after)
    assert np.array_equal(out[:, :40, :], before)


def test_side_by_side_right_region_matches_after():
    before = _solid_bgr(40, 30, (10, 20, 30))
    after = _solid_bgr(50, 30, (40, 50, 60))
    out = side_by_side(before, after)
    assert np.array_equal(out[:, 40 + DEFAULT_DIVIDER_WIDTH:, :], after)


# ---------------------------------------------------------------------------
# diff_overlay
# ---------------------------------------------------------------------------


def test_diff_is_zero_for_identical():
    a = _solid_bgr(20, 15, (123, 45, 67))
    out = diff_overlay(a, a)
    assert out.shape == a.shape
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_diff_is_nonzero_for_different():
    a = _solid_bgr(20, 15, (10, 20, 30))
    b = _solid_bgr(20, 15, (200, 100, 50))
    out = diff_overlay(a, b)
    assert np.any(out != 0)


def test_diff_default_gain():
    """Default gain (5.0) should brighten a subtle diff beyond raw absdiff."""
    a = _solid_bgr(20, 15, (100, 100, 100))
    b = a.copy()
    b[0, 0] = (101, 101, 101)  # 1-unit diff
    raw = np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)
    enhanced = diff_overlay(a, b)
    # Enhanced (gain 5.0) should be >= raw, and strictly > at the diff pixel
    assert enhanced[0, 0, 0] > raw[0, 0, 0]


def test_diff_preserves_shape():
    a = _solid_bgr(33, 17, (0, 0, 0))
    b = _solid_bgr(33, 17, (255, 255, 255))
    out = diff_overlay(a, b)
    assert out.shape == a.shape


def test_diff_handles_bgra():
    a = _solid_bgra(20, 15, (10, 20, 30, 200))
    b = _solid_bgra(20, 15, (50, 60, 70, 100))
    out = diff_overlay(a, b)
    # 4-channel preserved
    assert out.shape[2] == 4
    assert out.dtype == np.uint8


def test_diff_dtype_uint8():
    a = _solid_bgr(8, 8, (10, 20, 30))
    b = _solid_bgr(8, 8, (40, 50, 60))
    out = diff_overlay(a, b)
    assert out.dtype == np.uint8


def test_diff_raises_on_shape_mismatch():
    a = _solid_bgr(10, 10, (0, 0, 0))
    b = _solid_bgr(12, 10, (0, 0, 0))
    with pytest.raises(ValueError):
        diff_overlay(a, b)


def test_diff_clips_to_255():
    """Even at gain 5.0, the output must not exceed 255 anywhere."""
    a = np.zeros((5, 5, 3), dtype=np.uint8)
    b = np.full((5, 5, 3), 255, dtype=np.uint8)
    out = diff_overlay(a, b, gain=5.0)
    assert out.max() == 255
