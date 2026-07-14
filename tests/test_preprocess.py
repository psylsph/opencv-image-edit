"""Tests for app.pipeline.preprocess — image resize helpers (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.preprocess import get_image_dimensions, resize_if_needed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uint8() -> type[np.uint8]:
    return np.uint8


# ---------------------------------------------------------------------------
# resize_if_needed — passthrough (no resize needed)
# ---------------------------------------------------------------------------


def test_resize_passthrough_when_small() -> None:
    """Input 800x600, max_dim=1536 — output shape is (600, 800, 3).

    Largest dim (800) <= 1536, so no resizing. Returned as-is.
    """
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.shape == (600, 800, 3)
    # Should be the same data (no copy required, but value equality must hold)
    np.testing.assert_array_equal(out, img)


def test_resize_does_not_upscale() -> None:
    """Input 500x400, max_dim=1000 — passthrough (500 < 1000, 400 < 1000).

    The function must NEVER upscale. Even though 1000 is "allowed",
    500 and 400 are both below that ceiling, so output is unchanged.
    """
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1000)
    assert out.shape == (400, 500, 3)


def test_resize_exact_boundary_passthrough() -> None:
    """When longest == max_dim exactly, no resize should occur."""
    img = np.zeros((1000, 1536, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.shape == (1000, 1536, 3)


# ---------------------------------------------------------------------------
# resize_if_needed — downscale
# ---------------------------------------------------------------------------


def test_resize_downscales_oversized_landscape() -> None:
    """Input 3000x2000, max_dim=1536 → (1024, 1536, 3).

    Aspect preserved: 3000 * (1536/3000) = 1536, 2000 * (1536/3000) = 1024.
    """
    img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.shape == (1024, 1536, 3), f"expected (1024, 1536, 3), got {out.shape}"


def test_resize_downscales_oversized_portrait() -> None:
    """Input 2000x3000, max_dim=1536 → (1536, 1024, 3)."""
    img = np.zeros((3000, 2000, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.shape == (1536, 1024, 3), f"expected (1536, 1024, 3), got {out.shape}"


def test_resize_preserves_aspect_ratio() -> None:
    """Input 4000x3000, max_dim=1000 → (750, 1000, 3).

    4000 * (1000/4000) = 1000, 3000 * (1000/4000) = 750.
    """
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1000)
    assert out.shape == (750, 1000, 3), f"expected (750, 1000, 3), got {out.shape}"


# ---------------------------------------------------------------------------
# resize_if_needed — multi-channel handling
# ---------------------------------------------------------------------------


def test_resize_handles_bgra() -> None:
    """4-channel BGRA input → 4-channel output, same channel count."""
    img = np.zeros((600, 800, 4), dtype=np.uint8)
    # Fill alpha so we can verify it's preserved
    img[..., 3] = 200
    out = resize_if_needed(img, max_dim=100)  # forces downscale
    assert out.ndim == 3
    assert out.shape[2] == 4, f"expected 4 channels, got {out.shape[2]}"


def test_resize_handles_grayscale() -> None:
    """2D grayscale input (H, W) → still 2D output, same shape if no resize."""
    img = np.zeros((400, 600), dtype=np.uint8)  # shape (400, 600)
    out = resize_if_needed(img, max_dim=1000)  # passthrough
    assert out.ndim == 2, f"expected 2D output, got ndim={out.ndim}"
    assert out.shape == (400, 600)


def test_resize_grayscale_downscales_2d() -> None:
    """2D grayscale input — downscale keeps 2D shape."""
    img = np.zeros((800, 1600), dtype=np.uint8)  # longest=1600
    out = resize_if_needed(img, max_dim=400)
    # 1600 * (400/1600) = 400, 800 * (400/1600) = 200
    assert out.ndim == 2
    assert out.shape == (200, 400)


# ---------------------------------------------------------------------------
# resize_if_needed — array properties
# ---------------------------------------------------------------------------


def test_resize_returns_contiguous_array() -> None:
    """Output must be C-contiguous (required by downstream OpenCV calls)."""
    img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.flags["C_CONTIGUOUS"] is True, "output is not C-contiguous"


def test_resize_returns_contiguous_on_passthrough() -> None:
    """Even passthrough branch must return C-contiguous array."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.flags["C_CONTIGUOUS"] is True


def test_resize_dtype_preserved() -> None:
    """Output dtype equals input dtype (uint8 for both)."""
    img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    out = resize_if_needed(img, max_dim=1536)
    assert out.dtype == img.dtype == np.uint8


# ---------------------------------------------------------------------------
# get_image_dimensions
# ---------------------------------------------------------------------------


def test_get_image_dimensions_bgr() -> None:
    """get_image_dimensions returns (H, W) for a 3D BGR image."""
    img = np.zeros((123, 456, 3), dtype=np.uint8)
    h, w = get_image_dimensions(img)
    assert (h, w) == (123, 456)


def test_get_image_dimensions_grayscale() -> None:
    """get_image_dimensions returns (H, W) for a 2D image."""
    img = np.zeros((50, 75), dtype=np.uint8)
    h, w = get_image_dimensions(img)
    assert (h, w) == (50, 75)


# ---------------------------------------------------------------------------
# resize_if_needed — input validation
# ---------------------------------------------------------------------------


def test_resize_rejects_zero_max_dim() -> None:
    """max_dim=0 must raise ValueError."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_if_needed(img, max_dim=0)


def test_resize_rejects_negative_max_dim() -> None:
    """max_dim<0 must raise ValueError."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_if_needed(img, max_dim=-5)
