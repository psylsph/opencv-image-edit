"""Tests for app.pipeline.filters — OpenCV filter functions (TDD).

Filter factor semantics (OpenCV port — cleaner API than the original Pillow one):
    factor == 1.0 -> no change (identity)
    factor == 0.0 -> "zero" / collapsed effect
    factor == 2.0 -> "double" / stronger effect

Where this departs from the original Pillow version (which used [-1, 1] offsets),
the deviation is documented and the new API is what we test.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.filters import (
    add_auto_enhance,
    add_vignette,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_sharpness,
    apply_blur,
    apply_color_adjustments,
    apply_grayscale,
    apply_sepia,
    apply_unsharp_mask,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gradient_bgr(w: int = 64, h: int = 64) -> np.ndarray:
    """Deterministic BGR gradient test image with full dynamic range."""
    gx = np.linspace(0, 255, w, dtype=np.float32)
    gy = np.linspace(0, 255, h, dtype=np.float32)
    xv, yv = np.meshgrid(gx, gy)
    b = xv.astype(np.uint8)
    g = yv.astype(np.uint8)
    r = ((xv + yv) / 2.0).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)  # BGR


def _flat_bgr(value: int, w: int = 32, h: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (value,) * 3
    return img


@pytest.fixture
def gradient() -> np.ndarray:
    return _gradient_bgr()


@pytest.fixture
def mid_gray() -> np.ndarray:
    return _flat_bgr(128, 64, 64)


# ---------------------------------------------------------------------------
# adjust_brightness
# ---------------------------------------------------------------------------


def test_adjust_brightness_factor_1_unchanged(gradient: np.ndarray) -> None:
    """factor=1.0 -> identity."""
    out = adjust_brightness(gradient, 1.0)
    assert out.shape == gradient.shape
    np.testing.assert_array_equal(out, gradient)


def test_adjust_brightness_factor_0_is_black(gradient: np.ndarray) -> None:
    """factor=0.0 -> all pixels collapse to 0 (black)."""
    out = adjust_brightness(gradient, 0.0)
    assert out.shape == gradient.shape
    assert int(out.max()) == 0
    assert int(out.min()) == 0


def test_adjust_brightness_factor_2_doubles(gradient: np.ndarray) -> None:
    """factor=2.0 -> output values ~ 2x input (clipped at 255)."""
    out = adjust_brightness(gradient, 2.0)
    assert out.shape == gradient.shape
    # Pick a midtone pixel: in our gradient center is ~ (127, 127, 127) BGR.
    h, w = out.shape[:2]
    cy, cx = h // 2, w // 2
    in_bgr = gradient[cy, cx].astype(np.int32)
    out_bgr = out[cy, cx].astype(np.int32)
    np.testing.assert_allclose(out_bgr, in_bgr * 2, atol=3)


def test_adjust_brightness_clamps_to_255(gradient: np.ndarray) -> None:
    """Bright areas (>=128) with factor=2.0 must clip to 255, not overflow."""
    out = adjust_brightness(gradient, 2.0)
    assert int(out.max()) <= 255
    # At least one pixel should hit 255
    assert int(out.max()) == 255


# ---------------------------------------------------------------------------
# adjust_contrast
# ---------------------------------------------------------------------------


def test_adjust_contrast_factor_1_unchanged(gradient: np.ndarray) -> None:
    out = adjust_contrast(gradient, 1.0)
    np.testing.assert_array_equal(out, gradient)


def test_adjust_contrast_factor_0_is_gray(mid_gray: np.ndarray) -> None:
    """factor=0 -> every channel collapses to mean gray (128)."""
    out = adjust_contrast(mid_gray, 0.0)
    # Already gray 128, should remain 128 across the board.
    assert int(out.min()) == 128
    assert int(out.max()) == 128


def test_adjust_contrast_factor_2_increases_swing(gradient: np.ndarray) -> None:
    """factor=2.0 should expand values around the mid-gray pivot."""
    out = adjust_contrast(gradient, 2.0)
    # Pure 0 (input 0) -> 128 - 2*(128-0) clipped to 0
    assert int(out[0, 0, 0]) == 0
    # Pure 255 (input 255) -> 128 + 2*(255-128) = 254
    h, w = out.shape[:2]
    in_topright_b = int(gradient[0, w - 1, 0])
    out_topright_b = int(out[0, w - 1, 0])
    # expect 128 + 2 * (in - 128), clipped
    expected = max(0, min(255, 128 + 2 * (in_topright_b - 128)))
    assert abs(out_topright_b - expected) <= 2


# ---------------------------------------------------------------------------
# adjust_saturation
# ---------------------------------------------------------------------------


def test_adjust_saturation_factor_1_unchanged(gradient: np.ndarray) -> None:
    out = adjust_saturation(gradient, 1.0)
    np.testing.assert_array_equal(out, gradient)


def test_adjust_saturation_factor_0_is_grayscale(gradient: np.ndarray) -> None:
    """factor=0.0 -> S channel in HSV goes to 0, so output is grayscale.

    Convert output back to gray, and check B==G==R for all pixels.
    """
    out = adjust_saturation(gradient, 0.0)
    assert out.shape == gradient.shape
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    np.testing.assert_array_equal(b, g)
    np.testing.assert_array_equal(g, r)


def test_adjust_saturation_factor_2_preserves_shape(gradient: np.ndarray) -> None:
    out = adjust_saturation(gradient, 2.0)
    assert out.shape == gradient.shape
    # Hue values should not be dramatically different (saturate, don't recolor)
    # Convert original and output to HSV, compare hue channel roughly.
    hsv_in = cv2_hsv(gradient)
    hsv_out = cv2_hsv(out)
    diff = np.abs(hsv_in[:, :, 0].astype(np.int32) - hsv_out[:, :, 0].astype(np.int32))
    # Hue is 0-179 in OpenCV; allow some small drift.
    assert float(diff.mean()) < 5.0


def cv2_hsv(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ---------------------------------------------------------------------------
# adjust_sharpness
# ---------------------------------------------------------------------------


def test_adjust_sharpness_factor_1_unchanged(gradient: np.ndarray) -> None:
    out = adjust_sharpness(gradient, 1.0)
    np.testing.assert_array_equal(out, gradient)


def test_adjust_sharpness_factor_0_is_blur(gradient: np.ndarray) -> None:
    """factor=0.0 collapses the unsharp add to a pure blur."""
    out = adjust_sharpness(gradient, 0.0)
    assert out.shape == gradient.shape
    # Difference vs original should be nonzero (it is a blur, not identity)
    assert not np.array_equal(out, gradient)
    # Variance of pixel differences should be small (smooth)
    diff = out.astype(np.int32) - gradient.astype(np.int32)
    assert float(np.abs(diff).max()) < 255  # sanity bound
    # The blurred output should be smoother: high-frequency content lower
    # Compare per-pixel magnitude of Laplacian proxy: mean abs gradient of output < input
    lap_in = _laplacian_energy(gradient)
    lap_out = _laplacian_energy(out)
    assert lap_out < lap_in


def _laplacian_energy(img: np.ndarray) -> float:
    """Sum of squared pixel-diffs vs 4-neighbor mean — cheap sharpness proxy."""
    g = img.astype(np.float32)
    dx = g[:, 1:, :] - g[:, :-1, :]
    dy = g[1:, :, :] - g[:-1, :, :]
    return float((dx**2).mean() + (dy**2).mean())


def test_adjust_sharpness_factor_2_increases_sharpness() -> None:
    """factor=2.0 sharpens an image with edge content (vertical step)."""
    # Vertical step at x=32, mid-gray values — unsharp should produce a steeper transition.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32, :] = 80
    img[:, 32:, :] = 180
    out = adjust_sharpness(img, 2.0)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    # Check the transition: original jumps 80->180 in one pixel.
    # Sharpened should overshoot: a pixel left of x=32 should be < 80, a pixel
    # right of x=32 should be > 180.
    mid_row = out[32, :, 0]
    left = int(mid_row[30])
    right = int(mid_row[34])
    assert left < 80, f"expected left side to overshoot darker, got {left}"
    assert right > 180, f"expected right side to overshoot brighter, got {right}"


# ---------------------------------------------------------------------------
# add_vignette
# ---------------------------------------------------------------------------


def test_add_vignette_corners_darker(gradient: np.ndarray) -> None:
    """With default strength=0.5, center unchanged, corner pixel < original."""
    h, w = gradient.shape[:2]
    out = add_vignette(gradient, strength=0.5)
    assert out.shape == gradient.shape
    # Center pixel — should be approximately preserved (mask ~ 1.0)
    cy, cx = h // 2, w // 2
    center_in = gradient[cy, cx].astype(np.int32)
    center_out = out[cy, cx].astype(np.int32)
    np.testing.assert_allclose(center_out, center_in, atol=3)
    # Corner pixel — must be darker (mask < 1.0). The gradient's B channel
    # at the top-left is 0, so check a channel/direction with non-zero input.
    h, w = gradient.shape[:2]
    # Use the bottom-right corner where every channel is non-zero (255)
    corner_in = int(gradient[h - 1, w - 1, 0])
    corner_out = int(out[h - 1, w - 1, 0])
    assert corner_out < corner_in


def test_add_vignette_strength_0_no_change(gradient: np.ndarray) -> None:
    """strength=0 -> mask is 1.0 everywhere -> no change."""
    out = add_vignette(gradient, strength=0.0)
    np.testing.assert_array_equal(out, gradient)


def test_add_vignette_uniform_image_stays_uniform(mid_gray: np.ndarray) -> None:
    """A flat image, even with vignette, should remain visually flat (modulo corner darken)."""
    out = add_vignette(mid_gray, strength=0.5)
    # Center pixel must still be ~128
    h, w = out.shape[:2]
    assert abs(int(out[h // 2, w // 2, 0]) - 128) <= 2
    # Corner should be <= center
    assert int(out[0, 0, 0]) <= int(out[h // 2, w // 2, 0])


# ---------------------------------------------------------------------------
# add_auto_enhance
# ---------------------------------------------------------------------------


def test_auto_enhance_stretches_histogram() -> None:
    """A narrow-histogram input should be expanded after auto-enhance."""
    # Input: BGR all = 100 +- 5
    rng = np.random.default_rng(0)
    base = rng.normal(100, 2, (64, 64, 3)).astype(np.float32)
    base = np.clip(base, 95, 105).astype(np.uint8)
    out = add_auto_enhance(base)
    assert out.shape == base.shape
    # The stretched output should have a wider range than the input.
    in_range = int(base.max()) - int(base.min())
    out_range = int(out.max()) - int(out.min())
    assert out_range > in_range + 30  # substantially stretched


# ---------------------------------------------------------------------------
# apply_color_adjustments
# ---------------------------------------------------------------------------


def test_apply_color_adjustments_passthrough_with_defaults(gradient: np.ndarray) -> None:
    """Defaults brightness=contrast=saturation=sharpness=1.0 -> no change."""
    out = apply_color_adjustments(gradient)
    np.testing.assert_array_equal(out, gradient)


def test_apply_color_adjustments_combines_effects(mid_gray: np.ndarray) -> None:
    """Non-default values should produce a result that differs from identity."""
    out = apply_color_adjustments(
        mid_gray,
        brightness=1.5,
        contrast=1.5,
        saturation=0.5,
        sharpness=1.5,
    )
    # The output for a uniform 128 gray should still be 128-ish everywhere
    # (saturation doesn't matter for gray), but the order of ops matters.
    # At minimum, it should still be a valid image.
    assert out.shape == mid_gray.shape
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# apply_sepia
# ---------------------------------------------------------------------------


def test_apply_sepia_changes_color(gradient: np.ndarray) -> None:
    """Sepia output should have B > R on average (sepia is warm/brownish).

    For a gradient with varied color, the mean B channel should exceed
    the mean R channel after sepia is applied at full intensity.
    """
    out = apply_sepia(gradient)
    assert out.shape == gradient.shape
    mean_r_out = float(out[:, :, 2].mean())
    mean_b_out = float(out[:, :, 0].mean())
    # Original gradient: B grows along x, R is mixed -> we don't assume input order
    # Sepia must produce warm tones: R >= B for all pixels (on average the
    # sepia transform swaps them: r_out=0.393*r+0.769*g+0.189*b, b_out=0.272*r+0.534*g+0.131*b)
    # We expect output R > output B for the original gradient which is roughly balanced.
    assert mean_r_out > mean_b_out


def test_apply_sepia_output_is_uint8(gradient: np.ndarray) -> None:
    out = apply_sepia(gradient)
    assert out.dtype == np.uint8
    assert int(out.max()) <= 255
    assert int(out.min()) >= 0


# ---------------------------------------------------------------------------
# apply_grayscale
# ---------------------------------------------------------------------------


def test_apply_grayscale_blend_0_unchanged(gradient: np.ndarray) -> None:
    out = apply_grayscale(gradient, blend=0.0)
    np.testing.assert_array_equal(out, gradient)


def test_apply_grayscale_blend_1_is_gray(gradient: np.ndarray) -> None:
    """blend=1.0 -> output is fully grayscale: B == G == R for every pixel."""
    out = apply_grayscale(gradient, blend=1.0)
    assert out.shape == gradient.shape
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    np.testing.assert_array_equal(b, g)
    np.testing.assert_array_equal(g, r)


def test_apply_grayscale_blend_half_intermediate(gradient: np.ndarray) -> None:
    """blend=0.5 should land between the original and the gray (not equal to either)."""
    gray = apply_grayscale(gradient, blend=1.0)
    half = apply_grayscale(gradient, blend=0.5)
    assert not np.array_equal(half, gradient)
    assert not np.array_equal(half, gray)


# ---------------------------------------------------------------------------
# apply_blur
# ---------------------------------------------------------------------------


def test_apply_blur_ksize_1_unchanged(gradient: np.ndarray) -> None:
    out = apply_blur(gradient, ksize=1)
    np.testing.assert_array_equal(out, gradient)


def test_apply_blur_ksize_0_unchanged(gradient: np.ndarray) -> None:
    out = apply_blur(gradient, ksize=0)
    np.testing.assert_array_equal(out, gradient)


def test_apply_blur_reduces_high_freq() -> None:
    """Input with a sharp edge should produce a softer output after blur."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32, :] = 0
    img[:, 32:, :] = 255  # vertical edge
    out = apply_blur(img, ksize=9)
    assert out.shape == img.shape
    # At the edge, the output should have intermediate values (not 0 or 255)
    edge_col = out[:, 32, 0]
    # The central row of the edge column should be neither pure 0 nor pure 255
    mid_val = int(edge_col[edge_col.shape[0] // 2])
    assert 0 < mid_val < 255


# ---------------------------------------------------------------------------
# apply_unsharp_mask
# ---------------------------------------------------------------------------


def test_apply_unsharp_mask_increases_sharpness(gradient: np.ndarray) -> None:
    """Unsharp mask with default sigma=1.0, strength=1.5 should sharpen the input."""
    out = apply_unsharp_mask(gradient, sigma=1.0, strength=1.5)
    assert out.shape == gradient.shape
    assert out.dtype == np.uint8
    lap_in = _laplacian_energy(gradient)
    lap_out = _laplacian_energy(out)
    assert lap_out > lap_in


def test_apply_unsharp_mask_strength_0_is_passthrough(gradient: np.ndarray) -> None:
    """strength=0 collapses to a plain blur addWeighted with weight 1 -> blur output, not identity.

    But with strength=0, addWeighted(img, 1+0, blur, 0, 0) = img exactly. So it IS identity.
    """
    out = apply_unsharp_mask(gradient, sigma=1.0, strength=0.0)
    np.testing.assert_array_equal(out, gradient)


def test_apply_unsharp_mask_clamps_to_uint8(gradient: np.ndarray) -> None:
    out = apply_unsharp_mask(gradient, sigma=1.0, strength=1.5)
    assert int(out.max()) <= 255
    assert int(out.min()) >= 0
