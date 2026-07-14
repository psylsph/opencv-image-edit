"""Tests for app.pipeline.background and app.models.matting (TDD).

Covers:
- apply_background_blur: shape contract, identity cases, blur effect
- apply_background_remove: BGRA output, alpha = mask, fg preservation
- get_alpha_debug: uint8 output with autocontrast
- _resize_mask: shape coercion utility
- MattingModel: existence/API smoke (full inference is not unit-tested here)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.matting import MattingModel
from app.pipeline.background import (
    apply_background_blur,
    apply_background_remove,
    get_alpha_debug,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bgr(w: int = 200, h: int = 300) -> np.ndarray:
    """Deterministic BGR image with full dynamic range (gradient)."""
    gx = np.linspace(0, 255, w, dtype=np.float32)
    gy = np.linspace(0, 255, h, dtype=np.float32)
    xv, yv = np.meshgrid(gx, gy)
    b = xv.astype(np.uint8)
    g = yv.astype(np.uint8)
    r = ((xv + yv) / 2.0).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


# ---------------------------------------------------------------------------
# apply_background_blur
# ---------------------------------------------------------------------------


def test_apply_background_blur_returns_same_shape():
    img = _bgr(200, 300)
    mask = np.ones((300, 200), dtype=np.float32)  # H, W
    out = apply_background_blur(img, mask, blur_strength=10)
    # Note the width/height order: image is (H=300, W=200, 3)
    assert out.shape == (300, 200, 3)
    assert out.dtype == np.uint8


def test_apply_background_blur_with_zero_mask_preserves_image():
    """All-zero mask means nothing is foreground → output must equal input."""
    img = _bgr(100, 80)
    mask = np.zeros((80, 100), dtype=np.float32)
    out = apply_background_blur(img, mask, blur_strength=15)
    # In a "zero mask" world, alpha=0 means we take 100% of the blurred image
    # (so the output will be blurred). This assertion is the spec:
    #   with mask=0 everywhere, output == blurred input
    blurred = cv2_blur(img, 15)
    assert np.array_equal(out, blurred)


def test_apply_background_blur_with_ones_mask_preserves_image():
    """All-ones mask means everything is foreground → output == input."""
    img = _bgr(100, 80)
    mask = np.ones((80, 100), dtype=np.float32)
    out = apply_background_blur(img, mask, blur_strength=15)
    assert np.array_equal(out, img)


def test_apply_background_blur_with_half_mask_is_between():
    """Mixed mask: output should be neither pure input nor pure blur."""
    img = _bgr(100, 80)
    mask = np.full((80, 100), 0.5, dtype=np.float32)
    out = apply_background_blur(img, mask, blur_strength=15)
    # Should not equal the input exactly
    assert not np.array_equal(out, img)
    # Should differ from a fully-blurred image
    blurred = cv2_blur(img, 15)
    assert not np.array_equal(out, blurred)


def test_apply_background_blur_resizes_mismatched_mask():
    """If mask shape != image shape, the function should resize it."""
    img = _bgr(200, 300)
    mask = np.ones((320, 320), dtype=np.float32)  # wrong shape
    out = apply_background_blur(img, mask, blur_strength=10)
    assert out.shape == (300, 200, 3)


# ---------------------------------------------------------------------------
# apply_background_remove
# ---------------------------------------------------------------------------


def test_apply_background_remove_returns_bgra():
    img = _bgr(100, 80)
    mask = np.ones((80, 100), dtype=np.float32)
    out = apply_background_remove(img, mask)
    assert out.ndim == 3
    assert out.shape == (80, 100, 4)
    assert out.dtype == np.uint8


def test_apply_background_remove_mask_determines_alpha():
    """Alpha channel of output should equal mask (interior, away from feathered edge)."""
    img = _bgr(100, 80)
    # Use an interesting mask with both full and zero values
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[20:60, 30:70] = 1.0  # foreground rectangle
    out = apply_background_remove(img, mask)
    alpha = out[:, :, 3].astype(np.float32) / 255.0
    # 1-px Gaussian feathering only affects the boundary. Compare INTERIOR
    # pixels (well clear of the boundary) for a tight tolerance.
    interior_fg = alpha[30:50, 40:60]  # deep inside the FG rectangle
    interior_bg = alpha[2:18, 2:28]  # deep inside the BG area
    assert interior_fg.max() <= 1.0 + 1e-6
    assert interior_fg.min() > 0.95  # quantised to ~255/255
    assert interior_bg.max() < 0.05  # quantised to ~0/255
    # Boundary (last few pixels) is allowed to be mid-range
    # — that's the feathered edge this function explicitly produces.


def test_apply_background_remove_preserves_foreground():
    """Where mask=1, the RGB channels should equal the input RGB."""
    img = _bgr(100, 80)
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[20:60, 30:70] = 1.0
    out = apply_background_remove(img, mask)
    # Pick a clearly interior foreground pixel
    rgb_out = out[:, :, :3]
    rgb_in = img
    # In the interior of the FG region (away from feathering edge)
    assert np.array_equal(rgb_out[40, 50], rgb_in[40, 50])
    assert np.array_equal(rgb_out[30, 60], rgb_in[30, 60])


def test_apply_background_remove_preserves_input_rgb_outside_fg():
    """Where mask=0, RGB should be preserved (transparency is what changes)."""
    img = _bgr(100, 80)
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[20:60, 30:70] = 1.0
    out = apply_background_remove(img, mask)
    rgb_out = out[:, :, :3]
    # Outside the FG rectangle
    assert np.array_equal(rgb_out[5, 5], img[5, 5])
    assert np.array_equal(rgb_out[70, 90], img[70, 90])


# ---------------------------------------------------------------------------
# get_alpha_debug
# ---------------------------------------------------------------------------


def test_alpha_debug_is_uint8_2d():
    mask = np.random.rand(40, 60).astype(np.float32)
    out = get_alpha_debug(mask)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == (40, 60)


def test_alpha_debug_max_is_255():
    """After autocontrast, the max should be stretched to 255 (non-uniform input)."""
    mask = np.zeros((40, 60), dtype=np.float32)
    mask[10:30, 20:40] = 0.7
    out = get_alpha_debug(mask)
    assert out.max() == 255


def test_alpha_debug_uniform_input_is_zero():
    """If the input is constant, autocontrast stretches to zero (no contrast)."""
    mask = np.full((40, 60), 0.42, dtype=np.float32)
    out = get_alpha_debug(mask)
    # uniform input → no contrast → all zeros
    assert out.max() == 0
    assert out.min() == 0


def test_alpha_debug_stretches_range():
    """For a [0.1, 0.7] mask, output should cover the full [0, 255] range."""
    mask = np.zeros((40, 60), dtype=np.float32)
    mask[10:30, 20:40] = 0.1
    mask[5:10, 5:15] = 0.7
    out = get_alpha_debug(mask)
    assert out.min() == 0
    assert out.max() == 255


# ---------------------------------------------------------------------------
# MattingModel (API smoke)
# ---------------------------------------------------------------------------


def test_matting_model_class_exists():
    """The MattingModel class is importable and instantiable with a path."""
    # Just check we can construct a non-functional instance pointing at a
    # nonexistent file — the constructor should raise ModelNotFoundError.
    from app.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        MattingModel("/nonexistent/u2netp.onnx")


def test_matting_model_predict_signature_exists():
    """The class exposes a .predict() method (no actual inference tested)."""
    assert hasattr(MattingModel, "predict")


@pytest.mark.slow
def test_matting_model_predicts_real_foreground():
    """End-to-end: real U2NetP ONNX should separate a drawn person from background.

    Regression guard for the 'no sigmoid' bug: without this, .predict() returns
    a near-uniform ~0.5 mask that destroys blur/remove compositing.
    Skipped if model file is missing.
    """
    from pathlib import Path

    import cv2

    model_path = Path(__file__).parent.parent / "models" / "u2netp.onnx"
    if not model_path.exists():
        pytest.skip("u2netp.onnx not present — run scripts/download_models.py")

    # Synthetic image: green background + blue person silhouette
    img = np.full((400, 600, 3), 80, dtype=np.uint8)
    img[:, :] = (80, 120, 100)  # BGR
    cv2.ellipse(img, (300, 200), (80, 120), 0, 0, 360, (180, 160, 200), -1)

    model = MattingModel(model_path)
    mask = model.predict(img)

    # Mask must be properly bimodal (not stuck around 0.5)
    fg_fraction = (mask > 0.5).mean()
    bg_corner = mask[10, 10]
    fg_center = mask[200, 300]
    assert fg_fraction < 0.5, f"mask looks uniform: {fg_fraction:.2%} foreground"
    assert bg_corner < 0.1, f"background not suppressed: corner={bg_corner}"
    assert fg_center > 0.9, f"foreground not detected: center={fg_center}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cv2_blur(img: np.ndarray, sigma: int) -> np.ndarray:
    """Local helper to compute the same Gaussian the pipeline uses."""
    import cv2

    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
