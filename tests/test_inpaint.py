"""Tests for app.pipeline.inpaint — OpenCV inpainting wrapper (TDD).

Covers:
- shape preservation across BGR / grayscale / BGRA inputs
- "no mask = no change" contract
- both TELEA and NS algorithms
- radius and algorithm validation
- masked region is modified
- unmasked region is preserved exactly
- dtype is uint8
- composability with other pipeline steps (upscale)
"""
from __future__ import annotations

import numpy as np
import pytest

from app.exceptions import ValidationError
from app.pipeline.inpaint import inpaint


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------


def test_inpaint_returns_same_shape_as_input():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    out = inpaint(img, mask, algorithm="telea")
    assert out.shape == (200, 300, 3)


def test_inpaint_dtype_uint8():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    out = inpaint(img, mask, algorithm="telea")
    assert out.dtype == np.uint8


def test_inpaint_handles_grayscale():
    img = np.zeros((80, 80), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    out = inpaint(img, mask, algorithm="telea")
    assert out.ndim == 2
    assert out.shape == (80, 80)
    assert out.dtype == np.uint8


def test_inpaint_handles_bgra():
    img = np.zeros((60, 60, 4), dtype=np.uint8)
    img[:, :, 3] = 200  # non-default alpha to verify preservation
    mask = np.zeros((60, 60), dtype=np.uint8)
    out = inpaint(img, mask, algorithm="telea")
    assert out.shape == (60, 60, 4)
    # Alpha must be preserved unchanged (no inpainting on alpha)
    assert np.array_equal(out[:, :, 3], img[:, :, 3])


# ---------------------------------------------------------------------------
# Identity contract
# ---------------------------------------------------------------------------


def test_inpaint_unchanged_when_mask_is_empty():
    """All-zero mask means: nothing to inpaint -> output == input (pixel-for-pixel)."""
    img = np.random.RandomState(0).randint(0, 256, (120, 140, 3), dtype=np.uint8)
    mask = np.zeros((120, 140), dtype=np.uint8)
    out = inpaint(img, mask, algorithm="telea")
    assert np.array_equal(out, img)


def test_inpaint_preserves_unmasked_region():
    """Pixels OUTSIDE the mask must be bit-identical to the input."""
    img = np.random.RandomState(1).randint(0, 256, (150, 150, 3), dtype=np.uint8)
    # White square in the middle
    img[60:90, 60:90, :] = 255
    mask = np.zeros((150, 150), dtype=np.uint8)
    mask[60:90, 60:90] = 255  # only the square is masked

    out = inpaint(img, mask, radius=5, algorithm="telea")

    # Build a "unmasked" boolean: True where mask == 0
    unmasked = mask == 0
    assert np.array_equal(out[unmasked], img[unmasked])


# ---------------------------------------------------------------------------
# Both algorithms run
# ---------------------------------------------------------------------------


def test_inpaint_telea_runs_without_error():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255  # 20x20 white square in the middle
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    out = inpaint(img, mask, algorithm="telea", radius=3)
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8


def test_inpaint_ns_runs_without_error():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    out = inpaint(img, mask, algorithm="ns", radius=3)
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8


def test_inpaint_modifies_masked_region():
    """The center of the inpainted square must NOT be solid white anymore —
    it should be filled with surrounding (background) colors.
    """
    # Distinct, repeating background pattern so inpainting has something to copy.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # B
    img[:, :, 1] = 100  # G
    img[:, :, 2] = 150  # R
    # White square in the middle
    img[40:60, 40:60, :] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    out = inpaint(img, mask, radius=5, algorithm="telea")

    # The exact center of the originally-white region should NOT be white anymore.
    # (It's surrounded by a 50x100x150 background, so inpainting should pull those in.)
    center = out[50, 50]
    original_center = img[50, 50]
    assert not np.array_equal(center, original_center), (
        f"inpainted center should differ from white input; got {center}"
    )
    # And it should be in a plausible color range (not pure black, not pure white)
    assert center.sum() > 0
    assert center.sum() < 255 * 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_inpaint_radius_too_small_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValidationError):
        inpaint(img, mask, algorithm="telea", radius=0)


def test_inpaint_radius_too_large_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValidationError):
        inpaint(img, mask, algorithm="telea", radius=200)


def test_inpaint_invalid_algorithm_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValidationError):
        inpaint(img, mask, algorithm="foo")


def test_inpaint_lama_default_algorithm(monkeypatch):
    """Default algorithm must be 'lama' (highest quality, on by default)."""
    from app.pipeline import inpaint as inpaint_mod

    # Mock LaMa to return the image unchanged (no model needed for this test)
    class _FakeLaMa:
        @staticmethod
        def get():
            return _FakeLaMa()

        @staticmethod
        def infer(img, mask, **kwargs):
            return img.copy()

    monkeypatch.setattr(inpaint_mod, "LaMa", _FakeLaMa)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    # Should NOT raise — just runs the default algorithm.
    out = inpaint(img, mask)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_inpaint_lama_dispatches_without_radius_error(monkeypatch):
    """radius param is only meaningful for TELEA/NS; LaMa path must accept
    any radius value without raising (it gets ignored)."""
    from app.pipeline import inpaint as inpaint_mod

    class _FakeLaMa:
        @staticmethod
        def get():
            return _FakeLaMa()

        @staticmethod
        def infer(img, mask, **kwargs):
            return img.copy()

    monkeypatch.setattr(inpaint_mod, "LaMa", _FakeLaMa)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    # radius=0 would normally fail TELEA validation, but LaMa ignores it
    out = inpaint(img, mask, algorithm="lama", radius=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_inpaint_lama_raises_when_model_missing(monkeypatch):
    """If the LaMa model file doesn't exist, raise ModelNotFoundError."""
    from app.exceptions import ModelNotFoundError
    from app.pipeline import inpaint as inpaint_mod

    def _raise(*args, **kwargs):
        raise ModelNotFoundError("LaMa model not found")

    class _MissingLaMa:
        get = staticmethod(_raise)

    monkeypatch.setattr(inpaint_mod, "LaMa", _MissingLaMa)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    with pytest.raises(ModelNotFoundError):
        inpaint(img, mask, algorithm="lama")


# ---------------------------------------------------------------------------
# Composes with other pipeline steps
# ---------------------------------------------------------------------------


def test_inpaint_chains_after_upscale():
    """Inpainting a 2x-upscaled image must still produce a valid uint8 output.

    The mask is also doubled to keep alignment with the upscaled image. Uses
    the LANCZOS4 fallback (no model on disk) so the test stays hermetic.
    """
    from pathlib import Path

    from app.pipeline.upscale import upscale

    base = np.random.RandomState(2).randint(0, 256, (50, 50, 3), dtype=np.uint8)
    base[20:30, 20:30, :] = 255  # a square to remove

    # No model on disk -> LANCZOS4 fallback (still 2x).
    upscaled = upscale(base, scale=2, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert upscaled.shape[0] == 100 and upscaled.shape[1] == 100

    # Build a matching mask (2x the size)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # doubled coords of 20:30

    out = inpaint(upscaled, mask, radius=5, algorithm="telea")
    assert out.shape == upscaled.shape
    assert out.dtype == np.uint8
    # The (now-upscaled) masked center should no longer be pure white
    assert not np.array_equal(out[50, 50], upscaled[50, 50])
