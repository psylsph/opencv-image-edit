"""Tests for app.pipeline.upscale (TDD).

Covers:
- Upscaler class: init, model caching, cache clearing
- upscale() function: shape contract, fallback behavior, dtype/channel preservation
- LANCZOS4 fallback when no model file is present
- Grayscale input handling (round-trips through BGR)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.pipeline.upscale import (
    SUPPORTED_ALGORITHMS,
    SUPPORTED_SCALES,
    Upscaler,
    upscale,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_upscaler_cache():
    """Make sure the Upscaler singleton cache is empty before every test."""
    Upscaler.clear_cache()
    yield
    Upscaler.clear_cache()


def _bgr(w: int = 100, h: int = 100) -> np.ndarray:
    """Deterministic BGR image (gradient) — no file I/O needed."""
    gx = np.linspace(0, 255, w, dtype=np.float32)
    gy = np.linspace(0, 255, h, dtype=np.float32)
    xv, yv = np.meshgrid(gx, gy)
    b = xv.astype(np.uint8)
    g = yv.astype(np.uint8)
    r = ((xv + yv) / 2.0).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def _empty_model_dir(tmp_path: Path) -> Path:
    """A directory that contains no model files — forces the LANCZOS4 fallback."""
    d = tmp_path / "no_models"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Upscaler class
# ---------------------------------------------------------------------------


def test_upscaler_init():
    """Upscaler() can be constructed (it's a static-method class)."""
    ups = Upscaler()
    assert ups is not None


def test_upscaler_fallback_when_model_missing(tmp_path):
    """upscale() with no model file present uses LANCZOS4, output shape is correct."""
    img = _bgr(100, 80)  # (80, 100, 3)
    out = upscale(img, scale=2, model_dir=_empty_model_dir(tmp_path))
    assert out.shape == (160, 200, 3)
    assert out.dtype == np.uint8


def test_upscaler_get_raises_model_not_found(tmp_path):
    """Upscaler.get() raises ModelNotFoundError when the .pb file is missing."""
    from app.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        Upscaler.get("edsr", 2, _empty_model_dir(tmp_path))


def test_upscaler_get_rejects_unsupported_algorithm(tmp_path):
    """Upscaler.get() rejects algorithms outside the SUPPORTED_ALGORITHMS set."""
    with pytest.raises(ValueError, match="unsupported algorithm"):
        Upscaler.get("not_a_real_algo", 2, _empty_model_dir(tmp_path))


def test_upscaler_get_rejects_unsupported_scale(tmp_path):
    """Upscaler.get() rejects scales outside the SUPPORTED_SCALES set."""
    with pytest.raises(ValueError, match="unsupported scale"):
        Upscaler.get("edsr", 5, _empty_model_dir(tmp_path))


# ---------------------------------------------------------------------------
# upscale() function — fallback (LANCZOS4) behavior
# ---------------------------------------------------------------------------


def test_upscaler_fallback_2x():
    """100x100 BGR -> 200x200 BGR via LANCZOS4 fallback."""
    img = _bgr(100, 100)
    out = upscale(img, scale=2, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.shape == (200, 200, 3)


def test_upscaler_fallback_4x():
    """100x100 BGR -> 400x400 BGR via LANCZOS4 fallback."""
    img = _bgr(100, 100)
    out = upscale(img, scale=4, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.shape == (400, 400, 3)


def test_upscaler_fallback_1x():
    """scale=1 is a no-op passthrough (100x100 -> 100x100)."""
    img = _bgr(100, 100)
    out = upscale(img, scale=1, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.shape == (100, 100, 3)
    # Passthrough returns the same data (or a copy).
    assert np.array_equal(out, img)


def test_upscaler_preserves_dtype():
    """Output dtype is uint8 regardless of input scale."""
    img = _bgr(64, 64)
    out = upscale(img, scale=3, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.dtype == np.uint8


def test_upscaler_preserves_channels():
    """Input BGR (H, W, 3) -> output BGR (H*scale, W*scale, 3) — not 4-channel."""
    img = _bgr(80, 60)  # (60, 80, 3)
    out = upscale(img, scale=2, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.shape == (120, 160, 3)


def test_upscaler_strips_alpha():
    """Input BGRA (H, W, 4) -> output BGR (H*scale, W*scale, 3) (alpha dropped)."""
    img = _bgr(40, 40)
    bgra = np.dstack([img, np.full((40, 40), 255, dtype=np.uint8)])  # (40, 40, 4)
    out = upscale(bgra, scale=2, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8


def test_upscaler_handles_grayscale():
    """Input 2D grayscale -> output 2D grayscale (LANCZOS4 fallback path)."""
    gray = np.linspace(0, 255, 100 * 100, dtype=np.uint8).reshape(100, 100)
    out = upscale(gray, scale=2, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out.ndim == 2
    assert out.shape == (200, 200)
    assert out.dtype == np.uint8


def test_upscaler_scale_one_returns_copy_not_alias():
    """scale=1 returns a copy (so callers can mutate it without touching the input)."""
    img = _bgr(50, 50)
    out = upscale(img, scale=1, model_dir=Path("/tmp/nonexistent_models_dir_xyz"))
    assert out is not img
    assert np.array_equal(out, img)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_supported_algorithms_contains_expected():
    """SUPPORTED_ALGORITHMS must include the four documented algorithms."""
    assert "edsr" in SUPPORTED_ALGORITHMS
    assert "espcn" in SUPPORTED_ALGORITHMS
    assert "fsrcnn" in SUPPORTED_ALGORITHMS
    assert "lapsrn" in SUPPORTED_ALGORITHMS


def test_supported_scales_contains_expected():
    """SUPPORTED_SCALES must include 2, 3, and 4."""
    assert 2 in SUPPORTED_SCALES
    assert 3 in SUPPORTED_SCALES
    assert 4 in SUPPORTED_SCALES


# ---------------------------------------------------------------------------
# Caching behavior (mocked)
# ---------------------------------------------------------------------------


def test_upscaler_caches_model(tmp_path):
    """Loading the same (algorithm, scale) twice should only call readModel once."""
    from unittest.mock import MagicMock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ESPCN_x2.pb").write_bytes(b"fake frozen graph bytes")

    sr_instance = MagicMock()
    with patch(
        "app.pipeline.upscale.cv2.dnn_superres.DnnSuperResImpl_create",
        return_value=sr_instance,
    ):
        a = Upscaler.get("espcn", 2, model_dir)
        b = Upscaler.get("espcn", 2, model_dir)

    # Same instance returned (cache hit on the second call)
    assert a is b
    # DnnSuperResImpl_create called exactly once (proves caching)
    assert sr_instance.readModel.call_count == 1
    assert sr_instance.setModel.call_count == 1


def test_upscaler_clear_cache(tmp_path):
    """clear_cache() resets the singleton cache so the next get() reloads."""
    from unittest.mock import MagicMock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ESPCN_x2.pb").write_bytes(b"fake")

    sr_a = MagicMock()
    sr_b = MagicMock()

    with patch("app.pipeline.upscale.cv2.dnn_superres.DnnSuperResImpl_create", return_value=sr_a):
        first = Upscaler.get("espcn", 2, model_dir)

    Upscaler.clear_cache()

    with patch("app.pipeline.upscale.cv2.dnn_superres.DnnSuperResImpl_create", return_value=sr_b):
        second = Upscaler.get("espcn", 2, model_dir)

    assert first is sr_a
    assert second is sr_b
    assert first is not second
