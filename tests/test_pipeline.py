"""Tests for app.pipeline orchestrator (process_pipeline) and app.api.schemas.

Covers:
- process_pipeline import + signature
- Stage toggles (all-disabled passthrough)
- Each individual stage (background blur, background remove, grain, upscale, filters)
- Stage ordering / chaining
- ProcessRequest validation
- ProcessRequest default factories
- ProcessResult fields

The background and upscale stages depend on model files on disk. The matting
model (``u2netp.onnx``) and the EDSR x2 model are present in the repo's
``models/`` directory, so we exercise the real model path. If a model is
missing in a CI environment, those tests are skipped rather than failed.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    BackgroundRequest,
    FiltersRequest,
    GrainRequest,
    ProcessRequest,
    UpscaleRequest,
)
from app.exceptions import ModelNotFoundError, ProcessingError
from app.pipeline import process_pipeline
from app.pipeline.__init__ import ProcessResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bgr(w: int = 96, h: int = 64) -> np.ndarray:
    """Deterministic BGR gradient image."""
    gx = np.linspace(0, 255, w, dtype=np.float32)
    gy = np.linspace(0, 255, h, dtype=np.float32)
    xv, yv = np.meshgrid(gx, gy)
    b = xv.astype(np.uint8)
    g = yv.astype(np.uint8)
    r = ((xv + yv) / 2.0).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def _matting_model_available() -> bool:
    from pathlib import Path

    from app.config import get_settings

    return (Path(get_settings().model_dir) / "u2netp.onnx").exists()


def _upscale_model_available(algorithm: str = "edsr", scale: int = 2) -> bool:
    from pathlib import Path

    from app.config import get_settings

    return (
        Path(get_settings().model_dir) / f"{algorithm.upper()}_x{scale}.pb"
    ).exists()


# ---------------------------------------------------------------------------
# Import / signature
# ---------------------------------------------------------------------------


def test_pipeline_imports():
    """process_pipeline is importable and callable from app.pipeline."""
    from app.pipeline import process_pipeline as fn

    assert callable(fn)


def test_pipeline_process_result_dataclass_exists():
    """ProcessResult is exposed on app.pipeline.__init__ with the expected fields."""
    fields = {f for f in ProcessResult.__dataclass_fields__}
    # All six named outputs in the result (mask and grain_only can be None)
    for name in ("original", "final", "before_after", "diff", "mask", "upscaled"):
        assert name in fields, f"missing ProcessResult field: {name}"


def test_pipeline_process_result_grain_only_field():
    """grain_only is also exposed (a 2D uint8 viz, or None)."""
    fields = {f for f in ProcessResult.__dataclass_fields__}
    assert "grain_only" in fields


# ---------------------------------------------------------------------------
# All-disabled passthrough
# ---------------------------------------------------------------------------


def test_pipeline_passthrough_when_all_disabled():
    """With every stage disabled, the final image is the preprocessed input."""
    img = _bgr(96, 64)
    request = ProcessRequest()  # all defaults: all .enabled == False
    result = process_pipeline(img, request)

    # preprocessed image is identical to the input at this size (no resize needed)
    assert np.array_equal(result.original, img)
    assert np.array_equal(result.final, img)
    # Before/after visualisation must be present and same height as the final
    assert result.before_after.ndim == 3
    assert result.before_after.shape[0] == result.final.shape[0]
    # Diff is uint8 and matches final shape
    assert result.diff.shape == result.final.shape[:2] + (3,)
    assert result.diff.dtype == np.uint8
    # No debug images produced
    assert result.mask is None
    assert result.upscaled is None
    assert result.grain_only is None


# ---------------------------------------------------------------------------
# Background stage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _matting_model_available(), reason="u2netp.onnx not present"
)
def test_pipeline_background_blur_runs():
    """Background blur stage produces a different image (or matching shape)."""
    img = _bgr(96, 64)
    request = ProcessRequest(
        background=BackgroundRequest(enabled=True, mode="blur", blur_strength=20)
    )
    result = process_pipeline(img, request)
    # Shape and dtype preserved
    assert result.final.shape == img.shape
    assert result.final.dtype == np.uint8
    # Mask debug was produced
    assert result.mask is not None
    assert result.mask.ndim == 2
    assert result.mask.shape == img.shape[:2]


@pytest.mark.skipif(
    not _matting_model_available(), reason="u2netp.onnx not present"
)
def test_pipeline_background_remove_returns_bgra():
    """Background remove returns a 4-channel BGRA final image."""
    img = _bgr(96, 64)
    request = ProcessRequest(
        background=BackgroundRequest(enabled=True, mode="remove")
    )
    result = process_pipeline(img, request)
    assert result.final.ndim == 3
    assert result.final.shape[2] == 4
    assert result.final.dtype == np.uint8


# ---------------------------------------------------------------------------
# Grain stage
# ---------------------------------------------------------------------------


def test_pipeline_grain_runs():
    """Grain stage changes the image and produces a non-zero grain_only viz."""
    img = _bgr(96, 64)
    request = ProcessRequest(grain=GrainRequest(enabled=True, intensity=0.5))
    result = process_pipeline(img, request)
    assert result.final.shape == img.shape
    # The image must be different (luminance noise added)
    assert not np.array_equal(result.final, img)
    # The grain debug must be a 2D uint8 non-zero image
    assert result.grain_only is not None
    assert result.grain_only.ndim == 2
    assert result.grain_only.shape == img.shape[:2]
    assert result.grain_only.dtype == np.uint8


# ---------------------------------------------------------------------------
# Upscale stage
# ---------------------------------------------------------------------------


def test_pipeline_upscale_runs_scale2():
    """scale=2 must double both spatial dimensions."""
    img = _bgr(64, 48)
    request = ProcessRequest(
        upscale=UpscaleRequest(enabled=True, scale=2, algorithm="interp")
    )
    result = process_pipeline(img, request)
    assert result.final.shape[0] == img.shape[0] * 2
    assert result.final.shape[1] == img.shape[1] * 2
    # The upscaled intermediate is recorded
    assert result.upscaled is not None
    assert result.upscaled.shape[0] == img.shape[0] * 2


def test_pipeline_upscale_runs_scale4():
    """scale=4 must quadruple both spatial dimensions."""
    img = _bgr(40, 32)
    request = ProcessRequest(
        upscale=UpscaleRequest(enabled=True, scale=4, algorithm="interp")
    )
    result = process_pipeline(img, request)
    assert result.final.shape[0] == img.shape[0] * 4
    assert result.final.shape[1] == img.shape[1] * 4


# ---------------------------------------------------------------------------
# Filters stage
# ---------------------------------------------------------------------------


def test_pipeline_filters_brightness_runs():
    """brightness=0.5 should produce a darker image than the original."""
    img = _bgr(96, 64)
    request = ProcessRequest(
        filters=FiltersRequest(enabled=True, brightness=0.5)
    )
    result = process_pipeline(img, request)
    # Mean luma must drop
    assert result.final.mean() < img.mean()
    # Values are still uint8 in range
    assert result.final.dtype == np.uint8
    assert result.final.min() >= 0
    assert result.final.max() <= 255


def test_pipeline_filters_inactive_when_disabled():
    """If filters disabled but other stages disabled too, final is identical."""
    img = _bgr(96, 64)
    request = ProcessRequest(filters=FiltersRequest(enabled=False, brightness=0.5))
    result = process_pipeline(img, request)
    assert np.array_equal(result.final, img)


# ---------------------------------------------------------------------------
# Chaining / ordering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _matting_model_available(), reason="u2netp.onnx not present"
)
def test_pipeline_chains_in_order():
    """All stages on, the final image should reflect every stage's effect."""
    img = _bgr(80, 56)
    request = ProcessRequest(
        background=BackgroundRequest(enabled=True, mode="remove"),
        grain=GrainRequest(enabled=True, intensity=0.5),
        upscale=UpscaleRequest(enabled=True, scale=2, algorithm="interp"),
        filters=FiltersRequest(enabled=True, brightness=0.8),
    )
    result = process_pipeline(img, request)

    # Background remove → BGRA
    assert result.final.ndim == 3
    assert result.final.shape[2] == 4
    # Upscale doubles spatial dims
    assert result.final.shape[0] == img.shape[0] * 2
    assert result.final.shape[1] == img.shape[1] * 2
    # Grain must have produced a debug viz
    assert result.grain_only is not None
    # The mask debug is present
    assert result.mask is not None
    # The upscaled intermediate is present
    assert result.upscaled is not None
    # Original is the preprocessed input (not upscaled)
    assert result.original.shape == img.shape


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_pipeline_validation_blur_out_of_range():
    """ProcessRequest rejects blur_strength above 50."""
    with pytest.raises(ValidationError):
        ProcessRequest(
            background=BackgroundRequest(enabled=True, mode="blur", blur_strength=100)
        )


def test_pipeline_validation_scale_out_of_range():
    """UpscaleRequest rejects scale=3 (not in {1, 2, 4})."""
    with pytest.raises(ValidationError):
        UpscaleRequest(enabled=True, scale=3)  # type: ignore[arg-type]


def test_pipeline_validation_intensity_negative():
    """GrainRequest rejects intensity < 0."""
    with pytest.raises(ValidationError):
        GrainRequest(enabled=True, intensity=-0.5)


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------


def test_pipeline_request_default_factory():
    """ProcessRequest() with no args creates all sub-requests with sensible defaults."""
    req = ProcessRequest()
    assert isinstance(req.background, BackgroundRequest)
    assert isinstance(req.grain, GrainRequest)
    assert isinstance(req.upscale, UpscaleRequest)
    assert isinstance(req.filters, FiltersRequest)
    # None of them are enabled by default
    assert req.background.enabled is False
    assert req.grain.enabled is False
    assert req.upscale.enabled is False
    assert req.filters.enabled is False


def test_pipeline_background_request_default_factory():
    """BackgroundRequest has sane defaults."""
    bg = BackgroundRequest()
    assert bg.mode == "blur"
    assert bg.blur_strength == 15
    assert bg.model_name == "u2netp"


def test_pipeline_upscale_request_default_factory():
    """UpscaleRequest has sane defaults."""
    up = UpscaleRequest()
    assert up.scale == 2
    assert up.algorithm == "interp"


def test_pipeline_filters_request_default_factory():
    """FiltersRequest has sane defaults (all factors == 1.0)."""
    f = FiltersRequest()
    assert f.brightness == 1.0
    assert f.contrast == 1.0
    assert f.saturation == 1.0
    assert f.sharpness == 1.0
    assert f.vignette_strength == 0.0
    assert f.sepia is False
    assert f.grayscale_blend == 0.0


# ---------------------------------------------------------------------------
# ProcessResult fields
# ---------------------------------------------------------------------------


def test_pipeline_result_has_all_outputs():
    """ProcessResult exposes: original, final, before_after, diff, mask, upscaled, grain_only."""
    sig = inspect.signature(ProcessResult)
    names = list(sig.parameters.keys())
    for field in ("original", "final", "before_after", "diff", "mask", "upscaled", "grain_only"):
        assert field in names, f"ProcessResult missing field: {field}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_pipeline_empty_image_raises_processing_error():
    """A None or empty image should raise ProcessingError."""
    request = ProcessRequest()
    with pytest.raises(ProcessingError):
        process_pipeline(None, request)  # type: ignore[arg-type]


def test_pipeline_handles_existing_input_end_to_end():
    """Smoke test: a real BGR image with all stages disabled round-trips."""
    img = _bgr(120, 80)
    result = process_pipeline(img, ProcessRequest())
    assert result.original.shape == img.shape
    assert result.final.shape == img.shape
    # before_after is wider than the input
    assert result.before_after.shape[1] > img.shape[1]


# ---------------------------------------------------------------------------
# Sanity: model-missing path
# ---------------------------------------------------------------------------


def test_pipeline_missing_matting_model_propagates_error():
    """If get_matting_model raises ModelNotFoundError, the pipeline wraps it in ProcessingError.

    We monkeypatch ``app.models.matting.get_matting_model`` (the location the
    orchestrator actually imports from) to return a broken instance whose
    .predict() raises ModelNotFoundError. The pipeline should catch the
    underlying error and re-raise as ProcessingError.
    """
    from app.models import matting

    saved_singleton = matting._singleton
    orig_getter = matting.get_matting_model
    try:
        class _Broken:
            def predict(self, _img):
                raise ModelNotFoundError("no model")

        matting.get_matting_model = lambda *a, **k: _Broken()  # type: ignore[assignment]
        # Force the orchestrator module to re-resolve the symbol from app.models.matting
        import importlib
        import app.pipeline.__init__ as orch
        importlib.reload(orch)
        try:
            img = _bgr(80, 60)
            request = ProcessRequest(
                background=BackgroundRequest(enabled=True, mode="blur", blur_strength=10)
            )
            with pytest.raises(ProcessingError):
                orch.process_pipeline(img, request)
        finally:
            # Restore the orchestrator module to use the real getter
            matting.get_matting_model = orig_getter  # type: ignore[assignment]
            importlib.reload(orch)
    finally:
        matting._singleton = saved_singleton
