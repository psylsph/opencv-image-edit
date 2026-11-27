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

    return (Path(get_settings().model_dir) / f"{algorithm.upper()}_x{scale}.pb").exists()


# ---------------------------------------------------------------------------
# Import / signature
# ---------------------------------------------------------------------------


def test_pipeline_imports():
    """process_pipeline is importable and callable from app.pipeline."""
    from app.pipeline import process_pipeline as fn

    assert callable(fn)


def test_pipeline_process_result_dataclass_exists():
    """ProcessResult is exposed on app.pipeline.__init__ with the expected fields."""
    fields = set(ProcessResult.__dataclass_fields__)
    # All six named outputs in the result (mask and grain_only can be None)
    for name in ("original", "final", "before_after", "diff", "mask", "upscaled"):
        assert name in fields, f"missing ProcessResult field: {name}"


def test_pipeline_process_result_grain_only_field():
    """grain_only is also exposed (a 2D uint8 viz, or None)."""
    fields = set(ProcessResult.__dataclass_fields__)
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


def test_pipeline_accepts_image_with_alpha_channel():
    """A BGRA input (transparent PNG) is flattened to 3-channel BGR before any stage runs."""
    img = np.dstack([_bgr(64, 96), np.full((96, 64), 255, np.uint8)])
    result = process_pipeline(img, ProcessRequest())
    assert result.original.shape == (96, 64, 3)
    assert result.final.shape == (96, 64, 3)


@pytest.mark.skipif(not _matting_model_available(), reason="u2netp.onnx not present")
def test_pipeline_background_with_alpha_input():
    """Regression: background removal on a BGRA photo crashed ('Expected: 3, Got: 4')."""
    img = np.dstack([_bgr(64, 96), np.full((96, 64), 255, np.uint8)])
    req = ProcessRequest(background={"enabled": True, "mode": "blur"})
    assert process_pipeline(img, req).final.shape[:2] == (96, 64)


# ---------------------------------------------------------------------------
# Background stage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _matting_model_available(), reason="u2netp.onnx not present")
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


@pytest.mark.skipif(not _matting_model_available(), reason="u2netp.onnx not present")
def test_pipeline_background_remove_returns_bgra():
    """Background remove returns a 4-channel BGRA final image."""
    img = _bgr(96, 64)
    request = ProcessRequest(background=BackgroundRequest(enabled=True, mode="remove"))
    result = process_pipeline(img, request)
    assert result.final.ndim == 3
    assert result.final.shape[2] == 4
    assert result.final.dtype == np.uint8


def test_pipeline_manual_background_mask_blurs_painted_area_without_model():
    """Painted background mask bypasses AI matting and marks painted pixels as BG."""
    img = _bgr(80, 50)
    bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bg_mask[:, :30] = 255  # user highlighted the left side as background
    request = ProcessRequest(
        background=BackgroundRequest(enabled=True, mode="blur", blur_strength=12)
    )

    result = process_pipeline(img, request, background_mask=bg_mask)

    assert result.final.shape == img.shape
    assert result.mask is not None
    # Debug mask is foreground alpha: painted background becomes black.
    assert result.mask[:, :30].mean() < 5
    assert result.mask[:, 35:].mean() > 250
    # Foreground/right side remains sharp/original because alpha=1 there.
    assert np.array_equal(result.final[:, 35:], img[:, 35:])


def test_pipeline_manual_background_remove_makes_painted_area_transparent():
    """Manual background paint can drive background removal alpha directly."""
    img = _bgr(40, 30)
    bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bg_mask[5:20, 10:25] = 255
    request = ProcessRequest(background=BackgroundRequest(enabled=True, mode="remove"))

    result = process_pipeline(img, request, background_mask=bg_mask)

    assert result.final.shape == img.shape[:2] + (4,)
    alpha = result.final[:, :, 3]
    assert alpha[8:17, 13:22].max() < 10
    assert alpha[:4, :4].min() > 245


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
    request = ProcessRequest(upscale=UpscaleRequest(enabled=True, scale=2, algorithm="interp"))
    result = process_pipeline(img, request)
    assert result.final.shape[0] == img.shape[0] * 2
    assert result.final.shape[1] == img.shape[1] * 2
    # The upscaled intermediate is recorded
    assert result.upscaled is not None
    assert result.upscaled.shape[0] == img.shape[0] * 2


def test_pipeline_upscale_runs_scale4():
    """scale=4 must quadruple both spatial dimensions."""
    img = _bgr(40, 32)
    request = ProcessRequest(upscale=UpscaleRequest(enabled=True, scale=4, algorithm="interp"))
    result = process_pipeline(img, request)
    assert result.final.shape[0] == img.shape[0] * 4
    assert result.final.shape[1] == img.shape[1] * 4


# ---------------------------------------------------------------------------
# Filters stage
# ---------------------------------------------------------------------------


def test_pipeline_filters_brightness_runs():
    """brightness=0.5 should produce a darker image than the original."""
    img = _bgr(96, 64)
    request = ProcessRequest(filters=FiltersRequest(enabled=True, brightness=0.5))
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


@pytest.mark.skipif(not _matting_model_available(), reason="u2netp.onnx not present")
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
        ProcessRequest(background=BackgroundRequest(enabled=True, mode="blur", blur_strength=100))


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

    saved_cache = matting._cache
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
        matting._cache = saved_cache


# ---------------------------------------------------------------------------
# Order and combining: background first, then a selection tool, then the rest
# ---------------------------------------------------------------------------


def test_selection_tool_runs_as_a_pipeline_stage():
    img = _bgr(64, 96)
    sel = np.zeros((96, 64), np.uint8)
    sel[20:50, 10:40] = 255
    req = ProcessRequest(edit={"kind": "blur", "blur_strength": 25})
    out = process_pipeline(img, req, edit_mask=sel).final
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(out[80:, 50:], img[80:, 50:])  # far from the selection: untouched


def test_selection_tool_needs_a_mask():
    from app.exceptions import ProcessingError

    with pytest.raises(ProcessingError):
        process_pipeline(_bgr(64, 96), ProcessRequest(edit={"kind": "inpaint"}))


def test_background_runs_before_the_selection_tool_and_filters(monkeypatch):
    calls = []
    import app.pipeline as P

    monkeypatch.setattr(P, "apply_background_blur", lambda img, m, s: (calls.append("bg"), img)[1])
    monkeypatch.setattr(
        "app.pipeline.edit.apply_edit", lambda img, m, e: (calls.append("edit"), (img, None))[1]
    )
    monkeypatch.setattr(P, "apply_grain", lambda img, i: (calls.append("grain"), (img, None))[1])
    monkeypatch.setattr(
        P,
        "get_matting_model",
        lambda n: type("M", (), {"predict": lambda s, i: np.ones(i.shape[:2], np.float32)})(),
    )
    req = ProcessRequest(
        background={"enabled": True, "mode": "blur"},
        edit={"kind": "blur"},
        grain={"enabled": True},
    )
    process_pipeline(_bgr(64, 96), req, edit_mask=np.full((96, 64), 255, np.uint8))
    assert calls == ["bg", "edit", "grain"]


def test_background_removal_keeps_transparency_through_later_stages(monkeypatch):
    import app.pipeline as P

    monkeypatch.setattr(
        P,
        "get_matting_model",
        lambda n: type(
            "M",
            (),
            {"predict": lambda s, i: np.pad(np.ones((48, 32), np.float32), ((24, 24), (16, 16)))},
        )(),
    )
    req = ProcessRequest(
        background={"enabled": True, "mode": "remove"},
        grain={"enabled": True},
        filters={"enabled": True, "sepia": True},
    )
    out = process_pipeline(_bgr(64, 96), req).final
    assert out.shape == (96, 64, 4) and out[0, 0, 3] == 0 and out[48, 32, 3] == 255


def test_stage_cache_reuses_earlier_stages(monkeypatch):
    import app.pipeline as P

    calls = []
    monkeypatch.setattr(
        P, "apply_background_blur", lambda img, m, s: (calls.append("bg"), img + 1)[1]
    )
    monkeypatch.setattr(
        P,
        "get_matting_model",
        lambda n: type("M", (), {"predict": lambda s, i: np.ones(i.shape[:2], np.float32)})(),
    )
    img = _bgr(64, 96)
    base = {"background": {"enabled": True, "mode": "blur"}, "grain": {"enabled": True}}
    r1 = process_pipeline(img, ProcessRequest(**base), use_cache=True)
    base["grain"]["intensity"] = 0.9  # only a late step changes
    r2 = process_pipeline(img, ProcessRequest(**base), use_cache=True)
    assert calls == ["bg"]  # background ran once
    assert [s["cached"] for s in r2.steps] == [True, False]
    assert [s["name"] for s in r1.steps] == ["Background", "Grain"]
    process_pipeline(img + 2, ProcessRequest(**base), use_cache=True)  # a different photo: no reuse
    assert calls == ["bg", "bg"]


def test_several_selection_tools_run_in_order_and_report_progress(monkeypatch):
    seen, progress = [], []
    monkeypatch.setattr(
        "app.pipeline.edit.apply_edit",
        lambda img, m, e: (seen.append(e.kind), (img, "note " + e.kind))[1],
    )
    sel = np.full((96, 64), 255, np.uint8)
    req = ProcessRequest(edits=[{"kind": "inpaint"}, {"kind": "blur"}])
    r = process_pipeline(_bgr(64, 96), req, edit_masks=[sel, sel], progress=progress.append)
    assert seen == ["inpaint", "blur"]
    assert r.notes == ["note inpaint", "note blur"]
    assert [p["current"] for p in progress] == ["Remove object", "Blur area", None]


@pytest.mark.parametrize("kind", ["inpaint", "tattoo", "blur"])
def test_apply_edit_returns_image_and_note_for_every_tool(kind, monkeypatch):
    """Each real tool branch returns (image, note); a stubbed test once hid a branch returning a bare image."""
    from app.api.schemas import EditRequest
    from app.pipeline import edit as E

    monkeypatch.setattr(E, "person_mask", lambda img: None)
    monkeypatch.setattr(
        E, "remove_tattoo_lama", lambda img, m, **k: (img.copy(), {"mode": "lama", "ink_pct": 0})
    )
    img = _bgr(64, 96)
    sel = np.zeros((96, 64), np.uint8)
    sel[30:50, 20:40] = 255
    algo = "telea" if kind == "inpaint" else "lama"
    out, note = E.apply_edit(img, sel, EditRequest(kind=kind, algorithm=algo))
    assert out.shape == img.shape
    assert (note is not None) == (kind == "tattoo")


def _counting_background(monkeypatch, calls):
    import app.pipeline as P

    monkeypatch.setattr(
        P, "apply_background_blur", lambda img, m, s: (calls.append("bg"), img + 1)[1]
    )
    monkeypatch.setattr(
        P,
        "get_matting_model",
        lambda n: type("M", (), {"predict": lambda s, i: np.ones(i.shape[:2], np.float32)})(),
    )


def test_none_entries_in_edits_take_no_mask(monkeypatch):
    """Masks pair with the active tools in order, as the API counts them; a "none" entry takes none."""
    seen = []
    monkeypatch.setattr(
        "app.pipeline.edit.apply_edit",
        lambda img, m, e: (seen.append((e.kind, int(m[0, 0]))), (img, None))[1],
    )
    a, b = np.full((96, 64), 10, np.uint8), np.full((96, 64), 20, np.uint8)
    req = ProcessRequest(
        edits=[{"kind": "none"}, {"kind": "inpaint"}, {"kind": "none"}, {"kind": "blur"}]
    )
    process_pipeline(_bgr(64, 96), req, edit_masks=[a, b])
    assert seen == [("inpaint", 10), ("blur", 20)]


def test_stage_cache_stays_within_its_memory_budget(monkeypatch):
    import app.pipeline as P
    from app.config import get_settings

    calls = []
    _counting_background(monkeypatch, calls)
    img = _bgr(64, 96)  # each snapshot ~18 KB (current + mask_debug)
    monkeypatch.setattr(get_settings(), "stage_cache_mb", 1)
    for i in range(16):
        process_pipeline(
            img + i, ProcessRequest(background={"enabled": True, "mode": "blur"}), use_cache=True
        )
    assert P._cache_bytes <= 1024 * 1024
    assert P._cache_bytes == sum(size for _, size in P._cache.values())

    big = _bgr(1024, 1024)  # 3 MB: larger than the whole budget, so it is not kept at all
    P.clear_stage_cache()
    process_pipeline(
        big, ProcessRequest(background={"enabled": True, "mode": "blur"}), use_cache=True
    )
    assert len(P._cache) == 0 and P._cache_bytes == 0

    monkeypatch.setattr(get_settings(), "stage_cache_mb", 0)  # 0 turns it off
    process_pipeline(
        img, ProcessRequest(background={"enabled": True, "mode": "blur"}), use_cache=True
    )
    assert len(P._cache) == 0


def test_snapshot_keeps_shared_arrays_shared():
    import app.pipeline as P

    up = np.zeros((100, 100, 3), np.uint8)
    snap = P._snapshot({"current": up, "upscaled": up, "notes": []})
    assert snap["current"] is snap["upscaled"] and snap["current"] is not up
    assert P._snapshot_bytes(snap) == up.nbytes  # counted once


def test_installing_a_model_invalidates_cached_results(monkeypatch, tmp_path):
    from app.config import get_settings

    calls = []
    _counting_background(monkeypatch, calls)
    monkeypatch.setattr(get_settings(), "model_dir", tmp_path)
    req = ProcessRequest(background={"enabled": True, "mode": "blur"})
    img = _bgr(64, 96)
    process_pipeline(img, req, use_cache=True)
    process_pipeline(img, req, use_cache=True)
    assert calls == ["bg"]
    (tmp_path / "gfpgan.onnx").write_bytes(b"new model")
    process_pipeline(img, req, use_cache=True)
    assert calls == ["bg", "bg"]


def test_unloading_idle_models_empties_the_stage_cache(monkeypatch):
    import app.pipeline as P
    from app.model_cache import clear_all_model_caches

    _counting_background(monkeypatch, [])
    process_pipeline(
        _bgr(64, 96), ProcessRequest(background={"enabled": True, "mode": "blur"}), use_cache=True
    )
    assert P._cache
    clear_all_model_caches()
    assert not P._cache and P._cache_bytes == 0
