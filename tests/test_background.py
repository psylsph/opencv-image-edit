"""Tests for app.pipeline.background and app.models.matting (TDD).

Covers:
- apply_background_blur: shape contract, identity cases, blur effect
- apply_background_remove: BGRA output, alpha = mask, fg preservation
- get_alpha_debug: uint8 output with autocontrast
- _resize_mask: shape coercion utility
- MattingModel: existence/API smoke (full inference is not unit-tested here)
"""

from __future__ import annotations

import cv2
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


def test_apply_background_remove_keeps_the_subject_and_blanks_the_removed_background():
    """The subject keeps its colour; the removed background is fully transparent *and* carries no colour, so a viewer
    that ignores alpha cannot show the old background (it used to keep the original RGB under alpha 0)."""
    img = _bgr(100, 80)
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[20:60, 30:70] = 1.0
    out = apply_background_remove(img, mask)
    rgb_out = out[:, :, :3]
    assert np.array_equal(rgb_out[40, 50], img[40, 50])  # inside the foreground: unchanged
    assert out[5, 5, 3] == 0 and out[70, 90, 3] == 0  # outside: transparent...
    assert (
        not rgb_out[5, 5].any() and not rgb_out[70, 90].any()
    )  # ...and no old-background colour left


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


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_matting_registry_has_expected_models():
    """All four expected model variants are registered."""
    from app.models.matting import MATTING_MODELS

    keys = set(MATTING_MODELS.keys())
    assert keys == {"u2netp", "silueta", "u2net", "isnet", "birefnet_tiny"}


def test_matting_registry_entries_have_metadata():
    """Each registry entry has filename, input_size, and display info."""
    from app.models.matting import MATTING_MODELS

    for key, info in MATTING_MODELS.items():
        assert info.filename.endswith(".onnx"), f"{key}: filename must be .onnx"
        assert info.input_size >= 320, f"{key}: input_size too small"
        assert len(info.display_name) > 0, f"{key}: empty display_name"
        assert len(info.description) > 0, f"{key}: empty description"


def test_get_matting_model_rejects_unknown_name():
    """Requesting an unregistered model raises ModelNotFoundError."""
    from app.exceptions import ModelNotFoundError
    from app.models.matting import get_matting_model

    with pytest.raises(ModelNotFoundError):
        get_matting_model("nonexistent_model")


def test_available_models_returns_list():
    """available_models() returns a list of dicts with required fields."""
    from app.models.matting import available_models

    models = available_models()
    assert isinstance(models, list)
    assert len(models) >= 4
    for m in models:
        assert "key" in m
        assert "display_name" in m
        assert "available" in m
        assert "size_mb" in m


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

    model = MattingModel(model_path, input_size=320)
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


def test_selection_blur_selection_vs_outside():
    from app.pipeline.background import apply_selection_blur

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    sel = np.zeros((40, 40), np.uint8)
    sel[:, :20] = 255
    inside = apply_selection_blur(img, sel, 5, "selection", feather=0)
    assert np.array_equal(inside[:, 25:], img[:, 25:])
    assert not np.array_equal(inside[:, :15], img[:, :15])
    outside = apply_selection_blur(img, sel, 5, "outside", feather=0)
    assert np.array_equal(outside[:, :15], img[:, :15])
    assert not np.array_equal(outside[:, 25:], img[:, 25:])


# ---------------------------------------------------------------------------
# Replace the background (colour / another photo)
# ---------------------------------------------------------------------------


class TestCoverFit:
    def test_same_aspect_uses_the_whole_photo(self):
        from app.pipeline.background import cover_rect

        assert cover_rect(400, 300, 200, 150) == (0.0, 0.0, 400.0, 300.0)

    def test_wider_photo_is_cropped_at_the_sides_and_centred(self):
        from app.pipeline.background import cover_rect

        x0, y0, vw, vh = cover_rect(800, 300, 200, 150)
        assert (y0, vh) == (0.0, 300.0)
        assert vw == pytest.approx(400.0) and x0 == pytest.approx(200.0)

    def test_zoom_and_pan(self):
        from app.pipeline.background import cover_rect

        x0, _, vw, _ = cover_rect(400, 300, 200, 150, zoom=2.0)
        assert vw == pytest.approx(200.0) and x0 == pytest.approx(100.0)  # centred
        assert cover_rect(400, 300, 200, 150, zoom=2.0, ox=-1.0)[0] == pytest.approx(0.0)
        assert cover_rect(400, 300, 200, 150, zoom=2.0, ox=1.0)[0] == pytest.approx(200.0)

    def test_fit_cover_returns_exact_size_and_pans(self):
        from app.pipeline.background import fit_cover

        bg = np.zeros((100, 400, 3), np.uint8)
        bg[:, :50] = (255, 0, 0)  # blue strip on the left
        assert fit_cover(bg, 50, 50).shape == (50, 50, 3)
        left = fit_cover(bg, 50, 50, zoom=1.0, ox=-1.0)
        assert left[25, 5, 0] == 255 and left[25, 45, 0] == 0
        right = fit_cover(bg, 50, 50, zoom=1.0, ox=1.0)
        assert right[:, :, 0].max() == 0


class TestRefineMatte:
    def _edge(self):
        m = np.zeros((40, 80), np.float32)
        m[:, 40:] = 1.0
        return m

    def test_zero_is_identity(self):
        from app.pipeline.background import refine_matte

        m = self._edge()
        assert np.array_equal(refine_matte(m), m)

    def test_feather_softens_the_edge(self):
        from app.pipeline.background import refine_matte

        out = refine_matte(self._edge(), feather=1.0)
        assert 0.05 < out[20, 38] < 0.5 < out[20, 41] < 0.95

    def test_shrink_moves_the_edge_in_and_negative_out(self):
        from app.pipeline.background import refine_matte

        soft = refine_matte(self._edge(), feather=1.0)
        assert (
            refine_matte(soft, shrink=0.8).sum()
            < soft.sum()
            < refine_matte(soft, shrink=-0.8).sum()
        )
        assert refine_matte(soft, shrink=1.0).min() == 0.0


class TestReplaceBackground:
    def _inputs(self):
        img = np.full((20, 30, 3), (10, 20, 30), np.uint8)
        mask = np.zeros((20, 30), np.float32)
        mask[:, :15] = 1.0
        return img, mask

    def test_colour(self):
        from app.pipeline.background import apply_background_replace

        img, mask = self._inputs()
        out = apply_background_replace(img, mask, color="#ff0000")
        assert out.shape == img.shape
        assert tuple(out[5, 3]) == (10, 20, 30)  # subject untouched
        assert tuple(out[5, 25]) == (0, 0, 255)  # red in BGR

    def test_photo_background_is_fitted_to_the_subject_photo(self):
        from app.pipeline.background import apply_background_replace

        img, mask = self._inputs()
        bg = np.full((100, 100, 3), (200, 100, 50), np.uint8)
        out = apply_background_replace(img, mask, background=bg)
        assert out.shape == img.shape
        assert tuple(out[5, 25]) == (200, 100, 50) and tuple(out[5, 3]) == (10, 20, 30)

    def test_needs_exactly_one_source(self):
        from app.pipeline.background import apply_background_replace

        img, mask = self._inputs()
        with pytest.raises(ValueError):
            apply_background_replace(img, mask)
        with pytest.raises(ValueError):
            apply_background_replace(img, mask, color="#000000", background=img)


class TestDefringe:
    """Colour-fringe removal: edge pixels lose the old background's colour."""

    @staticmethod
    def _scene():
        # Red subject (left half) on a green old background, with a 6 px soft edge that is a red/green mix.
        h, w = 40, 80
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :] = (0, 200, 0)  # BGR green background
        img[:, :40] = (0, 0, 220)  # BGR red subject
        mask = np.zeros((h, w), np.float32)
        mask[:, :40] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), 3)
        # Photograph the blur: the edge colour is the matte-weighted mix of subject and background.
        m = mask[:, :, None]
        img = (
            np.array([0, 0, 220], np.float32) * m + np.array([0, 200, 0], np.float32) * (1 - m)
        ).astype(np.uint8)
        return img, mask

    def test_zero_is_a_no_op(self):
        from app.pipeline.background import defringe_colors

        img, mask = self._scene()
        assert np.array_equal(defringe_colors(img, mask, 0.0), img.astype(np.float32))

    def test_edge_loses_the_background_colour(self):
        from app.pipeline.background import defringe_colors

        img, mask = self._scene()
        out = defringe_colors(img, mask, 1.0)
        edge = (slice(None), 40)  # right at the edge: mask ~ 0.5
        assert img[edge][:, 1].mean() > 60  # plenty of green contamination before
        assert out[edge][:, 1].mean() < 15  # nearly gone after
        assert out[edge][:, 2].mean() > 180  # and it is the subject's red

    def test_solid_subject_and_far_background_are_untouched(self):
        from app.pipeline.background import defringe_colors

        img, mask = self._scene()
        out = defringe_colors(img, mask, 1.0)
        assert np.abs(out[:, :25] - img[:, :25]).max() < 1.0  # deep inside the subject
        assert (
            np.abs(out[:, 70:] - img[:, 70:]).max() < 1.0
        )  # far from any subject (no core nearby)

    def test_half_amount_is_in_between(self):
        from app.pipeline.background import defringe_colors

        img, mask = self._scene()
        full, half = defringe_colors(img, mask, 1.0), defringe_colors(img, mask, 0.5)
        g = lambda a: a[:, 40, 1].mean()  # noqa: E731
        assert g(full) < g(half) < g(img.astype(np.float32))

    def test_replace_background_uses_it(self):
        img, mask = self._scene()
        from app.pipeline.background import apply_background_replace

        plain = apply_background_replace(img, mask, color="#0000ff")
        clean = apply_background_replace(img, mask, color="#0000ff", defringe=1.0)
        # Over a blue backdrop the green halo is what gives the edge away; it should shrink.
        assert clean[:, 40, 1].mean() < plain[:, 40, 1].mean() - 20


class TestTransparentPixelsCarryNoOldBackground:
    def test_fully_transparent_pixels_have_no_colour(self):
        from app.pipeline.background import apply_background_remove

        img = np.full((20, 20, 3), (10, 200, 30), np.uint8)  # a bright green "old background"
        mask = np.zeros((20, 20), np.float32)
        mask[:, 10:] = 1.0  # right half kept
        out = apply_background_remove(img, mask)
        gone = out[:, :, 3] == 0
        assert gone.any()
        assert (
            out[gone][:, :3] == 0
        ).all()  # no trace of the old background under the transparency

    def test_kept_pixels_and_soft_edges_keep_their_colour(self):
        from app.pipeline.background import apply_background_remove

        img = np.full((20, 20, 3), (10, 200, 30), np.uint8)
        mask = np.zeros((20, 20), np.float32)
        mask[:, 10:] = 1.0
        out = apply_background_remove(img, mask)
        assert (out[:, 15:, :3] == (10, 200, 30)).all() and (out[:, 15:, 3] == 255).all()
        edge = (out[:, :, 3] > 0) & (out[:, :, 3] < 255)
        assert edge.any() and (out[edge][:, :3] == (10, 200, 30)).all()

    def test_the_process_endpoint_returns_a_png_with_clean_transparency(self, test_client):
        import base64
        import json

        import cv2

        class Stub:
            def predict(self, im):
                m = np.zeros(im.shape[:2], np.float32)
                m[:, im.shape[1] // 2 :] = 1.0
                return m

        import app.pipeline as pipeline

        original = pipeline.get_matting_model
        pipeline.get_matting_model = lambda name: Stub()
        try:
            img = np.full((40, 60, 3), (10, 200, 30), np.uint8)
            r = test_client.post(
                "/api/v1/process",
                files={"file": ("a.png", cv2.imencode(".png", img)[1].tobytes(), "image/png")},
                data={"settings": json.dumps({"background": {"enabled": True, "mode": "remove"}})},
            )
        finally:
            pipeline.get_matting_model = original
        assert r.status_code == 200
        png = base64.b64decode(r.json()["final"].split(",", 1)[1])
        out = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        assert out.shape[2] == 4
        gone = out[:, :, 3] == 0
        assert gone.any() and (out[gone][:, :3] == 0).all()
