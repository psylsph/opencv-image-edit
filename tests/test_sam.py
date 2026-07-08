"""Tests for MobileSAM (point-prompt segmentation) — both model + API.

Covers:
- /api/v1/segment endpoint contract (file + x + y form-data)
- 400 for non-image bytes, 422 for out-of-bounds points
- 503 when model files are missing (ModelNotFoundError)
- End-to-end with the real model: synthetic image with a colored rectangle,
  click inside the rectangle, assert the mask covers most of the rectangle
  and avoids the background. Also asserts the overlay has red pixels in the
  masked area and the original BGR values outside it.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sam_synthetic_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a synthetic BGR test image with a colored rectangle on a uniform background.

    Returns (image, (x0, y0, x1, y1)) where (x0, y0) is the top-left and
    (x1, y1) the bottom-right of the rectangle.
    """
    # 800x1200 image, dark background
    img = np.full((800, 1200, 3), 50, dtype=np.uint8)
    # Draw a 400x400 bright-pink rectangle
    x0, y0, x1, y1 = 300, 200, 700, 600
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 100, 250), -1)
    return img, (x0, y0, x1, y1)


@pytest.fixture
def sam_synthetic_png_bytes(sam_synthetic_image) -> bytes:
    from app.pipeline.io import encode_png
    img, _ = sam_synthetic_image
    return encode_png(img)


@pytest.fixture
def require_sam_models() -> None:
    """Skip if the MobileSAM ONNX models aren't available."""
    settings = get_settings()
    enc = settings.model_dir / "mobile_sam.encoder.onnx"
    dec = settings.model_dir / "sam_vit_h_4b8939.decoder.onnx"
    if not enc.exists() or not dec.exists():
        pytest.skip(
            f"MobileSAM models missing: encoder={enc.exists()}, decoder={dec.exists()}"
        )


# ---------------------------------------------------------------------------
# Module-level / unit tests
# ---------------------------------------------------------------------------


def test_segment_module_api_exists() -> None:
    """The sam module exposes the documented public API surface."""
    from app.models import sam as sam_mod

    assert hasattr(sam_mod, "MobileSAM")
    assert hasattr(sam_mod, "segment_with_point")
    assert hasattr(sam_mod, "_MEAN")
    assert hasattr(sam_mod, "_STD")
    # mean/std must be float32
    assert sam_mod._MEAN.dtype == np.float32
    assert sam_mod._STD.dtype == np.float32
    # canonical SAM normalization values (compared with tolerance for float32
    # representation of these decimal constants)
    np.testing.assert_allclose(sam_mod._MEAN, [123.675, 116.28, 103.53], atol=1e-3)
    np.testing.assert_allclose(sam_mod._STD, [58.395, 57.12, 57.375], atol=1e-3)


# ---------------------------------------------------------------------------
# /api/v1/segment endpoint — error paths
# ---------------------------------------------------------------------------


def test_segment_endpoint_rejects_garbage(test_client: TestClient) -> None:
    """Bytes that aren't an image must return 400 (DecodeError)."""
    garbage = b"this is definitely not an image file"
    response = test_client.post(
        "/api/v1/segment",
        files={"file": ("garbage.png", garbage, "image/png")},
        data={"x": "100", "y": "100"},
    )
    assert response.status_code == 400
    body = response.json()
    detail = str(body.get("detail", "")).lower() if isinstance(body, dict) else ""
    assert "invalid image" in detail or "decode" in detail


def test_segment_endpoint_rejects_out_of_bounds_point(
    test_client: TestClient, sam_synthetic_png_bytes: bytes
) -> None:
    """A point outside the image bounds must return 422."""
    # sample image is 800x1200 (HxW); point (5000, 5000) is way out
    response = test_client.post(
        "/api/v1/segment",
        files={"file": ("sam_test.png", sam_synthetic_png_bytes, "image/png")},
        data={"x": "5000", "y": "5000"},
    )
    assert response.status_code == 422
    body = response.json()
    # The detail should mention the bounds
    detail = str(body).lower()
    assert "out" in detail or "bounds" in detail


def test_segment_endpoint_with_fake_model_returns_503(
    test_client: TestClient,
    sam_synthetic_png_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """With model files removed (or pointed elsewhere), endpoint should return 503.

    The MobileSAM singleton uses a class-level cache. We clear it and then
    monkeypatch get_settings to point at a non-existent model_dir.
    """
    # Clear cached singleton so the next get() actually looks at the model dir
    from app.models import sam as sam_mod
    sam_mod.MobileSAM.clear_cache()

    import app.models.sam as sam_mod_top
    monkeypatch.setattr(sam_mod_top, "MobileSAM", sam_mod_top.MobileSAM)  # no-op safety

    # Force get_settings to return a model_dir with no models
    from app import config as config_module
    real_settings = config_module.get_settings.__wrapped__()
    real_settings.model_dir = tmp_path

    def fake_get_settings():
        return real_settings

    monkeypatch.setattr(config_module, "get_settings", fake_get_settings)

    # Also override the one the singleton uses internally (it imports inside the method)
    # The endpoint itself doesn't import get_settings, but the model does. We've
    # already patched it at the top-level — and the sam module's reference will
    # also be updated because the segment module reaches sam -> MobileSAM.get
    # which calls get_settings from app.config.

    response = test_client.post(
        "/api/v1/segment",
        files={"file": ("sam_test.png", sam_synthetic_png_bytes, "image/png")},
        data={"x": "500", "y": "400"},
    )
    # ModelNotFoundError -> 503
    assert response.status_code == 503, response.text
    body_text = str(response.json()).lower()
    assert "modelnotfounderror" in body_text or "not found" in body_text

    # Restore singleton
    sam_mod.MobileSAM.clear_cache()


# ---------------------------------------------------------------------------
# /api/v1/segment endpoint — happy path with real model
# ---------------------------------------------------------------------------


def test_segment_with_point_real_model_isolates_foreground(
    test_client: TestClient,
    sam_synthetic_image: tuple[np.ndarray, tuple[int, int, int, int]],
    sam_synthetic_png_bytes: bytes,
    require_sam_models: None,
) -> None:
    """Real-model test: clicking inside the rectangle should mask >70% of the
    rectangle interior and <5% of the background corner."""
    img, (x0, y0, x1, y1) = sam_synthetic_image
    # Reset singleton so we pick up the (now present) model files
    from app.models import sam as sam_mod
    sam_mod.MobileSAM.clear_cache()

    # Click near the center of the rectangle
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    response = test_client.post(
        "/api/v1/segment",
        files={"file": ("sam_test.png", sam_synthetic_png_bytes, "image/png")},
        data={"x": str(cx), "y": str(cy)},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # Basic response shape
    assert "mask" in body and body["mask"].startswith("data:image/png;base64,")
    assert "overlay" in body and body["overlay"].startswith("data:image/png;base64,")
    assert isinstance(body["score"], float)
    assert body["point"] == {"x": cx, "y": cy}

    # Decode the mask
    b64 = body["mask"].split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    assert mask is not None, "mask PNG failed to decode"
    assert mask.shape == img.shape[:2]
    binary = mask > 0

    # Rectangle interior coverage: > 70% (MobileSAM with a single point
    # on a simple rectangle extends the mask slightly beyond the rectangle
    # bounds, so we assert a high but realistic threshold).
    rect_interior = np.zeros_like(binary, dtype=bool)
    rect_interior[y0 + 5 : y1 - 5, x0 + 5 : x1 - 5] = True
    coverage = binary[rect_interior].mean()
    assert coverage > 0.70, f"rectangle coverage {coverage:.3f} < 0.70"

    # Background corner (top-left 100x100): < 5% masked
    bg_corner = binary[:100, :100]
    bg_pct = bg_corner.mean()
    assert bg_pct < 0.05, f"background-corner mask pct {bg_pct:.3f} >= 0.05"

    # Score should be in a sane range (> 0)
    assert body["score"] > 0.0
    # And elapsed time should be > 0
    assert body["elapsed_seconds"] > 0.0


def test_segment_overlay_uses_red(
    test_client: TestClient,
    sam_synthetic_image: tuple[np.ndarray, tuple[int, int, int, int]],
    sam_synthetic_png_bytes: bytes,
    require_sam_models: None,
) -> None:
    """The overlay should have red pixels in the masked area and original BGR
    values close to input outside the masked area."""
    img, (x0, y0, x1, y1) = sam_synthetic_image
    from app.models import sam as sam_mod
    sam_mod.MobileSAM.clear_cache()

    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    response = test_client.post(
        "/api/v1/segment",
        files={"file": ("sam_test.png", sam_synthetic_png_bytes, "image/png")},
        data={"x": str(cx), "y": str(cy)},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    b64 = body["overlay"].split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    overlay = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert overlay is not None, "overlay PNG failed to decode"
    assert overlay.shape == img.shape

    # Decode the mask too so we can localize the assertions
    mask_b64 = body["mask"].split(",", 1)[1]
    mask_raw = base64.b64decode(mask_b64)
    mask_arr = np.frombuffer(mask_raw, dtype=np.uint8)
    mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    binary = mask > 0

    # In the masked region, the red channel (overlay[:, :, 2]) should be
    # higher than the blue channel (overlay[:, :, 0]) — i.e. red-shifted.
    if binary.any():
        masked_red = overlay[binary, 2].astype(np.int32)
        masked_blue = overlay[binary, 0].astype(np.int32)
        assert (masked_red > masked_blue).mean() > 0.90, (
            "overlay should be red-shifted in the masked area"
        )

    # Outside the masked region, the overlay should match the input image
    # closely (allow small alpha-blending tolerance at the boundary)
    unmasked = ~binary
    # Avoid the boundary by eroding the mask
    if binary.any():
        eroded = cv2.erode(
            binary.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
        ).astype(bool)
        safe_unmasked = unmasked & ~eroded  # we want truly unmasked, non-boundary
    else:
        safe_unmasked = unmasked
    if safe_unmasked.any():
        diff = np.abs(overlay[safe_unmasked].astype(np.int32) - img[safe_unmasked].astype(np.int32))
        # Strict: outside the boundary region, alpha is 0 so no color shift
        assert diff.max() <= 1, (
            f"overlay pixels outside the mask should match the input exactly (max diff {diff.max()})"
        )
