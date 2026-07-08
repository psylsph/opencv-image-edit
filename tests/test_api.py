"""Tests for the FastAPI HTTP layer.

Endpoints under test:
- ``GET  /health``              — liveness + model readiness
- ``GET  /presets``             — 4 default presets
- ``POST /api/v1/process``      — process an uploaded image
- ``GET  /metrics``             — Prometheus exposition format

The tests are designed to exercise the **real** code paths (including
the real ``u2netp.onnx`` matting model when present). The fixtures
``require_matting_model`` and ``require_edsr_x2`` skip individual
tests gracefully if a model file is missing.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.schemas import ProcessRequest


# ===========================================================================
# /health
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the ``GET /health`` endpoint."""

    def test_health_endpoint_returns_ok(self, test_client: TestClient):
        response = test_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    def test_health_includes_app_metadata(self, test_client: TestClient):
        response = test_client.get("/health")
        body = response.json()
        # app_name / version come from Settings
        assert "app" in body
        assert "version" in body
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0

    def test_health_includes_opencv_version(self, test_client: TestClient):
        response = test_client.get("/health")
        body = response.json()
        assert "opencv_version" in body
        # Should match the version OpenCV reports
        assert body["opencv_version"] == cv2.__version__

    def test_health_includes_models_status(self, test_client: TestClient):
        response = test_client.get("/health")
        body = response.json()
        assert "models" in body
        models = body["models"]
        assert isinstance(models, dict)
        # The matting model key is always reported
        assert "matting" in models
        # The status is one of "loaded" or "missing: <ErrorClass>"
        for key, status in models.items():
            assert status == "loaded" or status.startswith("missing:"), (
                f"unexpected status for {key!r}: {status!r}"
            )

    def test_health_includes_upscale_models(
        self, test_client: TestClient
    ):
        response = test_client.get("/health")
        body = response.json()
        models = body["models"]
        # The health endpoint checks at least the EDSR x2 model
        assert "upscale_edsr_x2" in models
        assert "upscale_edsr_x4" in models


# ===========================================================================
# /presets
# ===========================================================================


class TestPresetsEndpoint:
    """Tests for the ``GET /presets`` endpoint."""

    def test_presets_endpoint_returns_3_presets(self, test_client: TestClient):
        response = test_client.get("/presets")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"portrait", "landscape", "vintage"}

    def test_presets_have_label_and_description(self, test_client: TestClient):
        response = test_client.get("/presets")
        for name, meta in response.json().items():
            assert "label" in meta, f"preset {name!r} missing label"
            assert "description" in meta, f"preset {name!r} missing description"
            assert "settings" in meta, f"preset {name!r} missing settings"
            assert len(meta["label"]) > 0
            assert len(meta["description"]) > 0

    def test_presets_portrait_has_blur_enabled(self, test_client: TestClient):
        response = test_client.get("/presets")
        portrait = response.json()["portrait"]
        settings = portrait["settings"]
        assert settings["background"]["enabled"] is True
        assert settings["background"]["mode"] == "blur"
        assert settings["background"]["blur_strength"] >= 1

    def test_presets_landscape_has_vivid_filters(self, test_client: TestClient):
        response = test_client.get("/presets")
        landscape = response.json()["landscape"]
        settings = landscape["settings"]
        # Landscape: vivid colors → saturation/contrast > 1
        assert settings["filters"]["enabled"] is True
        assert settings["filters"]["saturation"] > 1.0
        assert settings["filters"]["contrast"] >= 1.0

    def test_presets_vintage_has_grain_and_sepia(self, test_client: TestClient):
        response = test_client.get("/presets")
        vintage = response.json()["vintage"]
        settings = vintage["settings"]
        assert settings["grain"]["enabled"] is True
        assert settings["grain"]["intensity"] > 0.0
        assert settings["filters"]["enabled"] is True
        assert settings["filters"]["sepia"] is True

    def test_presets_settings_are_valid_process_requests(
        self, test_client: TestClient
    ):
        response = test_client.get("/presets")
        for name, meta in response.json().items():
            # Each preset's settings must be a valid ProcessRequest
            try:
                ProcessRequest.model_validate(meta["settings"])
            except Exception as exc:  # pragma: no cover - sanity check
                pytest.fail(f"preset {name!r} has invalid settings: {exc}")


# ===========================================================================
# /api/v1/process — validation
# ===========================================================================


class TestProcessValidation:
    """Tests for ``POST /api/v1/process`` validation paths."""

    def test_process_requires_file(self, test_client: TestClient):
        response = test_client.post(
            "/api/v1/process",
            data={"settings": "{}"},
        )
        assert response.status_code in (400, 422)

    def test_process_requires_settings(self, test_client: TestClient, sample_png_bytes: bytes):
        # Multipart upload with a file but no settings field
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code in (400, 422)

    def test_process_validates_settings_json(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "not-valid-json"},
        )
        assert response.status_code == 422
        body = response.json()
        # Detail should mention the invalid settings
        assert "invalid settings" in str(body).lower()

    def test_process_validates_settings_schema(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        # Valid JSON but not a valid ProcessRequest (negative blur_strength)
        bad_settings = {
            "background": {
                "enabled": True,
                "mode": "blur",
                "blur_strength": 999,  # out of allowed range (1-50)
            }
        }
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": json.dumps(bad_settings)},
        )
        assert response.status_code == 422

    def test_process_rejects_too_large_file(
        self,
        test_client: TestClient,
        sample_png_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Override the settings to a tiny max size
        from app.config import get_settings

        get_settings.cache_clear()
        # Patch max_image_size_mb to 0 (effectively zero — any file is too large)
        # We use a different approach: set max bytes via a fixture.
        # Simpler: replace the cached settings with a stub that returns tiny limit
        from app import config as config_module

        real_settings = config_module.get_settings.__wrapped__()
        real_settings.max_image_size_mb = 0  # any non-empty file is too large

        def fake_get_settings():
            return real_settings

        monkeypatch.setattr(config_module, "get_settings", fake_get_settings)
        # The endpoint imports get_settings via "from app.config import get_settings"
        # inside the function — so we need to patch the import in app.api.process too.
        import app.api.process as proc_module

        monkeypatch.setattr(proc_module, "get_settings", fake_get_settings)

        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        assert response.status_code == 413
        body = response.json()
        assert "too large" in str(body).lower()

    def test_process_rejects_invalid_image_bytes(self, test_client: TestClient):
        # 1x1 PNG-valid-bytes would decode, so send actual garbage
        garbage = b"this is not a real image"
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", garbage, "image/png")},
            data={"settings": "{}"},
        )
        # 400 from DecodeError, or 422 from cv2
        assert response.status_code in (400, 422)
        body_text = str(response.json()).lower()
        # The detail should mention the decode problem
        assert (
            "decode" in body_text
            or "format" in body_text
            or "image" in body_text
            or "unsupported" in body_text
        )


# ===========================================================================
# /api/v1/process — happy paths
# ===========================================================================


class TestProcessEndpoint:
    """End-to-end happy-path tests for the process endpoint."""

    def test_process_handles_passthrough_request(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        # No stages enabled — should return the original BGR
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "final" in body
        assert body["final"].startswith("data:image/png;base64,")
        # Decode it and verify it's a valid PNG of the right size
        import base64

        payload = body["final"].split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        # The sample image is 100x100; passthrough should preserve that
        assert img.shape[0] == 100
        assert img.shape[1] == 100

    def test_process_returns_all_6_outputs(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # Must return all 6 image outputs + metadata
        for key in ("final", "before_after", "diff", "mask", "grain", "upscaled"):
            assert key in body, f"missing output: {key}"
        assert "elapsed_seconds" in body
        assert "output_size" in body
        assert "width" in body["output_size"]
        assert "height" in body["output_size"]

    def test_process_returns_elapsed_metadata(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        body = response.json()
        assert isinstance(body["elapsed_seconds"], (int, float))
        assert body["elapsed_seconds"] >= 0

    def test_process_with_filters_changes_pixels(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        import base64

        # Strong contrast should make some pixels clip to 0 or 255
        settings = json.dumps(
            {
                "filters": {
                    "enabled": True,
                    "brightness": 1.0,
                    "contrast": 2.0,  # max contrast → expect clipping
                    "saturation": 1.0,
                    "sharpness": 1.0,
                }
            }
        )
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": settings},
        )
        assert response.status_code == 200, response.text
        payload = response.json()["final"].split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        # Strong contrast should produce a mix of near-0 and near-255 pixels
        min_val = int(img.min())
        max_val = int(img.max())
        assert max_val - min_val > 100, (
            f"expected wide pixel range after contrast=2.0; got [{min_val}, {max_val}]"
        )

    def test_process_with_grain_modifies_image(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        import base64

        # Apply grain — should change pixel values from the passthrough
        passthrough = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        grain = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={
                "settings": json.dumps(
                    {"grain": {"enabled": True, "intensity": 0.9}}
                )
            },
        )
        assert passthrough.status_code == 200
        assert grain.status_code == 200, grain.text

        def _decode(data_url: str) -> np.ndarray:
            payload = data_url.split(",", 1)[1]
            arr = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        img_p = _decode(passthrough.json()["final"])
        img_g = _decode(grain.json()["final"])
        # The two images should differ
        assert not np.array_equal(img_p, img_g), "grain had no effect on image"

    def test_process_with_upscale_doubles_dimensions(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        settings = json.dumps(
            {
                "upscale": {
                    "enabled": True,
                    "scale": 2,
                    "algorithm": "interp",  # no model needed
                }
            }
        )
        response = test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": settings},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # 100x100 → 200x200
        assert body["output_size"]["width"] == 200
        assert body["output_size"]["height"] == 200


# ===========================================================================
# /metrics
# ===========================================================================


class TestMetricsEndpoint:
    """Tests for the ``GET /metrics`` Prometheus endpoint."""

    def test_metrics_endpoint_returns_prometheus_format(
        self, test_client: TestClient
    ):
        response = test_client.get("/metrics")
        assert response.status_code == 200
        # Prometheus exposition format starts with "# HELP" or "# TYPE" comments
        body = response.text
        assert "# HELP" in body or "# TYPE" in body
        # The content-type header should be the Prometheus text format
        ct = response.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_metrics_endpoint_includes_process_counter(
        self, test_client: TestClient, sample_png_bytes: bytes
    ):
        # Make a process request so the counter has data
        test_client.post(
            "/api/v1/process",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"settings": "{}"},
        )
        response = test_client.get("/metrics")
        body = response.text
        # The counter and histogram names should be present
        assert "image_process_total" in body
        assert "image_process_seconds" in body

    def test_metrics_endpoint_includes_request_size_histogram(
        self, test_client: TestClient
    ):
        response = test_client.get("/metrics")
        body = response.text
        assert "request_size_bytes" in body


# ===========================================================================
# Module-level sanity
# ===========================================================================


def test_app_loads_without_error():
    """The app module must be importable on its own."""
    from app.main import app

    assert app.title  # has a title set from settings
    assert len(app.routes) > 0
