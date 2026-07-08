"""Shared pytest fixtures for the test suite.

This conftest provides:
- ``test_app``: the FastAPI application instance.
- ``test_client``: a ``fastapi.testclient.TestClient`` bound to ``test_app``.
- ``sample_image``: a small BGR numpy array (100x100) suitable for the pipeline.
- ``settings_overrides`` / ``small_max_size_settings``: helpers for tuning
  per-test limits (e.g. to exercise size-validation paths).
- ``reset_singletons``: ensure model/upscaler caches are cleared between tests.

The fixtures are intentionally lightweight — they avoid loading the heavy
ONNX matting model at import time. Tests that need the matting model should
request it explicitly via ``require_matting_model``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_app():
    """Return the FastAPI app instance (imported lazily to avoid heavy imports at conftest-load)."""
    from app.main import app

    return app


@pytest.fixture(scope="session")
def test_client(test_app) -> TestClient:
    """Return a TestClient for the FastAPI app (session-scoped for speed)."""
    return TestClient(test_app)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a small 100x100 BGR uint8 image with a clear two-color pattern.

    The image has a darker top half and a lighter bottom half — this gives
    tests something visually meaningful to assert on (e.g. ``mean(top) <
    mean(bottom)``) and works well with the matting model since there's a
    clear foreground/background split.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, :] = (40, 60, 80)   # darker top half (BGR)
    img[50:, :, :] = (200, 180, 160)  # lighter bottom half
    # Draw a small "foreground" object for the matting model
    img[30:70, 30:70, :] = (10, 200, 30)
    return img


@pytest.fixture
def sample_png_bytes(sample_image) -> bytes:
    """Return the sample image as PNG-encoded bytes (BGR → PNG)."""
    from app.pipeline.io import encode_png

    return encode_png(sample_image)


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_settings_cache() -> Iterator[None]:
    """Clear the get_settings() LRU cache so tests can override env vars."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def require_matting_model():
    """Skip the test if the matting model file is not present on disk."""
    settings = get_settings()
    model_path = Path(settings.model_dir) / "u2netp.onnx"
    if not model_path.exists():
        pytest.skip(f"matting model missing: {model_path}")


@pytest.fixture
def require_edsr_x2():
    """Skip the test if the EDSR x2 model is not present on disk."""
    settings = get_settings()
    model_path = Path(settings.model_dir) / "EDSR_x2.pb"
    if not model_path.exists():
        pytest.skip(f"EDSR x2 model missing: {model_path}")


# ---------------------------------------------------------------------------
# Model cache reset (model loading is process-global)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def reset_model_caches() -> Iterator[None]:
    """Clear the matting + upscaler caches before and after a test.

    Not autouse — only request it explicitly when needed (e.g. in tests that
    mutate model state or in tests that need a clean singleton slate).
    """
    from app.models import matting
    from app.pipeline.upscale import Upscaler

    matting._singleton = None
    Upscaler.clear_cache()
    yield
    matting._singleton = None
    Upscaler.clear_cache()
