"""Tests for app.config settings module."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Default values are correct for all env-controlled fields."""
    s = Settings()
    # Server
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.debug is False
    assert s.log_level == "INFO"
    # Image limits
    assert s.max_image_size_mb == 10
    assert s.max_image_dimension == 1536
    # Model paths
    assert s.model_dir == Path("./models").expanduser().resolve()
    # Rate limiting
    assert s.rate_limit_requests == 10
    assert s.rate_limit_period == 60
    # Metrics
    assert s.enable_metrics is True
    assert s.metrics_port == 9090
    # Pipeline limits
    assert s.blur_strength_max == 50
    assert s.grain_intensity_max == 1.0
    # App metadata
    assert s.app_version == "1.0.0"
    assert s.app_name == "opencv-image-edit"


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override defaults."""
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_IMAGE_SIZE_MB", "25")
    monkeypatch.setenv("MAX_IMAGE_DIMENSION", "2048")
    monkeypatch.setenv("MODEL_DIR", "/tmp/models")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "50")
    monkeypatch.setenv("RATE_LIMIT_PERIOD", "30")
    monkeypatch.setenv("ENABLE_METRICS", "false")
    monkeypatch.setenv("METRICS_PORT", "9100")
    monkeypatch.setenv("BLUR_STRENGTH_MAX", "40")
    monkeypatch.setenv("GRAIN_INTENSITY_MAX", "0.75")
    monkeypatch.setenv("APP_VERSION", "2.0.0-rc1")
    monkeypatch.setenv("APP_NAME", "custom-app")
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 9000
    assert s.debug is True
    assert s.log_level == "DEBUG"
    assert s.max_image_size_mb == 25
    assert s.max_image_dimension == 2048
    assert s.model_dir == Path("/tmp/models")
    assert s.rate_limit_requests == 50
    assert s.rate_limit_period == 30
    assert s.enable_metrics is False
    assert s.metrics_port == 9100
    assert s.blur_strength_max == 40
    assert s.grain_intensity_max == 0.75
    assert s.app_version == "2.0.0-rc1"
    assert s.app_name == "custom-app"


def test_settings_cached() -> None:
    """get_settings() returns the same instance on repeated calls (lru_cache)."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_model_dir_is_path() -> None:
    """MODEL_DIR is a Path instance after instantiation."""
    s = Settings()
    assert isinstance(s.model_dir, Path)


def test_settings_invalid_max_image_size() -> None:
    """ValueError when MAX_IMAGE_SIZE_MB <= 0 (gt=0 constraint)."""
    with pytest.raises(ValidationError):
        Settings(max_image_size_mb=0)
    with pytest.raises(ValidationError):
        Settings(max_image_size_mb=-1)


def test_settings_invalid_blur_max() -> None:
    """ValueError when blur_strength_max > 50 (outside the UI slider range).

    The field is configured with ``le=100``, so values strictly greater than 50
    (and up to 100) are accepted by the field constraint, but the slider
    contract demands a hard ceiling at 50. The simplest expression of that is
    to confirm that any value above 50 is rejected at construction time, and
    that the boundary 50 itself remains valid.
    """
    # Boundary value is still valid.
    s = Settings(blur_strength_max=50)
    assert s.blur_strength_max == 50
    # Values just above the UI ceiling are rejected.
    with pytest.raises(ValidationError):
        Settings(blur_strength_max=51)
