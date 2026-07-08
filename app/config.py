"""Application configuration via pydantic-settings.

All settings are read from environment variables (or .env file).
Use `get_settings()` to access the cached singleton.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # Image limits
    max_image_size_mb: int = Field(default=10, gt=0, le=100)
    max_image_dimension: int = Field(default=1536, gt=64, le=8192)

    # Model paths
    model_dir: Path = Path("./models")

    # Rate limiting (per-IP, requests per period)
    rate_limit_requests: int = Field(default=10, gt=0, le=1000)
    rate_limit_period: int = Field(default=60, gt=0)

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Pipeline limits
    blur_strength_max: int = Field(default=50, gt=0, le=50)
    grain_intensity_max: float = Field(default=1.0, gt=0.0, le=2.0)

    # App metadata
    app_version: str = "1.1.0"
    app_name: str = "opencv-image-edit"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_upper

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
