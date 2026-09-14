"""Lightweight health check endpoint.

This route is polled by Docker every 30 seconds.  It must only inspect model
files; constructing inference sessions here would keep several gigabytes of
models resident while the editor is idle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import APIRouter

from app.config import get_settings
from app.models.lama import _MODEL_FILENAME
from app.models.matting import DEFAULT_MODEL, MATTING_MODELS

router = APIRouter()


def _file_status(path: Path) -> str:
    return "available" if path.is_file() else "missing: FileNotFoundError"


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    models_status = {
        "matting": _file_status(model_dir / MATTING_MODELS[DEFAULT_MODEL].filename),
        "inpaint_lama": _file_status(model_dir / _MODEL_FILENAME),
        "upscale_edsr_x2": _file_status(model_dir / "EDSR_x2.pb"),
        "upscale_edsr_x4": _file_status(model_dir / "EDSR_x4.pb"),
    }
    any_missing = any(status.startswith("missing") for status in models_status.values())
    return {
        "status": "degraded" if any_missing else "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "opencv_version": cv2.__version__,
        "models": models_status,
        "model_idle_timeout_seconds": settings.model_idle_timeout_seconds,
    }
