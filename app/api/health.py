"""Health check endpoint."""

from __future__ import annotations

import cv2
from fastapi import APIRouter

from app.config import get_settings
from app.models.matting import get_matting_model

router = APIRouter()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    models_status = {}
    # Check matting model
    try:
        get_matting_model(settings.model_dir)
        models_status["matting"] = "loaded"
    except Exception as exc:
        models_status["matting"] = f"missing: {exc.__class__.__name__}"
    # LaMa inpainting model
    try:
        from app.models.lama import LaMa

        LaMa.get(settings.model_dir)
        models_status["inpaint_lama"] = "loaded"
    except Exception as exc:
        models_status["inpaint_lama"] = f"missing: {exc.__class__.__name__}"
    # Upscale models
    from app.pipeline.upscale import Upscaler

    for algo in ("edsr",):
        for scale in (2, 4):
            key = f"upscale_{algo}_x{scale}"
            try:
                Upscaler.get(algo, scale, settings.model_dir)
                models_status[key] = "loaded"
            except Exception as exc:
                models_status[key] = f"missing: {exc.__class__.__name__}"
    # Overall status reflects model availability. We still return HTTP 200 so
    # the Docker healthcheck (which only checks the status code) treats a
    # missing-optional-model state as live; "degraded" is informational for
    # readiness checks and the UI.
    any_missing = any(
        isinstance(v, str) and v.startswith("missing") for v in models_status.values()
    )
    return {
        "status": "degraded" if any_missing else "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "opencv_version": cv2.__version__,
        "models": models_status,
    }
