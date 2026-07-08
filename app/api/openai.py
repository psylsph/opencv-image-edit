"""OpenAI provider status endpoint.

Checks whether the OpenAI API key is configured and the cloud
generative inpaint is available.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


@router.get("/api/v1/openai/status")
def openai_status() -> dict:
    """Check if OpenAI inpainting is configured and available."""
    settings = get_settings()
    has_key = bool(settings.openai_api_key)
    return {
        "available": has_key,
        "model": settings.openai_model,
    }
