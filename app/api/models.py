"""List available matting models for background removal."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.matting import available_models

router = APIRouter()


@router.get("/api/v1/models")
def list_models() -> dict:
    """Return all registered background-removal models and their availability."""
    return {"models": available_models()}
