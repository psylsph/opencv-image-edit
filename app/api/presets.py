"""Presets endpoint — returns the 4 default presets and their settings."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import (
    BackgroundRequest,
    FiltersRequest,
    GrainRequest,
    ProcessRequest,
    UpscaleRequest,
)

router = APIRouter()


PRESETS: dict[str, dict] = {
    "portrait": {
        "label": "👤 Portrait",
        "description": "Blur background + light grain + 2x upscale + slight color punch",
        "settings": ProcessRequest(
            background=BackgroundRequest(enabled=True, mode="blur", blur_strength=20),
            grain=GrainRequest(enabled=True, intensity=0.2),
            upscale=UpscaleRequest(enabled=True, scale=2, algorithm="interp"),
            filters=FiltersRequest(enabled=True, brightness=1.05, contrast=1.05, saturation=1.1, sharpness=1.1),
        ),
    },
    "landscape": {
        "label": "🏞️ Landscape",
        "description": "Vivid colors + auto-enhance + 2x upscale",
        "settings": ProcessRequest(
            grain=GrainRequest(enabled=False),
            upscale=UpscaleRequest(enabled=True, scale=2, algorithm="interp"),
            filters=FiltersRequest(enabled=True, brightness=1.0, contrast=1.15, saturation=1.25, sharpness=1.2),
        ),
    },
    "vintage": {
        "label": "🎬 Vintage",
        "description": "Strong grain + sepia + vignette + 1x (no upscale)",
        "settings": ProcessRequest(
            grain=GrainRequest(enabled=True, intensity=0.5),
            upscale=UpscaleRequest(enabled=False),
            filters=FiltersRequest(enabled=True, sepia=True, vignette_strength=0.6, contrast=1.1),
        ),
    },
}


@router.get("/presets")
def presets() -> dict:
    return {
        name: {"label": meta["label"], "description": meta["description"], "settings": meta["settings"].model_dump()}
        for name, meta in PRESETS.items()
    }
