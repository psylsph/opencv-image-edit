"""Pydantic schemas for the process API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BackgroundRequest(BaseModel):
    enabled: bool = False
    mode: Literal["blur", "remove"] = "blur"
    blur_strength: int = Field(default=15, ge=1, le=50)
    model_name: str = "birefnet_tiny"


class GrainRequest(BaseModel):
    enabled: bool = False
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)


class UpscaleRequest(BaseModel):
    enabled: bool = False
    scale: Literal[1, 2, 4] = 2
    algorithm: Literal["edsr", "espcn", "fsrcnn", "lapsrn", "interp"] = "interp"


class FiltersRequest(BaseModel):
    enabled: bool = False
    brightness: float = Field(default=1.0, ge=0.0, le=2.0)
    contrast: float = Field(default=1.0, ge=0.0, le=2.0)
    saturation: float = Field(default=1.0, ge=0.0, le=2.0)
    sharpness: float = Field(default=1.0, ge=0.0, le=2.0)
    vignette_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    sepia: bool = False
    grayscale_blend: float = Field(default=0.0, ge=0.0, le=1.0)
    cartoon: bool = False
    cartoon_color_levels: int = Field(default=8, ge=2, le=32)
    cartoon_edge_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    cartoon_smooth_passes: int = Field(default=2, ge=0, le=4)


class ProcessRequest(BaseModel):
    """Top-level request model that bundles all stage sub-requests together."""

    background: BackgroundRequest = Field(default_factory=BackgroundRequest)
    grain: GrainRequest = Field(default_factory=GrainRequest)
    upscale: UpscaleRequest = Field(default_factory=UpscaleRequest)
    filters: FiltersRequest = Field(default_factory=FiltersRequest)
