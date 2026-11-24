"""Pydantic schemas for the process API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BackgroundRequest(BaseModel):
    enabled: bool = False
    mode: Literal["blur", "remove", "color", "image"] = "blur"
    blur_strength: int = Field(default=15, ge=1, le=50)
    model_name: str = "u2netp"
    # Replacement background (modes "color" and "image"; "image" needs the ``background_image`` upload).
    color: str = Field(default="#00b140", pattern=r"^#[0-9a-fA-F]{6}$")
    zoom: float = Field(default=1.0, ge=1.0, le=4.0)
    offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    feather: float = Field(default=0.0, ge=0.0, le=1.0)  # soften the subject's edge
    shrink: float = Field(default=0.0, ge=-1.0, le=1.0)  # > 0 edge in, < 0 edge out
    defringe: float = Field(
        default=0.0, ge=0, le=1
    )  # remove old-background colour from the subject's edge


class FaceRequest(BaseModel):
    """Replace one face in the photo with a face from a donor photo (needs the ``face_donor`` upload)."""

    enabled: bool = False
    target: int = Field(default=0, ge=0, le=63)  # which face in the photo, counted left to right
    donor: int = Field(
        default=0, ge=0, le=63
    )  # which face in the donor photo, counted left to right
    blend: float = Field(default=0.5, ge=0, le=1)  # edge softness
    color_match: float = Field(
        default=0.8, ge=0, le=1
    )  # pull the new face's colour toward the photo's
    restore: float = Field(default=0.3, ge=0, le=1)  # face restorer strength (0 = off)
    hair: bool = False  # also bring the donor's hair (needs the face-parsing model)
    detail: float = Field(
        default=0.8, ge=0, le=1.5
    )  # give the new face the photo's sharpness and grain
    shape: float = Field(default=0.8, ge=0, le=1)  # 0 keep their face shape, 1 take the donor's


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
    anime: bool = False
    anime_style: Literal["hayao", "shinkai", "paprika", "hayao3", "jp_face"] = "hayao"
    painting: bool = False
    painting_style: Literal["oil", "painterly", "pointillism"] = "oil"


class ProcessRequest(BaseModel):
    """Top-level request model that bundles all stage sub-requests together."""

    face: FaceRequest = Field(default_factory=FaceRequest)
    background: BackgroundRequest = Field(default_factory=BackgroundRequest)
    grain: GrainRequest = Field(default_factory=GrainRequest)
    upscale: UpscaleRequest = Field(default_factory=UpscaleRequest)
    filters: FiltersRequest = Field(default_factory=FiltersRequest)
