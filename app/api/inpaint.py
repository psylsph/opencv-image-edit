"""Object removal / inpainting endpoint."""
from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.exceptions import DecodeError, ValidationError
from app.pipeline.inpaint import inpaint
from app.pipeline.io import decode_to_bgr, encode_png


router = APIRouter()


@router.post("/api/v1/inpaint")
async def inpaint_endpoint(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    radius: int = Form(default=3),
    algorithm: str = Form(default="telea"),
) -> dict:
    """Remove the masked region and fill it in with surrounding content."""
    img_bytes = await file.read()
    mask_bytes = await mask.read()

    try:
        img = decode_to_bgr(img_bytes)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    # Mask must be single-channel uint8
    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise HTTPException(status_code=400, detail="invalid mask image")
    if mask_img.shape != img.shape[:2]:
        raise HTTPException(
            status_code=400,
            detail=f"mask shape {mask_img.shape} != image shape {img.shape[:2]}",
        )

    try:
        result = inpaint(img, mask_img, radius=radius, algorithm=algorithm)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inpaint failed: {exc}") from exc

    png = encode_png(result)
    return {
        "final": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}",
        "radius": radius,
        "algorithm": algorithm,
        "output_size": {"width": result.shape[1], "height": result.shape[0]},
    }
