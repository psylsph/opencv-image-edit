"""Selective blur endpoint: blur a user-selected region or everything outside it."""

from __future__ import annotations

import base64
import time
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api._common import read_image_bytes
from app.exceptions import DecodeError
from app.monitoring import image_process_seconds, image_process_total
from app.pipeline.background import apply_selection_blur
from app.pipeline.io import decode_to_bgr, encode_png

router = APIRouter()


@router.post("/api/v1/blur")
async def blur_selection(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    strength: int = Form(default=15, ge=1, le=50),
    target: Literal["selection", "outside"] = Form(default="selection"),
) -> dict:
    """Blur the masked region (``target=selection``) or everything else (``outside``)."""
    img_bytes = await read_image_bytes(file)
    mask_bytes = await read_image_bytes(mask)

    try:
        img = decode_to_bgr(img_bytes)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    mask_img = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise HTTPException(status_code=400, detail="invalid mask image")
    if mask_img.shape != img.shape[:2]:
        raise HTTPException(
            status_code=400,
            detail=f"mask shape {mask_img.shape} != image shape {img.shape[:2]}",
        )

    started = time.perf_counter()
    result = apply_selection_blur(img, mask_img, strength, target)
    elapsed = time.perf_counter() - started
    image_process_total.labels(status="ok").inc()
    image_process_seconds.labels(status="ok").observe(elapsed)

    png = encode_png(result)
    return {
        "final": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}",
        "strength": strength,
        "target": target,
        "elapsed_seconds": elapsed,
        "output_size": {"width": result.shape[1], "height": result.shape[0]},
    }
