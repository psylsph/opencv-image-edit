"""Object removal / inpainting endpoint."""
from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.exceptions import DecodeError, ModelNotFoundError, ValidationError
from app.monitoring import image_process_seconds, image_process_total
from app.pipeline.inpaint import inpaint
from app.pipeline.io import decode_to_bgr, encode_png

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/api/v1/inpaint")
async def inpaint_endpoint(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    radius: int = Form(default=3),
    algorithm: str = Form(default="lama"),
    iterations: int = Form(default=1),
    prompt: str = Form(default=""),
) -> dict:
    """Remove the masked region and fill it in with surrounding content.

    algorithm: "lama" (default, fast AI), "sd" (local generative, slow),
               "telea" (fast), or "ns" (smoother).
    prompt: Text description for generative fill (algorithm="sd").
    """
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

    started = time.perf_counter()
    status = "ok"

    # Log mask diagnostics for debugging
    mask_ratio = float((mask_img > 0).mean())
    logger.info(
        "inpaint: img=%dx%d mask=%dx%d mask_coverage=%.1f%% algo=%s",
        img.shape[1], img.shape[0], mask_img.shape[1], mask_img.shape[0],
        mask_ratio * 100, algorithm,
    )

    try:
        result = inpaint(img, mask_img, radius=radius, algorithm=algorithm,
                         iterations=iterations, prompt=prompt)
    except ModelNotFoundError as exc:
        # LaMa model missing — count separately so /health + metrics reflect it
        image_process_total.labels(status="model_missing").inc()
        image_process_seconds.labels(status="model_missing").observe(
            time.perf_counter() - started
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        status = "inpaint_error"
        image_process_total.labels(status=status).inc()
        image_process_seconds.labels(status=status).observe(time.perf_counter() - started)
        raise HTTPException(status_code=500, detail=f"inpaint failed: {exc}") from exc
    image_process_total.labels(status=status).inc()
    image_process_seconds.labels(status=status).observe(time.perf_counter() - started)

    png = encode_png(result)
    return {
        "final": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}",
        "radius": radius,
        "algorithm": algorithm,
        "iterations": iterations,
        "elapsed_seconds": time.perf_counter() - started,
        "output_size": {"width": result.shape[1], "height": result.shape[0]},
    }
