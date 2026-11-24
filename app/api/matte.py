"""Matte endpoint: just the subject mask, so the browser can composite backgrounds itself.

The matte is the expensive part of a background edit and it doesn't change when the user moves a blur slider or
swaps the background, so the client fetches it once per photo + model and does the rest on the phone (web/gl.js).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.api._common import read_image_bytes
from app.config import get_settings
from app.exceptions import DecodeError
from app.models.matting import get_matting_model
from app.monitoring import image_process_seconds, image_process_total
from app.pipeline.io import decode_to_bgr
from app.pipeline.preprocess import resize_if_needed

router = APIRouter()


@router.post("/api/v1/matte")
async def matte(
    file: UploadFile = File(...),
    model_name: str = Form("birefnet_tiny"),
) -> Response:
    """Return the foreground matte as an 8-bit grayscale PNG (255 = subject, 0 = background).

    The matte has the size ``/api/v1/process`` works at (the photo shrunk to ``MAX_IMAGE_DIMENSION``), so a client
    compositing with it sees the same picture the server would.
    """
    body = await read_image_bytes(file)
    try:
        img = decode_to_bgr(body)
    except DecodeError as exc:
        image_process_total.labels(status="decode_error").inc()
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    started = time.perf_counter()
    img = resize_if_needed(img, get_settings().max_image_dimension)
    mask = get_matting_model(model_name).predict(img)
    image_process_seconds.labels(status="ok").observe(time.perf_counter() - started)
    image_process_total.labels(status="ok").inc()

    ok, png = cv2.imencode(".png", np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if not ok:  # pragma: no cover - imencode on a uint8 plane doesn't fail
        raise HTTPException(status_code=500, detail="could not encode matte")
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Matte-Width": str(img.shape[1]),
            "X-Matte-Height": str(img.shape[0]),
        },
    )
