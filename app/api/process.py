"""Image processing endpoint."""

from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api._common import read_image_bytes
from app.api.schemas import ProcessRequest
from app.exceptions import DecodeError, ProcessingError
from app.monitoring import (
    image_process_seconds,
    image_process_total,
    request_size_bytes,
)
from app.pipeline import ProcessResult, process_pipeline
from app.pipeline.io import decode_to_bgr, encode_png

router = APIRouter()


def _to_b64_png(img: np.ndarray | None) -> str | None:
    """Encode a BGR/BGRA image to a base64 data URL, or None if input is None."""
    if img is None:
        return None
    png_bytes = encode_png(img)
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


@router.post("/api/v1/process")
async def process(
    file: UploadFile = File(...),
    settings: str = Form(...),
) -> dict:
    """Process an uploaded image through the pipeline.

    Args:
        file: image file (PNG, JPEG, WebP, HEIC).
        settings: JSON-encoded ProcessRequest.

    Returns:
        JSON with base64-encoded PNGs for final + 5 debug outputs.
    """
    body = await read_image_bytes(file)
    request_size_bytes.observe(len(body))

    try:
        req = ProcessRequest.model_validate_json(settings)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid settings: {exc}") from exc

    try:
        img_bgr = decode_to_bgr(body)
    except DecodeError as exc:
        image_process_total.labels(status="decode_error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    status = "ok"
    try:
        result: ProcessResult = process_pipeline(img_bgr, req)
    except ProcessingError as exc:
        status = "processing_error"
        image_process_total.labels(status=status).inc()
        image_process_seconds.labels(status=status).observe(time.perf_counter() - started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = time.perf_counter() - started
    image_process_seconds.labels(status=status).observe(elapsed)
    image_process_total.labels(status=status).inc()

    return {
        "final": _to_b64_png(result.final),
        "before_after": _to_b64_png(result.before_after),
        "diff": _to_b64_png(result.diff),
        "mask": _to_b64_png(result.mask),
        "grain": _to_b64_png(result.grain_only),
        "upscaled": _to_b64_png(result.upscaled),
        "elapsed_seconds": elapsed,
        "output_size": {"width": result.final.shape[1], "height": result.final.shape[0]},
    }
