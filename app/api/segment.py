"""Point-prompt object segmentation endpoint (MobileSAM)."""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.exceptions import DecodeError, ModelNotFoundError
from app.monitoring import image_process_seconds, image_process_total
from app.models.sam import segment_with_point
from app.pipeline.io import decode_to_bgr, encode_png


router = APIRouter()

# MobileSAM boundaries can land a few pixels inside the object. Expand click
# selections so edge colour/fringing is included in the inpainted area.
_SELECTION_BORDER_PX = 8


def _expand_selection_mask(mask: np.ndarray, border_px: int = _SELECTION_BORDER_PX) -> np.ndarray:
    """Return a rounded, outward expansion of a binary selection mask."""
    if border_px <= 0:
        return mask.copy()
    ksize = border_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel)


@router.post("/api/v1/segment")
async def segment(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
) -> dict:
    """Segment the object at point (x, y) in the image.

    Returns:
        JSON with the binary mask (PNG, base64 data URL) + IoU score.
    """
    body = await file.read()
    try:
        img = decode_to_bgr(body)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise HTTPException(
            status_code=422,
            detail=f"point ({x},{y}) out of image bounds {w}x{h}",
        )

    status = "ok"
    started = time.perf_counter()
    try:
        mask, score = segment_with_point(img, (x, y))
        mask = _expand_selection_mask(mask)
    except ModelNotFoundError:
        # Don't count as a 500 or "segment_error" — model availability is
        # an operational concern, not a request failure. Let the AppError
        # exception handler convert this to 503.
        image_process_total.labels(status="model_missing").inc()
        image_process_seconds.labels(status="model_missing").observe(
            time.perf_counter() - started
        )
        raise
    except Exception as exc:
        status = "segment_error"
        image_process_total.labels(status=status).inc()
        image_process_seconds.labels(status=status).observe(time.perf_counter() - started)
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    image_process_total.labels(status=status).inc()
    image_process_seconds.labels(status=status).observe(elapsed)

    # Encode mask as PNG
    mask_png = encode_png(mask)
    # Compose a preview = image with mask overlaid (red translucent on masked area)
    red_layer = np.zeros_like(img)
    red_layer[:, :] = (60, 60, 230)  # BGR red
    alpha = (mask > 0).astype(np.float32)[:, :, np.newaxis] * 0.45
    overlay = (
        img.astype(np.float32) * (1 - alpha) + red_layer.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_png = encode_png(overlay)

    return {
        "mask": f"data:image/png;base64,{base64.b64encode(mask_png).decode('ascii')}",
        "overlay": f"data:image/png;base64,{base64.b64encode(overlay_png).decode('ascii')}",
        "score": score,
        "selection_border_px": _SELECTION_BORDER_PX,
        "elapsed_seconds": elapsed,
        "mask_area_pct": float((mask > 0).mean()),
        "point": {"x": x, "y": y},
    }
