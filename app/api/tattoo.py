"""Tattoo removal endpoint (selection mask in, clean skin out)."""

from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api._common import read_image_bytes
from app.exceptions import DecodeError, ModelNotFoundError, ValidationError
from app.monitoring import image_process_seconds, image_process_total
from app.pipeline.io import decode_to_bgr, encode_png
from app.pipeline.tattoo import remove_tattoo_info, remove_tattoo_lama

logger = logging.getLogger(__name__)
router = APIRouter()


def _person_mask(img):
    """The person (foreground) from the matting model, as a 0/1 mask, or None if the model is unavailable.

    Used to keep the tattoo tool off the backdrop: however skin-like a wall or curtain is, it is never sampled or painted.
    """
    try:
        from app.models.matting import get_matting_model

        alpha = get_matting_model("u2netp").predict(img)
    except Exception:  # no model, or it failed: the tool still works, without the extra protection
        return None
    fg = (alpha > 0.4).astype(np.uint8)
    if (
        fg.mean() < 0.02
    ):  # the model found no subject at all: do not let that blank out the whole photo
        return None
    return cv2.dilate(
        fg, np.ones((3, 3), np.uint8), iterations=2
    )  # keep a little margin at the person's edge


@router.post("/api/v1/tattoo")
async def tattoo(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    strength: float = Form(default=1.0),
    grow: float = Form(default=0.5),
    algorithm: str = Form(default="lama"),
) -> dict:
    """Remove the tattoo inside ``mask`` (white = the tattoo; it may overshoot onto the background).

    ``algorithm="lama"`` (default): the backdrop is temporarily replaced by skin colour and the selected skin is removed
    with AI object removal plus matching skin grain, then the backdrop is restored. Any other value uses the colour algorithm, where
    ``grow`` 0..1 is the sensitivity (higher catches fainter ink). ``strength`` 0..1 (1 = gone, less = faded).
    """
    try:
        img = decode_to_bgr(await read_image_bytes(file))
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    mask_img = cv2.imdecode(
        np.frombuffer(await read_image_bytes(mask), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if mask_img is None:
        raise HTTPException(status_code=400, detail="invalid mask image")
    if mask_img.shape != img.shape[:2]:
        raise HTTPException(
            status_code=400, detail=f"mask shape {mask_img.shape} != image shape {img.shape[:2]}"
        )
    if not (0.0 <= strength <= 1.0 and 0.0 <= grow <= 1.0):
        raise HTTPException(status_code=422, detail="strength and grow must be between 0 and 1")
    if not mask_img.any():
        raise HTTPException(
            status_code=422, detail="select the tattoo first (tap it or paint over it)"
        )

    foreground = _person_mask(img)
    started = time.perf_counter()
    try:
        info = None
        if algorithm == "lama":
            try:
                result, info = remove_tattoo_lama(
                    img, mask_img, strength=strength, foreground=foreground
                )
            except ModelNotFoundError:  # no LaMa model: fall back to the colour algorithm
                logger.warning("tattoo: LaMa model missing, using the colour algorithm")
        if info is None:
            result, info = remove_tattoo_info(
                img, mask_img, strength=strength, grow=grow, foreground=foreground
            )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc
    except Exception as exc:
        image_process_total.labels(status="tattoo_error").inc()
        raise HTTPException(status_code=500, detail=f"tattoo removal failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    image_process_total.labels(status="ok").inc()
    image_process_seconds.labels(status="ok").observe(elapsed)

    png = encode_png(result)
    return {
        "final": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}",
        "mode": info["mode"],
        "ink_pct": info["ink_pct"],
        "passes": info.get("passes", 1),
        "elapsed_seconds": elapsed,
        "output_size": {"width": result.shape[1], "height": result.shape[0]},
    }
