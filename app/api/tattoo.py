"""Tattoo removal endpoint (selection mask in, clean skin out)."""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api._common import read_image_bytes
from app.exceptions import DecodeError, ValidationError
from app.monitoring import image_process_seconds, image_process_total
from app.pipeline.io import decode_to_bgr, encode_png
from app.pipeline.tattoo import face_skin_hint, remove_tattoo_info

router = APIRouter()


def _face_context(img):
    """From the (largest) face: the person's skin colour, and a mask of hair/hat/clothing to leave alone.

    Both are optional: without the face models the tool still works, just with less protection."""
    try:
        from app.models.face import get_face_engine

        engine = get_face_engine()
        faces = engine.detect(img)
    except Exception:
        return None, None
    if not faces:
        return None, None
    big = max(faces, key=lambda f: f.box[2] * f.box[3])
    hint = face_skin_hint(img, big.kps)
    try:
        import cv2
        import numpy as np

        from app.pipeline import face as geo

        size = 640
        tpl = geo.head_template(size)
        crop, m = geo.align_crop(img, big.kps, tpl, size)
        labels = cv2.resize(
            engine.parse(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        covered = np.isin(labels, (2, 3, 4, 5, 6, 11, 12, 13, 17, 18)).astype(
            np.float32
        )  # brows, eyes, glasses, mouth, lips, hair, hat
        protect = (geo.warp_mask_to_image(covered, m, img.shape) > 0.5).astype(np.uint8)
    except Exception:
        protect = None
    return hint, protect


@router.post("/api/v1/tattoo")
async def tattoo(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    strength: float = Form(default=1.0),
    grow: float = Form(default=0.3),
    algorithm: str = Form(default="lama"),
) -> dict:
    """Remove the tattoo inside ``mask`` (white = search here; it may overshoot onto the background).

    Only tattoo-coloured pixels that are surrounded by skin are replaced, with the colour of the skin around them.
    ``strength`` 0..1 (1 = gone, less = faded), ``grow`` 0..1 (higher catches fainter ink and outlines).
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

    hint, protect = _face_context(img)
    started = time.perf_counter()
    try:
        result, info = remove_tattoo_info(
            img, mask_img, strength=strength, grow=grow, algorithm=algorithm
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
