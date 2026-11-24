"""Face detection endpoint: where the faces are, so the UI can let the user pick one."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api._common import read_image_bytes
from app.config import get_settings
from app.exceptions import DecodeError
from app.models.face import face_models_status, get_face_engine
from app.pipeline.io import decode_to_bgr
from app.pipeline.preprocess import resize_if_needed

router = APIRouter()


@router.get("/api/v1/faces/status")
def faces_status() -> dict:
    """Whether the face models are installed (``ready``) and whether the optional restorer is."""
    return face_models_status()


@router.post("/api/v1/faces")
async def faces(file: UploadFile = File(...)) -> dict:
    """Detect faces, ordered left to right (the same order ``face.target`` / ``face.donor`` count in).

    Coordinates are in the frame ``/api/v1/process`` works at (the photo shrunk to ``MAX_IMAGE_DIMENSION``);
    ``width``/``height`` give that frame's size so a client can scale them onto its own preview.
    """
    body = await read_image_bytes(file)
    try:
        img = decode_to_bgr(body)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = resize_if_needed(img, get_settings().max_image_dimension)
    found = get_face_engine().detect(img)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "faces": [
            {
                "box": [round(v, 1) for v in f.box],
                "score": round(f.score, 3),
                "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in f.kps],
            }
            for f in found
        ],
    }
