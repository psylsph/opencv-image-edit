"""AnimeGANv2 style-transfer models via ONNX Runtime.

Turns photos into anime/cartoon styles (Hayao, Shinkai, Paprika).
Models are lazy-loaded singletons, mirroring the matting registry pattern.

All AnimeGANv2 ONNX exports share the same I/O contract:
    input:  float32 NHWC, RGB, values in [-1, 1] ((pixel - 127.5) / 128)
    output: float32 NHWC, RGB, values in [-1, 1] (same de-normalisation)

Images are resized so H and W are multiples of 8 (network requirement),
processed, then resized back to the original dimensions.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


class AnimeStyleInfo(NamedTuple):
    """Static metadata for an AnimeGANv2 style variant."""

    filename: str
    display_name: str
    description: str
    size_mb: float


# Registry of supported styles. ``key`` is what the API/UI uses.
ANIME_STYLES: dict[str, AnimeStyleInfo] = {
    "hayao": AnimeStyleInfo(
        filename="AnimeGANv2_Hayao.onnx",
        display_name="Hayao",
        description="8.6 MB · classic anime, soft pastel colours",
        size_mb=8.6,
    ),
    "shinkai": AnimeStyleInfo(
        filename="AnimeGANv2_Shinkai.onnx",
        display_name="Shinkai",
        description="8.6 MB · vivid blues, 'Your Name' look",
        size_mb=8.6,
    ),
    "paprika": AnimeStyleInfo(
        filename="AnimeGANv2_Paprika.onnx",
        display_name="Paprika",
        description="8.6 MB · bold saturated colours",
        size_mb=8.6,
    ),
    "hayao3": AnimeStyleInfo(
        filename="AnimeGANv3_Hayao_36.onnx",
        display_name="Hayao v3",
        description="4.2 MB · Ghibli-style, soft hand-drawn 90s look",
        size_mb=4.2,
    ),
    "jp_face": AnimeStyleInfo(
        filename="AnimeGANv3_JP_face.onnx",
        display_name="JP Face",
        description="6.1 MB · stylised anime faces, cel-shaded 3D-cartoon feel",
        size_mb=6.1,
    ),
}

_SESSIONS: dict[str, ort.InferenceSession] = {}
_LOCK = Lock()

from app.config import get_settings

_MODELS_DIR = Path(get_settings().model_dir)


def get_anime_session(style: str) -> ort.InferenceSession:
    """Return (and cache) an ONNX session for the given style key."""
    if style not in ANIME_STYLES:
        raise ModelNotFoundError(
            f"unknown anime style '{style}'; valid: {sorted(ANIME_STYLES)}"
        )
    with _LOCK:
        if style not in _SESSIONS:
            path = _MODELS_DIR / ANIME_STYLES[style].filename
            if not path.exists():
                raise ModelNotFoundError(
                    f"model file not found: {path} (download it into models/)"
                )
            _SESSIONS[style] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        return _SESSIONS[style]


def apply_anime_style(img: np.ndarray, style: str) -> np.ndarray:
    """Run AnimeGANv2 style transfer on a BGR uint8 image.

    Returns a BGR uint8 image of the same dimensions.
    """
    session = get_anime_session(style)
    input_name = session.get_inputs()[0].name

    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]

    # Network requires H/W multiples of 8; avoid upscaling small inputs
    target_w = max(8, (w + 7) // 8 * 8)
    target_h = max(8, (h + 7) // 8 * 8)
    resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # BGR -> RGB, normalise to [-1, 1], NHWC
    x = ((resized[..., ::-1].astype(np.float32) - 127.5) / 128.0)[None]
    y = session.run(None, {input_name: x})[0][0]

    # De-normalise, RGB -> BGR
    out = np.clip(y * 128.0 + 127.5, 0, 255).astype(np.uint8)[..., ::-1]
    if (target_h, target_w) != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
