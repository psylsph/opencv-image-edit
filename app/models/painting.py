"""Fast Neural Style painting models via ONNX Runtime.

Turns photos into painterly styles (Oil Paint, Painterly Modern, Pointillism)
using the ONNX Model Zoo fast_neural_style transformer networks.

All three exports share the same I/O contract:
    input:  float32 NCHW, RGB, values in [0, 255]
    output: float32 NCHW, RGB, values in [0, 255] (clamp to be safe)

The networks expect exactly 224x224 input, so larger images are processed
as overlapping tiles and blended back together to avoid visible seams.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


class PaintStyleInfo(NamedTuple):
    """Static metadata for a painting style variant."""

    filename: str
    display_name: str
    description: str
    size_mb: float


# Registry of supported styles. ``key`` is what the API/UI uses.
PAINT_STYLES: dict[str, PaintStyleInfo] = {
    "oil": PaintStyleInfo(
        filename="fast_neural_style_rain_princess-9.onnx",
        display_name="Oil Paint",
        description="6.7 MB · rich oil-painting brushwork (Rain Princess)",
        size_mb=6.7,
    ),
    "painterly": PaintStyleInfo(
        filename="fast_neural_style_udnie-9.onnx",
        display_name="Painterly",
        description="6.7 MB · soft modern painterly look (Udnie)",
        size_mb=6.7,
    ),
    "pointillism": PaintStyleInfo(
        filename="fast_neural_style_pointilism-9.onnx",
        display_name="Pointillism",
        description="6.7 MB · dotted impressionist texture",
        size_mb=6.7,
    ),
}

_SESSIONS: dict[str, ort.InferenceSession] = {}
_LOCK = Lock()

from app.config import get_settings

_MODELS_DIR = Path(get_settings().model_dir)

_TILE = 224          # network input size (fixed)
_OVERLAP = 32        # pixels of overlap between tiles


def get_paint_session(style: str) -> ort.InferenceSession:
    """Return (and cache) an ONNX session for the given style key."""
    from app.model_cache import mark_model_used

    mark_model_used()
    if style not in PAINT_STYLES:
        raise ModelNotFoundError(
            f"unknown painting style '{style}'; valid: {sorted(PAINT_STYLES)}"
        )
    with _LOCK:
        if style not in _SESSIONS:
            path = _MODELS_DIR / PAINT_STYLES[style].filename
            if not path.exists():
                raise ModelNotFoundError(
                    f"model file not found: {path} (download it into models/)"
                )
            _SESSIONS[style] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        return _SESSIONS[style]


def clear_paint_cache() -> None:
    """Release all cached painting-style ONNX sessions."""
    with _LOCK:
        _SESSIONS.clear()


def _run_tile(session: ort.InferenceSession, tile_bgr: np.ndarray) -> np.ndarray:
    """Stylise one 224x224 BGR uint8 tile, returning BGR uint8."""
    input_name = session.get_inputs()[0].name
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = rgb.transpose(2, 0, 1)[None]  # HWC -> 1CHW
    y = session.run(None, {input_name: x})[0][0].transpose(1, 2, 0)
    out = np.clip(y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _blend_weights(size: int) -> np.ndarray:
    """1-D linear ramp weights (0..1) for feathering tile edges."""
    w = np.ones(size, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, _OVERLAP, dtype=np.float32)
    w[:_OVERLAP] = np.minimum(w[:_OVERLAP], ramp)
    w[-_OVERLAP:] = np.minimum(w[-_OVERLAP:], ramp[::-1])
    return w


def apply_paint_style(img: np.ndarray, style: str) -> np.ndarray:
    """Run fast-neural-style painting transfer on a BGR uint8 image.

    Tiles the image with overlap, stylises each 224x224 tile, and
    feather-blends the results. Returns a BGR uint8 image of the same
    dimensions.
    """
    session = get_paint_session(style)

    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]

    # Small images: pad up to one tile, stylise, crop back
    if h <= _TILE and w <= _TILE:
        pw, ph = max(w, _TILE), max(h, _TILE)
        padded = cv2.copyMakeBorder(
            bgr, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101
        )
        out = _run_tile(session, padded[:_TILE, :_TILE])
        return out[:h, :w].copy()

    stride = _TILE - _OVERLAP
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    wx = _blend_weights(min(_TILE, w))[None, :, None]
    wy = _blend_weights(min(_TILE, h))[:, None, None]
    tile_w = (wx * wy).astype(np.float32)  # HxWx1 feather mask

    ys = list(range(0, max(h - _TILE, 0) + 1, stride))
    xs = list(range(0, max(w - _TILE, 0) + 1, stride))
    if ys[-1] + _TILE < h:
        ys.append(h - _TILE)
    if xs[-1] + _TILE < w:
        xs.append(w - _TILE)

    for y0 in ys:
        for x0 in xs:
            tile = bgr[y0 : y0 + _TILE, x0 : x0 + _TILE]
            stylised = _run_tile(session, tile)
            th, tw = stylised.shape[:2]
            acc[y0 : y0 + th, x0 : x0 + tw] += stylised * tile_w[:th, :tw]
            weight[y0 : y0 + th, x0 : x0 + tw] += tile_w[:th, :tw, 0]

    weight = np.maximum(weight, 1e-6)[:, :, None]
    return (acc / weight).round().astype(np.uint8)
