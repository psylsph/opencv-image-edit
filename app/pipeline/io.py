"""Image decode/encode helpers (HEIC + standard formats).

All decode functions return BGR (or BGRA) numpy arrays — the format OpenCV
uses internally. All encode functions accept BGR/BGRA and return bytes.

Pillow is used ONLY as a HEIC/HEIF decoder shim. For all other formats we
use cv2.imdecode/cv2.imencode directly to keep OpenCV as the single source
of truth for image I/O.
"""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from app.exceptions import DecodeError, EncodeError


# Register HEIC opener once at import time
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - pillow-heif is in requirements
    pass


_HEIF_MAGIC = b"ftypheic"
_HEIF_MAGIC_ALT = b"ftypheix"
_HEIF_MAGIC_HIF = b"ftypmif1"


def _is_heic(data: bytes) -> bool:
    """Detect HEIC/HEIF by magic bytes (offset 4-11)."""
    if len(data) < 12:
        return False
    magic = data[4:12]
    return any(m in magic for m in (_HEIF_MAGIC, _HEIF_MAGIC_ALT, _HEIF_MAGIC_HIF))


def decode_to_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR numpy array.

    Supports PNG, JPEG, WebP, BMP, TIFF, GIF (first frame), and HEIC/HEIF
    via pillow-heif. Returns uint8 array with shape (H, W, 3) for BGR or
    (H, W, 4) for BGRA.

    Raises:
        DecodeError: If the data cannot be decoded as any supported format.
    """
    if not data:
        raise DecodeError("empty image data")

    if _is_heic(data):
        try:
            pil_img = Image.open(BytesIO(data))
            pil_img = pil_img.convert(
                "RGBA" if pil_img.mode in ("RGBA", "LA", "P") else "RGB"
            )
            arr_rgb = np.array(pil_img)
            if arr_rgb.ndim == 3 and arr_rgb.shape[2] == 4:
                return cv2.cvtColor(arr_rgb, cv2.COLOR_RGBA2BGRA)
            return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise DecodeError(f"failed to decode HEIC image: {exc}") from exc

    # Standard formats via OpenCV
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise DecodeError(
            "failed to decode image (unsupported format or corrupt data)"
        )
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def encode_png(img: np.ndarray) -> bytes:
    """Encode a BGR/BGRA image to PNG bytes."""
    return _encode(img, ".png", ())


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """Encode a BGR image to JPEG bytes (alpha will be dropped).

    Args:
        img: BGR numpy array (uint8). 4-channel BGRA is converted to BGR.
        quality: JPEG quality 1-100. Higher = better. Default 95.
    """
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    quality = max(1, min(100, quality))
    return _encode(img, ".jpg", (int(cv2.IMWRITE_JPEG_QUALITY), quality))


def encode_format(img: np.ndarray, ext: str, **kwargs) -> bytes:
    """Encode to arbitrary OpenCV-supported format.

    Args:
        img: BGR/BGRA numpy array.
        ext: file extension including leading dot, e.g. ".webp".
        **kwargs: passed to cv2.imencode (e.g. imwrite_params).
    """
    return _encode(img, ext, tuple(kwargs.items()))


def _encode(img: np.ndarray, ext: str, params: tuple) -> bytes:
    """Internal: encode via cv2.imencode."""
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise EncodeError(f"failed to encode image as {ext}")
    return bytes(buf.tobytes())
