"""Image inpainting (object removal) using OpenCV's built-in algorithms.

Two algorithms are available:
- cv2.INPAINT_TELEA — Fast Marching Method (Telea 2004), fast, good for small masks
- cv2.INPAINT_NS — Navier-Stokes fluid dynamics, better quality on large regions

Both take a uint8 mask where non-zero pixels are the areas to be removed.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ValidationError


SUPPORTED_ALGORITHMS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}
MIN_RADIUS = 1
MAX_RADIUS = 100


def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    algorithm: str = "telea",
) -> np.ndarray:
    """Remove masked regions from the image and fill them in.

    Args:
        img: uint8 image (2D, 3D BGR, or 3D BGRA). The inpainting is applied
            to the first 3 channels; alpha (if present) is preserved unchanged.
        mask: uint8 single-channel image of the same H, W as img. Non-zero
            pixels mark the area to be removed.
        radius: inpainting neighborhood radius in pixels (1-100). Larger
            values fill bigger holes but take longer and can blur.
        algorithm: "telea" (default, fast) or "ns" (slower, smoother).

    Returns:
        uint8 image, same shape and channel count as input.
    """
    if radius < MIN_RADIUS or radius > MAX_RADIUS:
        raise ValidationError(f"radius must be in [{MIN_RADIUS}, {MAX_RADIUS}], got {radius}")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValidationError(
            f"unsupported algorithm: {algorithm!r}; choose from {list(SUPPORTED_ALGORITHMS)}"
        )
    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img.shape[:2] != mask.shape[:2]:
        raise ValidationError(
            f"mask shape {mask.shape[:2]} does not match image shape {img.shape[:2]}"
        )

    # Inpaint only the color channels; preserve alpha
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    elif img.ndim == 2:
        bgr = img
    else:
        bgr = img

    out = cv2.inpaint(bgr, mask, float(radius), SUPPORTED_ALGORITHMS[algorithm])

    if img.ndim == 2:
        return out
    if alpha is not None:
        return np.dstack([out, alpha])
    return out
