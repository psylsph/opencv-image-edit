"""Image inpainting (object removal).

Three algorithms are available:
- ``telea`` — Fast Marching Method (Telea 2004), fast, classic cv2 inpaint
- ``ns``    — Navier-Stokes fluid dynamics, smoother than TELEA, classic cv2 inpaint
- ``lama``  — LaMa: Resolution-robust Large Mask Inpainting with Fourier
              Convolutions (WACV 2022). State-of-the-art quality via
              ONNX model from opencv/inpainting_lama. Default for v1.0.2+.

All three take a uint8 mask where non-zero pixels are the areas to be removed.
The default algorithm is "lama" (highest quality, ~50ms on 512×512 CPU).
"""
from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ValidationError
from app.models.lama import LaMa


SUPPORTED_ALGORITHMS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}
MIN_RADIUS = 1
MAX_RADIUS = 100
DEFAULT_ALGORITHM = "lama"


def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    algorithm: str = DEFAULT_ALGORITHM,
    iterations: int = 1,
) -> np.ndarray:
    """Remove masked regions from the image and fill them in.

    Args:
        img: uint8 image (2D, 3D BGR, or 3D BGRA). The inpainting is applied
            to the first 3 channels; alpha (if present) is preserved unchanged.
        mask: uint8 single-channel image of the same H, W as img. Non-zero
            pixels mark the area to be removed.
        radius: inpainting neighborhood radius in pixels (1-100). Larger
            values fill bigger holes but take longer and can blur. Only
            used by TELEA and NS; ignored by LaMa.
        algorithm: "lama" (default, best quality), "telea" (fast), or "ns"
            (smoother classic).
        iterations: Number of LaMa passes (default 1). Each pass feeds the
            previous output as input. 2-3 can improve stubborn removals.
            Ignored by TELEA/NS.

    Returns:
        uint8 image, same shape and channel count as input.
    """
    if algorithm not in SUPPORTED_ALGORITHMS and algorithm != "lama":
        raise ValidationError(
            f"unsupported algorithm: {algorithm!r}; "
            f"choose from {list(SUPPORTED_ALGORITHMS) + ['lama']}"
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

    if algorithm == "lama":
        # Local singleton lookup (lazy-loads the model on first use)
        lama = LaMa.get()
        out = lama.infer(bgr, mask, iterations=iterations)
    else:
        # radius only matters for cv2.inpaint; validate only when relevant
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            raise ValidationError(
                f"radius must be in [{MIN_RADIUS}, {MAX_RADIUS}], got {radius}"
            )
        out = cv2.inpaint(bgr, mask, float(radius), SUPPORTED_ALGORITHMS[algorithm])

    if img.ndim == 2:
        return out
    if alpha is not None:
        return np.dstack([out, alpha])
    return out
