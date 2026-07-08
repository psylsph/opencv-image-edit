"""Image inpainting (object removal).

Five algorithms are available:
- ``telea``  — Fast Marching Method (Telea 2004), fast, classic cv2 inpaint
- ``ns``     — Navier-Stokes fluid dynamics, smoother than TELEA, classic cv2 inpaint
- ``lama``   — LaMa: Resolution-robust Large Mask Inpainting with Fourier
               Convolutions (WACV 2022). Fast AI inpainting. Default.
- ``sd``     — Stable Diffusion 1.5 Inpainting: generative fill that invents
               realistic content via text prompt. Slowest (~30-120s CPU).
- ``openai`` — OpenAI Images API (gpt-image-1): cloud generative fill.
               Fast (~3-10s), high quality, requires API key + internet.

All five take a uint8 mask where non-zero pixels are the areas to be removed.
The default algorithm is "lama" (fast, good quality, ~4s on CPU).
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
    prompt: str = "",
) -> np.ndarray:
    """Remove masked regions from the image and fill them in.

    Args:
        img: uint8 image (2D, 3D BGR, or 3D BGRA).
        mask: uint8 single-channel mask, non-zero = to remove.
        radius: Inpainting neighborhood radius (1-100). TELEA/NS only.
        algorithm: "lama" (default), "sd" (local gen), "openai" (cloud gen),
                   "telea", or "ns".
        iterations: LaMa passes. Ignored by TELEA/NS/SD/OpenAI.
        prompt: Text prompt for generative fill (algorithm="sd" or "openai").

    Returns:
        uint8 image, same shape and channel count as input.
    """
    valid_algos = list(SUPPORTED_ALGORITHMS) + ["lama", "sd", "openai"]
    if algorithm not in SUPPORTED_ALGORITHMS and algorithm not in ("lama", "sd", "openai"):
        raise ValidationError(
            f"unsupported algorithm: {algorithm!r}; choose from {valid_algos}"
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
        lama = LaMa.get()
        out = lama.infer(bgr, mask, iterations=iterations)
    elif algorithm == "sd":
        from app.models.sd_inpaint import SDInpaint
        sd = SDInpaint.get()
        out = sd.inpaint(bgr, mask, prompt=prompt or "")
    elif algorithm == "openai":
        from app.models.openai_inpaint import OpenAIInpaint
        oai = OpenAIInpaint()
        out = oai.inpaint(bgr, mask, prompt=prompt)
    else:
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
