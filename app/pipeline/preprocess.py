"""Image pre-processing: resize oversized images down to a max dimension."""

from __future__ import annotations

import cv2
import numpy as np


def resize_if_needed(
    img: np.ndarray,
    max_dim: int = 1536,
) -> np.ndarray:
    """Downscale an image so its largest dimension is <= max_dim.

    Never upscales. Preserves aspect ratio. Preserves number of channels
    and dtype. Returns a C-contiguous array (cv2.resize requirement).

    Args:
        img: numpy array — 2D (grayscale), 3D (HWC BGR/BGRA), uint8 or float32.
        max_dim: maximum allowed value of max(H, W). Must be > 0.

    Returns:
        Resized image as a C-contiguous numpy array of the same dtype.
    """
    if max_dim <= 0:
        raise ValueError(f"max_dim must be positive, got {max_dim}")

    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_dim:
        # Ensure C-contiguous for downstream OpenCV calls
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        return img

    scale = max_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # INTER_AREA is best for downscaling (mosaicing); INTER_CUBIC for upscaling.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def get_image_dimensions(img: np.ndarray) -> tuple[int, int]:
    """Return (height, width) of an image."""
    h, w = img.shape[:2]
    return h, w
