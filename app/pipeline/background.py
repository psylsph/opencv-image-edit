"""Background processing: compositing a mask onto a BGR image.

We accept a pre-computed float32 mask (typically the output of
``MattingModel.predict``) and provide two compositing modes:

* ``apply_background_blur`` — sharp foreground, Gaussian-blurred background.
* ``apply_background_remove`` — foreground over a transparent alpha channel
  (returns BGRA).

The mask is always interpreted as ``alpha``:
  mask == 1.0  → fully foreground (sharp / opaque)
  mask == 0.0  → fully background (blurred / transparent)
"""

from __future__ import annotations

import cv2
import numpy as np


def _resize_mask_to_image(mask: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Resize a (H, W) float mask to the spatial size of the BGR image.

    Returns the original mask unchanged if the shape already matches.
    """
    h, w = img_shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_background_blur(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    blur_strength: int = 15,
) -> np.ndarray:
    """Blur the background (where mask is low) and composite over sharp foreground.

    Args:
        img_bgr: BGR uint8 image (H, W, 3). Grayscale inputs are promoted.
        mask: float32 (H, W) in [0, 1] where 1 = foreground (sharp).
              Mismatched shapes are resized automatically.
        blur_strength: Gaussian sigma. 1-50, typical 10-25.

    Returns:
        BGR uint8 image (H, W, 3) with blurred background and sharp foreground.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    mask = _resize_mask_to_image(mask, img_bgr.shape)

    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=blur_strength)
    alpha = mask[:, :, np.newaxis].astype(np.float32)
    out = img_bgr.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_background_remove(
    img_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Remove the background, returning a BGRA image with feathered alpha.

    Args:
        img_bgr: BGR uint8 image (H, W, 3). Grayscale inputs are promoted.
        mask: float32 (H, W) in [0, 1] where 1 = keep, 0 = transparent.
              Mismatched shapes are resized automatically.

    Returns:
        BGRA uint8 image (H, W, 4) with feathered alpha edge.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    mask = _resize_mask_to_image(mask, img_bgr.shape)
    # Feather the alpha edge (1px Gaussian)
    alpha = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_u8 = (alpha * 255).astype(np.uint8)
    return np.dstack([img_bgr, alpha_u8])


def get_alpha_debug(mask: np.ndarray) -> np.ndarray:
    """Convert float mask to uint8 debug visualization with autocontrast.

    Stretches ``[mask.min(), mask.max()]`` to ``[0, 255]`` so the full
    dynamic range of a low-contrast mask is visible in the debug image.
    A constant mask is mapped to all zeros (no contrast to stretch).
    """
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    lo, hi = float(mask.min()), float(mask.max())
    stretched = (mask - lo) * 255.0 / (hi - lo) if hi > lo else np.zeros_like(mask)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_selection_blur(
    img_bgr: np.ndarray,
    selection: np.ndarray,
    blur_strength: int = 15,
    target: str = "selection",
    feather: float = 3.0,
) -> np.ndarray:
    """Blur a user-chosen region (or everything outside it).

    Args:
        img_bgr: BGR (H, W, 3) or BGRA uint8 image. Alpha is passed through.
        selection: (H, W) mask, nonzero / > 0.5 = selected region.
        blur_strength: Gaussian sigma. 1-50.
        target: "selection" blurs the selected region; "outside" blurs
            everything else (selection stays sharp).
        feather: sigma of the edge softening applied to the selection.
    """
    if target not in ("selection", "outside"):
        raise ValueError(f"unknown blur target: {target!r}")
    alpha_ch = None
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        alpha_ch = img_bgr[:, :, 3]
        img_bgr = img_bgr[:, :, :3]
    sel = selection.astype(np.float32)
    if sel.max() > 1.0:
        sel = sel / 255.0
    sel = _resize_mask_to_image(sel, img_bgr.shape)
    if feather > 0:
        sel = cv2.GaussianBlur(sel, (0, 0), sigmaX=feather)
    sharp_mask = 1.0 - sel if target == "selection" else sel
    out = apply_background_blur(img_bgr, sharp_mask, blur_strength)
    if alpha_ch is not None:
        out = np.dstack([out, alpha_ch])
    return out
