"""Image filters: brightness, contrast, saturation, sharpness, vignette, sepia,
grayscale, blur, unsharp mask, auto-enhance. All OpenCV-based.

Factor convention (cleaner than the original Pillow [-1, 1] offsets):
    factor == 1.0 -> identity (no change)
    factor == 0.0 -> "zero" / collapsed effect
    factor == 2.0 -> doubled / stronger effect

Deviations from the original Pillow `ai-image-edit/app/filters.py`:
    - The original used PIL `ImageEnhance.enhance(factor)` where the input
      factor was an OFFSET (e.g. brightness=0.3 meant enhance(1.3)). The
      new OpenCV port uses the factor directly so 1.0 == no-op, which is
      more intuitive and easier to test. UI callers will need to remap
      their sliders (e.g. slider=-1..1 -> factor=1.0+s).
    - `apply_sepia` is now full-intensity by default (intensity baked in
      via the kernel). Pass a pre-blended image if a partial effect is
      needed downstream; the function returns a fully-sepia image.
    - `apply_unsharp_mask` uses the same sigma/strength API rather than
      PIL's radius/percent/threshold signature, for OpenCV native speed.
    - `add_vignette` is now a multiplicative radial mask (vs the original
      alpha-composite black layer), so transparent RGBA inputs are not
      preserved. Callers that need alpha preservation should composite
      themselves.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tone adjustments
# ---------------------------------------------------------------------------


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative brightness. factor=1.0 -> identity, 0.0 -> black, 2.0 -> double."""
    if factor == 1.0:
        return img
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around the 128 pivot. factor=1.0 -> identity, 0.0 -> flat gray, 2.0 -> high contrast."""
    if factor == 1.0:
        return img
    mean = np.full(img.shape[:2] + (1,) * (img.ndim - 2), 128.0, dtype=np.float32)
    out = mean + factor * (img.astype(np.float32) - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale S channel in HSV. factor=1.0 -> identity, 0.0 -> grayscale, 2.0 -> vivid."""
    if factor == 1.0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_sharpness(img: np.ndarray, factor: float) -> np.ndarray:
    """Blend between blur and unsharp. factor=1.0 -> identity, 0.0 -> pure blur, 2.0 -> unsharp(1.0)."""
    if factor == 1.0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    if factor < 1.0:
        # Blend toward blur: img*(factor) + blur*(1-factor)
        return cv2.addWeighted(img, factor, blurred, 1.0 - factor, 0)
    # factor > 1.0: unsharp mask with strength (factor - 1)
    # sharpened = img + (img - blur) * (factor - 1)
    return cv2.addWeighted(img, factor, blurred, -(factor - 1.0), 0)


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------


def add_vignette(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Multiplicative radial darkening. strength=0.0 -> identity."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    # Normalized radial distance: 0 at center, ~1 at corners (rect, not inscribed circle)
    r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    # Mask = 1.0 at center, falls toward 0 at corners scaled by strength.
    mask = np.clip(1.0 - r * strength, 0.0, 1.0).astype(np.float32)
    if img.ndim == 3:
        mask = mask[:, :, np.newaxis]
    return (img.astype(np.float32) * mask).astype(np.uint8)


# ---------------------------------------------------------------------------
# Auto-enhance
# ---------------------------------------------------------------------------


def add_auto_enhance(img: np.ndarray) -> np.ndarray:
    """Percentile stretch (1-99) on the L channel of LAB + light unsharp mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lo, hi = np.percentile(l_ch, (1, 99))
    if hi > lo:
        l_stretched = np.clip((l_ch.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(
            np.uint8
        )
    else:
        l_stretched = l_ch
    lab = cv2.merge([l_stretched, a_ch, b_ch])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return apply_unsharp_mask(out, sigma=1.0, strength=1.0)


# ---------------------------------------------------------------------------
# Combined adjustments
# ---------------------------------------------------------------------------


def apply_color_adjustments(
    img: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
) -> np.ndarray:
    """Chain the four tone adjustments. All defaults = 1.0 = identity."""
    out = adjust_brightness(img, brightness)
    out = adjust_contrast(out, contrast)
    out = adjust_saturation(out, saturation)
    return adjust_sharpness(out, sharpness)


# ---------------------------------------------------------------------------
# Color effects
# ---------------------------------------------------------------------------


def apply_sepia(img: np.ndarray) -> np.ndarray:
    """Apply full-intensity sepia via a 3x3 BGR kernel."""
    kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ],
        dtype=np.float32,
    )
    out = cv2.transform(img, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_grayscale(img: np.ndarray, blend: float = 1.0) -> np.ndarray:
    """Blend the input with its grayscale. blend=0.0 -> identity, blend=1.0 -> fully gray."""
    if blend <= 0:
        return img
    blend = min(1.0, blend)
    if img.ndim == 2:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.merge([gray, gray, gray])
    if blend >= 1.0:
        return gray_3ch
    return cv2.addWeighted(img, 1.0 - blend, gray_3ch, blend, 0)


# ---------------------------------------------------------------------------
# Blur / sharpen
# ---------------------------------------------------------------------------


def apply_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur with odd ksize. ksize<=1 -> identity."""
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1  # GaussianBlur requires odd ksize
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_unsharp_mask(img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Unsharp mask: img + strength * (img - gaussian_blur(img)).

    sigma controls blur radius; strength controls sharpening intensity.
    strength=0 -> identity; strength=1.5 (default) is a moderate sharpen.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
