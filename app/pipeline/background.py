"""Background processing: compositing a mask onto a BGR image.

We accept a pre-computed float32 mask (typically the output of
``MattingModel.predict``) and provide two compositing modes:

* ``apply_background_blur`` — sharp foreground, Gaussian-blurred background.
* ``apply_background_remove`` — foreground over a transparent alpha channel
  (returns BGRA).
* ``apply_background_replace`` — foreground over a flat colour or another
  photo (cover-fitted, with edge softening / edge in-out on the matte).

The browser live preview (web/gl.js) mirrors ``cover_rect`` and ``refine_matte``;
keep them in step (tools/parity/run.py checks it).

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
    return clear_transparent_colour(np.dstack([img_bgr, alpha_u8]))


def clear_transparent_colour(bgra: np.ndarray) -> np.ndarray:
    """Blank the colour of fully transparent pixels.

    A transparent pixel still carries RGB, which here is the *old background*. Anything that ignores the alpha channel
    (a viewer, a converter, an app that flattens the image) then shows the old background instead of the cut-out. With
    the colour cleared such a viewer shows plain black, and edge pixels do not pick up a halo of the old backdrop.
    Partly transparent edge pixels keep their colour.
    """
    out = bgra.copy()
    out[out[:, :, 3] == 0, :3] = 0
    return out


def background_paint_mask_to_foreground_alpha(
    painted_background_mask: np.ndarray,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Convert a user-painted background mask into foreground alpha.

    UI/background-paint convention is the inverse of the matting convention:
    white/non-zero pixels are explicitly marked as background. The background
    compositors expect ``1 = foreground`` and ``0 = background``, so invert the
    mask after resizing it to image size.
    """
    if painted_background_mask.ndim == 3:
        if painted_background_mask.shape[2] == 4:
            # Browser masks may arrive as transparent PNGs with white strokes.
            # Alpha is the most reliable signal for "painted" pixels.
            painted_background_mask = painted_background_mask[:, :, 3]
        else:
            painted_background_mask = cv2.cvtColor(
                painted_background_mask[:, :, :3], cv2.COLOR_BGR2GRAY
            )

    painted = painted_background_mask.astype(np.float32)
    if painted.max(initial=0) > 1.0:
        painted = painted / 255.0
    painted = _resize_mask_to_image(painted, img_shape)
    painted = np.clip(painted, 0.0, 1.0)
    return 1.0 - painted


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


# ---------------------------------------------------------------------------
# Replace the background with a colour or another photo
# ---------------------------------------------------------------------------

#: Blur applied by ``feather`` = 1.0, in pixels at the processed image size.
FEATHER_MAX_SIGMA = 12.0


def cover_rect(
    bw: int, bh: int, w: int, h: int, zoom: float = 1.0, ox: float = 0.0, oy: float = 0.0
) -> tuple[float, float, float, float]:
    """The part of a ``bw`` x ``bh`` photo that fills a ``w`` x ``h`` frame ("cover" fit).

    ``zoom`` >= 1 magnifies; ``ox``/``oy`` in [-1, 1] pan within the slack that is left
    (0 = centred). Returns ``(x0, y0, width, height)`` in source pixels.
    """
    scale = max(w / bw, h / bh) * max(1.0, zoom)
    vw, vh = w / scale, h / scale
    ox = min(1.0, max(-1.0, ox))
    oy = min(1.0, max(-1.0, oy))
    return (bw - vw) / 2 * (1 + ox), (bh - vh) / 2 * (1 + oy), vw, vh


def fit_cover(
    bg: np.ndarray, w: int, h: int, zoom: float = 1.0, ox: float = 0.0, oy: float = 0.0
) -> np.ndarray:
    """Crop + resize ``bg`` (BGR) to exactly ``w`` x ``h`` using :func:`cover_rect`."""
    bh, bw = bg.shape[:2]
    x0, y0, vw, vh = cover_rect(bw, bh, w, h, zoom, ox, oy)
    xa = int(min(max(round(x0), 0), bw - 1))
    ya = int(min(max(round(y0), 0), bh - 1))
    xb = int(min(max(round(x0 + vw), xa + 1), bw))
    yb = int(min(max(round(y0 + vh), ya + 1), bh))
    crop = bg[ya:yb, xa:xb]
    shrinking = (xb - xa) > w or (yb - ya) > h
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def refine_matte(mask: np.ndarray, feather: float = 0.0, shrink: float = 0.0) -> np.ndarray:
    """Soften and move the matte's edge. Both 0 = unchanged.

    ``feather`` in [0, 1]: gaussian blur of up to ``FEATHER_MAX_SIGMA`` px.
    ``shrink`` in [-1, 1]: > 0 pulls the edge in, < 0 pushes it out (a levels adjustment, so
    it also tightens or loosens the soft edge).
    """
    out = mask.astype(np.float32)
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=max(0.3, feather * FEATHER_MAX_SIGMA))
    if shrink > 0:
        s = 0.5 * min(1.0, shrink)
        out = (out - s) / (1.0 - s)
    elif shrink < 0:
        s = 0.5 * max(-1.0, shrink)
        out = out / (1.0 + s)
    return np.clip(out, 0.0, 1.0)


DEFRINGE_SIGMA = 6.0  # px at the processing size; the shader scales it with the output size
_DEFRINGE_CORE_LO, _DEFRINGE_CORE_SPAN = 0.9, 0.1


def defringe_colors(img_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Replace the old background's colour bleeding into the subject's edge with the subject's own nearby colour.

    "Core" = pixels the matte is sure about (> 0.9). The subject colour is spread outward from the core with a
    gaussian (normalised convolution: blur(colour * core) / blur(core)); edge pixels (core < 1) move toward it by
    ``amount`` (0..1), faded out where no core is close. Returns float32 BGR. ``mask`` is the raw matte.
    """
    img = img_bgr[:, :, :3].astype(np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img
    core = np.clip((mask.astype(np.float32) - _DEFRINGE_CORE_LO) / _DEFRINGE_CORE_SPAN, 0.0, 1.0)
    num = cv2.GaussianBlur(img * core[:, :, None], (0, 0), DEFRINGE_SIGMA)
    den = cv2.GaussianBlur(core, (0, 0), DEFRINGE_SIGMA)
    own = num / np.maximum(den, 1e-4)[:, :, None]
    weight = amount * (1.0 - core) * np.clip(den / 0.1, 0.0, 1.0)
    return img + (own - img) * weight[:, :, None]


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    """``'#rrggbb'`` -> ``(b, g, r)``."""
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def apply_background_replace(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: str | None = None,
    background: np.ndarray | None = None,
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    feather: float = 0.0,
    shrink: float = 0.0,
    defringe: float = 0.0,
) -> np.ndarray:
    """Put the subject (``mask`` = 1) over a flat ``color`` or a photo (``background``, BGR/BGRA).

    Exactly one of ``color`` / ``background`` must be given. Returns BGR uint8 at the size of
    ``img_bgr``.
    """
    if (color is None) == (background is None):
        raise ValueError("give exactly one of color or background")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    raw = _resize_mask_to_image(mask, img_bgr.shape)
    subject = defringe_colors(img_bgr, raw, defringe)
    mask = refine_matte(raw, feather, shrink)
    if background is not None:
        if background.ndim == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        bg = fit_cover(background[:, :, :3], w, h, zoom, offset_x, offset_y)
    else:
        bg = np.empty_like(img_bgr[:, :, :3])
        bg[:] = hex_to_bgr(color)
    alpha = mask[:, :, np.newaxis]
    out = subject * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
