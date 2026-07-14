"""Luminance-aware film grain (Rec.601 weighted).

Darker pixels get more grain than brighter pixels, mimicking the look
of film stock. Output dtype is uint8, clipped to [0, 255].
"""

from __future__ import annotations

import numpy as np

# Rec.601 luma weights for an RGB triplet (R, G, B).
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def apply_grain(
    img: np.ndarray,
    intensity: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply luminance-aware film grain to an image.

    Args:
        img: uint8 image — 2D (grayscale), 3D BGR/BGR(A).
        intensity: grain strength in [0.0, 1.0+]. 0.0 = passthrough.
        seed: optional RNG seed for reproducibility.

    Returns:
        Tuple of (grainy_image, grain_only_debug_image).
        Both are uint8 with the same H, W as the input.
        ``grain_only`` is an auto-contrasted visualisation of just the
        grain pattern (2D).
    """
    # Fast-path: nothing to do.
    if intensity <= 0.0:
        copy = img.copy()
        # grain_only must be a 2-D debug plane; return zeros matching H, W.
        grain_only = np.zeros(img.shape[:2], dtype=np.uint8)
        return copy, grain_only

    # Save the caller's RNG state so a one-off seed doesn't leak globally.
    if seed is not None:
        saved_state = np.random.get_state()
        np.random.seed(seed)

    try:
        # Figure out the "color" planes we'll actually add grain to.
        # For BGRA we keep the alpha untouched and grain only BGR.
        if img.ndim == 2:
            color = img
            alpha = None
        else:
            alpha = img[:, :, 3] if img.shape[2] == 4 else None
            color = img[:, :, :3]

        # 1. Luminance mask (1.0 for black, 0.0 for white) — dark = more grain.
        if color.ndim == 2:
            luma = color.astype(np.float32) / 255.0
        else:
            # Rec.601 expects RGB order; OpenCV hands us BGR.
            rgb = color[:, :, ::-1]
            luma = (rgb.astype(np.float32) @ _LUMA_RGB) / 255.0
        luma_mask = 1.0 - luma

        # 2. Gaussian noise (zero mean, sigma=1.0) — same shape as the
        #    color planes; alpha is not grained.
        noise = np.random.normal(0.0, 1.0, size=color.shape).astype(np.float32)

        # 3. Scale noise by intensity and the luma mask.
        scale = float(intensity) * 32.0
        if color.ndim == 3:
            scaled = noise * scale * luma_mask[:, :, np.newaxis]
        else:
            scaled = noise * scale * luma_mask

        # 4. Combine with the original color planes, clipping to uint8 range.
        grainy_color = color.astype(np.float32) + scaled
        grainy_color = np.clip(grainy_color, 0, 255).astype(np.uint8)

        if alpha is not None:
            grainy = np.concatenate([grainy_color, alpha[:, :, np.newaxis]], axis=2)
        else:
            grainy = grainy_color

        # 5. Grain-only debug: auto-contrast the per-pixel noise for viz.
        #    For 3D inputs the grain is the same per pixel across channels,
        #    so we collapse to a single 2D plane.
        flat = scaled if scaled.ndim == 2 else scaled[:, :, 0]
        lo, hi = float(flat.min()), float(flat.max())
        if hi - lo < 1e-6:
            grain_only = np.zeros(flat.shape, dtype=np.uint8)
        else:
            grain_only = ((flat - lo) / (hi - lo) * 255.0).astype(np.uint8)

        return grainy, grain_only
    finally:
        if seed is not None:
            np.random.set_state(saved_state)
