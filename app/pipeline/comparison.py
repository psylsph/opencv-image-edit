"""Before/After and Diff visualizations.

Pure OpenCV implementation — no Pillow dependency. Useful for QA of image
edits (side-by-side) and for highlighting what changed (diff overlay).
"""
from __future__ import annotations

import cv2
import numpy as np


DEFAULT_DIVIDER_COLOR: tuple[int, int, int] = (0, 255, 0)  # green, BGR
DEFAULT_DIVIDER_WIDTH: int = 3


def _normalize_channels(
    before: np.ndarray, after: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Make both images the same channel count (3 or 4).

    If one is BGR (3ch) and the other is BGRA (4ch), we drop the alpha
    on the BGRA side so the output is consistently 3-channel. Grayscale
    inputs (2D) are promoted to BGR.
    """
    if before.ndim == 2:
        before = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    if after.ndim == 2:
        after = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)

    if before.ndim != after.ndim:
        raise ValueError(
            f"ndim mismatch: before={before.ndim}, after={after.ndim}"
        )
    if before.ndim != 3:
        raise ValueError(f"expected 2D or 3D images, got ndim={before.ndim}")

    cb, ca = before.shape[2], after.shape[2]
    if cb == ca:
        return before, after, cb
    if cb == 4 and ca == 3:
        return cv2.cvtColor(before, cv2.COLOR_BGRA2BGR), after, 3
    if cb == 3 and ca == 4:
        return before, cv2.cvtColor(after, cv2.COLOR_BGRA2BGR), 3
    raise ValueError(f"unsupported channel counts: before={cb}, after={ca}")


def side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    divider_color: tuple[int, int, int] = DEFAULT_DIVIDER_COLOR,
    divider_width: int = DEFAULT_DIVIDER_WIDTH,
    add_labels: bool = False,
) -> np.ndarray:
    """Create a horizontal side-by-side composite of two images.

    Args:
        before: BGR or BGRA image (H, W, C).
        after:  BGR or BGRA image, same H as ``before``.
        divider_color: BGR color of the divider strip.
        divider_width: Width of the divider in pixels.
        add_labels: If True, render ``"BEFORE"`` / ``"AFTER"`` text with
            ``cv2.putText`` (3-channel input only).

    Returns:
        BGR or BGRA image, shape ``(H, W_before + divider_width + W_after, C)``,
        dtype ``uint8``.
    """
    if before.shape[0] != after.shape[0]:
        raise ValueError(
            f"height mismatch: before={before.shape[0]}, after={after.shape[0]}"
        )
    if divider_width < 1:
        raise ValueError(f"divider_width must be >= 1, got {divider_width}")

    before, after, channels = _normalize_channels(before, after)

    h = before.shape[0]
    w_b = before.shape[1]
    w_a = after.shape[1]
    canvas_w = w_b + divider_width + w_a

    canvas = np.zeros((h, canvas_w, channels), dtype=before.dtype)

    # Left half — ``before``
    canvas[:, :w_b] = before

    # Divider strip
    canvas[:, w_b : w_b + divider_width, :3] = divider_color
    if channels == 4:
        canvas[:, w_b : w_b + divider_width, 3] = 255

    # Right half — ``after``
    canvas[:, w_b + divider_width :] = after

    if add_labels and channels == 3:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600.0)
        thickness = max(1, int(scale * 2))
        cv2.putText(
            canvas, "BEFORE", (10, 30), font, scale, (0, 255, 0), thickness
        )
        cv2.putText(
            canvas,
            "AFTER",
            (w_b + divider_width + 10, 30),
            font,
            scale,
            (0, 255, 0),
            thickness,
        )

    return canvas


def diff_overlay(
    before: np.ndarray,
    after: np.ndarray,
    gain: float = 5.0,
) -> np.ndarray:
    """Compute ``|before - after| * gain`` to visualize changes.

    Args:
        before: BGR or BGRA image (uint8).
        after:  BGR or BGRA image, same shape as ``before`` (uint8).
        gain: Multiplier on the diff — improves visibility of subtle
            changes. Output is clipped to ``[0, 255]``.

    Returns:
        Image of the same shape and dtype (``uint8``).
    """
    if before.shape != after.shape:
        raise ValueError(
            f"shape mismatch: before={before.shape}, after={after.shape}"
        )

    diff = cv2.absdiff(before, after).astype(np.float32) * gain
    return np.clip(diff, 0, 255).astype(np.uint8)
