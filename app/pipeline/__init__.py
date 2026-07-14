"""Image processing pipeline orchestrator.

Chains stages in fixed order: pre-process → background → grain → upscale → filters.
Each stage can be individually enabled/disabled via the ProcessRequest.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.api.schemas import ProcessRequest
from app.config import get_settings
from app.exceptions import ProcessingError
from app.models.matting import get_matting_model
from app.pipeline.background import (
    apply_background_blur,
    apply_background_remove,
    get_alpha_debug,
)
from app.pipeline.comparison import diff_overlay, side_by_side
from app.pipeline.filters import (
    add_vignette,
    apply_color_adjustments,
    apply_grayscale,
    apply_sepia,
)
from app.pipeline.grain import apply_grain
from app.pipeline.preprocess import resize_if_needed
from app.pipeline.upscale import upscale


@dataclass
class ProcessResult:
    """Result of the processing pipeline. All images are BGR/BGRA uint8."""

    original: np.ndarray  # BGR preprocessed input
    final: np.ndarray  # BGR or BGRA final output
    before_after: np.ndarray  # BGR side-by-side
    diff: np.ndarray  # BGR diff overlay
    mask: np.ndarray | None  # 2D uint8 alpha debug (or None)
    upscaled: np.ndarray | None  # BGR upscaled intermediate (or None)
    grain_only: np.ndarray | None  # 2D uint8 grain visualization (or None)


def process_pipeline(img_bgr: np.ndarray, request: ProcessRequest) -> ProcessResult:
    """Run the full processing pipeline.

    Args:
        img_bgr: BGR uint8 image (H, W, 3).
        request: validated ProcessRequest specifying which stages to run.

    Returns:
        ProcessResult with the final image plus debug visualizations.
    """
    try:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            raise ProcessingError("empty image")

        # 1. Pre-process (downscale if oversized)
        original = resize_if_needed(img_bgr, get_settings().max_image_dimension)
        current = original.copy()
        upscaled_out: np.ndarray | None = None
        mask_debug: np.ndarray | None = None
        grain_only: np.ndarray | None = None

        # 2. Background
        if request.background.enabled:
            matting = get_matting_model()
            mask = matting.predict(current)
            mask_debug = get_alpha_debug(mask)
            if request.background.mode == "blur":
                current = apply_background_blur(current, mask, request.background.blur_strength)
            else:  # remove
                current = apply_background_remove(current, mask)

        # 3. Grain
        if request.grain.enabled:
            current, grain_only = apply_grain(current, request.grain.intensity)

        # 4. Upscale (run before filters for cleaner result)
        if request.upscale.enabled and request.upscale.scale > 1:
            algorithm = None if request.upscale.algorithm == "interp" else request.upscale.algorithm
            if algorithm is None:
                # Fast LANCZOS4 path — no model required
                upscaled_out = cv2.resize(
                    current,
                    None,
                    fx=request.upscale.scale,
                    fy=request.upscale.scale,
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                upscaled_out = upscale(current, request.upscale.scale, algorithm)
            current = upscaled_out

        # 5. Filters (only on BGR — split BGRA if background_remove was used)
        if request.filters.enabled:
            if current.ndim == 3 and current.shape[2] == 4:
                # Drop alpha for filters, reattach at end
                bgr = current[:, :, :3]
                alpha = current[:, :, 3]
            else:
                bgr = current
                alpha = None

            bgr = apply_color_adjustments(
                bgr,
                brightness=request.filters.brightness,
                contrast=request.filters.contrast,
                saturation=request.filters.saturation,
                sharpness=request.filters.sharpness,
            )
            if request.filters.vignette_strength > 0:
                bgr = add_vignette(bgr, request.filters.vignette_strength)
            if request.filters.sepia:
                bgr = apply_sepia(bgr)
            if request.filters.grayscale_blend > 0:
                bgr = apply_grayscale(bgr, request.filters.grayscale_blend)

            current = np.dstack([bgr, alpha]) if alpha is not None else bgr

        # 6. Comparisons (always)
        # For before/after, use the preprocessed original (no alpha) for left side
        before_for_compare = original
        if before_for_compare.ndim == 3 and before_for_compare.shape[2] == 4:
            before_for_compare = before_for_compare[:, :, :3]
        elif before_for_compare.ndim == 2:
            before_for_compare = cv2.cvtColor(before_for_compare, cv2.COLOR_GRAY2BGR)

        after_for_compare = current
        if after_for_compare.ndim == 3 and after_for_compare.shape[2] == 4:
            after_for_compare = after_for_compare[:, :, :3]
        elif after_for_compare.ndim == 2:
            after_for_compare = cv2.cvtColor(after_for_compare, cv2.COLOR_GRAY2BGR)

        # Match heights (after may be larger from upscale)
        if before_for_compare.shape[0] != after_for_compare.shape[0]:
            scale_y = after_for_compare.shape[0] / before_for_compare.shape[0]
            new_w = max(1, int(round(before_for_compare.shape[1] * scale_y)))
            before_for_compare = cv2.resize(
                before_for_compare,
                (new_w, after_for_compare.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        before_after = side_by_side(before_for_compare, after_for_compare, add_labels=True)

        # Diff (use before/after at same size)
        if before_for_compare.shape != after_for_compare.shape:
            before_for_diff = cv2.resize(
                before_for_compare,
                (after_for_compare.shape[1], after_for_compare.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            before_for_diff = before_for_compare
        diff = diff_overlay(before_for_diff, after_for_compare)

        return ProcessResult(
            original=original,
            final=current,
            before_after=before_after,
            diff=diff,
            mask=mask_debug,
            upscaled=upscaled_out,
            grain_only=grain_only,
        )
    except ProcessingError:
        raise
    except Exception as exc:
        raise ProcessingError(f"pipeline failed: {exc}") from exc
