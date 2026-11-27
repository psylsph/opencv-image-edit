"""Image processing pipeline orchestrator.

Chains stages in fixed order: pre-process → background → selection tool → face → grain → upscale → filters.
Each stage can be individually enabled/disabled via the ProcessRequest.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.api.schemas import ProcessRequest
from app.config import get_settings
from app.exceptions import ProcessingError, ValidationError
from app.models.matting import get_matting_model
from app.models.anime import apply_anime_style
from app.models.painting import apply_paint_style
from app.pipeline.background import (
    apply_background_blur,
    apply_background_remove,
    apply_background_replace,
    background_paint_mask_to_foreground_alpha,
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


def process_pipeline(
    img_bgr: np.ndarray,
    request: ProcessRequest,
    background_mask: Optional[np.ndarray] = None,
    background_image: Optional[np.ndarray] = None,
    matte: Optional[np.ndarray] = None,
    face_donor: Optional[np.ndarray] = None,
    face_donors_extra: Optional[list] = None,
    face_keep: Optional[np.ndarray] = None,
    edit_mask: Optional[np.ndarray] = None,
) -> ProcessResult:
    """Run the full processing pipeline.

    Args:
        img_bgr: BGR uint8 image (H, W, 3).
        request: validated ProcessRequest specifying which stages to run.
        background_mask: optional user-painted mask where non-zero pixels are
            explicitly background. Used instead of AI matting when provided.
        background_image: BGR photo for ``background.mode == "image"``.
        face_donor: BGR photo whose face replaces one in the image when ``face.enabled``.
        face_donors_extra: more photos of the same person; the identity is averaged over them.
        face_keep: optional grayscale mask (255 = put the original pixels back over the swapped face, e.g. a hand).
        edit_mask: selection for ``request.edit`` (non-zero = selected area).
        matte: optional 8-bit grayscale subject matte from ``/api/v1/matte`` (255 = subject); skips the model.

    Returns:
        ProcessResult with the final image plus debug visualizations.
    """
    try:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            raise ProcessingError("empty image")

        # 1. Pre-process (downscale if oversized)
        original = resize_if_needed(img_bgr, get_settings().max_image_dimension)
        # Every stage (matting, face, filters) expects plain 3-channel BGR: a PNG/WebP/HEIC
        # with an alpha channel used to crash the matting model ("Expected: 3, Got: 4").
        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.shape[2] == 4:
            a = original[:, :, 3:4].astype(np.float32) / 255.0
            white = np.full_like(original[:, :, :3], 255, dtype=np.float32)
            original = np.clip(original[:, :, :3] * a + white * (1 - a), 0, 255).astype(np.uint8)
        current = original.copy()
        upscaled_out: np.ndarray | None = None
        mask_debug: np.ndarray | None = None
        alpha_keep: np.ndarray | None = None
        grain_only: np.ndarray | None = None

        # 1b. Background first: every later step then works on the finished backdrop (and is quicker/easier)
        if request.background.enabled:
            if background_mask is not None:
                mask = background_paint_mask_to_foreground_alpha(background_mask, current.shape)
            elif matte is not None:
                m = matte[:, :, 0] if matte.ndim == 3 else matte
                if m.shape != current.shape[:2]:
                    m = cv2.resize(
                        m, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR
                    )
                mask = m.astype(np.float32) / 255.0
            else:
                matting = get_matting_model(request.background.model_name)
                mask = matting.predict(current)
            mask_debug = get_alpha_debug(mask)
            if request.background.mode == "blur":
                current = apply_background_blur(current, mask, request.background.blur_strength)
            elif request.background.mode in ("color", "image"):
                bg = request.background
                if bg.mode == "image" and background_image is None:
                    raise ProcessingError("background mode 'image' needs a background image")
                current = apply_background_replace(
                    current,
                    mask,
                    color=bg.color if bg.mode == "color" else None,
                    background=background_image if bg.mode == "image" else None,
                    zoom=bg.zoom,
                    offset_x=bg.offset_x,
                    offset_y=bg.offset_y,
                    feather=bg.feather,
                    shrink=bg.shrink,
                    defringe=bg.defringe,
                )
            else:  # remove
                current = apply_background_remove(current, mask)
            if current.ndim == 3 and current.shape[2] == 4:
                # Background removal gives transparency: carry it alongside and put it back at the end, so the
                # selection tools, face swap, grain, upscale and filters all see plain BGR.
                alpha_keep = current[:, :, 3]
                current = current[:, :, :3]

        # 1c. Selection tool (remove object / tattoo / blur an area), painted or tapped on the photo
        if request.edit.kind != "none":
            if edit_mask is None:
                raise ProcessingError("the selection tool needs an edit_mask")
            from app.pipeline.edit import apply_edit

            current = apply_edit(current, edit_mask, request.edit)

        # 1d. Face replacement (after background and selection tools; grain/filters then unify the pasted face)
        if request.face.enabled:
            if face_donor is None:
                raise ProcessingError("face replacement needs a donor photo")
            from app.models.face import get_face_engine

            f = request.face
            before_face = current
            current = get_face_engine().swap(
                current,
                face_donor,
                target_index=f.target,
                donor_index=f.donor,
                blend=f.blend,
                color_match=f.color_match,
                restore=f.restore,
                hair=f.hair,
                shape=f.shape,
                detail=f.detail,
                extra_donors=face_donors_extra or [],
                all_faces=f.all_faces,
                texture=f.texture,
            )
            if face_keep is not None:
                from app.pipeline.face import keep_original

                k = face_keep[:, :, 0] if face_keep.ndim == 3 else face_keep
                current = keep_original(current, before_face, k)

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
            if request.filters.anime:
                bgr = apply_anime_style(bgr, request.filters.anime_style)
            if request.filters.painting:
                bgr = apply_paint_style(bgr, request.filters.painting_style)

            current = np.dstack([bgr, alpha]) if alpha is not None else bgr

        if alpha_keep is not None:
            if alpha_keep.shape != current.shape[:2]:
                alpha_keep = cv2.resize(
                    alpha_keep,
                    (current.shape[1], current.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            current = np.dstack([current[:, :, :3], alpha_keep])

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
    except (ProcessingError, ValidationError):
        raise  # ValidationError keeps its 422 (e.g. no face found in the donor photo)
    except Exception as exc:
        raise ProcessingError(f"pipeline failed: {exc}") from exc
