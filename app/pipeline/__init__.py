"""Image processing pipeline orchestrator.

Chains stages in fixed order: pre-process → background → selection tool → face → grain → upscale → filters.
Each stage can be individually enabled/disabled via the ProcessRequest.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import cv2
import numpy as np

from app.api.schemas import EditRequest, ProcessRequest
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
    steps: list = field(default_factory=list)  # [{"name", "seconds", "cached"}] per stage that ran
    notes: list = field(default_factory=list)  # hints for the user (e.g. from tattoo removal)


# --- stage cache -----------------------------------------------------------------------------------------------
# The result after each stage is kept for the last few runs, keyed by everything that stage depends on (the input
# image, every earlier stage's settings and uploads, and its own). Changing a late step (grain, a filter) then reuses
# the background, selection tools and face swap instead of recomputing them (15-25 s -> about a second).

_CACHE_MAX = 16
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = Lock()


def clear_stage_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _digest(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        if part is None:
            h.update(b"\x00none")
        elif isinstance(part, np.ndarray):
            h.update(str((part.shape, part.dtype.str)).encode())
            h.update(np.ascontiguousarray(part).data)
        elif isinstance(part, (list, tuple)):
            h.update(_digest(*part).encode())
        else:
            h.update(str(part).encode())
        h.update(b"|")
    return h.hexdigest()


def _snapshot(state: dict) -> dict:
    return {
        k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
        for k, v in state.items()
    }


@dataclass
class _Stage:
    name: str  # shown in the progress line
    key: tuple  # what this stage depends on, besides the earlier stages
    run: Callable[[dict], None]  # updates the state in place


def _stages(
    request: ProcessRequest,
    background_mask,
    background_image,
    matte,
    face_donor,
    face_donors_extra,
    face_keep,
    edits: list[tuple[EditRequest, np.ndarray | None]],
) -> list[_Stage]:
    """The enabled stages, in the fixed order: background -> selection tools -> face -> grain -> upscale -> filters."""
    out: list[_Stage] = []

    if request.background.enabled:

        def run_background(st):
            current = st["current"]
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
                mask = get_matting_model(request.background.model_name).predict(current)
            st["mask_debug"] = get_alpha_debug(mask)
            bg = request.background
            if bg.mode == "blur":
                current = apply_background_blur(current, mask, bg.blur_strength)
            elif bg.mode in ("color", "image"):
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
                # Transparency is carried alongside and put back at the end, so every later stage sees plain BGR.
                st["alpha"] = current[:, :, 3]
                current = current[:, :, :3]
            st["current"] = current

        out.append(
            _Stage(
                "Background",
                (request.background.model_dump_json(), background_mask, background_image, matte),
                run_background,
            )
        )

    labels = {"inpaint": "Remove object", "tattoo": "Remove tattoo", "blur": "Blur area"}
    for edit, emask in edits:
        if emask is None:
            raise ProcessingError(f"the {labels[edit.kind].lower()} tool needs a selection mask")

        def run_edit(st, edit=edit, emask=emask):
            from app.pipeline.edit import apply_edit

            st["current"], note = apply_edit(st["current"], emask, edit)
            if note:
                st["notes"].append(note)

        out.append(_Stage(labels[edit.kind], (edit.model_dump_json(), emask), run_edit))

    if request.face.enabled:
        if face_donor is None:
            raise ProcessingError("face replacement needs a donor photo")

        def run_face(st):
            from app.models.face import get_face_engine

            f = request.face
            before = st["current"]
            current = get_face_engine().swap(
                before,
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
                current = keep_original(current, before, k)
            st["current"] = current

        out.append(
            _Stage(
                "Face swap",
                (request.face.model_dump_json(), face_donor, face_donors_extra or [], face_keep),
                run_face,
            )
        )

    if request.grain.enabled:

        def run_grain(st):
            st["current"], st["grain_only"] = apply_grain(st["current"], request.grain.intensity)

        out.append(_Stage("Grain", (request.grain.model_dump_json(),), run_grain))

    if request.upscale.enabled and request.upscale.scale > 1:

        def run_upscale(st):
            algorithm = None if request.upscale.algorithm == "interp" else request.upscale.algorithm
            if algorithm is None:  # fast LANCZOS4 path, no model
                up = cv2.resize(
                    st["current"],
                    None,
                    fx=request.upscale.scale,
                    fy=request.upscale.scale,
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                up = upscale(st["current"], request.upscale.scale, algorithm)
            st["current"] = st["upscaled"] = up

        out.append(_Stage("Upscale", (request.upscale.model_dump_json(),), run_upscale))

    if request.filters.enabled:

        def run_filters(st):
            fl = request.filters
            bgr = apply_color_adjustments(
                st["current"],
                brightness=fl.brightness,
                contrast=fl.contrast,
                saturation=fl.saturation,
                sharpness=fl.sharpness,
            )
            if fl.vignette_strength > 0:
                bgr = add_vignette(bgr, fl.vignette_strength)
            if fl.sepia:
                bgr = apply_sepia(bgr)
            if fl.grayscale_blend > 0:
                bgr = apply_grayscale(bgr, fl.grayscale_blend)
            if fl.anime:
                bgr = apply_anime_style(bgr, fl.anime_style)
            if fl.painting:
                bgr = apply_paint_style(bgr, fl.painting_style)
            st["current"] = bgr

        out.append(_Stage("Filters", (request.filters.model_dump_json(),), run_filters))
    return out


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
    edit_masks: Optional[list] = None,
    *,
    use_cache: bool = False,
    progress: Optional[Callable[[dict], None]] = None,
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
        edit_masks: one selection per entry of ``request.edits``, same order.
        matte: optional 8-bit grayscale subject matte from ``/api/v1/matte`` (255 = subject); skips the model.
        use_cache: reuse stage results from earlier runs with the same inputs (the API turns this on).
        progress: called before each stage with ``{"steps": [names], "done": n, "current": name}``.

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

        edits: list[tuple[EditRequest, np.ndarray | None]] = []
        if request.edit.kind != "none":
            edits.append((request.edit, edit_mask))
        masks = list(edit_masks or [])
        for i, e in enumerate(request.edits):
            if e.kind != "none":
                edits.append((e, masks[i] if i < len(masks) else None))

        stages = _stages(
            request,
            background_mask,
            background_image,
            matte,
            face_donor,
            face_donors_extra,
            face_keep,
            edits,
        )
        state: dict = {
            "current": original.copy(),
            "alpha": None,
            "mask_debug": None,
            "grain_only": None,
            "upscaled": None,
            "notes": [],
        }
        steps: list[dict] = []
        key = _digest(original)
        names = [s.name for s in stages]
        for i, stage in enumerate(stages):
            key = _digest(key, stage.name, stage.key)
            if progress is not None:
                progress({"steps": names, "done": i, "current": stage.name})
            started = time.perf_counter()
            hit = None
            if use_cache:
                with _cache_lock:
                    hit = _cache.get(key)
                    if hit is not None:
                        _cache.move_to_end(key)
            if hit is not None:
                state = _snapshot(hit)
            else:
                stage.run(state)
                if use_cache:
                    with _cache_lock:
                        _cache[key] = _snapshot(state)
                        while len(_cache) > _CACHE_MAX:
                            _cache.popitem(last=False)
            steps.append(
                {
                    "name": stage.name,
                    "seconds": round(time.perf_counter() - started, 2),
                    "cached": hit is not None,
                }
            )
        if progress is not None:
            progress({"steps": names, "done": len(stages), "current": None})

        current = state["current"]
        alpha_keep = state["alpha"]
        mask_debug = state["mask_debug"]
        grain_only = state["grain_only"]
        upscaled_out = state["upscaled"]
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
            steps=steps,
            notes=state["notes"],
        )
    except (ProcessingError, ValidationError):
        raise  # ValidationError keeps its 422 (e.g. no face found in the donor photo)
    except Exception as exc:
        raise ProcessingError(f"pipeline failed: {exc}") from exc
