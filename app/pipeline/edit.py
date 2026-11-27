"""Selection tools (object removal, tattoo removal, selective blur) as a pipeline stage."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.api.schemas import EditRequest
from app.exceptions import ModelNotFoundError, ValidationError
from app.pipeline.background import apply_selection_blur
from app.pipeline.inpaint import inpaint
from app.pipeline.tattoo import person_mask, remove_tattoo_info, remove_tattoo_lama

logger = logging.getLogger(__name__)


def apply_edit(
    img: np.ndarray, mask: np.ndarray, edit: EditRequest
) -> tuple[np.ndarray, str | None]:
    """Run the selected tool on ``img`` (BGR) inside ``mask`` (any size; white / non-zero = selected).

    Returns the image and an optional hint for the user.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not mask.any():
        raise ValidationError("the selection is empty: tap or paint the area first")

    if edit.kind == "inpaint":
        out = inpaint(
            img, mask, radius=edit.radius, algorithm=edit.algorithm, iterations=edit.iterations
        )
        return out, None
    if edit.kind == "blur":
        return apply_selection_blur(img, mask, edit.blur_strength, edit.target), None
    if edit.kind == "tattoo":
        foreground = person_mask(img)
        info = None
        if edit.algorithm == "lama":
            try:
                out, info = remove_tattoo_lama(
                    img, mask, strength=edit.strength, foreground=foreground
                )
            except ModelNotFoundError:
                logger.warning("tattoo: LaMa model missing, using the colour algorithm")
        if info is None:
            out, info = remove_tattoo_info(
                img, mask, strength=edit.strength, grow=edit.grow, foreground=foreground
            )
        return out, tattoo_note(info)
    return img, None


def tattoo_note(info: dict) -> str:
    """What to tell the user after a tattoo removal (the wording the tool always used)."""
    return {
        "lama": "Tattoo: if some is left, paint over it again (or a little closer around it) and run it once more.",
        "detected": f"Tattoo: removed {info.get('ink_pct')}% of the painted area. If some is left, raise the "
        "sensitivity or paint a little closer around it.",
        "none": "Tattoo: found no ink in that area. Raise the sensitivity, or paint over the tattoo itself.",
    }.get(info.get("mode"), "Tattoo removed.")
