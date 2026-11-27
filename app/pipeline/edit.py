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


def apply_edit(img: np.ndarray, mask: np.ndarray, edit: EditRequest) -> np.ndarray:
    """Run the selected tool on ``img`` (BGR) inside ``mask`` (any size; white / non-zero = selected)."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not mask.any():
        raise ValidationError("the selection is empty: tap or paint the area first")

    if edit.kind == "inpaint":
        return inpaint(
            img, mask, radius=edit.radius, algorithm=edit.algorithm, iterations=edit.iterations
        )
    if edit.kind == "blur":
        return apply_selection_blur(img, mask, edit.blur_strength, edit.target)
    if edit.kind == "tattoo":
        foreground = person_mask(img)
        if edit.algorithm == "lama":
            try:
                return remove_tattoo_lama(img, mask, strength=edit.strength, foreground=foreground)[
                    0
                ]
            except ModelNotFoundError:
                logger.warning("tattoo: LaMa model missing, using the colour algorithm")
        return remove_tattoo_info(
            img, mask, strength=edit.strength, grow=edit.grow, foreground=foreground
        )[0]
    return img
