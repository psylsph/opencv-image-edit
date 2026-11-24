"""Tattoo removal: replace the inked skin with clean skin that matches its surroundings.

Plain inpainting of a tattoo mask leaves three tell-tales: the mask usually misses the faint ink halo and fine
outlines, the filled patch is smoother than real skin, and the edge is visible. So:

1. **Skin sample**: the colour (LAB) of a ring of skin around the selection, robustly (median, outliers dropped).
2. **Catch the ink**: pixels near the selection that differ from that skin colour and touch the selection are added
   to the mask (``grow`` sets how eagerly).
3. **Fill** with the usual inpainting (LaMa by default).
4. **Skin texture**: the fill is smooth, so noise at the ring's own fine-grain level is added back inside the patch.
5. **Blend** with a soft edge; ``strength`` < 1 only fades the tattoo instead of removing it.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ValidationError
from app.pipeline.inpaint import inpaint

MAX_GROWTH = 6.0  # the grown mask may be at most this many times the selection's area, else fall back to a dilation


def _radius(shape: tuple[int, ...]) -> int:
    return max(4, int(round(0.012 * max(shape[:2]))))


def skin_sample(lab: np.ndarray, mask: np.ndarray, r: int) -> tuple[np.ndarray, float]:
    """Median LAB colour of the ring of skin around ``mask`` and a robust spread (median distance to it)."""
    k = np.ones((3, 3), np.uint8)
    # Start well clear of the selection: it usually stops short of the ink's soft edge, and that ink must not
    # count as "skin".
    near = cv2.dilate(mask, k, iterations=2 * r)
    far = cv2.dilate(mask, k, iterations=6 * r)
    ring = (far > 0) & (near == 0)
    px = lab[ring]
    if len(px) < 50:  # selection fills the frame: use everything outside it
        px = lab[mask == 0]
    if len(px) == 0:
        raise ValidationError("the selection covers the whole picture")
    inliers = px
    for _ in range(2):  # drop the outliers (leftover ink, moles, hair), twice
        med = np.median(inliers, axis=0)
        d = np.linalg.norm(inliers - med, axis=1)
        inliers = inliers[d <= max(np.percentile(d, 75), 1.0)]
    med = np.median(inliers, axis=0)
    spread = float(max(np.median(np.linalg.norm(inliers - med, axis=1)), 1.5))
    return med.astype(np.float32), spread


def grow_selection(
    lab: np.ndarray, mask: np.ndarray, med: np.ndarray, spread: float, grow: float, r: int
) -> np.ndarray:
    """The selection plus nearby pixels that look like skin mixed with this tattoo's ink and touch the selection.

    "Looks like": the pixel's colour lies near the line from the skin colour to the ink colour (the ink colour is
    measured inside the selection). Hair, shadows and clothing sit off that line, so they are left alone.
    """
    grow = float(np.clip(grow, 0.0, 1.0))
    k = np.ones((3, 3), np.uint8)
    base = cv2.dilate(mask, k, iterations=max(1, r // 3))
    if grow > 0:
        sel_px = lab[mask > 0]
        far = np.linalg.norm(sel_px - med, axis=1)
        inked = sel_px[far > max(spread * 3.0, 6.0)]
        if len(inked) >= 20:
            ink = np.median(inked, axis=0)
            axis = ink - med
            span = float(np.linalg.norm(axis))
            zone = cv2.dilate(mask, k, iterations=max(2, int(r * (1.0 + 3.0 * grow))))
            rel = lab - med
            t = np.clip((rel @ axis) / max(span * span, 1e-6), 0.0, 1.0)  # 0 = skin, 1 = full ink
            resid = np.linalg.norm(rel - t[:, :, None] * axis, axis=2)
            min_t = 0.45 - 0.33 * grow  # how faint a trace still counts
            tol = max(spread * 2.5, 0.22 * span)
            extra = ((t > min_t) & (resid < tol) & (zone > 0)).astype(np.uint8)
            cand = ((base > 0) | (extra > 0)).astype(np.uint8)
            n, comp = cv2.connectedComponents(cand, connectivity=8)
            touching = np.unique(comp[mask > 0])
            keep = np.isin(comp, touching[touching > 0])
            base = np.maximum(base, keep.astype(np.uint8))
            base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k, iterations=2)
            base = cv2.dilate(base, k, iterations=max(1, r // 4))
    if base.sum() > MAX_GROWTH * max(int(mask.sum()), 1):
        base = cv2.dilate(mask, k, iterations=max(2, r // 2))
    return base


def add_skin_texture(
    result: np.ndarray,
    original: np.ndarray,
    patch: np.ndarray,
    ring: np.ndarray,
    amount: float = 1.0,
) -> np.ndarray:
    """Add fine grain to the smooth fill, at the level measured on the skin ring (deterministic noise)."""
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_l = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    fine = orig_l - cv2.GaussianBlur(orig_l, (0, 0), 1.5)
    if ring.sum() < 50:
        return result
    fine_ring = fine[ring > 0]
    # Robust: skin grain only (median absolute deviation), not the hair and edges that happen to be in the ring.
    target = float(np.median(np.abs(fine_ring - np.median(fine_ring))) * 1.4826)
    fill_l = lab[:, :, 0]
    have = (
        float((fill_l - cv2.GaussianBlur(fill_l, (0, 0), 1.5))[patch > 0.5].std())
        if (patch > 0.5).any()
        else 0.0
    )
    need = float(np.sqrt(max(target**2 - have**2, 0.0))) * float(np.clip(amount, 0.0, 1.5))
    if need < 0.3:
        return result
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.standard_normal(fill_l.shape).astype(np.float32), (0, 0), 0.7)
    noise /= max(float(noise.std()), 1e-6)
    lab[:, :, 0] = np.clip(fill_l + noise * need * patch, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def remove_tattoo(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    grow: float = 0.3,
    algorithm: str = "lama",
    iterations: int = 2,
) -> np.ndarray:
    """Remove the tattoo under ``mask`` (uint8, non-zero = tattoo). Returns BGR uint8, same size."""
    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValidationError("mask size does not match the image")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    binary = (mask > 0).astype(np.uint8)
    if not binary.any() or strength <= 0:
        return img_bgr.copy()

    r = _radius(img_bgr.shape)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med, spread = skin_sample(lab, binary, r)
    grown = grow_selection(lab, binary, med, spread, grow, r)

    filled = inpaint(
        img_bgr, grown * 255, radius=max(3, r // 2), algorithm=algorithm, iterations=iterations
    )

    k = np.ones((3, 3), np.uint8)
    ring = (
        (cv2.dilate(grown, k, iterations=3 * r) > 0) & (cv2.dilate(grown, k, iterations=r) == 0)
    ).astype(np.uint8)
    sigma = max(1.2, r / 4.0)
    soft = np.clip(cv2.GaussianBlur(grown.astype(np.float32), (0, 0), sigma) * 1.6, 0.0, 1.0)
    skin_like = (ring > 0) & (np.linalg.norm(lab - med, axis=2) < max(spread * 5.0, 10.0))
    filled = add_skin_texture(filled, img_bgr, soft, skin_like.astype(np.uint8))

    a = (soft * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    out = img_bgr.astype(np.float32) * (1.0 - a) + filled.astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
