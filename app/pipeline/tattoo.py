"""Tattoo removal: find the ink inside the selection and repaint just those pixels with the surrounding skin.

The selection is only the *search area*, so it is fine to paint loosely and run off the arm onto the background:

1. **Skin colour**: the dominant skin-coloured pixels in and around the selection (a generic skin-tone prior in
   YCrCb narrows it to skin, then the median is trimmed of outliers such as ink, moles and hair).
2. **Skin region**: everything near the selection that is close to the *local* skin colour (so shading is followed),
   plus every hole completely surrounded by it. Ink on an arm is such a hole; hair, clothing and the background beside
   the arm connect to the outside, so they are not.
3. **Ink** = selected pixels that are inside the skin region but not skin-coloured (plus their soft edge). Background
   pixels inside the selection are not enclosed by skin, so they are never touched.
4. **Replace** only the ink by inpainting it (LaMa, or TELEA if the model is missing). Each patch is enclosed by
   skin, so what the inpainter copies from around it is skin and no background colour can bleed in. Then skin-like
   grain is added and the result blended softly.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ModelNotFoundError, ValidationError
from app.pipeline.inpaint import inpaint


def _radius(shape: tuple[int, ...]) -> int:
    return max(3, int(round(0.008 * max(shape[:2]))))


def skin_prior(bgr: np.ndarray) -> np.ndarray:
    """Pixels that are plausibly skin of any tone (YCrCb ranges): used only to find the dominant skin colour."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    return (cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 130)


def skin_colour(lab: np.ndarray, prior: np.ndarray, zone: np.ndarray) -> tuple[np.ndarray, float]:
    """The dominant skin colour (LAB) in ``zone`` and a robust spread.

    "Dominant" = the densest peak of skin-looking colours: a limb is a big smooth patch of one colour, while ink,
    hair, the background and its look-alikes are scattered over many different colours or are a small share.
    """
    pool = zone & prior
    if pool.sum() < 50:
        pool = zone  # no skin-looking pixels at all: fall back to everything in the zone
    px = lab[pool]
    if len(px) == 0:
        raise ValidationError("the selection covers the whole picture")
    if len(px) > 40000:  # plenty to find a peak; keeps big photos fast and the result deterministic
        px = px[np.linspace(0, len(px) - 1, 40000).astype(int)]
    q = np.stack([px[:, 0] * L_WEIGHT, px[:, 1], px[:, 2]], axis=1)
    bins = np.floor(q / 6.0).astype(np.int64)
    keys, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    # Smooth the histogram a little: a bin's score is its own count plus its 26 neighbours'.
    score = counts.astype(np.float64).copy()
    lookup = {tuple(kk): i for i, kk in enumerate(keys)}
    for i, kk in enumerate(keys):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx or dy or dz:
                        j = lookup.get((kk[0] + dx, kk[1] + dy, kk[2] + dz))
                        if j is not None:
                            score[i] += 0.5 * counts[j]
    peak = keys[int(np.argmax(score))]
    near = np.max(np.abs(bins - peak), axis=1) <= 1
    inliers = px[near]
    for _ in range(2):
        med = np.median(inliers, axis=0)
        d = _dist(inliers, med)
        keep = inliers[d <= max(np.percentile(d, 80), 1.0)]
        if len(keep) < 20:
            break
        inliers = keep
    med = np.median(inliers, axis=0)
    spread = float(max(np.median(_dist(inliers, med)), 1.5))
    return med.astype(np.float32), spread


REACH_FACTOR = (
    0.25  # how far to look for skin, as a multiple of the selection's size (sqrt of its area)
)
NEED_SIDES = 6  # of 8 directions must have skin within reach for a pixel to count as ink-on-skin
L_WEIGHT = (
    0.5  # brightness counts half as much as colour when deciding "is this skin": limbs are shaded
)


def _dist(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = lab - ref
    return np.sqrt((L_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def skin_around(skin: np.ndarray, reach: int, need: int) -> np.ndarray:
    """Pixels that have skin within ``reach`` px in at least ``need`` of the 8 compass directions (bool map).

    Ink on an arm has skin on both sides; hair edges, the background beside the arm and clothing have skin in only
    the few directions that face the body.
    """
    n = reach + 1
    h, w = skin.shape
    pad = reach
    src = cv2.copyMakeBorder(
        skin.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    eye = np.eye(n, dtype=np.uint8)
    kernels = []
    row = np.ones((1, n), np.uint8)
    col = np.ones((n, 1), np.uint8)
    # (kernel, anchor): the anchor sits at the pixel's own end of the ray, so the ray extends one way only.
    kernels.append((row, (0, 0)))  # right
    kernels.append((row, (reach, 0)))  # left
    kernels.append((col, (0, 0)))  # down
    kernels.append((col, (0, reach)))  # up
    kernels.append((eye, (0, 0)))  # down-right
    kernels.append((eye, (reach, reach)))  # up-left
    kernels.append((np.fliplr(eye).copy(), (reach, 0)))  # down-left
    kernels.append((np.fliplr(eye).copy(), (0, reach)))  # up-right
    count = np.zeros((h, w), np.uint8)
    for kern, anchor in kernels:
        hit = cv2.dilate(src, kern, anchor=anchor)
        count += hit[pad : pad + h, pad : pad + w]
    return count >= need


def find_ink(
    img_bgr: np.ndarray, mask: np.ndarray, grow: float, level: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(ink, skin, lab)``: uint8 0/1 masks of the tattoo pixels and of the clean skin, plus the LAB image.

    ``level`` 0 is the careful detector; 1 is relaxed (looser skin test, fewer directions, any colour difference);
    2 treats the whole selection as the tattoo (the caller asks for it when nothing was found).
    """
    if level >= 2:
        binary2 = (mask > 0).astype(np.uint8)
        return (
            binary2,
            np.zeros_like(binary2),
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32),
        )
    if level == 1:
        grow = 1.0
    grow = float(np.clip(grow, 0.0, 1.0))
    r = _radius(img_bgr.shape)
    k = np.ones((3, 3), np.uint8)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    binary = (mask > 0).astype(np.uint8)
    roi = cv2.dilate(binary, k, iterations=4 * r) > 0
    prior = skin_prior(img_bgr)
    ring = (cv2.dilate(binary, k, iterations=3 * r) > 0) & (
        cv2.dilate(binary, k, iterations=1) == 0
    )
    med, spread = skin_colour(lab, prior, ring if (ring & prior).sum() >= 100 else roi)

    # Skin shading changes across a limb, so compare each pixel with the skin colour *near it*: first a loose pass
    # against the global colour, then the local skin colour spread outward from those pixels (normalised convolution).
    loose = (_dist(lab, med) < max(spread * 6.0, 14.0)) & roi
    w = loose.astype(np.float32)
    sigma = 6.0 * r
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    local = cv2.GaussianBlur(lab * w[:, :, None], (0, 0), sigma) / np.maximum(den, 1e-6)[:, :, None]
    local = np.where((den > 0.02)[:, :, None], local, med[None, None, :])
    dist = _dist(lab, local)
    sp = float(max(np.median(dist[loose]) if loose.any() else spread, 1.5))
    thr = max(
        sp * (5.0 - 3.0 * grow), 6.0
    )  # more "catch faint ink" = a stricter idea of what is skin
    skin = ((dist < thr) & roi).astype(np.uint8)

    # Candidate ink: selected pixels that are not skin-coloured. It only counts where skin is close by on (nearly)
    # all sides, so hair edges, the background next to the arm and clothing are never mistaken for ink.
    # How far to look for skin: far enough to cross the biggest tattoo the selection could hold, no further.
    reach = int(
        min(max(6 * r, REACH_FACTOR * float(np.sqrt(binary.sum())), 12), 0.3 * max(binary.shape))
    )
    need = NEED_SIDES if grow < 0.75 else NEED_SIDES - 1
    if level == 1:
        reach, need = min(2 * reach, int(0.4 * max(binary.shape))), 4
    near_skin = skin_around(skin > 0, reach, need)
    # Ink absorbs light: it is darker than the skin around it, or strongly coloured. Lighter, skin-ish colours (a pale
    # wall, a highlight) are never ink.
    darker = (local[..., 0] - lab[..., 0]) > 6.0
    vivid = np.hypot(lab[..., 1] - local[..., 1], lab[..., 2] - local[..., 2]) > 22.0
    differs = np.ones_like(darker) if level == 1 else (darker | vivid)
    ink = ((binary > 0) & (skin == 0) & near_skin & differs).astype(np.uint8)
    region = near_skin.astype(np.uint8)
    # Tiny specks are grain, not tattoo.
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(4, r):
            ink[comp == i] = 0
    # The ink's soft anti-aliased edge, but only inside the selection and the skin region.
    ink = cv2.dilate(ink, k, iterations=max(1, r // 2)) & binary & region.astype(np.uint8)
    return ink.astype(np.uint8), skin, lab


def add_skin_texture(
    filled_bgr: np.ndarray, original: np.ndarray, patch: np.ndarray, skin: np.ndarray
) -> np.ndarray:
    """Add fine grain to the smooth fill at the level measured on clean skin (deterministic noise)."""
    lab = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_l = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    fine = orig_l - cv2.GaussianBlur(orig_l, (0, 0), 1.5)
    px = fine[skin > 0]
    if px.size < 50:
        return filled_bgr
    need = float(np.median(np.abs(px - np.median(px))) * 1.4826)  # robust grain level of the skin
    if need < 0.3:
        return filled_bgr
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.standard_normal(orig_l.shape).astype(np.float32), (0, 0), 0.7)
    noise /= max(float(noise.std()), 1e-6)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + noise * need * patch, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def remove_tattoo_info(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    grow: float = 0.3,
    algorithm: str = "lama",
) -> tuple[np.ndarray, dict]:
    """Remove the tattoo inside ``mask`` (uint8, non-zero = search here).

    Returns ``(image, info)``; ``info["mode"]`` says how the ink was found: ``detected``, ``relaxed`` (the careful
    detector found too little) or ``selection`` (nothing was found, so the whole selection was treated as tattoo).
    ``info["ink_pct"]`` is the share of the selection that was repainted. Never silently does nothing.

    Pixels outside the selection, and selected pixels that are not ink on skin (the background, clothing), are
    returned unchanged.
    """
    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValidationError("mask size does not match the image")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    binary = (mask > 0).astype(np.uint8)
    if not binary.any() or strength <= 0:
        return img_bgr.copy(), {"mode": "none", "ink_pct": 0.0}

    r = _radius(img_bgr.shape)
    sel_area = int(binary.sum())
    enough = max(30, int(0.02 * sel_area))  # less than this is probably noise, not the tattoo
    for level, mode in ((0, "detected"), (1, "relaxed"), (2, "selection")):
        ink, skin, lab = find_ink(img_bgr, mask, grow, level)
        if level == 0:
            skin0 = skin
        if int(ink.sum()) >= enough:
            break
        if level == 1 and ((skin0 > 0) & (binary > 0)).sum() >= 0.95 * sel_area:
            return img_bgr.copy(), {
                "mode": "clean",
                "ink_pct": 0.0,
            }  # the selection is plain skin: nothing to remove

    k = np.ones((3, 3), np.uint8)
    clean = (skin > 0) & (
        cv2.dilate(ink, k, iterations=2) == 0
    )  # skin that is certainly not ink or its edge
    hole = (cv2.dilate(ink, k, iterations=1) & binary).astype(np.uint8)
    # Every ink patch is enclosed by skin, so what an inpainter copies from around it is skin.
    try:
        filled = inpaint(img_bgr, hole * 255, radius=max(3, r), algorithm=algorithm, iterations=2)
    except ModelNotFoundError:
        filled = inpaint(img_bgr, hole * 255, radius=max(3, r), algorithm="telea")

    soft = np.clip(
        cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(0.8, r / 3.0)) * 1.5, 0.0, 1.0
    )
    soft *= binary.astype(np.float32)  # never outside the selection
    filled = add_skin_texture(filled, img_bgr, soft, clean.astype(np.uint8))

    a = (soft * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    out = img_bgr.astype(np.float32) * (1.0 - a) + filled.astype(np.float32) * a
    info = {"mode": mode, "ink_pct": round(100.0 * float(ink.sum()) / max(sel_area, 1), 1)}
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info


def remove_tattoo(img_bgr: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """:func:`remove_tattoo_info` without the info."""
    return remove_tattoo_info(img_bgr, mask, **kwargs)[0]
