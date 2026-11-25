"""Tattoo removal: find the dark ink inside the selection and repaint it with the flesh tone next to it.

Deliberately simple and predictable:

1. **Search area** = the painted selection. Nothing outside it is ever touched.
2. **Ink** = selected pixels that are much *darker* than the skin around them and not strongly coloured: dark brown,
   black, grey-blue. "The skin around them" is the local skin brightness, estimated from the skin-looking, not-dark
   pixels near the selection and spread over it (so shading across a limb is followed). The ``sensitivity`` slider is
   the darkness margin: the higher it is, the fainter the ink that is caught, and it only ever removes more.
3. **Replacement** = the local flesh colour, estimated *only from skin pixels* (never from the background, hair or the
   ink itself), plus skin-like grain, blended with a soft edge. So the fill is always a flesh tone and a backdrop
   beside the arm can never tint it.

So paint over the tattoo; going off the arm is fine for backdrops that are not dark and neutral, but dark hair or a dark
jacket inside the painted area counts as "dark" too, so keep the stroke reasonably close.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ValidationError

# Darkness margin (LAB L units below the local skin) that counts as ink: 40 at sensitivity 0 down to 6 at sensitivity 1.
MARGIN_AT_0, MARGIN_AT_1 = 40.0, 6.0
# Ink is dark and not strongly coloured: its LAB chroma (distance from neutral grey) is at most this. A saturated blue wall
# or a red curtain is darker than skin too, but is not brown/black/grey ink.
CHROMA_MAX = 32.0
# The skin reference is built once with this fixed darkness margin, so changing the sensitivity only moves the
# threshold and never changes what counts as clean skin (which once made the slider go the wrong way).
REFERENCE_MARGIN = 18.0
# However sensitive, never accept a darkness margin below this many times the photo's own grain.
GRAIN_FLOOR = 3.0


def _radius(shape: tuple[int, ...]) -> int:
    return max(3, int(round(0.008 * max(shape[:2]))))


def skin_prior(bgr: np.ndarray) -> np.ndarray:
    """Pixels whose hue is plausibly skin of any tone, lit or shaded, even ruddy (generous YCrCb ranges).

    Used to choose *which* colour is the skin: without it the most common colour near a loose selection (a flat wall)
    would win. Blue and green backdrops fail it; a pink wall can pass, which colour alone cannot fix.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    return (cr >= 130) & (cr <= 195) & (cb >= 70) & (cb <= 135)


def _norm_conv(
    values: np.ndarray, weight: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-weighted average of ``values`` over the pixels where ``weight`` is set, and how much weight was there."""
    w = weight.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    num = cv2.GaussianBlur(values * w[..., None], (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None], den


def local_skin(lab: np.ndarray, skin_ok: np.ndarray, hole: np.ndarray, r: int) -> np.ndarray:
    """The flesh colour (LAB) for every ``hole`` pixel, from ``skin_ok`` pixels only, filled inward from the edge.

    Peels the hole from its border toward the middle: each round fills the pixels next to known colour with the local
    average and counts them as known for the next, so every pixel takes the shading right beside it (not the average of
    skin far away). Pixels still unreached at the end fall back to the median skin colour.
    """
    fallback = np.median(lab[skin_ok], axis=0) if skin_ok.any() else lab.reshape(-1, 3).mean(0)
    out = np.broadcast_to(fallback, lab.shape).copy()
    todo = hole > 0
    if not todo.any():
        return out
    ys, xs = np.nonzero(todo)
    pad = int(14 * r)
    y0, y1 = max(0, ys.min() - pad), min(lab.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(lab.shape[1], xs.max() + 1 + pad)
    vals = lab[y0:y1, x0:x1].copy()
    known = skin_ok[y0:y1, x0:x1].astype(np.float32)
    sub_todo = todo[y0:y1, x0:x1].copy()
    res = out[y0:y1, x0:x1]
    sigma = max(1.5, 0.7 * r)
    for _ in range(120):
        est, den = _norm_conv(vals, known, sigma)
        fill = sub_todo & (den > 0.12)
        if not fill.any():
            sigma *= 1.5  # nothing in reach: look a little farther
            if sigma > 12 * r:
                break
            continue
        res[fill] = est[fill]
        vals[fill] = est[fill]
        known[fill] = 1.0
        sub_todo &= ~fill
        if not sub_todo.any():
            break
    out[y0:y1, x0:x1] = res
    return out


def margin_for(sensitivity: float) -> float:
    """Darkness margin (L units) for a sensitivity in 0..1: the higher the sensitivity, the smaller the margin."""
    s = float(np.clip(sensitivity, 0.0, 1.0))
    return MARGIN_AT_0 + (MARGIN_AT_1 - MARGIN_AT_0) * s


def _dist(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Colour distance with brightness counted half as much as colour (limbs are shaded)."""
    d = lab - ref
    return np.sqrt((0.5 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _dominant_colour(px: np.ndarray) -> tuple[np.ndarray, float]:
    """The densest colour peak of ``px`` (N x 3 LAB) and its spread."""
    if len(px) > 40000:
        px = px[np.linspace(0, len(px) - 1, 40000).astype(int)]
    q = np.stack([px[:, 0] * 0.5, px[:, 1], px[:, 2]], axis=1)
    bins = np.floor(q / 6.0).astype(np.int64)
    keys, counts = np.unique(bins, axis=0, return_counts=True)
    score = counts.astype(np.float64)
    lookup = {tuple(kk): i for i, kk in enumerate(keys)}
    for i, kk in enumerate(keys):  # smooth: a bin also gets half of its 26 neighbours
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx or dy or dz:
                        j = lookup.get((kk[0] + dx, kk[1] + dy, kk[2] + dz))
                        if j is not None:
                            score[i] += 0.5 * counts[j]
    peak = keys[int(np.argmax(score))]
    inliers = px[np.max(np.abs(bins - peak), axis=1) <= 1]
    med = np.median(inliers, axis=0).astype(np.float32)
    return med, float(max(np.median(_dist(inliers, med)), 1.5))


class _Prep:
    """Everything about the selection that does not depend on the sensitivity."""

    def __init__(
        self, img_bgr: np.ndarray, mask: np.ndarray, foreground: np.ndarray | None = None
    ) -> None:
        self.r = _radius(img_bgr.shape)
        k = np.ones((3, 3), np.uint8)
        # `foreground` (from background matting) is the person. Everything outside it is off limits: not searched for
        # ink, not sampled for skin colour, never repainted. So no backdrop colour, however skin-like, can get in.
        self.fg = np.ones(img_bgr.shape[:2], bool) if foreground is None else (foreground > 0)
        self.lab = lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.luma = luma = lab[:, :, 0]
        self.binary = binary = ((mask > 0) & self.fg).astype(np.uint8)
        roi = (cv2.dilate(binary, k, iterations=5 * self.r) > 0) & self.fg
        if not roi.any() or roi.all() and not (binary == 0).any():
            raise ValidationError("the selection covers the whole picture")
        # The skin colour comes from what is actually around the tattoo, not from a generic skin-tone range (shaded or
        # ruddy skin would fall outside it): the painted area plus a ring around it, minus the dark pixels (the ink).
        ring = (cv2.dilate(binary, k, iterations=3 * self.r) > 0) & (
            cv2.dilate(binary, k, iterations=1) == 0
        )
        prior = skin_prior(img_bgr)
        near = (binary > 0) | ring
        skinlike = near & prior
        if skinlike.sum() < 50:
            skinlike = (
                near  # nothing looks like skin by hue: fall back to everything near the selection
            )
        cut = float(np.percentile(luma[skinlike], 50)) - 30.0
        cand = skinlike & (luma > cut)
        med, spread = _dominant_colour(lab[cand if cand.sum() >= 50 else skinlike])
        # Skin = skin-hued pixels near the selection of that same tone (any brightness: lit and shaded alike).
        pool = roi & prior & (_dist(lab, med) < max(3.0 * spread, 22.0))
        if pool.sum() < 50:
            pool = roi & (_dist(lab, med) < max(3.0 * spread, 22.0))
        if pool.sum() < 50:
            pool = roi
        self.pool = pool
        skin = pool.copy()
        sigma = 4.0 * self.r
        for _ in range(3):
            ref, _den = _norm_conv(lab, skin, sigma)
            dark = (((ref[..., 0] - luma) > REFERENCE_MARGIN) & (binary > 0)).astype(np.uint8)
            skin = pool & (cv2.dilate(dark, k, iterations=2) == 0)
        self.skin = skin
        self.ref_luma = ref[..., 0]
        # Only where some skin is within reach at all: a dark region far from any skin is not a tattoo on skin.
        _far, far_den = _norm_conv(lab, skin, 10.0 * self.r)
        self.near_skin = far_den > 0.01
        self.chroma_ok = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0) <= CHROMA_MAX
        fine = luma - cv2.GaussianBlur(luma, (0, 0), 1.5)
        px = fine[skin]
        self.grain = float(np.median(np.abs(px - np.median(px))) * 1.4826) if px.size >= 50 else 2.0


def find_ink(
    img_bgr: np.ndarray, mask: np.ndarray, sensitivity: float, foreground: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(ink, skin, lab)``: 0/1 masks of the ink pixels and of the clean skin, plus the LAB image.

    Ink = selected pixels darker than the skin around them by the sensitivity's margin (never below a few times the
    photo's grain) that are not strongly coloured. The skin reference does not depend on the sensitivity, so raising it
    can only add ink.
    """
    prep = _Prep(img_bgr, mask, foreground)
    s = float(np.clip(sensitivity, 0.0, 1.0))
    margin = max(margin_for(s), GRAIN_FLOOR * prep.grain)
    dark = (prep.ref_luma - prep.luma) > margin
    ink = (dark & prep.chroma_ok & prep.near_skin & (prep.binary > 0)).astype(np.uint8)
    # Specks are grain; the ink's soft edge is added back (more at high sensitivity), inside the selection only.
    k = np.ones((3, 3), np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(4, prep.r):
            ink[comp == i] = 0
    ink = cv2.dilate(ink, k, iterations=1 + int(round(2 * s))) & prep.binary
    # Clean skin for the fill: every skin-looking pixel near the selection (shaded ones included) except anything even
    # slightly darker than the skin around it, so the ink's faint halo does not tint the replacement.
    faint = (prep.ref_luma - prep.luma) > 8.0
    clean = prep.pool & ~faint & prep.fg
    return ink.astype(np.uint8), clean.astype(np.uint8), prep.lab


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
    grow: float = 0.5,
    algorithm: str = "",  # accepted and ignored: older clients still send it
    foreground: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Remove the dark tattoo ink inside ``mask`` (uint8, non-zero = search here).

    ``foreground`` (optional, non-zero = the person, e.g. from background matting) keeps everything else out of bounds:
    the skin colour is sampled only from the person and only the person's pixels are ever repainted.
    ``grow`` is the sensitivity (0..1): the higher, the fainter the ink that is caught. ``strength`` (0..1) fades
    instead of erasing. Returns ``(image, info)``; ``info["mode"]`` is ``detected`` or ``none`` (no ink darker than the
    margin in the selection). ``info["ink_pct"]`` is the share of the selection that was repainted.
    Pixels outside the selection are never changed.
    """
    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValidationError("mask size does not match the image")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    binary = (mask > 0).astype(np.uint8)
    if foreground is not None:
        binary = binary & (foreground > 0).astype(np.uint8)  # only the person is searched
    if not binary.any() or strength <= 0:
        return img_bgr.copy(), {"mode": "none", "ink_pct": 0.0, "passes": 0}

    r = _radius(img_bgr.shape)
    sel_area = int(binary.sum())
    enough = max(20, int(0.002 * sel_area))
    current = img_bgr
    union = np.zeros(img_bgr.shape[:2], np.float32)
    total = np.zeros(img_bgr.shape[:2], np.uint8)
    k = np.ones((3, 3), np.uint8)
    passes = 0
    for _ in range(
        2
    ):  # a second look catches the faint halo that only shows once the dark core is gone
        ink, clean, lab = find_ink(current, mask, grow, foreground)
        if int(ink.sum()) < (enough if passes == 0 else max(10, enough // 4)):
            break
        passes += 1
        total = np.maximum(total, ink)
        clean = (clean > 0) & (
            cv2.dilate(ink, k, iterations=3) == 0
        )  # skin that is certainly not ink or its edge
        fill_lab = local_skin(
            lab, clean, ink, r
        )  # the flesh colour next to each ink pixel, from skin pixels only
        filled = cv2.cvtColor(np.clip(fill_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        soft = np.clip(
            cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(0.8, r / 3.0)) * 1.5, 0.0, 1.0
        )
        soft *= binary.astype(np.float32)  # never outside the selection
        if foreground is not None:
            soft *= (foreground > 0).astype(np.float32)  # and never on the backdrop
        filled = add_skin_texture(filled, current, soft, clean.astype(np.uint8))
        out_f = (
            current.astype(np.float32) * (1.0 - soft[:, :, None])
            + filled.astype(np.float32) * soft[:, :, None]
        )
        current = np.clip(out_f + 0.5, 0, 255).astype(np.uint8)
        union = np.maximum(union, soft)
    if passes == 0:
        return img_bgr.copy(), {"mode": "none", "ink_pct": 0.0, "passes": 1}
    a = (union * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    out = img_bgr.astype(np.float32) * (1.0 - a) + current.astype(np.float32) * a
    info = {
        "mode": "detected",
        "ink_pct": round(100.0 * float(total.sum()) / max(sel_area, 1), 1),
        "passes": passes,
    }
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info


def remove_tattoo(img_bgr: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """:func:`remove_tattoo_info` without the info."""
    return remove_tattoo_info(img_bgr, mask, **kwargs)[0]
