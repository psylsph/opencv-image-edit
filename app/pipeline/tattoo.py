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
# Shaded skin (shadows, creases, wrinkles) is darker than lit skin but keeps skin's reddish hue; ink does not. A pixel is
# ink if it is very dark (more than this many L units below the local skin) or its hue is off the skin's by at least a gap
# that shrinks as the sensitivity rises (14 -> 6 colour units).
VERY_DARK = 60.0
HUE_GAP_AT_0, HUE_GAP_AT_1 = 14.0, 6.0
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


def _ink_colour(lab: np.ndarray, cand: np.ndarray) -> np.ndarray | None:
    """The tattoo's own colour: the densest colour peak among the darkest half of the candidate ink pixels."""
    px = lab[cand]
    if len(px) < 20:
        return None
    px = px[px[:, 0] <= np.percentile(px[:, 0], 50)]  # the dark core, not the soft edge
    q = np.stack([px[:, 0] * 0.5, px[:, 1], px[:, 2]], axis=1)
    bins = np.floor(q / 5.0).astype(np.int64)
    keys, counts = np.unique(bins, axis=0, return_counts=True)
    peak = keys[int(np.argmax(counts))]
    near = np.max(np.abs(bins - peak), axis=1) <= 1
    return np.median(px[near], axis=0).astype(np.float32)


def _on_skin_ink_line(
    lab: np.ndarray, ref: np.ndarray, ink_colour: np.ndarray, tol: float
) -> np.ndarray:
    """Pixels whose colour is a mix of the local skin colour ``ref`` and the ink colour (bool map).

    Ink on skin, including its soft anti-aliased edge, lies on the straight line between those two colours. A different
    dark colour (a hair, a freckle, a shadow, an eyebrow) is off that line and is left alone.
    """
    axis = ink_colour[None, None, :] - ref
    rel = lab - ref
    w = np.array([0.5, 1.0, 1.0], np.float32)
    axis_w, rel_w = axis * w, rel * w
    span2 = np.maximum((axis_w**2).sum(-1), 1e-6)
    t = np.clip((rel_w * axis_w).sum(-1) / span2, 0.0, 1.2)
    resid = np.linalg.norm(rel_w - t[..., None] * axis_w, axis=-1)
    return (t > 0.1) & (resid < np.maximum(tol, 0.22 * np.sqrt(span2)))


def find_ink(
    img_bgr: np.ndarray, mask: np.ndarray, sensitivity: float, foreground: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(ink, skin, lab)``: 0/1 masks of the ink pixels and of the clean skin, plus the LAB image.

    Ink = selected pixels that (1) are darker than the skin around them by the sensitivity's margin, (2) are not
    strongly coloured, (3) are the *tattoo's own colour* (a mix of skin and the dominant dark colour: so a hair, a
    freckle or an eyebrow of a different colour is not touched), and (4) belong to a patch bigger than a speck of
    skin texture. Pixel by pixel: nothing else in the area changes.
    """
    prep = _Prep(img_bgr, mask, foreground)
    s = float(np.clip(sensitivity, 0.0, 1.0))
    margin = max(margin_for(s), GRAIN_FLOOR * prep.grain)
    k = np.ones((3, 3), np.uint8)
    sel = prep.binary > 0
    drop = prep.ref_luma - prep.luma
    # Hue gap from the skin (a, b only): shadows and creases stay on the skin's hue, ink does not.
    ab_ref = (
        np.median(prep.lab[prep.skin][:, 1:], axis=0)
        if prep.skin.any()
        else np.array([145.0, 150.0], np.float32)
    )
    off_hue = np.hypot(prep.lab[..., 1] - ab_ref[0], prep.lab[..., 2] - ab_ref[1])
    gap = HUE_GAP_AT_0 + (HUE_GAP_AT_1 - HUE_GAP_AT_0) * s
    inkish = (drop > VERY_DARK) | (off_hue > gap)
    dark = (drop > margin) & inkish & prep.chroma_ok & prep.near_skin & sel
    cand = dark
    # The ink colour comes from the strongest (not the faintest) candidates, so it does not move with the slider.
    strong = (
        ((prep.ref_luma - prep.luma) > max(margin_for(0.5), GRAIN_FLOOR * prep.grain))
        & ((drop > VERY_DARK) | (off_hue > HUE_GAP_AT_0))
        & prep.chroma_ok
        & prep.near_skin
        & sel
    )
    # A faint tattoo has no 'strong' pixels: then the ink colour comes from the candidates at the current sensitivity.
    colour = _ink_colour(
        prep.lab, strong if strong.sum() >= max(30, int(0.001 * sel.sum())) else dark
    )
    if colour is not None:
        ref = np.stack(
            [
                prep.ref_luma,
                *[
                    np.full_like(prep.ref_luma, c)
                    for c in np.median(prep.lab[prep.skin][:, 1:], axis=0)
                ],
            ],
            axis=-1,
        )
        cand = dark & _on_skin_ink_line(prep.lab, ref, colour, tol=6.0 + 10.0 * s)
    ink = cand.astype(np.uint8)
    # Skin texture is full of 1-2 px dark specks (pores, stubble, grain): ignore anything narrower than a few pixels,
    # and anything too small to be part of a drawing.
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        smallest = max(8, prep.r * prep.r // 2, int(0.01 * areas.max()))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < smallest:
                ink[comp == i] = 0
    # Just the ink's own soft edge (one pixel), so nothing beyond the drawing is touched.
    ink = cv2.dilate(ink, k, iterations=1) & prep.binary
    # Clean skin for the fill: every skin-looking pixel near the selection, lit or shaded alike (the caller drops the
    # pixels right around the ink). Leaving shaded skin out biased the replacement lighter than its surroundings.
    clean = prep.pool & prep.fg
    return ink.astype(np.uint8), clean.astype(np.uint8), prep.lab


def nearest_skin_colour(lab: np.ndarray, skin_ok: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """For every ``hole`` pixel, the colour (LAB) of the nearest real skin pixel: laser-style, pixel for pixel.

    A copy of an actual neighbouring skin pixel, so pores and grain come along and no smooth patch appears. Works on the
    area around the hole only. Pixels with no skin anywhere near get the median skin colour.
    """
    out = np.broadcast_to(
        np.median(lab[skin_ok], axis=0) if skin_ok.any() else lab.reshape(-1, 3).mean(0), lab.shape
    ).copy()
    ys, xs = np.nonzero(hole > 0)
    if ys.size == 0:
        return out
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(lab.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(lab.shape[1], xs.max() + 1 + pad)
    ok = skin_ok[y0:y1, x0:x1] & (hole[y0:y1, x0:x1] == 0)
    if not ok.any():
        return out
    src = np.where(ok, 0, 1).astype(np.uint8)  # zero = a skin pixel to copy from
    _d, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    zy, zx = np.nonzero(ok)  # labels count the zero pixels in row-major order, starting at 1
    idx = np.clip(labels - 1, 0, len(zy) - 1)
    near = lab[y0:y1, x0:x1][zy[idx], zx[idx]]
    region = out[y0:y1, x0:x1]
    h = hole[y0:y1, x0:x1] > 0
    region[h] = near[h]
    out[y0:y1, x0:x1] = region
    return out


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
        # Pixel for pixel: each ink pixel becomes the colour of the nearest real skin pixel (pores and grain come along),
        # softened a little by the smooth local skin colour so single copied pixels do not show as streaks.
        copied = nearest_skin_colour(lab, clean, ink)
        smooth = local_skin(lab, clean, ink, r)
        fill_lab = 0.65 * copied + 0.35 * smooth
        filled = cv2.cvtColor(np.clip(fill_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # Ink pixels are replaced completely; only a 1 px feathered rim beyond them.
        soft = np.clip(cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 0.7) * 1.3, 0.0, 1.0)
        soft = np.where(ink > 0, 1.0, soft).astype(np.float32)
        soft *= binary.astype(np.float32)  # never outside the selection
        if foreground is not None:
            soft *= (foreground > 0).astype(np.float32)  # and never on the backdrop
        filled = add_skin_texture(
            filled, current, soft, clean.astype(np.uint8)
        )  # grain the copied pixels lack
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
