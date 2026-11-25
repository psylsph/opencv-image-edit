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


def skin_colour(
    lab: np.ndarray, prior: np.ndarray, zone: np.ndarray, hint: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """The dominant skin colour (LAB) in ``zone`` and a robust spread.

    "Dominant" = the densest peak of skin-looking colours: a limb is a big smooth patch of one colour, while ink,
    hair, the background and its look-alikes are scattered over many different colours or are a small share.
    """
    pool = zone & prior
    if hint is not None:
        # The person's own skin colour (from their face) anchors the search: a skin-coloured wall is not skin.
        near_hint = _dist(lab, np.asarray(hint, np.float32)) < HINT_RANGE
        anchored = zone & near_hint
        if anchored.sum() >= 50:
            pool = anchored & prior if (anchored & prior).sum() >= 50 else anchored
        else:
            return np.asarray(hint, np.float32), 4.0
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


# Tunables. Comments sit above the values so the formatter leaves each assignment on one line.
# A selection counts as "on skin" if this share of it is skin...
MIN_INSIDE_SKIN = 0.25
# ...or this share of the ring around it is (a tight selection around a solid patch of ink has no skin inside).
MIN_RING_SKIN = 0.4
# A patch of ink must have skin along at least this share of its edge.
SKIN_RIM_MIN = 0.5
# How many of the 8 compass directions must have skin within reach for a pixel to count as ink-on-skin.
NEED_SIDES = 6
# How far (in the weighted LAB distance) other skin of the same person may be from their face colour:
# shading, highlights and a tan line, but not a differently coloured wall.
HINT_RANGE = 38.0
# Spreading into a big tattoo's interior may grow the detected ink by at most this factor, else it is skipped.
SPREAD_MAX_FACTOR = 12.0
# Brightness counts half as much as colour when deciding "is this skin": limbs are shaded.
L_WEIGHT = 0.5


def _dist(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = lab - ref
    return np.sqrt((L_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _is_one_colour(lab: np.ndarray, binary: np.ndarray, tol: float = 18.0) -> bool:
    """True if the selected pixels are (nearly) one colour: a solid patch of ink, not a mix of skin, hair and wall."""
    px = lab[binary > 0]
    if len(px) < 30:
        return False
    return float(np.median(_dist(px, np.median(px, axis=0)))) < tol


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


def _ink_colour(lab: np.ndarray, ink: np.ndarray, local: np.ndarray) -> np.ndarray | None:
    """The tattoo's own colour: the densest colour peak among the detected ink pixels that differ most from the skin."""
    core = ink > 0
    if core.sum() < 20:
        return None
    px, ref = lab[core], local[core]
    d = _dist(px, ref)
    strong = px[d >= np.percentile(d, 55)]  # the ink proper, not its anti-aliased edge
    q = np.stack([strong[:, 0] * L_WEIGHT, strong[:, 1], strong[:, 2]], axis=1)
    bins = np.floor(q / 5.0).astype(np.int64)
    keys, counts = np.unique(bins, axis=0, return_counts=True)
    peak = keys[int(np.argmax(counts))]
    near = np.max(np.abs(bins - peak), axis=1) <= 1
    return np.median(strong[near], axis=0).astype(np.float32)


def _on_skin_ink_line(
    lab: np.ndarray, local: np.ndarray, ink_colour: np.ndarray, sp: float
) -> np.ndarray:
    """Pixels whose colour is a mix of the local skin colour and the ink colour (bool map).

    Ink on skin, including its soft anti-aliased edge, lies on the straight line between those two colours. Lips, teeth,
    eyes, a red curtain or a dark jacket sit off that line, so a loose selection cannot drag them in.
    """
    axis = ink_colour[None, None, :] - local
    axis_w = np.stack([axis[..., 0] * L_WEIGHT, axis[..., 1], axis[..., 2]], axis=-1)
    rel = lab - local
    rel_w = np.stack([rel[..., 0] * L_WEIGHT, rel[..., 1], rel[..., 2]], axis=-1)
    span2 = np.maximum((axis_w**2).sum(-1), 1e-6)
    t = np.clip((rel_w * axis_w).sum(-1) / span2, 0.0, 1.2)
    resid = np.linalg.norm(rel_w - t[..., None] * axis_w, axis=-1)
    span = np.sqrt(span2)
    return (t > 0.12) & (resid < np.maximum(2.5 * sp, 0.22 * span))


def _spread_into_ink(
    ink: np.ndarray,
    lab: np.ndarray,
    binary: np.ndarray,
    skin: np.ndarray,
    local: np.ndarray,
    sp: float,
    near_skin_far: np.ndarray,
) -> np.ndarray:
    """Big tattoos: the interior is far from any skin, so only the rim passes the skin-around test. Spread into
    connected selected pixels that lie on the skin-to-ink colour line and still have skin somewhere around them, and
    never by more than a few times the area already found."""
    colour = _ink_colour(lab, ink, local)
    if colour is None:
        return ink
    cand = (
        (binary > 0) & (skin == 0) & near_skin_far & _on_skin_ink_line(lab, local, colour, sp)
    ).astype(np.uint8)
    n, comp = cv2.connectedComponents(np.maximum(cand, ink), connectivity=8)
    touching = np.unique(comp[ink > 0])
    grown = np.isin(comp, touching[touching > 0]).astype(np.uint8) & cand
    out = np.maximum(ink, grown)
    if out.sum() > SPREAD_MAX_FACTOR * max(int(ink.sum()), 1):
        return ink  # it would swallow too much: do not spread at all
    return out


def find_ink(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    grow: float,
    level: int = 0,
    skin_hint: np.ndarray | None = None,
    protect: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(ink, skin, lab)``: uint8 0/1 masks of the tattoo pixels and of the clean skin, plus the LAB image.

    ``level`` 0 is the careful detector; 1 is relaxed (looser skin test, fewer directions, any colour difference);
    2 treats the whole selection as the tattoo (the caller only asks for it when the selection is one compact colour).
    """
    if level >= 2:
        binary2 = (mask > 0).astype(np.uint8)
        return (
            binary2,
            np.zeros_like(binary2),
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32),
        )
    if level >= 1:
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
    med, spread = skin_colour(lab, prior, ring if (ring & prior).sum() >= 100 else roi, skin_hint)

    # Skin shading changes across a limb, so compare each pixel with the skin colour *near it*: first a loose pass
    # against the global colour, then the local skin colour spread outward from those pixels (normalised convolution).
    loose = (_dist(lab, med) < max(spread * 6.0, 14.0)) & roi
    sigma = (
        2.5 * r
    )  # short steps: skin can brighten quickly (a highlight), so only trust close neighbours
    local = np.broadcast_to(med, lab.shape).copy()
    # Skin brightens and darkens gradually across a limb, so let the skin region *grow*: each round re-estimates the
    # local skin colour from the skin found so far and accepts pixels that match it, reaching highlights and shadows
    # that are far from the typical colour but close to their neighbours.
    for _ in range(12):
        w = loose.astype(np.float32)
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        est = (
            cv2.GaussianBlur(lab * w[:, :, None], (0, 0), sigma) / np.maximum(den, 1e-6)[:, :, None]
        )
        local = np.where((den > 0.01)[:, :, None], est, med[None, None, :])
        grown = ((_dist(lab, local) < max(spread * 3.5, 9.0)) & roi) | loose
        if grown.sum() <= loose.sum() * 1.01:
            loose = grown
            break
        loose = grown
    w = loose.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    local = np.where(
        (den > 0.01)[:, :, None],
        cv2.GaussianBlur(lab * w[:, :, None], (0, 0), sigma) / np.maximum(den, 1e-6)[:, :, None],
        med[None, None, :],
    )
    dist = _dist(lab, local)
    sp = float(max(np.median(dist[loose]) if loose.any() else spread, 1.5))
    # "Catch faint ink" (grow) tightens every test at once: what still counts as skin, how much darker or more colourful
    # than skin a pixel must be, and how many directions need skin around it. At 1 even a faint stain qualifies.
    thr = max(
        sp * (5.0 - 3.0 * grow), 6.5
    )  # never below ~6.5: grain and shading must still count as skin
    skin = ((dist < thr) & roi).astype(np.uint8)

    # Candidate ink: selected pixels that are not skin-coloured. It only counts where skin is close by on (nearly)
    # all sides, so hair edges, the background next to the arm and clothing are never mistaken for ink.
    # How far to look for skin: far enough to cross the biggest tattoo the selection could hold, no further.
    reach = int(
        min(max(6 * r, 12), 0.1 * max(binary.shape))
    )  # a fixed distance: big tattoos are handled by spreading
    need = max(5, int(round(NEED_SIDES - 1.0 * grow)))
    if level == 1:
        reach, need = min(int(1.5 * reach), int(0.15 * max(binary.shape))), 5
    near_skin = skin_around(skin > 0, reach, need)
    # Ink absorbs light: it is darker than the skin around it, or strongly coloured. Lighter, skin-ish colours (a pale
    # wall, a highlight) are never ink.
    darker = (local[..., 0] - lab[..., 0]) > (6.0 - 4.5 * grow)
    vivid = np.hypot(lab[..., 1] - local[..., 1], lab[..., 2] - local[..., 2]) > (
        22.0 - 14.0 * grow
    )
    differs = darker | vivid  # at every level: a pale wall or a highlight is never ink
    cand_ink = (binary > 0) & (skin == 0) & near_skin & differs
    first = cand_ink.astype(np.uint8)
    colour = _ink_colour(lab, first, local)
    if colour is not None:
        cand_ink &= _on_skin_ink_line(lab, local, colour, sp)
    ink = cand_ink.astype(np.uint8)
    if protect is not None:
        ink[protect > 0] = 0  # hair, clothing: never repainted
    region = near_skin.astype(np.uint8)
    # Tiny specks are grain, not tattoo; and a patch of ink has skin around most of its edge, whereas the background
    # beside an arm touches skin along one side only.
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    skin_b = skin > 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(4, r):
            ink[comp == i] = 0
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = r + 3
        y0, y1 = max(0, y - pad), min(ink.shape[0], y + h_ + pad)
        x0, x1 = max(0, x - pad), min(ink.shape[1], x + w_ + pad)
        me = (comp[y0:y1, x0:x1] == i).astype(np.uint8)
        rim = (cv2.dilate(me, k, iterations=max(2, r // 2)) > 0) & (me == 0)
        if rim.sum() and skin_b[y0:y1, x0:x1][rim].mean() < SKIN_RIM_MIN:
            ink[comp == i] = 0
    # The ink's soft anti-aliased edge, but only inside the selection and the skin region.
    # (wider at high sensitivity: faint ink bleeds into the skin around it)
    ink = (
        cv2.dilate(ink, k, iterations=max(1, int(round(r * (0.5 + 0.8 * grow)))))
        & binary
        & region.astype(np.uint8)
    )
    # Big tattoos: bring in the solid interior (same colour as the ink found, connected to it).
    near_far = skin_around(skin > 0, min(3 * reach, int(0.3 * max(binary.shape))), 3)
    ink = _spread_into_ink(ink, lab, binary, skin, local, sp, near_far)
    return ink.astype(np.uint8), skin, lab


def skin_reference(lab: np.ndarray, clean: np.ndarray, hole: np.ndarray, r: int) -> np.ndarray:
    """The flesh colour (LAB) each ``hole`` pixel should have, from clean skin pixels only.

    Normalised convolution over ``clean`` at growing radii, so nothing but skin contributes: a nearby wall, hair or
    clothing cannot tint it. Where no skin is anywhere near, falls back to the median skin colour.
    """
    w = clean.astype(np.float32)
    ref = np.broadcast_to(
        np.median(lab[clean > 0], axis=0) if clean.any() else lab.mean((0, 1)), lab.shape
    ).copy()
    todo = hole > 0
    for sigma in (max(2.0, 2.0 * r), 6.0 * r, 16.0 * r):
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        est = (
            cv2.GaussianBlur(lab * w[:, :, None], (0, 0), sigma) / np.maximum(den, 1e-6)[:, :, None]
        )
        ok = todo & (den > 0.03)
        ref[ok] = est[ok]
        todo &= ~ok
        if not todo.any():
            break
    return ref


def bound_to_skin(
    filled_bgr: np.ndarray, img_lab: np.ndarray, clean: np.ndarray, hole: np.ndarray, r: int
) -> np.ndarray:
    """Keep the replacement in flesh tones: fill pixels that stray from the local skin colour are pulled onto it.

    Inpainting copies from whatever borders the hole, which near the edge of an arm includes the background. A fill
    pixel close to the skin reference is kept (it carries the inpainter's structure); the further it strays, the more
    it is replaced by the skin-only reference, completely beyond a modest distance.
    """
    if not (hole > 0).any():
        return filled_bgr
    ref = skin_reference(img_lab, clean, hole, r)
    f_lab = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = _dist(f_lab, ref)
    w = np.clip((dist - 8.0) / 14.0, 0.0, 1.0) * (hole > 0)
    out_lab = f_lab * (1.0 - w[:, :, None]) + ref * w[:, :, None]
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    keep = (hole > 0)[:, :, None]
    return np.where(keep, out, filled_bgr)


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


def face_skin_hint(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray | None:
    """Median LAB colour of the skin between the eyes and the mouth of a face (cheeks and nose), or None.

    ``kps`` are the 5 face points (eyes, nose, mouth corners). The person's skin elsewhere (an arm, a shoulder) is
    expected to be within a shading/highlight range of this colour.
    """
    k = np.asarray(kps, np.float32)
    h, w = img_bgr.shape[:2]
    x0, x1 = int(min(k[0][0], k[3][0])), int(max(k[1][0], k[4][0]))
    y0 = int((k[0][1] + k[1][1]) / 2 + 0.15 * (k[3][1] - k[0][1]))
    y1 = int((k[3][1] + k[4][1]) / 2 - 0.1 * (k[3][1] - k[0][1]))
    x0, x1, y0, y1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    patch = img_bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    ok = skin_prior(patch).reshape(-1)
    px = lab[ok] if ok.sum() >= 30 else lab
    return np.median(px, axis=0).astype(np.float32)


def remove_tattoo_info(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    grow: float = 0.3,
    algorithm: str = "lama",
    skin_hint: np.ndarray | None = None,
    protect: np.ndarray | None = None,
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
    ink0, skin0, _ = find_ink(img_bgr, mask, grow, 0, skin_hint, protect)
    k3 = np.ones((3, 3), np.uint8)
    ring = (cv2.dilate(binary, k3, iterations=3 * r) > 0) & (
        cv2.dilate(binary, k3, iterations=1) == 0
    )
    inside_share = float((skin0 > 0)[binary > 0].mean())
    ring_share = float((skin0 > 0)[ring].mean()) if ring.any() else 0.0
    if inside_share < MIN_INSIDE_SKIN and ring_share < MIN_RING_SKIN:
        # Neither the selection nor its surroundings are skin (a wall, clothing): never paint flesh there.
        return img_bgr.copy(), {"mode": "noskin", "ink_pct": 0.0}
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mode = "detected"
    for level, mode in ((0, "detected"), (1, "relaxed")):
        ink, skin, lab = (
            (ink0, skin0, lab_img)
            if level == 0
            else find_ink(img_bgr, mask, grow, level, skin_hint, protect)
        )
        if int(ink.sum()) >= enough:
            break
    else:
        if inside_share >= 0.95:
            return img_bgr.copy(), {
                "mode": "clean",
                "ink_pct": 0.0,
            }  # plain skin: nothing to remove
        if not _is_one_colour(lab_img, binary):
            # Mixed content (skin + hair + background...): treating it all as ink would paint flesh over the rest.
            return img_bgr.copy(), {"mode": "unsure", "ink_pct": 0.0}
        ink, skin, lab = find_ink(img_bgr, mask, grow, 2, skin_hint)
        mode = "selection"

    current = img_bgr
    union = np.zeros(img_bgr.shape[:2], np.float32)
    total_ink = ink.copy()
    passes = (
        1 if grow < 0.5 else 2 if grow < 0.85 else 3
    )  # high sensitivity: look again at the cleaned result
    for n_pass in range(passes):
        if n_pass:  # faint remnants often only show up once the obvious ink is gone
            ink, skin, lab = find_ink(current, mask, grow, min(level, 2), skin_hint, protect)
            if int(ink.sum()) < max(20, int(0.005 * sel_area)):
                break
            total_ink = np.maximum(total_ink, ink)
        current, soft = _repaint(current, binary, ink, skin, algorithm, r)
        union = np.maximum(union, soft)

    a = (union * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    out = img_bgr.astype(np.float32) * (1.0 - a) + current.astype(np.float32) * a
    info = {
        "mode": mode,
        "ink_pct": round(100.0 * float(total_ink.sum()) / max(sel_area, 1), 1),
        "passes": n_pass + 1,
    }
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info


def _repaint(
    img: np.ndarray, binary: np.ndarray, ink: np.ndarray, skin: np.ndarray, algorithm: str, r: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint the ``ink`` pixels, add skin grain, and return ``(image, soft_mask)`` (the mask of what changed)."""
    k = np.ones((3, 3), np.uint8)
    clean = (skin > 0) & (
        cv2.dilate(ink, k, iterations=2) == 0
    )  # skin that is certainly not ink or its edge
    hole = (cv2.dilate(ink, k, iterations=1) & binary).astype(np.uint8)
    # Every ink patch is surrounded by skin, so what an inpainter copies from around it is skin.
    try:
        filled = inpaint(img, hole * 255, radius=max(3, r), algorithm=algorithm, iterations=2)
    except ModelNotFoundError:
        filled = inpaint(img, hole * 255, radius=max(3, r), algorithm="telea")
    # The replacement must be flesh-coloured, whatever the inpainter copied from beside the hole.
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    filled = bound_to_skin(filled, img_lab, clean.astype(np.uint8), hole, r)
    soft = np.clip(
        cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(0.8, r / 3.0)) * 1.5, 0.0, 1.0
    )
    soft *= binary.astype(np.float32)  # never outside the selection
    soft *= ((skin > 0) | (ink > 0)).astype(
        np.float32
    )  # and never on background: only skin or ink pixels change
    filled = add_skin_texture(filled, img, soft, clean.astype(np.uint8))
    out = (
        img.astype(np.float32) * (1.0 - soft[:, :, None])
        + filled.astype(np.float32) * soft[:, :, None]
    )
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), soft


def remove_tattoo(img_bgr: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """:func:`remove_tattoo_info` without the info."""
    return remove_tattoo_info(img_bgr, mask, **kwargs)[0]
