"""Tattoo removal: find the ink inside the selection, take its colour cast and darkness out, keep the skin underneath.

Ink on skin changes two things: it makes the skin *less warm* (grey, blue-grey or black ink against pink/tan skin) and
darker. Faded tattoos are mostly the first: a soft grey wash over a whole shoulder is only a few colour units from skin,
and barely darker than shaded skin. So detection and repair are built on colour, not on "much darker than the
neighbours" (which cannot work when the tattoo is bigger than the skin around it):

1. **Search area** = the painted selection (and the person, when a matting mask is given). Nothing else is touched.
2. **Skin model** = how a pixel's skin colour (LAB a, b) varies with its brightness (L), learned from clean skin in a
   ring around the selection. Shaded skin is darker but keeps the curve, so shadows, creases and freckles are never ink;
   ink falls off the curve towards grey.
3. **Ink score** = how much less warm a pixel is than skin of the same brightness, in units of the clean skin's own
   noise. The ``sensitivity`` slider lowers the score at which ink starts to count; it only ever removes more.
4. **Repair** = put the skin colour back at the pixel's own brightness, take the ink's darkness out in proportion to its
   strength, then flatten what is left of the drawing (shapes tens of pixels wide) into the smooth shading of the limb,
   keeping the fine grain. No copied patches, so no seams.

Paint over the tattoo; going a little onto bare skin is fine, and with the person mask the backdrop is never touched.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.exceptions import ValidationError

# Ink starts to count when its score reaches ``k * sigma`` (sigma = the clean skin's own colour noise), with k falling
# from ONSET_K_AT_0 to ONSET_K_AT_1 as the sensitivity rises. Never below ONSET_FLOOR colour units: below that is noise.
ONSET_K_AT_0, ONSET_K_AT_1 = 3.0, 1.4
ONSET_FLOOR = 2.5
# Full strength is reached this far (colour units, at least RAMP_SIGMAS * sigma) above the onset.
RAMP_MIN, RAMP_SIGMAS = 4.0, 3.0
SIGMA_FLOOR = 1.0
# Ink is not strongly coloured: a saturated blue wall or red curtain is not brown/black/grey ink.
CHROMA_MAX = 32.0
# Ink is never lighter than the lit skin: a pale neutral wall (when there is no person mask) is not ink.
BRIGHT_MARGIN = 8.0
# Next to a drawing, ink this much fainter than the onset (the soft edge of the ink) is taken too.
HALO_ONSET = 0.7
# How far (L units) past the brightness range of the clean skin the skin model keeps following its trend.
EXTRAPOLATE = 30.0
# Darker than the darkest clean skin by more than DARK_GAP L units counts as ink on darkness alone.
DARK_GAP, DARK_GAIN = 35.0, 0.8
# Ink still this far (L units) below the wide skin envelope after the smoothing is lifted to within this of it.
WIDE_SLACK = 12.0


def _radius(shape: tuple[int, ...]) -> int:
    return max(3, int(round(0.008 * max(shape[:2]))))


def skin_prior(bgr: np.ndarray) -> np.ndarray:
    """Pixels whose hue is plausibly skin of any tone, lit or shaded, even ruddy (generous YCrCb ranges).

    Blue and green backdrops fail it; a pink wall can pass, which colour alone cannot fix (the person mask does).
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    return (cr >= 130) & (cr <= 195) & (cb >= 70) & (cb <= 135)


def onset_for(sensitivity: float, sigma: float) -> float:
    """The ink score at which ink starts to count, for a sensitivity in 0..1: the higher the sensitivity, the lower."""
    s = float(np.clip(sensitivity, 0.0, 1.0))
    k = ONSET_K_AT_0 + (ONSET_K_AT_1 - ONSET_K_AT_0) * s
    return max(ONSET_FLOOR, k * max(sigma, SIGMA_FLOOR))


class SkinModel:
    """The skin's colour (LAB a, b) as a function of brightness (L), learned from clean skin pixels."""

    def __init__(self, lab: np.ndarray, clean: np.ndarray) -> None:
        lum = lab[:, :, 0][clean]
        self.lo, self.hi = (float(v) for v in np.percentile(lum, [2, 98]))
        span = self.hi - self.lo
        degree = 2 if (span >= 40 and len(lum) >= 300) else 1 if span >= 8 else 0
        self.coefs = []
        self.slopes = []  # the overall trend per L unit, for brightness beyond the data
        x = (lum - 128.0) / 64.0
        for ch in (1, 2):
            v = lab[:, :, ch][clean]
            keep = np.ones(len(v), bool)
            co = np.array([np.median(v)])
            for _ in range(
                4
            ):  # robust: drop what is not skin-like (hair, lips, a stray backdrop pixel)
                if keep.sum() < 10:
                    break
                co = np.polyfit(x[keep], v[keep], degree)
                res = v - np.polyval(co, x)
                s = 1.4826 * np.median(np.abs(res[keep]))
                keep = np.abs(res) < 2.5 * max(s, 0.5)
            self.coefs.append(co)
            self.slopes.append(
                float(np.polyfit(x[keep], v[keep], 1)[0]) / 64.0
                if span >= 8 and keep.sum() >= 10
                else 0.0
            )

    def reference(self, lum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Skin's (a, b) at brightness ``lum``.

        Beyond the brightness range seen in clean skin (a shadow darker than anything around the tattoo) the trend is
        carried on in a straight line for a while, then held: shaded skin keeps drifting the way it was drifting
        instead of being compared with the colour of the darkest skin that was seen.
        """
        lum = np.asarray(lum, np.float32)
        out = []
        for co, slope in zip(self.coefs, self.slopes, strict=True):
            val = np.polyval(co, (np.clip(lum, self.lo, self.hi) - 128.0) / 64.0)
            beyond = np.clip(lum - self.hi, 0.0, EXTRAPOLATE) - np.clip(
                self.lo - lum, 0.0, EXTRAPOLATE
            )
            out.append(val + slope * beyond)
        return out[0], out[1]


def _norm_conv(
    values: np.ndarray, weight: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-weighted average of ``values`` over the pixels where ``weight`` is set, and how much weight was there."""
    w = weight.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    num = cv2.GaussianBlur(values * w[..., None], (0, 0), sigma).reshape(values.shape)
    return num / np.maximum(den, 1e-6)[..., None], den


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def _envelope(
    lum: np.ndarray, valid: np.ndarray, se: int, avoid: np.ndarray | None = None
) -> np.ndarray:
    """The smooth upper envelope of the brightness: what the skin would be without dark features narrower than ``se``.

    A grey-scale closing (dark drawing lines and blobs vanish, the limb's own shading stays) then a blur, computed on a
    reduced copy because the result is smooth. Pixels that are not valid skin (backdrop) are filled from their
    surroundings first (but never from ``avoid``, the ink), so a bright wall never lifts the skin beside it.
    """
    h, w = lum.shape
    f = max(1, int(round(se / 21)))
    size = (max(2, w // f), max(2, h // f))
    small = cv2.resize(lum, size, interpolation=cv2.INTER_AREA)
    ok = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5
    se_d = max(3, _odd(se / f))
    if not ok.all():
        fill_value = float(np.median(small[ok])) if ok.any() else float(small.mean())
        src = ok
        if avoid is not None:
            src = ok & ~(
                cv2.resize(avoid.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.2
            )
            if not src.any():
                src = ok
        est, den = _norm_conv(small[..., None], src, max(2.0, se_d / 4))
        est = np.where(den[..., None] > 1e-3, est, fill_value)
        small = np.where(ok, small, est[..., 0]).astype(np.float32)
    closed = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_d, se_d))
    )
    closed = cv2.GaussianBlur(closed, (0, 0), max(1.0, se_d / 4))
    return cv2.resize(closed, (w, h), interpolation=cv2.INTER_LINEAR)


class _Analysis:
    """Where the ink is and how strong, for one sensitivity; everything the repair needs."""

    def __init__(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        sensitivity: float,
        foreground: np.ndarray | None,
    ) -> None:
        self.r = r = _radius(img_bgr.shape)
        self.lab = lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        fg = np.ones(img_bgr.shape[:2], bool) if foreground is None else (foreground > 0)
        self.fg = fg
        self.sel = sel = (mask > 0) & fg
        self.alpha = np.zeros(sel.shape, np.float32)
        self.ok = False
        if not sel.any():
            return
        k = np.ones((3, 3), np.uint8)
        sel_u8 = sel.astype(np.uint8)
        near = cv2.dilate(sel_u8, k, iterations=10 * r) > 0
        ring = near & ~(cv2.dilate((mask > 0).astype(np.uint8), k, iterations=2) > 0)
        clean = ring & skin_prior(img_bgr) & fg
        if (
            clean.sum() < 100
        ):  # a selection that covers the skin around it: use skin anywhere outside it
            clean = ~(mask > 0) & skin_prior(img_bgr) & fg
        if clean.sum() < 100:
            return
        self.model = model = SkinModel(lab, clean)
        lum = lab[:, :, 0]
        ra, rb = model.reference(lum)
        warm = np.array(
            [np.median(ra[clean]) - 128.0, np.median(rb[clean]) - 128.0], np.float32
        )  # the way skin leans away from neutral grey
        warmth = float(np.hypot(*warm))
        if warmth < 3.0:  # a grey-scale photo: there is no colour cast to find
            return
        axis = warm / warmth
        # How much less warm than skin of the same brightness this pixel is (positive = towards grey).
        score = (ra - lab[:, :, 1]) * axis[0] + (rb - lab[:, :, 2]) * axis[1]
        # Far darker than any skin around the tattoo there is no colour to judge by (everything goes neutral in the
        # dark): black ink is told by its darkness alone.
        score = np.maximum(score, DARK_GAIN * (model.lo - DARK_GAP - lum))
        self.score = score
        sigma = max(
            SIGMA_FLOOR, 1.4826 * float(np.median(np.abs(score[clean] - np.median(score[clean]))))
        )
        self.sigma = sigma
        self.onset = onset = onset_for(sensitivity, sigma)
        ramp = max(RAMP_MIN, RAMP_SIGMAS * sigma)
        chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
        eligible = (
            sel & (chroma <= max(CHROMA_MAX, 1.3 * warmth)) & (lum <= model.hi + BRIGHT_MARGIN)
        )
        alpha = np.clip((score - onset) / ramp, 0.0, 1.0) * eligible

        # Keep only drawings, not specks: skin texture is full of 1-2 px colour flecks.
        core = cv2.morphologyEx(
            (alpha > 0.5).astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        n, comp, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        keep = np.zeros(n, bool)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            smallest = max(8, r * r // 2, int(0.01 * areas.max()))
            keep[1:] = areas >= smallest
        self.core = core = keep[comp] if n > 1 else np.zeros(sel.shape, bool)
        enough = max(20, int(0.002 * sel.sum()))
        if int(core.sum()) < enough:
            return
        # The faint halo around a kept drawing comes with it (the soft, fading edge of the ink); a faint blotch on its own,
        # or far from the drawing, does not.
        reach = cv2.dilate(
            core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * r + 1,) * 2)
        )
        smooth_score = cv2.GaussianBlur(
            score.astype(np.float32), (0, 0), 1.5
        )  # noise averages out, a real edge does not
        soft = (
            np.clip((smooth_score - HALO_ONSET * onset) / ramp, 0.0, 1.0) * eligible * (reach > 0)
        )
        soft = cv2.GaussianBlur(soft.astype(np.float32), (0, 0), 2.0)
        alpha = np.maximum(alpha, soft)
        n2, comp2 = cv2.connectedComponents((alpha > 0.12).astype(np.uint8), connectivity=8)
        wanted = np.zeros(n2, bool)
        wanted[np.unique(comp2[core])] = True
        wanted[0] = False
        alpha = alpha * wanted[comp2]
        self.alpha = (cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.0) * sel).astype(
            np.float32
        )
        self.ok = True
        self.clean = clean
        self.ink_pct = round(100.0 * float(core.sum()) / max(int(sel.sum()), 1), 1)


def find_ink(
    img_bgr: np.ndarray, mask: np.ndarray, sensitivity: float, foreground: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(ink, clean)``: the ink opacity (float 0..1, zero outside the selection) and the clean skin mask."""
    a = _Analysis(img_bgr, mask, sensitivity, foreground)
    if not a.ok:
        return np.zeros(mask.shape, np.float32), np.zeros(mask.shape, np.uint8)
    return a.alpha, a.clean.astype(np.uint8)


def _repair(a: _Analysis, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the repaired image and the per-pixel weight it should be blended in with."""
    lab, alpha, sel, fg, model = a.lab, a.alpha, a.sel, a.fg, a.model
    h, w = alpha.shape
    lum = lab[:, :, 0]
    se = int(np.clip(_odd(0.05 * max(h, w)), 31, 151))
    inked = alpha > 0.05
    # Skin to take the shading from: clean skin and anything that is skin-coloured, but not backdrop.
    ra, rb = model.reference(lum)
    skin_like = (np.hypot(lab[:, :, 1] - ra, lab[:, :, 2] - rb) < 3.5 * a.sigma) | inked
    valid = fg & skin_like

    # 1. Take out the ink's darkness in proportion to its strength. How dark ink is per unit of score is read off the
    # picture itself: how far the strongly-inked pixels sit below the skin envelope.
    big = min(3 * se, _odd(0.5 * min(h, w)))  # wide: a big drawing is bigger than `se`
    env0 = _envelope(lum, valid, big, inked)
    strong = alpha > 0.5
    drop = np.clip(env0 - lum, 0.0, None)
    ratio = drop[strong] / np.maximum(a.score[strong], 1.0)
    kd = float(np.clip(np.median(ratio), 0.0, 8.0)) if ratio.size else 0.0
    unmixed = lum + alpha * kd * np.clip(a.score, 0.0, None)
    unmixed = np.minimum(unmixed, np.maximum(lum, env0))

    # 2. What is left of the drawing is mid-sized shapes: flatten them into the limb's smooth shading. Keep the fine
    # grain (clipped so a drawing's sharp outline cannot leak back in).
    env = _envelope(unmixed, valid, se, inked)
    fine = unmixed - cv2.GaussianBlur(unmixed, (0, 0), 2.0)
    clean_fine = (lum - cv2.GaussianBlur(lum, (0, 0), 2.0))[a.clean]
    grain = (
        float(1.4826 * np.median(np.abs(clean_fine - np.median(clean_fine))))
        if clean_fine.size
        else 1.0
    )
    fine = np.clip(fine, -3.0 * max(grain, 0.5), 3.0 * max(grain, 0.5))
    # Weight for brightness: the ink itself, plus the bare-skin islands inside a big drawing (so they do not stand out
    # as dark blobs once their surroundings are lifted).
    solid = (alpha > 0.5).astype(np.uint8)
    closed = cv2.morphologyEx(
        solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    )
    w_l = cv2.GaussianBlur(np.maximum(alpha, closed.astype(np.float32) * sel), (0, 0), 1.5)
    w_l = np.clip(w_l, 0.0, 1.0) * sel
    lift = env + fine - unmixed
    cap = 10.0 + 300.0 * alpha  # skin that is not ink may be lifted only a little
    lift = np.clip(lift, -cap, cap)
    new_l = np.clip(unmixed + w_l * lift, 0.0, 255.0)
    # A very dark, very big drawing (solid black ink) can be wider than `se`: whatever is still far below the wide skin
    # envelope is lifted up to just under it.
    wide = _envelope(unmixed, valid, big, inked)
    new_l = new_l + alpha * np.clip(wide - WIDE_SLACK - new_l, 0.0, None)

    # Grain: the drawing's own fine detail is mostly gone, so give the repaired skin the grain of the clean skin around it.
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.7)
    noise /= max(float(noise.std()), 1e-6)
    new_l = np.clip(new_l + noise * max(grain, 0.0) * 0.8 * w_l, 0.0, 255.0)

    # 3. Skin colour at the new brightness, in proportion to the ink (and for the islands inside a big drawing).
    na, nb = model.reference(new_l)
    w_c = np.maximum(alpha, w_l)
    new_a = lab[:, :, 1] + w_c * (na - lab[:, :, 1])
    new_b = lab[:, :, 2] + w_c * (nb - lab[:, :, 2])
    out_lab = np.stack([new_l, new_a, new_b], axis=-1)
    out = cv2.cvtColor(np.clip(out_lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, np.maximum(alpha, w_l).astype(np.float32)


def remove_tattoo_info(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    grow: float = 0.5,
    algorithm: str = "",  # accepted and ignored: older clients still send it
    foreground: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Remove the tattoo ink inside ``mask`` (uint8, non-zero = search here).

    ``foreground`` (optional, non-zero = the person, e.g. from background matting) keeps everything else out of bounds:
    the skin colour is learned only from the person and only the person's pixels are ever repainted.
    ``grow`` is the sensitivity (0..1): the higher, the fainter the ink that is caught. ``strength`` (0..1) fades
    instead of erasing. Returns ``(image, info)``; ``info["mode"]`` is ``detected`` or ``none`` (no ink found in the
    selection). ``info["ink_pct"]`` is the share of the selection that is solid ink.
    Pixels outside the selection are never changed.
    """
    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValidationError("mask size does not match the image")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    none = {"mode": "none", "ink_pct": 0.0, "passes": 0}
    sel_any = (mask > 0) if foreground is None else ((mask > 0) & (foreground > 0))
    if not sel_any.any() or strength <= 0:
        return img_bgr.copy(), none

    # Work on the selection and what is around it, not the whole photo.
    r = _radius(img_bgr.shape)
    se = int(np.clip(_odd(0.05 * max(img_bgr.shape[:2])), 31, 151))
    pad = 10 * r + se
    ys, xs = np.nonzero(sel_any)
    y0, y1 = max(0, ys.min() - pad), min(img_bgr.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img_bgr.shape[1], xs.max() + 1 + pad)
    crop = np.s_[y0:y1, x0:x1]
    sub_img = img_bgr[crop]
    sub_fg = None if foreground is None else foreground[crop]
    a = _Analysis(sub_img, mask[crop], grow, sub_fg)
    if not a.ok:
        return img_bgr.copy(), {**none, "passes": 1}
    fixed, weight = _repair(a, sub_img)
    k = float(np.clip(strength, 0.0, 1.0))
    wgt = (weight * k)[:, :, None]
    blended = sub_img.astype(np.float32) * (1.0 - wgt) + fixed.astype(np.float32) * wgt
    out = img_bgr.copy()
    out[crop] = np.where(wgt > 0, np.clip(blended + 0.5, 0, 255).astype(np.uint8), sub_img)
    return out, {"mode": "detected", "ink_pct": a.ink_pct, "passes": 1}


# Measured: more passes give a byte-identical fill (LaMa ignores what is inside the hole) and cost 5x the time.
LAMA_PASSES = 1


def _diffuse_colour(img_bgr: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Smooth colour (LAB a, b) for the hole: the skin around it diffused inwards (harmonic fill on a small grid)."""
    h, w = hole.shape
    f = max(1.0, max(h, w) / 96.0)
    sw, sh = max(2, round(w / f)), max(2, round(h / f))
    small = cv2.cvtColor(
        cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB
    )
    ab = small[:, :, 1:].astype(np.float32)
    unknown = (
        cv2.resize(
            cv2.dilate(hole, np.ones((5, 5), np.uint8)), (sw, sh), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    if unknown.all() or not unknown.any():
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 1:]
    known = ~unknown
    x = ab.copy()
    x[unknown] = ab[known].mean(axis=0)
    for _ in range(1500):
        x = cv2.blur(x, (3, 3), borderType=cv2.BORDER_REPLICATE)
        x[known] = ab[known]
    big = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(big, (0, 0), max(1.5, 3.0 / f))


def _restore_grain(
    orig: np.ndarray, repaired: np.ndarray, hole: np.ndarray, fg: np.ndarray | None
) -> np.ndarray:
    """Two-stage repair: keep the model's fill for colour and tone only, then put real skin texture back on top.

    LaMa works on a 512 px copy of the area and scales the fill back up, so what it paints is smooth and looks
    "painted". Like frequency separation in a retouching workflow, the fill is low-passed and fine grain with the
    measured strength of the clean skin around the selection is added (per LAB channel).
    """
    h, w = hole.shape
    lab_o = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB).astype(np.float32)
    fine = lab_o - cv2.GaussianBlur(lab_o, (0, 0), 2.0)
    m = hole > 0
    k = int(np.clip(0.05 * max(h, w), 9, 61)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = (cv2.dilate(hole, ker) > 0) & ~(cv2.dilate(hole, np.ones((5, 5), np.uint8)) > 0)
    if fg is not None and (ring & (fg > 0)).sum() >= 200:
        ring &= fg > 0  # skin of the person only, when there is enough of it
    if ring.sum() < 200:  # tiny ring: any clean pixel will do
        ring = ~m
    if ring.sum() < 50:
        return repaired
    sig = np.empty(3, np.float32)
    for c in range(3):
        v = fine[:, :, c][ring]
        sig[c] = 1.4826 * np.median(np.abs(v - np.median(v)))
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.standard_normal((h, w, 3)).astype(np.float32), (0, 0), 0.7)
    noise /= np.maximum(noise.reshape(-1, 3).std(axis=0), 1e-6)
    low = cv2.GaussianBlur(lab_r, (0, 0), 1.5)
    # LaMa's own colour mottles over a big hole (grey/green/orange blotches): take the colour (a, b) from a smooth
    # diffusion of the skin around the hole instead, and keep LaMa for the light and shade (L).
    low[:, :, 1:] = _diffuse_colour(orig, hole)
    textured = low + noise * sig
    wt = np.clip(cv2.GaussianBlur(hole.astype(np.float32) / 255.0, (0, 0), 1.5), 0, 1)[:, :, None]
    lab = lab_r * (1 - wt) + textured * wt
    return cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _temporary_background(img_bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """``img_bgr`` with the backdrop (where ``fg`` is zero) swapped for the person's own nearest colour, smoothed.

    This is the "temporary background removal": the inpainting model then sees only skin, so it can never pull a wall,
    curtain or highlight into the repair. The real backdrop is put back afterwards.
    """
    # The matting mask is generous at the person's edge (the API even grows it): sample colour only from well inside it.
    er = max(3, int(round(0.01 * max(fg.shape[:2]))))
    inner = cv2.erode(
        (fg > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1)),
    )
    person = (inner if inner.any() else (fg > 0)).astype(np.float32)
    if person.all() or not person.any():
        return img_bgr.copy()
    src = img_bgr.astype(np.float32) * person[:, :, None]
    filled = np.zeros_like(src)
    done = person > 0
    # The person's average colour at the smallest scale that reaches each backdrop pixel: smooth, and always skin.
    for sigma in (3, 6, 12, 24, 48, 96, 192, 384):
        den = cv2.GaussianBlur(person, (0, 0), sigma)
        num = cv2.GaussianBlur(src, (0, 0), sigma)
        take = ~done & (den > 0.02)
        filled[take] = num[take] / den[take][:, None]
        done |= take
        if done.all():
            break
    filled[~done] = src[person > 0].mean(axis=0)
    return np.where(person[:, :, None] > 0, img_bgr, np.clip(filled + 0.5, 0, 255).astype(np.uint8))


def remove_tattoo_lama(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    foreground: np.ndarray | None = None,
    passes: int = LAMA_PASSES,
) -> tuple[np.ndarray, dict]:
    """Remove the selection with AI object removal (LaMa, ``passes`` passes) plus real skin grain, with the backdrop temporarily removed.

    Only the selected part of the person is repainted; with ``foreground`` the backdrop is swapped for skin colour
    while the model runs and restored afterwards, so a highlighted or skin-like background cannot leak in.
    Raises ``ModelNotFoundError`` when the LaMa model is missing.
    """
    from app.models.lama import LaMa

    if mask.dtype != np.uint8:
        raise ValidationError(f"mask must be uint8, got {mask.dtype}")
    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValidationError("mask size does not match the image")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    sel = (mask > 0) if foreground is None else ((mask > 0) & (foreground > 0))
    if not sel.any() or strength <= 0:
        return img_bgr.copy(), {"mode": "none", "ink_pct": 0.0, "passes": 0}

    ys, xs = np.nonzero(sel)
    pad = max(64, int(0.75 * max(ys.max() - ys.min(), xs.max() - xs.min())))
    y0, y1 = max(0, ys.min() - pad), min(img_bgr.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img_bgr.shape[1], xs.max() + 1 + pad)
    crop = np.s_[y0:y1, x0:x1]
    sub = img_bgr[crop]
    hole = sel[crop].astype(np.uint8) * 255
    work = sub if foreground is None else _temporary_background(sub, foreground[crop])

    repaired = LaMa.get().infer(work, hole, iterations=max(1, int(passes)))
    repaired = _restore_grain(
        work, repaired, hole, None if foreground is None else foreground[crop]
    )

    w = np.clip(cv2.GaussianBlur(hole.astype(np.float32) / 255.0, (0, 0), 1.5), 0.0, 1.0)
    if foreground is not None:
        w = w * (foreground[crop] > 0)  # the real backdrop is never repainted
    w = (w * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
    blended = sub.astype(np.float32) * (1.0 - w) + repaired.astype(np.float32) * w
    out = img_bgr.copy()
    out[crop] = np.where(w > 0, np.clip(blended + 0.5, 0, 255).astype(np.uint8), sub)
    return out, {
        "mode": "lama",
        "ink_pct": round(100.0 * float(sel.sum()) / max(1, int((mask > 0).sum())), 1),
        "passes": int(passes),
    }


def remove_tattoo(img_bgr: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """:func:`remove_tattoo_info` without the info."""
    return remove_tattoo_info(img_bgr, mask, **kwargs)[0]
