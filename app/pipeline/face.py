"""Face geometry: alignment, masks, colour matching and paste-back for face replacement.

Everything here is plain numpy/OpenCV so it can be tested without any model files. The neural parts (detector,
embedder, swapper, restorer) live in :mod:`app.models.face`.

Landmarks are the five YuNet / InsightFace points in image coordinates, ordered
``[left eye, right eye, nose tip, left mouth corner, right mouth corner]`` ("left" = smaller x).
"""

from __future__ import annotations

import cv2
import numpy as np

# ArcFace's canonical landmark layout on a 112x112 crop (used by the embedder and the 128 px swapper).
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# FFHQ layout as fractions of a 512 px crop (what GFPGAN was trained on).
FFHQ_FRACTIONS = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)


# The layout the HyperSwap / InSwapper generators were trained with (fractions of the crop): a little more margin
# than ArcFace's own 112 layout, so forehead, jaw and ears are inside the crop. Measured: identity transfer to the
# donor went from 0.67 to 0.80 (ArcFace cosine) just by using this layout.
SWAP_FRACTIONS = np.array(
    [
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ],
    dtype=np.float32,
)


def swap_template(size: int) -> np.ndarray:
    return SWAP_FRACTIONS * float(size)


def head_template(size: int = 640) -> np.ndarray:
    """A wide crop around the face (face about 40% of the crop) so long hair, ears and the top of the head fit."""
    a = ARCFACE_112
    k = size / 112.0 * 0.4
    return ((a - a.mean(0)) * k + np.array([size * 0.5, size * 0.5])).astype(np.float32)


def border_fade(size: int, frac: float = 0.07) -> np.ndarray:
    """1 in the middle of a ``size`` px crop, ramping to 0 at its border, so a patch never ends in a hard edge."""
    n = max(2, int(size * frac))
    ramp = np.clip(np.arange(size, dtype=np.float32) / n, 0, 1)
    edge = np.minimum(ramp, ramp[::-1])
    return np.minimum(edge[:, None], edge[None, :])


def arcface_template(size: int) -> np.ndarray:
    """ArcFace landmarks scaled to a ``size`` px crop (112 for the embedder, 128 for the swapper)."""
    return ARCFACE_112 * (size / 112.0)


def ffhq_template(size: int = 512) -> np.ndarray:
    return FFHQ_FRACTIONS * float(size)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 similarity (rotation + uniform scale + translation) taking ``src`` points to ``dst``.

    Umeyama's closed form, so it is deterministic and never shears the face.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0**2).sum() / len(src)
    if var < 1e-12:
        raise ValueError("degenerate landmarks")
    cov = d0.T @ s0 / len(src)
    u, sing, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = (sing * d).sum() / var
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_crop(img: np.ndarray, kps: np.ndarray, template: np.ndarray, size: int):
    """Warp ``img`` so ``kps`` land on ``template``; returns ``(crop, M)`` with ``M`` mapping image -> crop."""
    m = similarity_transform(kps, template)
    crop = cv2.warpAffine(
        img,
        m,
        (size, size),
        flags=cv2.INTER_AREA if _shrinks(m) else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, m


def _shrinks(m: np.ndarray) -> bool:
    return float(np.hypot(m[0, 0], m[1, 0])) < 1.0


def _ellipse_mask(
    size: int, center: tuple[float, float], axes: tuple[float, float], feather: float
) -> np.ndarray:
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (int(round(axes[0])), int(round(axes[1]))),
        0,
        0,
        360,
        1.0,
        -1,
        lineType=cv2.LINE_AA,
    )
    sigma = float(np.clip(feather, 0.0, 1.0)) * 0.045 * size
    if sigma >= 0.5:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = np.clip(mask / max(float(mask.max()), 1e-6), 0.0, 1.0)
    return mask


def swap_mask(size: int, feather: float = 0.5, model_mask: np.ndarray | None = None) -> np.ndarray:
    """Paste mask on a :func:`swap_template` crop: the generator's own face mask (if it gives one), pulled in a little,
    times a soft ellipse so the paste never reaches the crop border (a hard line shows at the forehead otherwise)."""
    ell = _ellipse_mask(size, (0.5 * size, 0.495 * size), (0.375 * size, 0.43 * size), feather)
    if model_mask is None:
        return ell
    k = max(1, int(round(size * 0.02))) | 1
    inner = cv2.erode(np.clip(model_mask, 0, 1).astype(np.float32), np.ones((k, k), np.uint8))
    sigma = float(np.clip(feather, 0.0, 1.0)) * 0.03 * size
    if sigma >= 0.5:
        inner = cv2.GaussianBlur(inner, (0, 0), sigma)
    return np.clip(inner * ell, 0.0, 1.0)


# Face-parsing labels (CelebAMask-HQ / BiSeNet): what counts as hair and as face when moving a head.
HAIR_LABELS = (17,)  # hair (hats stay where they are)
FACE_LABELS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
)  # skin, brows, eyes, glasses, ears, nose, mouth, lips


def donor_face_mask(
    size: int,
    feather: float = 0.6,
    labels: np.ndarray | None = None,
    crop: np.ndarray | None = None,
) -> np.ndarray:
    """Where the donor's own face pixels are pre-pasted into the target crop (before the generator cleans it up).

    With face-parsing ``labels`` of the donor crop: skin, brows, eyes, nose, mouth, lips (no hair, hat, ears or
    background, which left stray strands and scars). Without: a plain ellipse. Always kept inside the swap ellipse.
    """
    ell = _ellipse_mask(size, (0.5 * size, 0.495 * size), (0.36 * size, 0.42 * size), 0.0)
    if labels is None:
        base = ell
    else:
        lab = cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)
        parts = np.isin(lab, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)).astype(np.uint8)
        if crop is not None:
            # A thin strand of hair across the cheek is often labelled "skin". Skin that is far darker than the rest
            # of the donor's skin is hair, a shadow or a stain: leave it out rather than paste it as a scar.
            luma = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2LAB)[:, :, 0].astype(
                np.float32
            )
            skin = lab == 1
            if skin.sum() > 100:
                parts[skin & (luma < np.median(luma[skin]) - 26.0)] = 0
        parts = cv2.morphologyEx(parts, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        base = parts.astype(np.float32) * ell
    sigma = float(np.clip(feather, 0.0, 1.0)) * 0.03 * size
    if sigma >= 0.5:
        base = cv2.GaussianBlur(base, (0, 0), sigma)
    return np.clip(base, 0.0, 1.0)


def prepaste(
    crop: np.ndarray, donor_crop: np.ndarray, mask: np.ndarray, shape: float
) -> np.ndarray:
    """Mix the donor's (colour-matched) face into the target crop by ``shape`` 0..1 inside ``mask``."""
    a = (mask * float(np.clip(shape, 0.0, 1.0)))[:, :, None]
    return np.clip(
        crop.astype(np.float32) * (1.0 - a) + donor_crop.astype(np.float32) * a + 0.5, 0, 255
    ).astype(np.uint8)


# Parts of the picture inside the face crop that are the face (or continue it): skin, brows, eyes, ears, nose, mouth,
# lips, neck. Anything else there (hair, hat, glasses, clothing, a hand, background) is in front of / beside the face.
VISIBLE_FACE_LABELS = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14)


def visible_face_mask(labels: np.ndarray, size: int, feather: float = 2.0) -> np.ndarray:
    """Float mask on a ``size`` px crop: 1 where the face is visible, 0 where something covers it (hair, glasses, hand)."""
    lab = cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)
    vis = np.isin(lab, VISIBLE_FACE_LABELS).astype(np.uint8)
    vis = cv2.morphologyEx(
        vis, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )  # parser speckle inside the face
    out = vis.astype(np.float32)
    return cv2.GaussianBlur(out, (0, 0), feather) if feather > 0 else out


def skin_color_transfer(
    patch: np.ndarray, crop: np.ndarray, skin: np.ndarray, strength: float
) -> np.ndarray:
    """Make the swapped face's *skin* match the person's own skin (LAB mean and spread over skin pixels only).

    ``patch`` is the generator's output, ``crop`` the original target face in the same alignment and ``skin`` a 0/1
    mask of skin pixels (no eyes, brows, lips, hair). Matching skin to skin keeps the eyes and mouth honest, and
    stops a pale face on a dark neck. ``strength`` 0..1.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    m = skin > 0
    if strength <= 0 or m.sum() < 64:
        return patch
    p = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = p.copy()
    for c in range(3):
        mp, sp = float(p[..., c][m].mean()), float(p[..., c][m].std()) + 1e-3
        mt, st = float(t[..., c][m].mean()), float(t[..., c][m].std()) + 1e-3
        gain = (
            float(np.clip(st / sp, 0.8, 1.25)) if c == 0 else 1.0
        )  # keep the face's own contrast, mostly
        matched = (p[..., c] - mp) * gain + mt
        out[..., c] = p[..., c] * (1.0 - strength) + matched * strength
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_detail(
    result: np.ndarray, original: np.ndarray, face: np.ndarray, amount: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Give the pasted face the sharpness and grain of the photo it was pasted into.

    ``face`` is a full-size float mask of the pasted face. The generator's output is smoother than a camera's, which
    is the "airbrushed" look. Compare fine-detail energy (a high-pass of the brightness) inside the face in the
    original and the result; then sharpen, and add matching grain for what sharpening cannot supply.
    """
    amount = float(np.clip(amount, 0.0, 1.5))
    core = face > 0.6
    if amount <= 0 or core.sum() < 200:
        return result
    ys, xs = np.nonzero(face > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    res = result[y0:y1, x0:x1].astype(np.float32)
    org = original[y0:y1, x0:x1].astype(np.float32)
    f = face[y0:y1, x0:x1]
    c = core[y0:y1, x0:x1]
    sigma = max(1.0, 0.0035 * max(result.shape[:2]))
    lum = lambda a: a @ np.array([0.114, 0.587, 0.299], np.float32)  # noqa: E731  (BGR luma)
    hp = lambda a: lum(a) - cv2.GaussianBlur(lum(a), (0, 0), sigma)  # noqa: E731

    def energy(a):
        x = hp(a)[c]
        return float(np.median(np.abs(x - np.median(x))) * 1.4826) + 1e-3

    want, have = energy(org), energy(res)
    if have >= want:
        return result
    # 1) sharpen toward the target's detail level (unsharp mask), capped so it never rings
    blur = cv2.GaussianBlur(res, (0, 0), sigma * 1.3)
    gain = float(np.clip((want / have - 1.0) * 0.6 * amount, 0.0, 1.6))
    sharp = res + (res - blur) * gain
    # 2) grain for the energy still missing
    have2 = energy(sharp)
    need = float(np.sqrt(max(want**2 - have2**2, 0.0))) * amount
    if need > 0.2:
        rng = np.random.default_rng(seed)
        n = cv2.GaussianBlur(rng.standard_normal(res.shape[:2]).astype(np.float32), (0, 0), 0.6)
        n /= max(float(n.std()), 1e-6)
        sharp = sharp + (n * need)[:, :, None]
    out = result.copy()
    mixed = res * (1.0 - f[:, :, None]) + sharp * f[:, :, None]
    out[y0:y1, x0:x1] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
    return out


def landmarks_to_5(lm68: np.ndarray) -> np.ndarray:
    """The five points the aligners use (eye centres, nose tip, mouth corners) from 68 face landmarks."""
    lm = np.asarray(lm68, np.float32)
    return np.stack([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]]).astype(
        np.float32
    )


def yaw_estimate(kps: np.ndarray) -> float:
    """Rough head turn from the 5 points: how far the nose tip sits from the midpoint of the eyes, in eye-distances.

    About 0 for a frontal face, +/-0.2..0.5 for a three-quarter view (sign says which way)."""
    k = np.asarray(kps, np.float32)
    eye_dist = float(np.linalg.norm(k[1] - k[0])) + 1e-6
    mid = (k[0] + k[1]) / 2.0
    across = (k[1] - k[0]) / eye_dist
    return float(np.dot(k[2] - mid, across) / eye_dist)


def shape_for_poses(shape: float, target_kps: np.ndarray, donor_kps: np.ndarray) -> float:
    """Take less of the donor's face *shape* when the two heads are turned differently: a face turned 30 degrees pasted
    onto a frontal one comes out distorted. Same pose: unchanged. Very different: about a third of the asked amount."""
    d = abs(yaw_estimate(target_kps) - yaw_estimate(donor_kps))
    return float(np.clip(shape, 0.0, 1.0) * np.clip(1.25 - 2.5 * d, 0.35, 1.0))


def piecewise_affine(
    img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, size: int
) -> np.ndarray:
    """Warp ``img`` (a ``size`` px square) so each ``src_pts`` point lands on its ``dst_pts`` point, triangle by triangle.

    Points on the crop's border map to themselves, so the warp stays inside the crop. Each Delaunay triangle gets its
    own affine map, which moves the eyes, nose and mouth independently (a similarity transform can only turn and
    scale the whole face).
    """
    hi = size - 1.0
    anchors = np.array(
        [[0, 0], [hi / 2, 0], [hi, 0], [0, hi / 2], [hi, hi / 2], [0, hi], [hi / 2, hi], [hi, hi]],
        np.float32,
    )
    src = np.vstack([np.asarray(src_pts, np.float32), anchors])
    dst = np.vstack([np.asarray(dst_pts, np.float32), anchors])
    rect = (0, 0, size, size)
    sub = cv2.Subdiv2D(rect)
    index = {}
    for i, (x, y) in enumerate(dst):
        key = (round(float(x), 3), round(float(y), 3))
        if key in index:  # two points on top of each other: keep the first
            continue
        index[key] = i
        sub.insert((float(np.clip(x, 0, hi)), float(np.clip(y, 0, hi))))
    out = np.zeros_like(img)
    filled = np.zeros((size, size), np.uint8)
    lookup = {
        (round(float(np.clip(x, 0, hi)), 3), round(float(np.clip(y, 0, hi)), 3)): i
        for i, (x, y) in enumerate(dst)
    }
    for t in sub.getTriangleList():
        pts = t.reshape(3, 2)
        ids = [lookup.get((round(float(x), 3), round(float(y), 3))) for x, y in pts]
        if any(i is None for i in ids):
            continue
        d = dst[ids].astype(np.float32)
        sr = src[ids].astype(np.float32)
        x, y, w, h = cv2.boundingRect(d)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w + 1, size), min(y + h + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        m = cv2.getAffineTransform(sr, d - np.array([x0, y0], np.float32))
        warped = cv2.warpAffine(
            img, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(
            tri_mask, np.round(d - np.array([x0, y0], np.float32)).astype(np.int32), 1
        )
        region = out[y0:y1, x0:x1]
        region[tri_mask > 0] = warped[tri_mask > 0]
        filled[y0:y1, x0:x1][tri_mask > 0] = 1
    out[filled == 0] = img[filled == 0]
    return out


def label_mask(labels: np.ndarray, which) -> np.ndarray:
    return np.isin(labels, which).astype(np.uint8)


def clean_hair_mask(crop_bgr: np.ndarray, hair: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Tidy a parsed hair mask: drop pixels coloured like the person's own skin (the parser sometimes calls a shoulder
    "hair") and pieces too small to be hair. Returns uint8 0/1."""
    hair = hair.astype(np.uint8)
    if not hair.any():
        return hair
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sk = skin > 0
    if sk.sum() > 50:
        mean = lab[sk].mean(0)
        dist = np.linalg.norm(lab - mean, axis=2)
        hair = hair * (dist > 14.0).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    keep = np.zeros_like(hair)
    area_min = hair.shape[0] * hair.shape[1] * 0.004
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            keep[comp == i] = 1
    return keep


def warp_mask_to_image(mask: np.ndarray, m: np.ndarray, shape) -> np.ndarray:
    """A crop-space mask (aligned with matrix ``m``: image -> crop) mapped back onto the full image, float 0..1."""
    h, w = shape[:2]
    inv = cv2.invertAffineTransform(m)
    return np.clip(
        cv2.warpAffine(mask.astype(np.float32), inv, (w, h), flags=cv2.INTER_LINEAR), 0.0, 1.0
    )


def face_mask(size: int, feather: float = 0.5) -> np.ndarray:
    """Soft face-shaped float mask on a ``size`` px ArcFace-aligned crop (1 = take the swapped face).

    An ellipse from brow line to chin, a bit narrower than the crop so hair and background at the crop corners are
    never pasted. ``feather`` 0..1 sets the edge softness, from a hard cut to about 4.5% of the crop size.
    """
    k = size / 112.0
    return _ellipse_mask(size, (56 * k, 56 * k), (39 * k, 44 * k), feather)


def ffhq_face_mask(size: int = 512, feather: float = 0.5) -> np.ndarray:
    """Same idea for the FFHQ-aligned crop the face restorer works on (it sees more hair and neck)."""
    k = size / 512.0
    return _ellipse_mask(size, (256 * k, 265 * k), (155 * k, 180 * k), feather)


def match_color(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Reinhard colour transfer in LAB: move ``src``'s mean/std (inside ``mask``) toward ``ref``'s.

    ``amount`` 0 returns ``src`` untouched, 1 matches ``ref`` fully. Both are uint8 BGR crops of the same size.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return src
    w = mask > 0.5
    if w.sum() < 16:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for c in range(3):
        ms, ss = s[..., c][w].mean(), s[..., c][w].std() + 1e-6
        mr, sr = r[..., c][w].mean(), r[..., c][w].std() + 1e-6
        matched = (s[..., c] - ms) * (sr / ss) + mr
        out[..., c] = s[..., c] * (1 - amount) + matched * amount
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def paste_back(
    img: np.ndarray, patch: np.ndarray, m: np.ndarray, mask: np.ndarray, opacity: float = 1.0
) -> np.ndarray:
    """Blend an aligned ``patch`` (and its float ``mask``) back into ``img`` through the inverse of ``m``.

    Only the face's bounding box is warped and blended, so a 12 MP photo costs the same as a small one.
    ``m`` maps image -> patch, as returned by :func:`align_crop`.
    """
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    pts = corners @ inv[:, :2].T + inv[:, 2]
    x0 = max(int(np.floor(pts[:, 0].min())) - 2, 0)
    y0 = max(int(np.floor(pts[:, 1].min())) - 2, 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 2, w)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return img
    shifted = inv.copy()
    shifted[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    # `inv` maps patch -> image, so no WARP_INVERSE_MAP here.
    warped = cv2.warpAffine(
        patch, shifted, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    alpha = cv2.warpAffine(
        mask, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    alpha = (np.clip(alpha, 0.0, 1.0) * float(np.clip(opacity, 0.0, 1.0)))[..., None]
    out = img.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(
        roi * (1 - alpha) + warped.astype(np.float32) * alpha + 0.5, 0, 255
    ).astype(np.uint8)
    return out
