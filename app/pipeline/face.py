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
