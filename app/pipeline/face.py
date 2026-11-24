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
