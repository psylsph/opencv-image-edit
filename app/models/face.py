"""Face detection, neural face swap and hair transplant (ONNX Runtime + OpenCV's YuNet).

Models, all optional downloads (see ``scripts/download_models.py``):

* **YuNet** (`cv2.FaceDetectorYN`): boxes + 5 landmarks.
* **ArcFace** ``w600k_r50``: 512-d identity embedding of the donor face.
* **HyperSwap 1c (256 px)**: the swapper. Sharper and more faithful than the older InSwapper, which is still used if
  it is the only swapper installed (``inswapper_128.onnx``; non-commercial research weights).
* **GFPGAN 1.4** (optional): face restorer.
* **BiSeNet** (optional): face parsing, needed to move the donor's hair too.

The geometry around them (alignment, masks, colour match, paste-back) is in :mod:`app.pipeline.face`.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError, ValidationError
from app.pipeline import face as geo

DETECT_MAX_SIDE = (
    1024  # detect on a copy no larger than this; landmarks are scaled back to the full image
)
MIN_SCORE = 0.7


class FaceModelInfo(NamedTuple):
    filename: str
    display_name: str
    size_mb: float
    required: bool


FACE_MODELS: dict[str, FaceModelInfo] = {
    "detector": FaceModelInfo(
        "face_detection_yunet_2023mar.onnx", "YuNet face detector", 0.2, True
    ),
    "embedder": FaceModelInfo("arcface_w600k_r50.onnx", "ArcFace identity embedder", 166.0, True),
    "swapper": FaceModelInfo("hyperswap_1c_256.onnx", "HyperSwap 1c face swapper", 384.0, False),
    "swapper_legacy": FaceModelInfo(
        "inswapper_128.onnx", "InSwapper 128 face swapper", 530.0, False
    ),
    "restorer": FaceModelInfo("gfpgan_1.4.onnx", "GFPGAN 1.4 face restorer", 325.0, False),
    "parser": FaceModelInfo("bisenet_resnet_34.onnx", "BiSeNet face parser (hair)", 89.0, False),
}


class Face(NamedTuple):
    box: tuple[float, float, float, float]  # x, y, w, h in image pixels
    kps: np.ndarray  # (5, 2) float32, image pixels
    score: float


def face_models_status(model_dir: Path | str | None = None) -> dict:
    """Which face models are on disk.

    ``ready``: detection + swapping work. ``swapper``: which generator will be used. ``restorer`` / ``hair``: the
    optional extras are installed.
    """
    if model_dir is None:
        from app.config import get_settings

        model_dir = get_settings().model_dir
    model_dir = Path(model_dir)
    has = {key: (model_dir / info.filename).exists() for key, info in FACE_MODELS.items()}
    missing = [FACE_MODELS[k].filename for k in ("detector", "embedder") if not has[k]]
    swapper = "hyperswap" if has["swapper"] else "inswapper" if has["swapper_legacy"] else None
    if swapper is None:
        missing.append(FACE_MODELS["swapper"].filename)
    return {
        "ready": not missing,
        "swapper": swapper,
        "restorer": has["restorer"],
        "hair": has["parser"],
        "missing": missing,
    }


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 0
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class FaceEngine:
    """Detector + embedder + swapper (+ restorer, parser), loaded lazily and kept together."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self._detector_lock = Lock()
        self._detector = None
        self._embedder: ort.InferenceSession | None = None
        self._swapper: ort.InferenceSession | None = None
        self._swapper_kind: str | None = None
        self._emap: np.ndarray | None = None
        self._restorer: ort.InferenceSession | None = None
        self._restorer_checked = False
        self._parser: ort.InferenceSession | None = None

    def _path(self, key: str) -> Path:
        path = self.model_dir / FACE_MODELS[key].filename
        if not path.exists():
            raise ModelNotFoundError(
                f"face model not found: {FACE_MODELS[key].filename} (run scripts/download_models.py)"
            )
        return path

    # -- detection ---------------------------------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        """Faces in the image, ordered left to right (so an index means the same thing on every call)."""
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._path("detector")), "", (320, 320), MIN_SCORE, 0.3, 5000
            )
        h, w = img_bgr.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = (
            img_bgr
            if scale == 1.0
            else cv2.resize(
                img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        )
        with self._detector_lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._detector.detect(small)
        faces: list[Face] = []
        for r in rows if rows is not None else []:
            kps = (r[4:14].reshape(5, 2) / scale).astype(np.float32)
            box = tuple(float(v) / scale for v in r[:4])
            faces.append(Face(box, kps, float(r[14])))
        faces.sort(key=lambda f: f.box[0] + f.box[2] / 2)
        return faces

    # -- identity ----------------------------------------------------------------------------------------------

    def _embedding(self, donor_bgr: np.ndarray, face: Face) -> np.ndarray:
        """Unit-length 512-d ArcFace identity of the donor face."""
        if self._embedder is None:
            self._embedder = _session(self._path("embedder"))
        crop, _ = geo.align_crop(donor_bgr, face.kps, geo.arcface_template(112), 112)
        blob = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        emb = self._embedder.run(None, {self._embedder.get_inputs()[0].name: blob})[0].reshape(-1)
        return (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)

    def _load_swapper(self) -> None:
        if self._swapper is not None:
            return
        status = face_models_status(self.model_dir)
        if status["swapper"] == "hyperswap":
            self._swapper = _session(self._path("swapper"))
            self._swapper_kind = "hyperswap"
        elif status["swapper"] == "inswapper":
            path = self._path("swapper_legacy")
            import onnx  # only to read the embedding map stored in the swapper's last initializer

            model = onnx.load(str(path), load_external_data=False)
            self._emap = onnx.numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
            self._swapper = _session(path)
            self._swapper_kind = "inswapper"
        else:
            raise ModelNotFoundError(
                f"face model not found: {FACE_MODELS['swapper'].filename} (run scripts/download_models.py)"
            )

    def _render(self, crop: np.ndarray, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the generator on an aligned target crop: returns (patch BGR uint8, the model's own face mask or None)."""
        if self._swapper_kind == "hyperswap":
            blob = ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[
                None
            ]
            out, mask = self._swapper.run(None, {"source": emb[None], "target": blob})
            patch = np.clip((out[0].transpose(1, 2, 0)[:, :, ::-1] + 1.0) * 127.5 + 0.5, 0, 255)
            return patch.astype(np.uint8), np.clip(mask[0, 0], 0, 1).astype(np.float32)
        latent = emb[None] @ self._emap
        latent = (latent / (np.linalg.norm(latent) + 1e-9)).astype(np.float32)
        blob = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self._swapper.run(None, {"target": blob, "source": latent})[0][0]
        patch = np.clip(out.transpose(1, 2, 0)[:, :, ::-1] * 255.0 + 0.5, 0, 255)
        return patch.astype(np.uint8), None

    # -- optional extras ---------------------------------------------------------------------------------------

    def _restore_session(self) -> ort.InferenceSession | None:
        if not self._restorer_checked:
            self._restorer_checked = True
            path = self.model_dir / FACE_MODELS["restorer"].filename
            if path.exists():
                self._restorer = _session(path)
        return self._restorer

    def _restore(
        self,
        img: np.ndarray,
        kps: np.ndarray,
        session: ort.InferenceSession,
        blend: float,
        strength: float,
    ) -> np.ndarray:
        crop, m = geo.align_crop(img, kps, geo.ffhq_template(512), 512)
        blob = ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
        restored = np.clip((out.transpose(1, 2, 0)[:, :, ::-1] + 1.0) * 127.5 + 0.5, 0, 255).astype(
            np.uint8
        )
        return geo.paste_back(img, restored, m, geo.ffhq_face_mask(512, blend), opacity=strength)

    def parse(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel face-part labels (BiSeNet, 19 classes) for a 512x512 crop."""
        if self._parser is None:
            self._parser = _session(self._path("parser"))
        blob = ((crop_bgr[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[
            None
        ]
        out = self._parser.run(None, {self._parser.get_inputs()[0].name: blob})[0]
        return out[0].argmax(0).astype(np.uint8)

    def _transplant_hair(
        self, result: np.ndarray, target: np.ndarray, donor: np.ndarray, tface: Face, dface: Face
    ) -> np.ndarray:
        """Replace the target's hair with the donor's, scaled and rotated to the target's head.

        The target's own hair is erased by inpainting (LaMa) first, so a shorter donor haircut doesn't leave the old
        one showing; then the donor's hair is pasted over, leaving the swapped face untouched.
        """
        size = 640
        tpl = geo.head_template(size)
        tcrop, mt = geo.align_crop(target, tface.kps, tpl, size)
        dcrop, _ = geo.align_crop(donor, dface.kps, tpl, size)
        # BiSeNet works at 512; parse there and bring the labels back to the crop's resolution.
        up = lambda lab: cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)  # noqa: E731
        tl = up(self.parse(cv2.resize(tcrop, (512, 512), interpolation=cv2.INTER_AREA)))
        dl = up(self.parse(cv2.resize(dcrop, (512, 512), interpolation=cv2.INTER_AREA)))

        t_hair = geo.clean_hair_mask(
            tcrop, geo.label_mask(tl, geo.HAIR_LABELS), geo.label_mask(tl, (1,))
        )
        d_hair = geo.clean_hair_mask(
            dcrop, geo.label_mask(dl, geo.HAIR_LABELS), geo.label_mask(dl, (1,))
        )
        if d_hair.sum() < size * size * 0.002:
            return result  # a bald donor: nothing to bring over (the target's hair stays)

        out = result
        # A huge old-hair area (long hair over a busy scene) inpaints into a smear; leave it and paste over it instead.
        if t_hair.any() and t_hair.mean() < 0.22:
            # Erase the old hair (a generous margin, so no halo of it is left), but never the face itself.
            k = int(size * 0.045) | 1
            grow = cv2.dilate(t_hair, np.ones((k, k), np.uint8))
            keep = cv2.dilate(geo.label_mask(tl, geo.FACE_LABELS), np.ones((3, 3), np.uint8))
            erase = geo.warp_mask_to_image((grow * (1 - keep)).astype(np.float32), mt, out.shape)
            hole = (erase > 0.3).astype(np.uint8) * 255
            if hole.any():
                from app.models.lama import LaMa

                out = LaMa.get(self.model_dir).infer(out, hole, dilate_px=2)

        soft = cv2.GaussianBlur(d_hair.astype(np.float32), (0, 0), size * 0.004)
        # Hair never covers the eyes, nose or mouth, and fades out before the crop's border.
        features = cv2.GaussianBlur(
            geo.label_mask(dl, (4, 5, 10, 11, 12, 13)).astype(np.float32), (0, 0), 3.0
        )
        soft = soft * (1.0 - features) * geo.border_fade(size)
        return geo.paste_back(out, dcrop, mt, soft)

    # -- the swap ----------------------------------------------------------------------------------------------

    def swap(
        self,
        target_bgr: np.ndarray,
        donor_bgr: np.ndarray,
        *,
        target_index: int = 0,
        donor_index: int = 0,
        blend: float = 0.5,
        color_match: float = 0.3,
        restore: float = 0.3,
        hair: bool = False,
    ) -> np.ndarray:
        """Put the donor's face (and with ``hair`` their hair) on one face of ``target_bgr``; same size back."""
        target_faces = self.detect(target_bgr)
        if not target_faces:
            raise ValidationError("no face found in the photo")
        if target_index >= len(target_faces):
            raise ValidationError(
                f"the photo has {len(target_faces)} face(s); face {target_index + 1} does not exist"
            )
        donor_faces = self.detect(donor_bgr)
        if not donor_faces:
            raise ValidationError("no face found in the donor photo")
        if donor_index >= len(donor_faces):
            raise ValidationError(
                f"the donor photo has {len(donor_faces)} face(s); face {donor_index + 1} does not exist"
            )
        if hair:
            self._path("parser")  # fail early with a clear message if hair swapping isn't installed

        target, donor = target_faces[target_index], donor_faces[donor_index]
        self._load_swapper()
        emb = self._embedding(donor_bgr, donor)

        size = 256 if self._swapper_kind == "hyperswap" else 128
        crop, m = geo.align_crop(target_bgr, target.kps, geo.swap_template(size), size)
        patch, model_mask = self._render(crop, emb)
        mask = geo.swap_mask(size, blend, model_mask)
        patch = geo.match_color(patch, crop, (mask > 0.5).astype(np.float32), color_match)
        result = geo.paste_back(target_bgr, patch, m, mask)

        restorer = self._restore_session() if restore > 0 else None
        if restorer is not None:
            result = self._restore(result, target.kps, restorer, blend, restore)
        if hair:
            result = self._transplant_hair(result, target_bgr, donor_bgr, target, donor)
        return result


# --- singleton ------------------------------------------------------------------------------------------------

_engine: FaceEngine | None = None
_engine_lock = Lock()


def get_face_engine(model_dir: Path | str | None = None) -> FaceEngine:
    from app.model_cache import mark_model_used

    mark_model_used()
    global _engine
    with _engine_lock:
        if _engine is None:
            if model_dir is None:
                from app.config import get_settings

                model_dir = get_settings().model_dir
            _engine = FaceEngine(model_dir)
        return _engine


def clear_face_cache() -> None:
    global _engine
    with _engine_lock:
        _engine = None
