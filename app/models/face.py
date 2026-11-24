"""Face detection and neural face swap (ONNX Runtime + OpenCV's YuNet).

Four models, all optional downloads (see ``scripts/download_models.py``):

* **YuNet** (`cv2.FaceDetectorYN`): boxes + 5 landmarks.
* **ArcFace** ``w600k_r50``: 512-d identity embedding of the donor face.
* **inswapper_128**: generator that renders the donor identity onto the aligned (128 px) target face. Its weights are
  published for non-commercial research use only.
* **GFPGAN 1.4** (optional): face restorer that sharpens the 128 px result at full resolution.

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
    "swapper": FaceModelInfo("inswapper_128.onnx", "InSwapper 128 face swapper", 530.0, True),
    "restorer": FaceModelInfo("gfpgan_1.4.onnx", "GFPGAN 1.4 face restorer", 325.0, False),
}


class Face(NamedTuple):
    box: tuple[float, float, float, float]  # x, y, w, h in image pixels
    kps: np.ndarray  # (5, 2) float32, image pixels
    score: float


def face_models_status(model_dir: Path | str | None = None) -> dict:
    """Which face models are on disk. ``ready`` means detection + swapping work; ``restorer`` is an optional extra."""
    if model_dir is None:
        from app.config import get_settings

        model_dir = get_settings().model_dir
    model_dir = Path(model_dir)
    files = {key: (model_dir / info.filename).exists() for key, info in FACE_MODELS.items()}
    missing = [
        FACE_MODELS[k].filename for k, ok in files.items() if not ok and FACE_MODELS[k].required
    ]
    return {"ready": not missing, "restorer": files["restorer"], "missing": missing}


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 0
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class FaceEngine:
    """Detector + embedder + swapper (+ restorer), loaded lazily and kept together."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self._detector_lock = Lock()
        self._detector = None
        self._embedder: ort.InferenceSession | None = None
        self._swapper: ort.InferenceSession | None = None
        self._emap: np.ndarray | None = None
        self._restorer: ort.InferenceSession | None = None
        self._restorer_checked = False

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

    # -- identity + generator ----------------------------------------------------------------------------------

    def _latent(self, donor_bgr: np.ndarray, face: Face) -> np.ndarray:
        if self._embedder is None:
            self._embedder = _session(self._path("embedder"))
        crop, _ = geo.align_crop(donor_bgr, face.kps, geo.arcface_template(112), 112)
        blob = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        emb = self._embedder.run(None, {self._embedder.get_inputs()[0].name: blob})[0].reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        if self._swapper is None:
            self._load_swapper()
        latent = emb[None].astype(np.float32) @ self._emap
        return (latent / (np.linalg.norm(latent) + 1e-9)).astype(np.float32)

    def _load_swapper(self) -> None:
        path = self._path("swapper")
        import onnx  # only needed once, to read the embedding map stored in the swapper's last initializer

        model = onnx.load(str(path), load_external_data=False)
        self._emap = onnx.numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
        self._swapper = _session(path)

    def _restore_session(self) -> ort.InferenceSession | None:
        if not self._restorer_checked:
            self._restorer_checked = True
            path = self.model_dir / FACE_MODELS["restorer"].filename
            if path.exists():
                self._restorer = _session(path)
        return self._restorer

    # -- the swap ----------------------------------------------------------------------------------------------

    def swap(
        self,
        target_bgr: np.ndarray,
        donor_bgr: np.ndarray,
        *,
        target_index: int = 0,
        donor_index: int = 0,
        blend: float = 0.5,
        color_match: float = 0.5,
        restore: float = 0.8,
    ) -> np.ndarray:
        """Put the donor's face on one face of ``target_bgr`` and return the new image (same size)."""
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

        target = target_faces[target_index]
        latent = self._latent(donor_bgr, donor_faces[donor_index])

        crop, m = geo.align_crop(target_bgr, target.kps, geo.arcface_template(128), 128)
        blob = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self._swapper.run(None, {"target": blob, "source": latent})[0][0]
        swapped = np.clip(out.transpose(1, 2, 0)[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)

        mask = geo.face_mask(128, blend)
        swapped = geo.match_color(swapped, crop, geo.face_mask(128, 0.0), color_match)
        result = geo.paste_back(target_bgr, swapped, m, mask)

        restorer = self._restore_session() if restore > 0 else None
        if restorer is not None:
            result = self._restore(result, target.kps, restorer, blend, restore)
        return result

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
