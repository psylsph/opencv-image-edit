"""U2NetP ONNX matting model wrapper using ONNX Runtime.

Loads a U2NetP portrait matting model via ONNX Runtime and runs inference
on a BGR image, returning a single-channel float32 mask in [0, 1].

All models in this app use ONNX Runtime for consistency — it avoids the
OpenCV 5 graph engine's incomplete operator support and gives us a single,
optimized inference path.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


_MODEL_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MattingModel:
    """U2NetP portrait matting via ONNX Runtime.

    The model is single-channel output: each pixel is foreground probability.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise ModelNotFoundError(f"matting model not found: {self.model_path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = use all available cores
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run matting on a BGR image, return float32 mask (H, W) in [0, 1].

        The U2NetP ONNX (from rembg) outputs a (1, 1, 320, 320) probability
        mask in [0, 1] — no sigmoid needed.
        """
        h, w = img_bgr.shape[:2]
        # BGR -> RGB, resize, normalize (ImageNet stats, [0,1] pixel range)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (_MODEL_SIZE, _MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized - _MEAN) / _STD
        # HWC -> NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        # Run inference via ORT
        input_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {input_name: blob})[0]
        mask = out[0, 0]  # (320, 320) float32, already a probability
        # Resize to original
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)


_singleton: MattingModel | None = None
_lock = Lock()


def get_matting_model(model_dir: Path | str | None = None) -> MattingModel:
    """Get the cached matting model singleton (thread-safe)."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _lock:
        if _singleton is not None:
            return _singleton
        if model_dir is None:
            from app.config import get_settings
            model_dir = get_settings().model_dir
        model_path = Path(model_dir) / "u2netp.onnx"
        _singleton = MattingModel(model_path)
        return _singleton


def clear_matting_cache() -> None:
    """Reset the singleton (used by tests)."""
    global _singleton
    with _lock:
        _singleton = None
