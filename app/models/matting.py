"""U2NetP ONNX matting model wrapper using OpenCV DNN.

Loads a U2NetP portrait matting model via cv2.dnn.readNetFromONNX and runs
inference on a BGR image, returning a single-channel float32 mask in [0, 1].

We deliberately do NOT depend on onnxruntime/rembg — the OpenCV DNN module
is bundled with the base opencv-python wheel and is plenty fast for a
4.7 MB model on CPU.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.exceptions import ModelNotFoundError


_MODEL_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MattingModel:
    """U2NetP portrait matting via cv2.dnn.

    The model is single-channel output: each pixel is foreground probability.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise ModelNotFoundError(f"matting model not found: {self.model_path}")
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run matting on a BGR image, return float32 mask (H, W) in [0, 1].

        The U2NetP ONNX (from rembg) outputs a (1, 1, 320, 320) probability
        mask in [0, 1] — no sigmoid needed. cv2.dnn collapses the multi-output
        U^2-Net graph into a single tensor.
        """
        h, w = img_bgr.shape[:2]
        # BGR -> RGB, resize, normalize (ImageNet stats, [0,1] pixel range)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (_MODEL_SIZE, _MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized - _MEAN) / _STD
        # HWC -> NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        self._net.setInput(blob)
        out = self._net.forward()  # shape (1, 1, 320, 320), values in [0, 1]
        mask = out[0, 0]  # (320, 320) float32, already a probability
        # Resize to original
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)


_singleton: MattingModel | None = None


def get_matting_model(model_dir: Path | str | None = None) -> MattingModel:
    """Get the cached matting model singleton."""
    global _singleton
    if _singleton is None:
        if model_dir is None:
            from app.config import get_settings
            model_dir = get_settings().model_dir
        model_path = Path(model_dir) / "u2netp.onnx"
        _singleton = MattingModel(model_path)
    return _singleton
