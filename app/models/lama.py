"""LaMa (Large Mask Inpainting) wrapper using ONNX Runtime.

The model is from the official `opencv/inpainting_lama` Hugging Face repo:
  https://huggingface.co/opencv/inpainting_lama

LaMa uses Fourier convolutions to handle very large masks and to reproduce
periodic structure (brick walls, window grids, etc.) that patch-propagation
methods (TELEA, NS) cannot. It is the standard modern quality upgrade for
object removal.

We use ONNX Runtime instead of cv2.dnn because OpenCV 5's new graph engine
emits a "Targets not supported" warning for this model and silently produces
incorrect (near-identity) outputs. ORT runs the same model correctly.

This wrapper:
- Lazy-loads the ONNX model on first use (singleton, thread-safe)
- Preprocesses: image -> BGR blob scaled 1/255, mask -> binarized 0/1 blob
  Both resized to the model's fixed input size of 512x512.
- Runs inference via ONNX Runtime
- Postprocesses: CHW->HWC, uint8, resize back to original image dimensions
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


_MODEL_FILENAME = "inpainting_lama_2025jan.onnx"
_INPUT_SIZE = 512  # model expects 512x512 inputs


class LaMa:
    """Lazy-loaded, thread-safe singleton wrapper around the LaMa ONNX model."""

    _instance: "LaMa | None" = None
    _lock = Lock()

    def __init__(self, model_path: Path | str) -> None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = use all available cores
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def get(cls, model_dir: Path | str | None = None) -> "LaMa":
        """Get or construct the LaMa singleton for the given model directory."""
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            if model_dir is None:
                from app.config import get_settings
                model_dir = get_settings().model_dir
            model_dir = Path(model_dir)
            model_path = model_dir / _MODEL_FILENAME
            if not model_path.exists():
                raise ModelNotFoundError(f"LaMa model not found: {model_path}")
            cls._instance = LaMa(model_path)
            return cls._instance

    @staticmethod
    def clear_cache() -> None:
        """Reset the singleton (used by tests)."""
        with LaMa._lock:
            LaMa._instance = None

    # ------------------------------------------------------------------
    # Pre/post processing helpers (public for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _padding_to_multiple_of_eight(h: int, w: int) -> tuple[int, int]:
        """How much padding (rows, cols) to add so H, W are multiples of 8.

        Kept for backwards compatibility with earlier tests; the model
        itself does internal resizing, so this is unused in practice.
        """
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        return pad_h, pad_w

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 [0,255] -> 4D float32 blob in NCHW, scaled by 1/255."""
        return cv2.dnn.blobFromImage(
            img_bgr, scalefactor=1.0 / 255.0, size=(_INPUT_SIZE, _INPUT_SIZE),
            mean=(0, 0, 0), swapRB=False, crop=False,
        )

    @staticmethod
    def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """uint8 single-channel mask -> 4D float32 binarized blob."""
        blob = cv2.dnn.blobFromImage(
            mask, scalefactor=1.0, size=(_INPUT_SIZE, _INPUT_SIZE),
            mean=(0,), swapRB=False, crop=False,
        )
        return (blob > 0).astype(np.float32)

    @staticmethod
    def _postprocess(out_nchw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """NCHW -> HWC uint8, resized to target dimensions.

        The model output is in display range (0..~255) — we just clip
        defensively and convert. Resize back to the original input size.
        """
        chw = out_nchw[0]
        hwc = chw.transpose(1, 2, 0)
        hwc_u8 = np.clip(hwc, 0, 255).astype(np.uint8)
        if hwc_u8.shape[0] != target_h or hwc_u8.shape[1] != target_w:
            hwc_u8 = cv2.resize(hwc_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return hwc_u8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint img_bgr using the given mask (non-zero = to inpaint).

        Args:
            img_bgr: BGR uint8 image (H, W, 3).
            mask:    uint8 single-channel mask (H, W), non-zero = hole.

        Returns:
            BGR uint8 image of the same shape as img_bgr.
        """
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]

        img_blob = self._preprocess(img_bgr)   # (1, 3, 512, 512)
        mask_blob = self._preprocess_mask(mask)  # (1, 1, 512, 512)

        out = self._session.run(None, {"image": img_blob, "mask": mask_blob})[0]

        return self._postprocess(out, h, w)
