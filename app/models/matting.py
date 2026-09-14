"""ONNX matting model wrapper using ONNX Runtime.

Supports multiple background-removal models (U2NetP, U2Net, IS-Net, Silueta)
via a model registry. Each model is loaded lazily and cached as a singleton.

All models use ONNX Runtime for consistency — it avoids the OpenCV 5 graph
engine's incomplete operator support and gives us a single, optimized
inference path.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError

# ImageNet normalization (all rembg-family models use this).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MattingModelInfo(NamedTuple):
    """Static metadata for a matting model variant."""

    filename: str
    input_size: int
    display_name: str
    description: str
    size_mb: float


# Registry of supported matting models.
# Ordered by quality (best last). The ``key`` is what the API/UI uses.
MATTING_MODELS: dict[str, MattingModelInfo] = {
    "u2netp": MattingModelInfo(
        filename="u2netp.onnx",
        input_size=320,
        display_name="U2NetP (fast)",
        description="4.7 MB · 320px · lightweight, best for portraits/people",
        size_mb=4.7,
    ),
    "silueta": MattingModelInfo(
        filename="silueta.onnx",
        input_size=320,
        display_name="Silueta (balanced)",
        description="44 MB · 320px · good speed/quality balance for general objects",
        size_mb=44.0,
    ),
    "u2net": MattingModelInfo(
        filename="u2net.onnx",
        input_size=320,
        display_name="U2Net (general)",
        description="176 MB · 320px · the standard rembg default, all-round performer",
        size_mb=176.0,
    ),
    "isnet": MattingModelInfo(
        filename="isnet_general_use.onnx",
        input_size=1024,
        display_name="IS-Net (high quality)",
        description="174 MB · 1024px · state-of-the-art, high accuracy for complex scenes",
        size_mb=174.0,
    ),
    "birefnet_tiny": MattingModelInfo(
        filename="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        input_size=1024,
        display_name="BiRefNet Tiny (best quality)",
        description="213 MB · 1024px · 2024 SOTA transformer, finest edge detail",
        size_mb=213.0,
    ),
}

DEFAULT_MODEL = "birefnet_tiny"


def available_models(model_dir: Path | str | None = None) -> list[dict]:
    """List all registered models with availability info.

    Returns a list of dicts suitable for JSON serialization:
    ``{key, filename, display_name, description, size_mb, available}``.
    """
    from app.config import get_settings

    if model_dir is None:
        model_dir = get_settings().model_dir
    model_dir = Path(model_dir)

    result = []
    for key, info in MATTING_MODELS.items():
        result.append(
            {
                "key": key,
                "filename": info.filename,
                "display_name": info.display_name,
                "description": info.description,
                "size_mb": info.size_mb,
                "available": (model_dir / info.filename).exists(),
            }
        )
    return result


class MattingModel:
    """Background matting via ONNX Runtime.

    Each model is single-channel output: each pixel is foreground probability.
    Works with any model in :data:`MATTING_MODELS`.
    """

    def __init__(self, model_path: Path | str, input_size: int = 320) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise ModelNotFoundError(f"matting model not found: {self.model_path}")
        self.input_size = input_size
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = use all available cores
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run matting on a BGR image, return float32 mask (H, W) in [0, 1].

        All rembg-family models output a (1, 1, S, S) probability mask in
        [0, 1] — no sigmoid needed.
        """
        h, w = img_bgr.shape[:2]
        size = self.input_size
        # BGR -> RGB, resize, normalize (ImageNet stats, [0,1] pixel range)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        normalized = (resized - _MEAN) / _STD
        # HWC -> NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        # Run inference via ORT
        input_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {input_name: blob})[0]
        mask = out[0, 0]  # (S, S) float32, already a probability
        # Resize to original
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)


# --- Per-model singleton cache -------------------------------------------------

_cache: dict[str, MattingModel] = {}
_cache_lock = Lock()


def get_matting_model(
    model_name: str = DEFAULT_MODEL,
    model_dir: Path | str | None = None,
) -> MattingModel:
    """Get a cached matting model singleton (thread-safe).

    Each model variant is loaded lazily on first use and kept in a per-key
    cache, so switching models in the UI doesn't reload previously used ones.
    """
    from app.model_cache import mark_model_used

    mark_model_used()
    if model_name not in MATTING_MODELS:
        raise ModelNotFoundError(
            f"unknown matting model '{model_name}'. "
            f"Available: {', '.join(MATTING_MODELS)}"
        )

    cached = _cache.get(model_name)
    if cached is not None:
        return cached

    with _cache_lock:
        # Double-checked locking
        cached = _cache.get(model_name)
        if cached is not None:
            return cached

        if model_dir is None:
            from app.config import get_settings

            model_dir = get_settings().model_dir

        info = MATTING_MODELS[model_name]
        model_path = Path(model_dir) / info.filename
        _cache[model_name] = MattingModel(model_path, input_size=info.input_size)
        return _cache[model_name]


def clear_matting_cache() -> None:
    """Reset the entire model cache (used by tests)."""
    global _cache
    with _cache_lock:
        _cache = {}
