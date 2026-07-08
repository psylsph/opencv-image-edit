"""AI upscaling via cv2.dnn_superres, with cv2.INTER_LANCZOS4 fallback.

OpenCV 5.0.0 exposes the dnn_superres module at the top level:
    cv2.dnn_superres.DnnSuperResImpl_create()

Supported algorithms: "edsr" (best quality, slow), "espcn" (fast),
"fsrcnn" (faster), "lapsrn" (fast, good).

If the corresponding ``.pb`` model file is not present in the model
directory, ``upscale()`` silently falls back to ``cv2.INTER_LANCZOS4``
so the pipeline stays usable without any neural-network weights on disk.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from app.exceptions import ModelNotFoundError


SUPPORTED_ALGORITHMS = ("edsr", "espcn", "fsrcnn", "lapsrn")
SUPPORTED_SCALES = (2, 3, 4)


class Upscaler:
    """Lazy-loaded, thread-safe singleton cache of ``cv2.dnn_superres`` upscalers.

    The cache is keyed by ``(algorithm, scale)`` so e.g. an EDSR x4 and an
    ESPCN x2 can coexist. Calling ``clear_cache()`` (e.g. in tests) resets
    the cache so the next ``get()`` will re-read the model from disk.
    """

    _instances: dict[tuple[str, int], "cv2.dnn_superres.DnnSuperResImpl"] = {}
    _lock = Lock()

    @classmethod
    def get(
        cls,
        algorithm: str,
        scale: int,
        model_dir: Path,
    ) -> "cv2.dnn_superres.DnnSuperResImpl":
        """Get or create a cached ``DnnSuperResImpl`` for ``(algorithm, scale)``.

        Raises:
            ValueError: if ``algorithm`` or ``scale`` is not supported.
            ModelNotFoundError: if the expected ``.pb`` file is missing.
        """
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"unsupported algorithm: {algorithm!r}; choose from {SUPPORTED_ALGORITHMS}"
            )
        if scale not in SUPPORTED_SCALES:
            raise ValueError(
                f"unsupported scale: {scale}; choose from {SUPPORTED_SCALES}"
            )

        key = (algorithm, scale)
        if key in cls._instances:
            return cls._instances[key]

        with cls._lock:
            # Re-check under the lock (double-checked locking).
            if key in cls._instances:
                return cls._instances[key]
            model_path = Path(model_dir) / f"{algorithm.upper()}_x{scale}.pb"
            if not model_path.exists():
                raise ModelNotFoundError(f"upscale model not found: {model_path}")
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model_path))
            sr.setModel(algorithm, scale)
            cls._instances[key] = sr
            return sr

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the singleton cache (used in tests)."""
        with cls._lock:
            cls._instances.clear()


def upscale(
    img: np.ndarray,
    scale: int = 2,
    algorithm: str = "edsr",
    model_dir: Path | str | None = None,
) -> np.ndarray:
    """Upscale an image by the given factor.

    If the model file is missing or ``scale == 1``, falls back to
    ``cv2.INTER_LANCZOS4`` (a classic, dependency-free resampler).

    Args:
        img: ``uint8`` image — 2D grayscale, 3D BGR, or 3D BGRA.
        scale: 1, 2, 3, or 4. ``1`` is a no-op (returns a copy of the input).
        algorithm: One of ``"edsr"``, ``"espcn"``, ``"fsrcnn"``, ``"lapsrn"``.
        model_dir: Directory containing the ``.pb`` model file. Defaults to
            ``settings.model_dir``.

    Returns:
        Upscaled image with spatial dims multiplied by ``scale``, same
        number of channels as the input (``alpha`` is dropped if present).
    """
    # --- passthrough -------------------------------------------------------
    if scale == 1:
        return img.copy() if isinstance(img, np.ndarray) else img

    # --- resolve model dir -------------------------------------------------
    if model_dir is None:
        from app.config import get_settings
        model_dir = get_settings().model_dir
    model_dir = Path(model_dir)

    # --- normalize input to 3D BGR (DnnSuperResImpl rejects 2D / 4D) --------
    if img.ndim == 2:
        input_was_2d = True
        img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        input_was_2d = False
        img_3d = img[:, :, :3]  # drop alpha if present

    # --- run model OR fall back to LANCZOS4 --------------------------------
    try:
        sr = Upscaler.get(algorithm, scale, model_dir)
        out = sr.upsample(img_3d)
    except ModelNotFoundError:
        out = cv2.resize(
            img_3d,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )

    # --- restore 2D output if the input was grayscale ----------------------
    if input_was_2d:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    return out
