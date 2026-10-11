"""Central lifecycle for lazily loaded inference models.

Models remain cached for a short idle window so consecutive edits are fast, then
all model sessions are released together.  The health endpoint must never call
``mark_model_used``: liveness checks should not load or retain inference models.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import threading
import time

logger = logging.getLogger(__name__)

_condition = threading.Condition()
_last_used: float | None = None
_worker: threading.Thread | None = None


def _idle_timeout_seconds() -> float:
    from app.config import get_settings

    return float(get_settings().model_idle_timeout_seconds)


def mark_model_used() -> None:
    """Reset the model-cache idle timer, starting its daemon on first use."""
    global _last_used, _worker
    with _condition:
        _last_used = time.monotonic()
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_eviction_loop,
                name="model-cache-evictor",
                daemon=True,
            )
            _worker.start()
        _condition.notify_all()


def _eviction_loop() -> None:
    global _last_used
    while True:
        with _condition:
            if _last_used is None:
                return
            remaining = _idle_timeout_seconds() - (time.monotonic() - _last_used)
            if remaining > 0:
                _condition.wait(timeout=remaining)
                continue
            _last_used = None

        clear_all_model_caches()
        logger.info("Unloaded idle model caches after %d seconds", int(_idle_timeout_seconds()))
        return


def clear_all_model_caches() -> None:
    """Release every cached model session and return free arenas to Linux."""
    from app.models.anime import clear_anime_cache
    from app.models.lama import LaMa
    from app.models.matting import clear_matting_cache
    from app.models.painting import clear_paint_cache
    from app.models.sam import MobileSAM
    from app.pipeline.upscale import Upscaler

    clear_matting_cache()
    LaMa.clear_cache()
    MobileSAM.clear_cache()
    Upscaler.clear_cache()
    clear_anime_cache()
    clear_paint_cache()
    gc.collect()

    # ONNX Runtime's CPU allocator may leave released arenas with glibc.
    # malloc_trim is Linux/glibc-specific, so failure is harmless elsewhere.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass
