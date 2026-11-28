"""One model-heavy request at a time.

Matting, SAM, LaMa, face swap and upscaling each hold large models and full-size images; two at once (a double tap, a
second tab, a live-preview matte during a Process run) could push the container past its memory limit. Every heavy
request takes this lock in its worker thread, so the event loop (and progress polling) keeps answering meanwhile.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Lock
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.exceptions import AppError, ProcessingError

_lock = Lock()
MAX_WAIT_SECONDS = 300.0


class RunCancelledError(ProcessingError):
    """The client went away before or during its run (nobody is waiting for the answer)."""

    status_code = 499


class BusyError(AppError):
    """Waited too long for the runs ahead of this one."""

    status_code = 503


def run_exclusive(
    fn: Callable[..., Any],
    *args,
    cancel: Event | None = None,
    on_queued: Callable[[], None] | None = None,
    max_wait: float = MAX_WAIT_SECONDS,
    **kwargs,
) -> tuple[Any, float]:
    """Run ``fn`` holding the lock. Returns ``(result, seconds spent waiting for the lock)``.

    Raises RunCancelledError if ``cancel`` is set while waiting (an abandoned request gives up its turn), BusyError after
    ``max_wait`` seconds.
    """
    started = time.perf_counter()
    if not _lock.acquire(blocking=False):
        if on_queued is not None:
            on_queued()
        while not _lock.acquire(timeout=0.25):
            if cancel is not None and cancel.is_set():
                raise RunCancelledError("the request was cancelled while waiting")
            if time.perf_counter() - started > max_wait:
                raise BusyError("the server is busy with other edits; try again in a moment")
    waited = time.perf_counter() - started
    try:
        if cancel is not None and cancel.is_set():
            raise RunCancelledError("the request was cancelled while waiting")
        return fn(*args, **kwargs), waited
    finally:
        _lock.release()


async def heavy(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """``await heavy(model_call, ...)`` from an endpoint: runs it in a worker thread, one at a time."""
    result, _ = await run_in_threadpool(run_exclusive, fn, *args, **kwargs)
    return result
