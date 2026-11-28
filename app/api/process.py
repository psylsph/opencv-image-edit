"""Image processing endpoint."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from threading import Event, Lock

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api._common import read_image_bytes
from app.api.schemas import ProcessRequest
from app.exceptions import AppError, DecodeError
from app.monitoring import (
    image_process_seconds,
    image_process_total,
    request_size_bytes,
)
from app.pipeline import ProcessResult, process_pipeline
from app.pipeline.io import decode_to_bgr, encode_png
from app.run_lock import RunCancelledError, run_exclusive

router = APIRouter()

# Progress of runs in flight, by the client's job id (polled by the page while a long run is going).
_PROGRESS_MAX = 64
_progress: dict[str, dict] = {}
_progress_lock = Lock()


def _run_one_at_a_time(
    job: str | None, cancel: Event, *args, **kwargs
) -> tuple[ProcessResult, float]:
    """The pipeline under the shared run lock (``app.run_lock``). Returns ``(result, seconds it ran)``.

    A run whose page has gone away gives up its place in the queue, and stops between stages if it already started
    (what finished stays in the stage cache, so a retry picks up from there).
    """

    def report(info: dict) -> None:
        if cancel.is_set():
            raise RunCancelledError("the request was cancelled")
        if job:
            _set_progress(job, info)

    def queued() -> None:
        if job:
            _set_progress(job, {"steps": [], "done": 0, "current": None, "queued": True})

    def timed():
        t = time.perf_counter()
        return process_pipeline(*args, **kwargs, progress=report), time.perf_counter() - t

    (result, ran), _waited = run_exclusive(timed, cancel=cancel, on_queued=queued)
    return result, ran


async def _watch_disconnect(request: Request, cancel: Event) -> None:
    while not cancel.is_set():
        if await request.is_disconnected():
            cancel.set()
            return
        await asyncio.sleep(0.5)


def _set_progress(job: str, info: dict) -> None:
    with _progress_lock:
        _progress[job] = {**info, "t": time.time()}
        if len(_progress) > _PROGRESS_MAX:
            for old in sorted(_progress, key=lambda k: _progress[k]["t"])[
                : len(_progress) - _PROGRESS_MAX
            ]:
                del _progress[old]


@router.get("/api/v1/progress/{job}")
def progress(job: str) -> dict:
    """Which stage a ``/api/v1/process`` run with this ``job`` id is on (``steps``, ``done``, ``current``)."""
    with _progress_lock:
        info = _progress.get(job)
    if info is None:
        raise HTTPException(status_code=404, detail="no such job")
    return {k: v for k, v in info.items() if k != "t"}


def _to_b64_png(img: np.ndarray | None) -> str | None:
    """Encode a BGR/BGRA image to a base64 data URL, or None if input is None."""
    if img is None:
        return None
    png_bytes = encode_png(img)
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


@router.post("/api/v1/process")
async def process(
    request: Request,
    file: UploadFile = File(...),
    settings: str = Form(...),
    background_mask: UploadFile | None = File(default=None),
    background_image: UploadFile | None = File(default=None),
    matte: UploadFile | None = File(default=None),
    face_donor: UploadFile | None = File(default=None),
    face_donor_extra: list[UploadFile] = File(default=[]),  # noqa: B008
    face_keep: UploadFile | None = File(default=None),
    edit_mask: UploadFile | None = File(default=None),
    edit_masks: list[UploadFile] = File(default=[]),  # noqa: B008
    job: str | None = Form(default=None),
) -> dict:
    """Process an uploaded image through the pipeline.

    Args:
        file: image file (PNG, JPEG, WebP, HEIC).
        settings: JSON-encoded ProcessRequest.
        matte: subject matte previously returned by ``/api/v1/matte``; avoids running the model again.
        face_donor: photo whose face replaces one in ``file``, required when ``face.enabled``.
        face_donor_extra: more photos of the same person (optional, repeatable) to average the identity over.
        face_keep: optional mask (white = keep the original pixels, for hair/glasses/hands over the face).
        background_image: replacement background, required when ``background.mode`` is ``image``.

    Returns:
        JSON with base64-encoded PNGs for final + 5 debug outputs.
    """
    body = await read_image_bytes(file)
    request_size_bytes.observe(len(body))

    try:
        req = ProcessRequest.model_validate_json(settings)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid settings: {exc}") from exc

    try:
        img_bgr = decode_to_bgr(body)
    except DecodeError as exc:
        image_process_total.labels(status="decode_error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bg_mask_img = None
    if background_mask is not None:
        try:
            bg_mask_img = decode_to_bgr(await background_mask.read())
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid background mask: {exc}") from exc

    matte_img = None
    if matte is not None:
        try:
            matte_img = decode_to_bgr(await read_image_bytes(matte))
            matte_img = matte_img[:, :, 0] if matte_img.ndim == 3 else matte_img
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid matte: {exc}") from exc

    bg_image = None
    if req.background.enabled and req.background.mode == "image":
        if background_image is None:
            raise HTTPException(
                status_code=422, detail="background mode 'image' needs a background_image file"
            )
        try:
            bg_image = decode_to_bgr(await read_image_bytes(background_image))
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid background image: {exc}") from exc

    donor_img = None
    if req.face.enabled:
        if face_donor is None:
            raise HTTPException(status_code=422, detail="face replacement needs a face_donor file")
        try:
            donor_img = decode_to_bgr(await read_image_bytes(face_donor))
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid donor photo: {exc}") from exc
        if donor_img.ndim == 3 and donor_img.shape[2] == 4:
            donor_img = donor_img[:, :, :3]

    extra_imgs: list[np.ndarray] = []
    keep_img = None
    if req.face.enabled:
        for up in face_donor_extra:
            try:
                im = decode_to_bgr(await read_image_bytes(up))
            except DecodeError as exc:
                image_process_total.labels(status="decode_error").inc()
                raise HTTPException(status_code=400, detail=f"invalid donor photo: {exc}") from exc
            extra_imgs.append(im[:, :, :3] if im.ndim == 3 and im.shape[2] == 4 else im)
        if face_keep is not None:
            try:
                keep_img = decode_to_bgr(await read_image_bytes(face_keep))
            except DecodeError as exc:
                image_process_total.labels(status="decode_error").inc()
                raise HTTPException(status_code=400, detail=f"invalid keep mask: {exc}") from exc
            keep_img = keep_img[:, :, 0] if keep_img.ndim == 3 else keep_img

    edit_img = None
    if req.edit.kind != "none":
        if edit_mask is None:
            raise HTTPException(
                status_code=422, detail="the selection tool needs an edit_mask file"
            )
        try:
            edit_img = decode_to_bgr(await read_image_bytes(edit_mask))
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid selection mask: {exc}") from exc
        edit_img = edit_img[:, :, 0] if edit_img.ndim == 3 else edit_img
    edit_imgs: list[np.ndarray] = []
    for up in edit_masks:
        try:
            im = decode_to_bgr(await read_image_bytes(up))
        except DecodeError as exc:
            image_process_total.labels(status="decode_error").inc()
            raise HTTPException(status_code=400, detail=f"invalid selection mask: {exc}") from exc
        edit_imgs.append(im[:, :, 0] if im.ndim == 3 else im)
    if len(edit_imgs) < sum(e.kind != "none" for e in req.edits):
        raise HTTPException(
            status_code=422, detail="every selection tool needs its own edit_masks file"
        )
    if job is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job):
        raise HTTPException(status_code=422, detail="invalid job id")

    started = time.perf_counter()
    status = "ok"
    cancel = Event()
    watcher = asyncio.create_task(_watch_disconnect(request, cancel))
    try:
        result, elapsed = await run_in_threadpool(
            _run_one_at_a_time,
            job,
            cancel,
            img_bgr,
            req,
            background_mask=bg_mask_img,
            background_image=bg_image,
            matte=matte_img,
            face_donor=donor_img,
            face_donors_extra=extra_imgs,
            face_keep=keep_img,
            edit_mask=edit_img,
            edit_masks=edit_imgs,
            use_cache=True,
        )
    except RunCancelledError as exc:
        image_process_total.labels(status="cancelled").inc()
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except AppError as exc:
        status = "processing_error"
        image_process_total.labels(status=status).inc()
        image_process_seconds.labels(status=status).observe(time.perf_counter() - started)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    finally:
        cancel.set()  # also stops the watcher
        watcher.cancel()
    # ``elapsed`` is the run itself, not the time spent queued behind another one.
    image_process_seconds.labels(status=status).observe(elapsed)
    image_process_total.labels(status=status).inc()

    return {
        "final": _to_b64_png(result.final),
        "before_after": _to_b64_png(result.before_after),
        "diff": _to_b64_png(result.diff),
        "mask": _to_b64_png(result.mask),
        "grain": _to_b64_png(result.grain_only),
        "upscaled": _to_b64_png(result.upscaled),
        "elapsed_seconds": elapsed,
        "steps": result.steps,
        "notes": result.notes,
        "output_size": {"width": result.final.shape[1], "height": result.final.shape[0]},
    }
