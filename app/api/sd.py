"""Stable Diffusion model management endpoints.

SD 1.5 models are ~4GB and optional. These endpoints let the frontend
check availability and trigger an on-demand download.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running download processes: task_id -> Popen
_downloads: dict[str, subprocess.Popen] = {}
# Track task metadata: task_id -> {log_path, log_fp, started_at, reaped}
_download_meta: dict[str, dict] = {}
# Keep at most this many finished tasks around for status polling.
_MAX_FINISHED_TASKS = 10
_FINISHED_TTL_SECONDS = 3600


def _reap_finished() -> None:
    """Close log file handles and evict old finished download tasks."""
    now = time.time()
    finished_ids = []
    for tid, proc in _downloads.items():
        meta = _download_meta.get(tid, {})
        # Close the log file handle once the process has exited.
        if proc.poll() is not None and meta.get("log_fp") is not None:
            with suppress(OSError):
                meta["log_fp"].close()
            meta["log_fp"] = None
            meta["reaped"] = True
        if meta.get("reaped"):
            finished_ids.append((tid, meta.get("started_at", now)))

    # Evict oldest finished entries beyond the retention cap or TTL.
    finished_ids.sort(key=lambda item: item[1])
    evict = [tid for tid, started in finished_ids if now - started > _FINISHED_TTL_SECONDS]
    over_cap = [tid for tid, _ in finished_ids if tid not in evict][
        : max(0, len(finished_ids) - _MAX_FINISHED_TASKS)
    ]
    for tid in evict + over_cap:
        meta = _download_meta.pop(tid, {})
        fp = meta.get("log_fp")
        if fp is not None:
            with suppress(OSError):
                fp.close()
        _downloads.pop(tid, None)


def _sd_dir() -> Path:
    return get_settings().model_dir / "sd-inpainting"


def _sd_size_mb() -> int:
    """Total size of SD model files in MB, or 0 if not present."""
    sd = _sd_dir()
    if not sd.exists():
        return 0
    total = 0
    for f in sd.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total // (1024 * 1024)


@router.get("/api/v1/sd/status")
def sd_status() -> dict:
    """Check if SD models are downloaded and ready."""
    from app.models.sd_inpaint import SDInpaint

    available = SDInpaint.is_available(get_settings().model_dir)
    return {
        "available": available,
        "size_mb": _sd_size_mb(),
        "path": str(_sd_dir()),
    }


@router.post("/api/v1/sd/download")
def sd_download() -> dict:
    """Start downloading SD models in the background.

    Returns a task_id that can be polled via /api/v1/sd/download/{task_id}/status.
    Only one download can run at a time.
    """
    _reap_finished()

    # Check if already running
    for tid, proc in list(_downloads.items()):
        if proc.poll() is None:
            return {"started": False, "task_id": tid, "message": "download already running"}

    settings = get_settings()
    log_dir = Path("/tmp/opencv-image-edit/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex[:12]
    log_path = log_dir / f"sd_download_{task_id}.log"

    cmd = [sys.executable, "scripts/download_models.py", str(settings.model_dir), "--sd-only"]

    logger.info("Starting SD model download (task_id=%s)", task_id)
    # Open in a context where we own the handle so it can be closed once the
    # child process exits (see _reap_finished). The child inherits the fd via
    # stdout=; we keep the handle open until then because the child is still
    # writing to it.
    log_fp = open(log_path, "w")  # noqa: SIM115 - outlives this function; closed in _reap_finished
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    _downloads[task_id] = proc
    _download_meta[task_id] = {
        "log_path": str(log_path),
        "log_fp": log_fp,
        "started_at": time.time(),
        "reaped": False,
    }

    return {"started": True, "task_id": task_id}


@router.get("/api/v1/sd/download/{task_id}/status")
def sd_download_status(task_id: str) -> dict:
    """Check the status of a running or completed download."""
    if task_id not in _download_meta:
        raise HTTPException(status_code=404, detail="unknown task_id")

    # Close the log handle as soon as we notice the process has finished.
    _reap_finished()

    meta = _download_meta.get(task_id)
    proc = _downloads.get(task_id)

    if meta is None or proc is None:
        return {"status": "unknown", "returncode": None, "log_tail": ""}

    rc = proc.poll()
    if rc is None:
        status = "running"
    elif rc == 0:
        status = "completed"
    else:
        status = "failed"

    # Read last 50 lines of log
    log_path = Path(meta["log_path"])
    log_tail = ""
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        log_tail = "\n".join(lines[-50:])

    elapsed = time.time() - meta["started_at"]

    return {
        "status": status,
        "returncode": rc,
        "elapsed_seconds": round(elapsed, 1),
        "log_tail": log_tail,
    }
