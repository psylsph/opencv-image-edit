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
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running download processes: task_id -> Popen
_downloads: dict[str, subprocess.Popen] = {}
# Track task metadata: task_id -> {log_path, started_at}
_download_meta: dict[str, dict] = {}


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
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    _downloads[task_id] = proc
    _download_meta[task_id] = {"log_path": str(log_path), "started_at": time.time()}

    return {"started": True, "task_id": task_id}


@router.get("/api/v1/sd/download/{task_id}/status")
def sd_download_status(task_id: str) -> dict:
    """Check the status of a running or completed download."""
    if task_id not in _download_meta:
        raise HTTPException(status_code=404, detail="unknown task_id")

    meta = _download_meta[task_id]
    proc = _downloads.get(task_id)

    if proc is None:
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
