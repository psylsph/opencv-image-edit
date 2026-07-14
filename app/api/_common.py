"""Shared helpers for the API routers."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

from app import config


async def read_image_bytes(file: UploadFile) -> bytes:
    """Read an uploaded file, enforcing the configured ``MAX_IMAGE_SIZE_MB`` limit.

    All file-upload endpoints go through this so the size cap is applied
    uniformly (previously only ``/api/v1/process`` checked it).

    Looks up settings via ``app.config.get_settings()`` at call time (not via
    a top-level ``from ... import get_settings``) so tests that patch
    ``app.config.get_settings`` propagate to every endpoint uniformly.

    Raises:
        HTTPException(413): if the payload exceeds ``max_image_size_mb``.
    """
    body = await file.read()
    settings = config.get_settings()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(f"file too large: {len(body)} bytes > {settings.max_image_size_mb} MB"),
        )
    return body
