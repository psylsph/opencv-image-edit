"""Tests for the SD model management endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import create_app
    return TestClient(create_app())


def test_sd_status_when_available(client):
    """SD status reports available=True when model files exist."""
    # SD models are already downloaded in models/sd-inpainting/
    with patch("app.config.get_settings") as mock_settings:
        mock_settings.return_value.model_dir = Path("models")
        resp = client.get("/api/v1/sd/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["size_mb"] > 0


def test_sd_status_when_not_available(client, tmp_path):
    """SD status reports available=False when model files don't exist."""
    with patch("app.models.sd_inpaint.SDInpaint.is_available", return_value=False):
        with patch("app.api.sd._sd_dir", return_value=tmp_path / "sd-inpainting"):
            resp = client.get("/api/v1/sd/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["size_mb"] == 0


def test_download_status_unknown_task(client):
    """Unknown task_id returns 404."""
    resp = client.get("/api/v1/sd/download/nonexistent/status")
    assert resp.status_code == 404


def test_download_starts_and_returns_task_id(client):
    """POST /download returns started=True with a task_id."""
    import app.api.sd as sd_module

    # Mock subprocess.Popen to capture without actually running
    class FakeProc:
        def __init__(self, *a, **kw):
            self.returncode = None
        def poll(self):
            return None

    with patch.object(sd_module.subprocess, "Popen", FakeProc):
        resp = client.post("/api/v1/sd/download")

    assert resp.status_code == 200
    data = resp.json()
    assert data["started"] is True
    assert "task_id" in data
    assert len(data["task_id"]) > 0

    # Clean up the tracked download
    task_id = data["task_id"]
    sd_module._downloads.pop(task_id, None)
    sd_module._download_meta.pop(task_id, None)
