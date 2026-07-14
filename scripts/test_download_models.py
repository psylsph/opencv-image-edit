"""Tests for scripts/download_models.py.

Validates the model registry and the idempotency contract of the download
script. Does NOT perform real network downloads — that is verified manually
during the build pipeline. These tests are safe to run in any environment.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Make scripts/ importable so we can import the download_models module directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from download_models import MODELS, _sha256  # noqa: E402

EXPECTED_KEYS = {"u2netp.onnx", "EDSR_x2.pb", "EDSR_x4.pb", "mobile_sam_20230629.zip"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = "0" * 64


def test_models_dict_has_all_entries():
    assert EXPECTED_KEYS.issubset(MODELS.keys()), (
        f"Missing keys: {EXPECTED_KEYS - set(MODELS.keys())}"
    )
    for name in EXPECTED_KEYS:
        meta = MODELS[name]
        assert "url" in meta, f"{name} missing 'url'"
        assert "sha256" in meta, f"{name} missing 'sha256'"
        assert "size_mb" in meta, f"{name} missing 'size_mb'"
        assert meta["url"].startswith("https://"), f"{name} url must be https"


def test_sha256_format_valid():
    """Every pinned sha256 must be 64 lowercase hex chars (or the 0*64 placeholder)."""
    for name, meta in MODELS.items():
        sha = meta["sha256"]
        assert SHA256_RE.match(sha), f"{name}: sha256 '{sha}' is not 64 lowercase hex chars"
        if sha == PLACEHOLDER:
            # Allow the placeholder while we're filling in the registry, but log it.
            print(f"  note: {name} still has placeholder hash — pin real hash before release")


def test_no_duplicate_urls():
    urls = [m["url"] for m in MODELS.values()]
    assert len(urls) == len(set(urls)), f"Duplicate URLs found: {urls}"


def test_verify_idempotent_in_process(tmp_path, monkeypatch):
    """_verify is a no-op for a file whose hash matches the registry.

    This exercises the idempotency contract directly, with no subprocess and
    no network access, so it runs identically in any environment.
    """
    import download_models as dm

    content = b"deterministic bytes for a fake model file"
    name = "fake_model.onnx"
    dest = tmp_path / name
    dest.write_bytes(content)
    sha = dm._sha256(dest)

    monkeypatch.setattr(
        dm,
        "MODELS",
        {name: {"url": "https://example.invalid/x", "sha256": sha, "size_mb": 1.0}},
    )

    assert dm._verify(dest, sha) is True
    before = dest.stat().st_mtime_ns
    assert dm._verify(dest, sha) is True  # second check is a no-op
    assert dest.stat().st_mtime_ns == before


def test_target_dir_creation_is_idempotent(tmp_path):
    """Running the script against a populated dir twice is a no-op the 2nd time.

    Requires the real base models to be present (populated CI / a dev box that
    ran ``scripts/download_models.py``). Skipped otherwise — the offline
    idempotency primitive is covered by ``test_verify_idempotent_in_process``.
    """
    target = REPO_ROOT / "models"
    if not (target / "u2netp.onnx").exists():
        pytest.skip("base models not present; idempotency verified in-process instead")

    # Real run: against the populated dir, must be idempotent.
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "download_models.py"), str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"download_models failed: {result.stderr}"
    assert "OK (already present and valid)" in result.stdout, (
        f"expected idempotent skip, got:\n{result.stdout}"
    )


def test_sha256_helper_round_trip(tmp_path):
    """_sha256 should produce the same hex string as a re-read of the same file."""
    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello world")
    h1 = _sha256(f)
    h2 = _sha256(f)
    assert h1 == h2
    assert SHA256_RE.match(h1)
    # Known SHA256 of "hello world"
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
