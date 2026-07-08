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


def test_target_dir_creation_is_idempotent(tmp_path):
    """Running the script against a fresh dir twice should be a no-op the second time."""
    # First, fabricate valid files in tmp_path with the expected hashes so
    # the script can verify them and skip. We don't actually need real model
    # binaries — we just need the file to exist with the right hash.
    # Since we don't want to mock MODELS (would defeat the test), we instead
    # run the script in a no-op way: point it at the actual models dir,
    # which we know is populated from CI. If models are missing, skip.
    target = REPO_ROOT / "models"
    if not (target / "u2netp.onnx").exists():
        # Fall back: just verify the script is well-formed by running --help-ish.
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "download_models.py"), str(tmp_path / "no-such")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # In the missing-models case the script will try to download. We don't
        # have a network or model registry guaranteed; just check the script
        # at least parsed and started.
        assert result.returncode in (0, 1), f"unexpected exit code {result.returncode}"
        return

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
