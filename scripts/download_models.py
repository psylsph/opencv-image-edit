#!/usr/bin/env python3
"""Download ONNX and .pb model files for opencv-image-edit.

Downloads U2NetP (matting) and EDSR (AI upscale) to a target directory.
Idempotent — skips files that already exist and match the expected SHA256.

Usage:
    python scripts/download_models.py [target_dir]
    python scripts/download_models.py /path/to/models

Environment:
    MODEL_DIR — default target directory (defaults to ./models)
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


# Pinned model URLs and expected SHA256 hashes.
# Hashes are filled in after the first successful download.
MODELS: dict[str, dict] = {
    "u2netp.onnx": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        "size_mb": 4.7,
    },
    "EDSR_x2.pb": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
        "sha256": "585623221baa070279a0d1e7e113a4c3faba0f318ca7fdd9a65d9afc0763d9b4",
        "size_mb": 38.5,
    },
    "EDSR_x4.pb": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "sha256": "dd35ce3cae53ecee2d16045e08a932c3e7242d641bb65cb971d123e06904347f",
        "size_mb": 38.5,
    },
}


CHUNK_SIZE = 64 * 1024  # 64 KB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream-download url to dest, printing progress."""
    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "opencv-image-edit/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with dest.open("wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(f"    {mb:6.1f} / {total_mb:6.1f} MB  ({pct:5.1f}%)", end="\r")
            print()  # newline after progress
    except urllib.error.URLError as exc:
        raise SystemExit(f"ERROR: failed to download {url}: {exc}") from exc


def _verify(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    if expected_sha256 == "0" * 64:
        # TODO placeholder — accept whatever's there
        return True
    actual = _sha256(path)
    return actual == expected_sha256


def main() -> int:
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        target_dir = Path(os.environ.get("MODEL_DIR", "./models")).expanduser().resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")
    print()

    failed = []
    for name, meta in MODELS.items():
        dest = target_dir / name
        print(f"[{name}] expected ~{meta['size_mb']} MB")
        if _verify(dest, meta["sha256"]):
            print(f"  OK (already present and valid)")
            print()
            continue
        try:
            _download(meta["url"], dest)
            if meta["sha256"] != "0" * 64:
                actual = _sha256(dest)
                if actual != meta["sha256"]:
                    dest.unlink()
                    raise SystemExit(
                        f"ERROR: SHA256 mismatch for {name}: "
                        f"expected {meta['sha256']}, got {actual}"
                    )
            print(f"  OK")
        except SystemExit:
            raise
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  FAILED: {exc}")
        print()

    if failed:
        print(f"ERROR: {len(failed)} model(s) failed to download:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print("All models downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
