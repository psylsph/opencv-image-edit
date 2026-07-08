#!/usr/bin/env python3
"""Download ONNX and .pb model files for opencv-image-edit.

Downloads U2NetP (matting), EDSR (AI upscale), and MobileSAM (segmentation) to
a target directory. Idempotent — skips files that already exist and match the
expected SHA256.

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
import time
import urllib.error
import urllib.request
import zipfile
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
    "mobile_sam_20230629.zip": {
        "url": "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip",
        "sha256": "41aff2660b7531becfee21fb257c49933ddc892c554507bdb775bf504d443942",
        "size_mb": 35.0,
        "is_zip": True,
        # Only extract the encoder from this zip.
        "extract": {
            "mobile_sam.encoder.onnx": "mobile_sam.encoder.onnx",
        },
    },
    "mobile_sam_mask_decoder.onnx": {
        # NanoSAM MobileSAM mask decoder — pairs correctly with the
        # TinyViT encoder above. NOT the SAM-H decoder from the zip.
        "url": "https://huggingface.co/dragonSwing/nanosam/resolve/main/mobile_sam_mask_decoder.onnx",
        "sha256": "41e49a298099048186ce109a4518286b8972959898a02577414405efa5c3b247",
        "size_mb": 16.5,
    },
    "inpainting_lama_2025jan.onnx": {
        # LaMa (Large Mask Inpainting) — official OpenCV packaging.
        # Used for high-quality object removal (replaces TELEA/NS as default).
        "url": "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
        "sha256": "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2",
        "size_mb": 88.3,
    },
}


CHUNK_SIZE = 64 * 1024  # 64 KB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, max_retries: int = 10) -> None:
    """Stream-download url to dest with resume + retry support."""
    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    for attempt in range(1, max_retries + 1):
        # Resume from existing partial file
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "opencv-image-edit/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            print(f"  Resuming from {existing / 1024 / 1024:.1f} MB")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Get total size from headers
                if resp.status == 206:
                    # Partial content — total is existing + content range
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                    else:
                        total = existing + int(resp.headers.get("Content-Length", 0))
                else:
                    # Server didn't support Range — start fresh
                    if existing:
                        existing = 0
                    total = int(resp.headers.get("Content-Length", 0))

                downloaded = existing
                mode = "ab" if existing and resp.status == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with dest.open(mode) as f:
                    while True:
                        try:
                            chunk = resp.read(CHUNK_SIZE)
                        except TimeoutError:
                            print(f"\n  Read timeout, flushing and retrying... ({attempt}/{max_retries})")
                            f.flush()
                            break
                        if not chunk:
                            return  # done!
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and downloaded % (10 * 1024 * 1024) < CHUNK_SIZE:
                            pct = 100 * downloaded / total
                            mb = downloaded / 1024 / 1024
                            total_mb = total / 1024 / 1024
                            print(f"    {mb:6.1f} / {total_mb:6.1f} MB  ({pct:5.1f}%)")
                    else:
                        # Loop ended normally (chunk empty)
                        return
            # If we broke out of inner loop due to timeout, retry
            print(f"  Retry {attempt}/{max_retries}...")
            time.sleep(2 * attempt)
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as exc:
            print(f"\n  Error: {exc} — retry {attempt}/{max_retries}...")
            time.sleep(min(2 * attempt, 30))
    raise SystemExit(f"ERROR: failed to download {url} after {max_retries} retries")


def _verify(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    if expected_sha256 == "0" * 64:
        # TODO placeholder — accept whatever's there
        return True
    actual = _sha256(path)
    return actual == expected_sha256


def _process_zip_entry(name: str, meta: dict, target_dir: Path) -> None:
    """Handle an is_zip=True entry: download zip, verify, extract mapped files."""
    extract_map: dict[str, str] = meta["extract"]
    # All target files must exist with correct hash to consider done.
    all_present = True
    for target_name in extract_map.values():
        dest = target_dir / target_name
        if not dest.exists():
            all_present = False
            break
        if meta["sha256"] != "0" * 64:
            # We hash the zip itself, not the extracted files, for the entry-level check.
            # Per-file integrity is checked after extraction.
            pass
    # Quick exit: if all target files exist, do a full per-file hash check using
    # a sidecar .sha256 file we write next to each extracted file. We track hashes
    # in a small JSON sidecar so the registry can re-verify on subsequent runs.
    if all_present:
        sidecar = target_dir / f".{name}.sha256.json"
        if sidecar.exists():
            import json
            try:
                known = json.loads(sidecar.read_text())
                ok = True
                for src, tgt in extract_map.items():
                    expected = known.get(src)
                    if not expected:
                        ok = False
                        break
                    actual = _sha256(target_dir / tgt)
                    if actual != expected:
                        ok = False
                        break
                if ok:
                    print("  OK (already present and valid)")
                    return
            except Exception:
                # Corrupt sidecar — re-extract.
                pass
        else:
            # No sidecar; verify each extracted file's size > 0 to be safe,
            # then re-extract to refresh sidecar data. This avoids the silent
            # "trust existing" trap where a partial/corrupt extraction would
            # never be detected on subsequent runs.
            all_present = all(
                (target_dir / tgt_name).exists() and (target_dir / tgt_name).stat().st_size > 0
                for tgt_name in extract_map.values()
            )
            if not all_present:
                print("  Files missing or empty — re-extracting")
            else:
                print("  sidecar missing — re-extracting to refresh metadata")

    # Download zip to a temp file inside target_dir, then extract.
    tmp_zip = target_dir / f".{name}.tmp.zip"
    try:
        _download(meta["url"], tmp_zip)
        if meta["sha256"] != "0" * 64:
            actual = _sha256(tmp_zip)
            if actual != meta["sha256"]:
                tmp_zip.unlink()
                raise SystemExit(
                    f"ERROR: SHA256 mismatch for {name}: "
                    f"expected {meta['sha256']}, got {actual}"
                )
        # Extract mapped files
        import json
        per_file_hashes: dict[str, str] = {}
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            names_in_zip = set(zf.namelist())
            for src_name, tgt_name in extract_map.items():
                if src_name not in names_in_zip:
                    raise SystemExit(
                        f"ERROR: {name}: expected {src_name!r} in zip, got {sorted(names_in_zip)}"
                    )
                target_path = target_dir / tgt_name
                with zf.open(src_name) as src_f, target_path.open("wb") as out_f:
                    out_f.write(src_f.read())
                per_file_hashes[src_name] = _sha256(target_path)
                print(f"  extracted {src_name} -> {tgt_name}")
        # Write sidecar for future idempotent runs
        sidecar = target_dir / f".{name}.sha256.json"
        sidecar.write_text(json.dumps(per_file_hashes, indent=2, sort_keys=True))
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


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
        print(f"[{name}] expected ~{meta['size_mb']} MB")
        try:
            if meta.get("is_zip"):
                _process_zip_entry(name, meta, target_dir)
                print("  OK")
            else:
                dest = target_dir / name
                if _verify(dest, meta["sha256"]):
                    print("  OK (already present and valid)")
                    continue
                _download(meta["url"], dest)
                if meta["sha256"] != "0" * 64:
                    actual = _sha256(dest)
                    if actual != meta["sha256"]:
                        dest.unlink()
                        raise SystemExit(
                            f"ERROR: SHA256 mismatch for {name}: "
                            f"expected {meta['sha256']}, got {actual}"
                        )
                print("  OK")
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
    print("All base models downloaded successfully.")

    # --- SD models (optional, ~4GB) ---
    print()
    print("=" * 60)
    print("Stable Diffusion 1.5 Inpainting (optional, ~4GB)")
    print("=" * 60)
    download_sd_models(target_dir)

    return 0


# --- Stable Diffusion model downloads ---

_SD_REPO = "modularai/stable-diffusion-1.5-onnx"
_SD_FILES = {
    "text_encoder/model.onnx": ("text_encoder/model.onnx", 469),
    "unet/model.onnx": ("unet/model.onnx", 1),
    "unet/model.onnx_data": ("unet/model.onnx_data", 3278),
    "vae_encoder/model.onnx": ("vae_encoder/model.onnx", 130),
    "vae_decoder/model.onnx": ("vae_decoder/model.onnx", 188),
    "tokenizer/vocab.json": ("tokenizer/vocab.json", 1),
    "tokenizer/merges.txt": ("tokenizer/merges.txt", 1),
    "tokenizer/tokenizer_config.json": ("tokenizer/tokenizer_config.json", 1),
    "tokenizer/special_tokens_map.json": ("tokenizer/special_tokens_map.json", 1),
}


def download_sd_models(target_dir: Path) -> None:
    """Download SD 1.5 ONNX models (~4GB). Optional — LaMa is the default."""
    sd_dir = target_dir / "sd-inpainting"

    # Check if already downloaded
    unet_data = sd_dir / "unet" / "model.onnx_data"
    if unet_data.exists() and unet_data.stat().st_size > 1_000_000_000:
        print("  SD models already present.")
        return

    print(f"  Downloading to: {sd_dir}")
    print(f"  Source: {_SD_REPO}")
    print(f"  Total: ~4GB (this will take several minutes)")
    print()

    base_url = f"https://huggingface.co/{_SD_REPO}/resolve/main"
    for subdir_path, (repo_path, size_mb) in _SD_FILES.items():
        dest = sd_dir / subdir_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 100:
            print(f"  [{subdir_path}] already present ({dest.stat().st_size / 1024 / 1024:.0f} MB)")
            continue

        # If partial is <100 bytes, treat as missing and start fresh (avoid 416)
        if dest.exists():
            dest.unlink()

        print(f"  [{subdir_path}] downloading (~{size_mb} MB)...")
        _download(f"{base_url}/{repo_path}", dest)
        print(f"  [{subdir_path}] OK")

    print()
    print("  SD models downloaded successfully.")


if __name__ == "__main__":
    sys.exit(main())
