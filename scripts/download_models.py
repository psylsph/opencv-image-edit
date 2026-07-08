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
    print("All models downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
