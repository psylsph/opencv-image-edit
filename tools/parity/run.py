#!/usr/bin/env python3
"""Does the browser's live preview (web/gl.js, WebGL2) look like the server's pipeline?

    python3 tools/parity/run.py [case ...] [--keep DIR] [--chrome /path/to/chromium]

Each case renders the same synthetic photo through app/pipeline (grain.py, filters.py) and through gl.js in headless
Chromium (SwiftShader, so no GPU needed) and compares the pictures. Needs numpy, opencv and Pillow (see
requirements.txt) and a Chromium/Chrome binary. Not part of pytest: it needs a browser. Run it after changing gl.js or
the filter/grain maths. Exit status 1 when a case drifts more than its threshold.

Grain can't match per pixel (different random generators), so it is compared by statistics: how strongly the grain
changes the picture (std of the difference) and that it is darker-pixel-weighted like the server's.
"""

import argparse
import base64
import functools
import http.server
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    """Load app/pipeline/<name>.py by path: importing the package would pull in the whole server (onnxruntime...)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"parity_{name}", ROOT / "app" / "pipeline" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_filters, _grain = _load("filters"), _load("grain")
add_vignette, apply_color_adjustments = _filters.add_vignette, _filters.apply_color_adjustments
apply_grayscale, apply_sepia, apply_grain = (
    _filters.apply_grayscale,
    _filters.apply_sepia,
    _grain.apply_grain,
)

# mean absolute difference (0-255); 8-bit quantisation and blur edge handling account for ~1
THRESHOLD = 2.5
LOOSE = {
    "sharp_up": 3.5,
    "sharp_down": 3.5,
}  # blur edge handling (clamp vs reflect) shows at the borders

F = lambda **kw: {
    "filters": {
        "enabled": True,
        "brightness": 1,
        "contrast": 1,
        "saturation": 1,
        "sharpness": 1,
        "vignette_strength": 0,
        "sepia": False,
        "grayscale_blend": 0,
        **kw,
    }
}
CASES = {
    "identity": F(),
    "brightness_up": F(brightness=1.4),
    "brightness_down": F(brightness=0.5),
    "contrast_up": F(contrast=1.6),
    "contrast_down": F(contrast=0.4),
    "saturation_up": F(saturation=1.8),
    "saturation_down": F(saturation=0.3),
    "sharp_up": F(sharpness=1.8),
    "sharp_down": F(sharpness=0.3),
    "vignette": F(vignette_strength=0.6),
    "sepia": F(sepia=True),
    "gray": F(grayscale_blend=0.7),
    "combined": F(
        brightness=1.2,
        contrast=1.3,
        saturation=0.8,
        sharpness=1.4,
        vignette_strength=0.4,
        sepia=True,
    ),
    "grain": {"grain": {"enabled": True, "intensity": 0.6}},
}


def photo(w=640, h=480):
    """A synthetic photo: smooth colour gradients, a hard-edged circle and square, a dark and a bright patch."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack(
        [(x / w) * 220 + 20, (y / h) * 200 + 30, ((x + y) / (w + h)) * 180 + 40], -1
    )  # RGB
    cv2.circle(img, (w // 3, h // 2), 90, (230, 60, 60), -1)
    cv2.rectangle(img, (w // 2, h // 4), (w // 2 + 150, h // 4 + 120), (30, 200, 90), -1)
    img[10:70, 10:100] = 15
    img[h - 70 : h - 10, w - 110 : w - 10] = 240
    return np.clip(img, 0, 255).astype(np.uint8)  # RGB


def server_render(rgb, s):
    bgr = rgb[:, :, ::-1].copy()
    g = s.get("grain")
    if g and g["enabled"]:
        bgr, _ = apply_grain(bgr, g["intensity"], seed=1)
    f = s.get("filters")
    if f and f["enabled"]:
        bgr = apply_color_adjustments(
            bgr, f["brightness"], f["contrast"], f["saturation"], f["sharpness"]
        )
        if f["vignette_strength"] > 0:
            bgr = add_vignette(bgr, f["vignette_strength"])
        if f["sepia"]:
            bgr = apply_sepia(bgr)
        if f["grayscale_blend"] > 0:
            bgr = apply_grayscale(bgr, f["grayscale_blend"])
    return bgr[:, :, ::-1]


def browser_render(chrome, base, img_path, s):
    url = f"{base}/tools/parity/page.html?img={urllib.parse.quote(img_path)}&s={urllib.parse.quote(json.dumps(s))}"
    for attempt in range(3):  # headless Chromium sometimes dumps the DOM before the module has run
        profile = tempfile.mkdtemp(prefix="parity-chrome-")
        try:
            r = _run_chrome(chrome, profile, url)
        except subprocess.TimeoutExpired:  # headless Chromium occasionally never exits
            continue
        finally:
            shutil.rmtree(profile, ignore_errors=True)
        m = base64_png(r.stdout)
        if m is not None:
            return m
    raise RuntimeError("browser gave no picture after 3 tries")


def _run_chrome(chrome, profile, url):
    return subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=8000",
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )


def base64_png(dom):
    i = dom.find("data:image/png;base64,")
    if i < 0:
        return None
    j = dom.find("<", i)
    data = dom[i + len("data:image/png;base64,") : j].strip()
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def judge(name, want, got):
    d = np.abs(want.astype(np.float32) - got.astype(np.float32))
    mean = float(d.mean())
    if name == "grain":
        base = photo().astype(np.float32)
        sw, sg = float((want - base).std()), float((got - base).std())
        # grain is dark-weighted: the dark patch must move more than the bright one
        dark = float(np.abs(got[10:70, 10:100].astype(np.float32) - base[10:70, 10:100]).mean())
        bright = float(
            np.abs(got[-70:-10, -110:-10].astype(np.float32) - base[-70:-10, -110:-10]).mean()
        )
        ok = abs(sw - sg) / sw < 0.2 and dark > bright * 2
        return (
            ok,
            f"grain std server {sw:.2f} / browser {sg:.2f}, dark patch {dark:.2f} vs bright {bright:.2f}",
        )
    limit = LOOSE.get(name, THRESHOLD)
    return mean <= limit, f"mean diff {mean:.2f} (max {d.max():.0f}, limit {limit})"


def find_chrome(arg):
    for c in ([arg] if arg else []) + [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
    ]:
        if c and shutil.which(c):
            return shutil.which(c)
    sys.exit("no Chromium/Chrome found; pass --chrome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--keep")
    ap.add_argument("--chrome")
    a = ap.parse_args()
    chrome = find_chrome(a.chrome)
    names = a.cases or list(CASES)
    work = Path(tempfile.mkdtemp(prefix="parity-"))
    (work / "web").symlink_to(ROOT / "web")
    (work / "tools").symlink_to(ROOT / "tools")
    rgb = photo()
    Image.fromarray(rgb).save(work / "photo.png")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(work))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    keep = Path(a.keep) if a.keep else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    failed = 0
    for n in names:
        s = CASES[n]
        want = server_render(rgb, s)
        try:
            got = browser_render(chrome, base, "/photo.png", s)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {n}: {e}")
            failed += 1
            continue
        ok, msg = judge(n, want, got)
        print(f"{'ok  ' if ok else 'FAIL'} {n}: {msg}")
        failed += not ok
        if keep:
            diff = np.clip(np.abs(want.astype(int) - got.astype(int)) * 8, 0, 255).astype(np.uint8)
            Image.fromarray(np.hstack([want, got, diff])).save(keep / f"{n}.png")
    srv.shutdown()
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
