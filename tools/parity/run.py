#!/usr/bin/env python3
"""Does the browser's live preview (web/gl.js, WebGL2) look like the server's pipeline?

    python3 tools/parity/run.py [case ...] [--keep DIR] [--chrome /path/to/chromium]

Each case renders the same synthetic photo through app/pipeline (background.py, grain.py, filters.py) and through gl.js in headless
Chromium (SwiftShader, so no GPU needed) and compares the pictures. Needs numpy, opencv, Pillow and websockets and a
Chromium/Chrome binary. Not part of pytest: it needs a browser. Run it after changing gl.js or
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
import time
import urllib.request
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


_filters, _grain, _bg = _load("filters"), _load("grain"), _load("background")
add_vignette, apply_color_adjustments = _filters.add_vignette, _filters.apply_color_adjustments
apply_background_blur, apply_background_replace = (
    _bg.apply_background_blur,
    _bg.apply_background_replace,
)
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
    "bg_blur_small": {"background": {"enabled": True, "mode": "blur", "blur_strength": 4}},
    "bg_blur_large": {"background": {"enabled": True, "mode": "blur", "blur_strength": 25}},
    "bg_blur_plus_filters": {
        "background": {"enabled": True, "mode": "blur", "blur_strength": 12},
        "filters": F(brightness=1.2, saturation=1.3)["filters"],
    },
}


def _bgcase(**kw):
    return {"background": {"enabled": True, "mode": "image", **kw}}


CASES.update(
    {
        "bg_color": _bgcase(mode="color", color="#2040c0"),
        "bg_color_shrink_out": _bgcase(mode="color", color="#2040c0", shrink=-0.6),
        "bg_color_defringe": _bgcase(mode="color", color="#2040c0", defringe=1.0),
        "bg_image_defringe_feather": _bgcase(defringe=0.7, feather=0.3, shrink=-0.3),
        "bg_image": _bgcase(),
        "bg_image_zoom_pan": _bgcase(zoom=2.0, offset_x=-0.5, offset_y=0.3),
        "bg_image_feather_shrink": _bgcase(feather=0.5, shrink=0.4),
        "bg_image_and_filters": {
            **_bgcase(feather=0.3),
            "filters": F(contrast=1.3, sepia=True)["filters"],
        },
    }
)
LOOSE.update(
    {
        "bg_blur_small": 3.0,
        "bg_blur_large": 4.0,
        "bg_blur_plus_filters": 4.0,
        "bg_image_and_filters": 4.0,
    }
)


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


def backdrop(w=800, h=300):
    """A wide synthetic backdrop (different aspect from the photo, so the cover fit crops it)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack(
        [255 - x / w * 200, (np.sin(x / 40) * 0.5 + 0.5) * 200 + 30, y / h * 220 + 20], -1
    )
    cv2.rectangle(img, (w // 2 - 60, 40), (w // 2 + 60, 140), (250, 240, 20), -1)
    cv2.circle(img, (120, 220), 50, (20, 20, 20), -1)
    return np.clip(img, 0, 255).astype(np.uint8)  # RGB


def matte(w=640, h=480):
    """A soft-edged 8-bit matte (what /api/v1/matte returns): a disc around the red circle."""
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, (w // 3, h // 2), 100, 1.0, -1)
    return (cv2.GaussianBlur(m, (0, 0), 6) * 255 + 0.5).astype(np.uint8)


def server_render(rgb, s):
    bgr = rgb[:, :, ::-1].copy()
    b = s.get("background")
    if b and b["enabled"]:
        m = matte().astype(np.float32) / 255.0
        if b["mode"] in ("color", "image"):
            bgr = apply_background_replace(
                bgr,
                m,
                color=b["color"] if b["mode"] == "color" else None,
                background=backdrop()[:, :, ::-1].copy() if b["mode"] == "image" else None,
                zoom=b.get("zoom", 1.0),
                offset_x=b.get("offset_x", 0.0),
                offset_y=b.get("offset_y", 0.0),
                feather=b.get("feather", 0.0),
                shrink=b.get("shrink", 0.0),
                defringe=b.get("defringe", 0.0),
            )
        else:
            bgr = apply_background_blur(bgr, m, b["blur_strength"])
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


def browser_render_all(chrome, url, timeout=180):
    """Open the parity page in headless Chromium (SwiftShader), wait for it to finish and return its JSON."""
    from websockets.sync.client import connect

    port = 9445
    profile = tempfile.mkdtemp(prefix="parity-chrome-")
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={port}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        tabs = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        if not tabs:
            raise RuntimeError("Chromium did not start")
        page = next(t for t in tabs if t["type"] == "page")
        with connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            n = 0

            def call(method, **params):
                nonlocal n
                n += 1
                ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", msg)

            call("Page.navigate", url=url)
            deadline = time.time() + timeout
            while time.time() < deadline:
                time.sleep(0.5)
                v = call(
                    "Runtime.evaluate",
                    expression="document.getElementById('out') && document.getElementById('out').textContent",
                    returnByValue=True,
                )["result"].get("value")
                if v and v != "pending":
                    data = json.loads(v)
                    if "__error" in data:
                        raise RuntimeError(data["__error"])
                    return data
            raise RuntimeError("the parity page did not finish in time")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def decode_png(data_url):
    if not data_url.startswith("data:image/png;base64,"):
        raise RuntimeError(data_url[:200])
    return np.array(
        Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
    )


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
    Image.fromarray(matte()).save(work / "matte.png")
    Image.fromarray(backdrop()).save(work / "backdrop.png")

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
    cases = {}
    for n in names:
        bgs = CASES[n].get("background") or {}
        cases[n] = {
            "settings": CASES[n],
            "mask": "/matte.png" if bgs else None,
            "bg": "/backdrop.png" if bgs.get("mode") == "image" else None,
        }
    (work / "cases.json").write_text(json.dumps(cases))
    failed = 0
    try:
        results = browser_render_all(
            chrome, f"{base}/tools/parity/page.html?img=/photo.png&cases=/cases.json"
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {e}")
        results = {}
        failed = len(names)
    for n in names if results else []:
        want = server_render(rgb, CASES[n])
        try:
            got = decode_png(results[n])
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
