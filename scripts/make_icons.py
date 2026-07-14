"""Generate PWA icons for OpenCV Image Editor.

Creates two PNGs:
  - web/icons/icon-192.png
  - web/icons/icon-512.png

Design: dark rounded square, accent-blue background, "CV" wordmark in the foreground color.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/home/stuart/hermes/opencv-image-edit/web/icons"
os.makedirs(OUT_DIR, exist_ok=True)

BG = (15, 15, 18)  # --bg
ACCENT = (110, 168, 254)  # --accent
FG = (10, 10, 12)  # --accent-fg

# Try a few common font locations, fall back to default
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def find_font() -> str | None:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def make_icon(size: int) -> None:
    img = Image.new("RGB", (size, size), color=BG)
    draw = ImageDraw.Draw(img)

    # Rounded square in accent color
    margin = size // 8
    radius = size // 6
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=ACCENT,
    )

    # "CV" wordmark
    font_path = find_font()
    if font_path:
        font = ImageFont.truetype(font_path, size=int(size * 0.42))
    else:
        font = ImageFont.load_default()

    text = "CV"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1] - int(size * 0.02)  # optical centering
    draw.text((x, y), text, font=font, fill=FG)

    out_path = os.path.join(OUT_DIR, f"icon-{size}.png")
    img.save(out_path, "PNG", optimize=True)
    print(f"Saved {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    for size in (192, 512):
        make_icon(size)
