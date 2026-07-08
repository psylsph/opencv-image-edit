"""OpenAI Images API wrapper for generative inpainting.

Uses the OpenAI Images API (/v1/images/edits) to fill masked regions
with AI-generated content based on a text prompt.

Advantages over local SD:
- No 4GB model download needed
- Much faster (~3-10s vs 30-120s on CPU)
- Higher quality generative fill
- Requires internet + API key

The OpenAI mask format: PNG with alpha channel where fully transparent
(alpha=0) pixels indicate areas to edit. We convert our single-channel
uint8 mask (non-zero = remove) to this format.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

# Max image size OpenAI accepts (25MB for gpt-image-1)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Max dimension to keep the PNG payload reasonable
_MAX_DIM = 1536
# Request timeout (seconds)
_TIMEOUT = 120


class OpenAIInpaint:
    """Wrapper around the OpenAI Images API for generative inpainting."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model
        self._base_url = settings.openai_base_url.rstrip("/")
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key not configured. Set OPENAI_API_KEY env var."
            )

    @staticmethod
    def is_available() -> bool:
        """Check if OpenAI inpainting is available (API key configured)."""
        return bool(get_settings().openai_api_key)

    def inpaint(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        prompt: str = "",
    ) -> np.ndarray:
        """Fill masked regions using OpenAI's generative image edit API.

        Args:
            img_bgr: BGR uint8 image (H, W, 3).
            mask: uint8 single-channel mask, non-zero = areas to edit.
            prompt: Text describing what to generate in masked areas.

        Returns:
            BGR uint8 image of the same shape as img_bgr.
        """
        if not prompt.strip():
            prompt = "Fill with surrounding content, seamless and natural."

        # Downscale if too large (OpenAI will resize anyway, but we save bandwidth)
        h, w = img_bgr.shape[:2]
        if max(h, w) > _MAX_DIM:
            scale = _MAX_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = img_bgr.shape[:2]

        # Prepare image PNG (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ok, img_png = cv2.imencode(".png", img_rgb)
        if not ok:
            raise RuntimeError("Failed to encode image to PNG")
        img_bytes = img_png.tobytes()

        # Prepare mask PNG with alpha channel
        # OpenAI expects: transparent (alpha=0) = edit this area
        # Our mask: non-zero = remove/edit
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        edit_area = mask > 0
        # Keep areas (mask==0): opaque with any RGB (OpenAI ignores RGB of transparent)
        mask_rgba[~edit_area, 3] = 255  # opaque
        mask_rgba[edit_area, 3] = 0     # transparent
        ok, mask_png = cv2.imencode(".png", mask_rgba)
        if not ok:
            raise RuntimeError("Failed to encode mask to PNG")
        mask_bytes = mask_png.tobytes()

        # Build multipart form body
        boundary = "----opencv-image-edit-boundary"
        body = self._build_multipart(
            boundary,
            image_bytes=img_bytes,
            mask_bytes=mask_bytes,
            prompt=prompt,
            model=self._model,
            size="auto",
        )

        # Send request
        url = f"{self._base_url}/images/edits"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        logger.info(
            "OpenAI inpaint: img=%dx%d mask_coverage=%.1f%% model=%s prompt=%r",
            w, h, float(edit_area.mean()) * 100, self._model,
            prompt[:80],
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            logger.error("OpenAI API error %d: %s", exc.code, error_body[:500])
            raise RuntimeError(
                f"OpenAI API returned {exc.code}: {error_body[:200]}"
            ) from exc

        # Parse response — gpt-image-1 returns b64_json, dall-e-2 may return url
        images = response_data.get("data", [])
        if not images:
            raise RuntimeError("OpenAI returned no images")

        if "b64_json" in images[0]:
            import base64
            result_bytes = base64.b64decode(images[0]["b64_json"])
        elif "url" in images[0]:
            # Download the URL
            img_url = images[0]["url"]
            with urllib.request.urlopen(img_url, timeout=_TIMEOUT) as img_resp:
                result_bytes = img_resp.read()
        else:
            raise RuntimeError("OpenAI response missing b64_json and url")

        # Decode result to BGR
        result_arr = np.frombuffer(result_bytes, dtype=np.uint8)
        result_bgr = cv2.imdecode(result_arr, cv2.IMREAD_COLOR)
        if result_bgr is None:
            raise RuntimeError("Failed to decode OpenAI result image")

        # The result may be a different size than the original (OpenAI returns
        # square or specified dimensions). Resize back to the original crop size.
        if result_bgr.shape[:2] != (h, w):
            result_bgr = cv2.resize(
                result_bgr, (w, h), interpolation=cv2.INTER_LINEAR
            )

        return result_bgr

    @staticmethod
    def _build_multipart(
        boundary: str,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        model: str,
        size: str = "auto",
    ) -> bytes:
        """Build a multipart/form-data body for the OpenAI images/edits API."""
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode("utf-8")
            )

        def add_file(name: str, filename: str, content: bytes) -> None:
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: image/png\r\n\r\n".encode("utf-8")
            )
            parts.append(content)
            parts.append(b"\r\n")

        add_field("model", model)
        add_field("prompt", prompt)
        add_field("size", size)
        add_field("n", "1")
        add_file("image", "image.png", image_bytes)
        add_file("mask", "mask.png", mask_bytes)
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(parts)
