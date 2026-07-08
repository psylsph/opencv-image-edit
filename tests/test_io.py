"""Tests for app.pipeline.io — image decode/encode helpers (HEIC + standard)."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from app.exceptions import DecodeError
from app.pipeline.io import (
    decode_to_bgr,
    encode_format,
    encode_jpeg,
    encode_png,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_bgr() -> np.ndarray:
    """A 4x3 BGR image with a known red pixel at (0, 0)."""
    # BGR order: pixel (0, 0) = (B=10, G=20, R=30) i.e. RGB(30, 20, 10)
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    img[0, 0] = (10, 20, 30)
    img[1, 1] = (200, 100, 50)  # RGB(50, 100, 200)
    return img


@pytest.fixture
def small_bgra() -> np.ndarray:
    """A 4x3 BGRA image with a known alpha value."""
    img = np.zeros((3, 4, 4), dtype=np.uint8)
    img[0, 0] = (10, 20, 30, 128)
    img[1, 1] = (200, 100, 50, 255)
    return img


# ---------------------------------------------------------------------------
# decode_to_bgr
# ---------------------------------------------------------------------------


def test_decode_png_returns_bgr(small_bgr: np.ndarray) -> None:
    """PNG bytes -> BGR ndarray with matching pixel at (0, 0)."""
    ok, buf = cv2.imencode(".png", small_bgr)
    assert ok, "fixture: cv2.imencode PNG failed"

    result = decode_to_bgr(bytes(buf))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3  # BGR
    assert result.shape[:2] == small_bgr.shape[:2]
    # Pixel (0, 0) BGR should match exactly: B=10, G=20, R=30
    np.testing.assert_array_equal(result[0, 0], np.array([10, 20, 30], dtype=np.uint8))


def test_decode_jpeg_returns_bgr() -> None:
    """JPEG bytes -> BGR ndarray, dtype/shape correct, BGR order verified.

    JPEG is lossy so we don't assert exact pixel equality — we use a
    large image with distinct B/G/R values and verify the channel order
    (blue should be 0 in a pure-green image, red should be 0 in a
    pure-blue image, etc.).
    """
    # 64x64 solid blue image: B=200, G=50, R=30
    blue = np.zeros((64, 64, 3), dtype=np.uint8)
    blue[:] = (200, 50, 30)  # BGR
    ok, buf = cv2.imencode(
        ".jpg", blue, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    assert ok

    result = decode_to_bgr(bytes(buf))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape[2] == 3
    assert result.shape[:2] == blue.shape[:2]
    # BGR order check: blue channel (index 0) must be the largest
    # because we encoded a blue image. If OpenCV returned RGB by mistake
    # the red channel (index 2) would be the largest, not the blue.
    sample = result[32, 32]
    assert int(sample[0]) > int(sample[1])  # B > G
    assert int(sample[0]) > int(sample[2])  # B > R
    # Sanity: image isn't all zero
    assert result.max() > 0


def test_decode_rejects_invalid_bytes() -> None:
    """Garbage bytes -> DecodeError."""
    with pytest.raises(DecodeError):
        decode_to_bgr(b"not an image")


def test_decode_rejects_empty() -> None:
    """Empty bytes -> DecodeError."""
    with pytest.raises(DecodeError):
        decode_to_bgr(b"")


# ---------------------------------------------------------------------------
# encode_png
# ---------------------------------------------------------------------------


def test_encode_png_roundtrip(small_bgr: np.ndarray) -> None:
    """Encode then decode a BGR image — shapes match exactly."""
    encoded = encode_png(small_bgr)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0

    decoded = decode_to_bgr(encoded)
    assert decoded.shape == small_bgr.shape
    np.testing.assert_array_equal(decoded, small_bgr)


# ---------------------------------------------------------------------------
# encode_jpeg
# ---------------------------------------------------------------------------


def test_encode_jpeg_quality_param(small_bgr: np.ndarray) -> None:
    """Higher quality -> larger file."""
    low = encode_jpeg(small_bgr, quality=10)
    high = encode_jpeg(small_bgr, quality=95)
    assert isinstance(low, bytes) and len(low) > 0
    assert isinstance(high, bytes) and len(high) > 0
    assert len(low) < len(high), (
        f"q=10 ({len(low)} bytes) should be smaller than q=95 ({len(high)} bytes)"
    )


# ---------------------------------------------------------------------------
# encode_png BGRA
# ---------------------------------------------------------------------------


def test_encode_bgra_handles_alpha(small_bgra: np.ndarray) -> None:
    """4-channel BGRA encode->decode preserves alpha (decoded via PIL)."""
    encoded = encode_png(small_bgra)
    assert isinstance(encoded, bytes)

    # PNG preserves alpha; decode via PIL and compare.
    pil = PILImage.open(BytesIO(encoded))
    assert pil.mode == "RGBA"
    arr = np.array(pil)  # RGB-A

    # Pixel (0, 0): source was BGR(10,20,30), A=128 -> RGB(30,20,10), A=128
    assert tuple(int(v) for v in arr[0, 0]) == (30, 20, 10, 128)
    # Pixel (1, 1): BGR(200,100,50), A=255 -> RGB(50,100,200), A=255
    assert tuple(int(v) for v in arr[1, 1]) == (50, 100, 200, 255)


# ---------------------------------------------------------------------------
# encode_format / generic helpers
# ---------------------------------------------------------------------------


def test_encode_to_bytes_returns_bytes(small_bgr: np.ndarray) -> None:
    """encode_format / encode_png return bytes with length > 0."""
    out_png = encode_png(small_bgr)
    out_jpg = encode_jpeg(small_bgr)
    out_webp = encode_format(small_bgr, ".webp")

    for out in (out_png, out_jpg, out_webp):
        assert isinstance(out, bytes)
        assert len(out) > 0
