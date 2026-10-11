"""Tests for the painting style-transfer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.models.painting import (
    PAINT_STYLES,
    ModelNotFoundError,
    apply_paint_style,
)


def _fake_session_factory():
    """Return a stub ORT session that returns the input tile unchanged."""

    class FakeInput:
        name = "input1"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _, feeds):
            x = list(feeds.values())[0]
            return [x]

    return FakeSession()


def test_registry_has_three_styles():
    assert sorted(PAINT_STYLES) == ["oil", "painterly", "pointillism"]


def test_unknown_style_raises():
    img = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ModelNotFoundError):
        apply_paint_style(img, "van_gogh")


def test_small_image_padded_and_cropped():
    img = np.full((100, 60, 3), 200, np.uint8)
    with patch(
        "app.models.painting.get_paint_session", side_effect=lambda _: _fake_session_factory()
    ):
        out = apply_paint_style(img, "oil")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # identity stub: content preserved
    assert np.array_equal(out, img)


def test_large_image_tiled_same_shape():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    with patch(
        "app.models.painting.get_paint_session", side_effect=lambda _: _fake_session_factory()
    ):
        out = apply_paint_style(img, "painterly")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # identity stub with feathered blending stays close to the original
    diff = np.abs(out.astype(int) - img.astype(int)).mean()
    assert diff < 2.0


def test_grayscale_input_promoted():
    img = np.full((50, 50), 128, np.uint8)
    with patch(
        "app.models.painting.get_paint_session", side_effect=lambda _: _fake_session_factory()
    ):
        out = apply_paint_style(img, "pointillism")
    assert out.ndim == 3 and out.shape[:2] == img.shape
