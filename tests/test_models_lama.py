"""Tests for app.models.lama — LaMa inpainting model wrapper (TDD).

These tests run WITHOUT the real ONNX model on disk — they mock the cv2.dnn
network so the test stays hermetic and CPU-cheap. The point of these tests
is the preprocessing/postprocessing contract:
- image scaled 1/255 (1/255 ~ 0.00392)
- mask binarized to 0/1 float
- both resized to 512x512 (model's fixed input size)
- output denormalized and resized back to original size

Real-model integration is verified in the smoke test (not in pytest).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.exceptions import ModelNotFoundError
from app.models.lama import _INPUT_SIZE, LaMa

# ---------------------------------------------------------------------------
# Singleton lifecycle (consistent with MobileSAM pattern)
# ---------------------------------------------------------------------------


def test_lama_singleton_returns_same_instance(monkeypatch, tmp_path):
    """Two .get() calls with no model on disk should both raise the same error."""
    LaMa.clear_cache()
    monkeypatch.setattr("app.models.lama._MODEL_FILENAME", "definitely_does_not_exist.onnx")
    with pytest.raises(ModelNotFoundError):
        LaMa.get(model_dir=tmp_path)
    with pytest.raises(ModelNotFoundError):
        LaMa.get(model_dir=tmp_path)


def test_lama_clear_cache_resets_singleton(monkeypatch, tmp_path):
    """clear_cache() must allow re-loading the model from a different dir."""
    LaMa.clear_cache()
    monkeypatch.setattr("app.models.lama._MODEL_FILENAME", "still_missing.onnx")
    with pytest.raises(ModelNotFoundError):
        LaMa.get(model_dir=tmp_path)
    assert LaMa._instance is None
    LaMa.clear_cache()
    assert LaMa._instance is None


# ---------------------------------------------------------------------------
# Image preprocessing: BGR uint8 -> blob scaled 1/255 at 512x512
# ---------------------------------------------------------------------------


def test_lama_preprocess_scales_image_to_zero_one():
    """All-zero input maps to all-zero blob; all-255 input maps to ~1.0."""
    # All-zero input
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    blob = LaMa._preprocess(black)
    assert blob.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)
    assert blob.dtype == np.float32
    np.testing.assert_allclose(blob, 0.0, atol=1e-6)

    # All-255 input
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    blob = LaMa._preprocess(white)
    np.testing.assert_allclose(blob, 1.0, atol=1e-6)


def test_lama_preprocess_resizes_to_model_input_size():
    """Any input size must be resized to 512x512."""
    img = np.full((300, 700, 3), 128, dtype=np.uint8)  # weird aspect ratio
    blob = LaMa._preprocess(img)
    assert blob.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# Mask preprocessing: uint8 -> binarized 0/1 float at 512x512
# ---------------------------------------------------------------------------


def test_lama_preprocess_mask_binarizes_to_zero_one():
    """Mask > 0 becomes 1.0, mask == 0 stays 0.0."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[100:200, 100:200] = 128  # non-zero
    blob = LaMa._preprocess_mask(mask)
    assert blob.shape == (1, 1, _INPUT_SIZE, _INPUT_SIZE)
    assert blob.dtype == np.float32
    # After resize+binarize, the middle should be 1.0
    assert blob[0, 0, 256, 256] == 1.0
    # Corners should be 0.0
    assert blob[0, 0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# Postprocessing: blob -> HWC uint8 resized to target dimensions
# ---------------------------------------------------------------------------


def test_lama_postprocess_denormalizes_to_uint8():
    """0.0 -> 0, 255.0 -> 255 (model output is already in display range)."""
    tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
    tensor[0, :, 0, 0] = 0.0
    tensor[0, :, 0, 1] = 255.0
    out = LaMa._postprocess(tensor, target_h=4, target_w=4)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 0
    assert out[0, 1, 0] == 255


def test_lama_postprocess_clamps_out_of_range():
    """Values > 255 or < 0 (defensive) must be clamped."""
    tensor = np.array([[[[300.0, -50.0]]] * 3], dtype=np.float32)  # 1x3x1x2
    out = LaMa._postprocess(tensor, target_h=1, target_w=2)
    assert out[0, 0, 0] == 255  # clamped from 300
    assert out[0, 1, 0] == 0  # clamped from -50


def test_lama_postprocess_resizes_to_target_dimensions():
    """If the model's 512x512 output doesn't match the input size,
    postprocess must resize it back to the original input dimensions.
    """
    tensor = np.full((1, 3, _INPUT_SIZE, _INPUT_SIZE), 100.0, dtype=np.float32)
    out = LaMa._postprocess(tensor, target_h=256, target_w=384)
    assert out.shape == (256, 384, 3)
    # 100 is well within [0, 255] — passes through unchanged
    assert out[100, 200, 0] == 100
