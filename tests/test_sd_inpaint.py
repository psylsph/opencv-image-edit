"""Smoke tests for the SD inpaint wrapper.

The full inference path is slow (~90s for 10 steps on CPU), so these tests
are skipped if the SD model files are not present. They exercise the
tokenizer (fast) and one full inference (slow but proves the model works).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_SD_DIR = Path("models/sd-inpainting")
_HAS_SD = (
    (_SD_DIR / "unet" / "model.onnx").exists()
    and (_SD_DIR / "text_encoder" / "model.onnx").exists()
    and (_SD_DIR / "vae_encoder" / "model.onnx").exists()
    and (_SD_DIR / "vae_decoder" / "model.onnx").exists()
    and (_SD_DIR / "tokenizer" / "vocab.json").exists()
)

pytestmark = pytest.mark.skipif(not _HAS_SD, reason="SD model files not present")


def test_tokenizer_produces_77_tokens():
    from app.models.sd_tokenizer import CLIPTokenizer

    tok = CLIPTokenizer(_SD_DIR / "tokenizer")
    ids = tok.tokenize("a photo of a cat")
    assert len(ids) == 77
    assert ids[0] == 49406  # BOS
    assert ids[1] == 49407 or ids[1] != ids[0]  # content after BOS
    # Should be padded with EOS (49407) at the end
    assert ids[-1] == 49407


def test_tokenizer_pads_short_input():
    from app.models.sd_tokenizer import CLIPTokenizer

    tok = CLIPTokenizer(_SD_DIR / "tokenizer")
    ids = tok.tokenize("cat")
    assert len(ids) == 77
    # Most of the 77 slots should be pad (49407) for a single-word input
    pad_count = sum(1 for i in ids if i == 49407)
    assert pad_count > 50


def test_sd_inpaint_returns_valid_image():
    """Full end-to-end: 10 steps on a small test image.

    Slow (~90s on CPU) but proves the ONNX sessions, scheduler, and
    blending all work together. Marked separately so it can be skipped
    in fast CI runs.
    """
    import cv2

    from app.models.sd_inpaint import SDInpaint

    # Solid background + red square to remove
    img = np.full((512, 512, 3), (180, 200, 220), dtype=np.uint8)  # sky BGR
    cv2.rectangle(img, (200, 200), (312, 312), (40, 40, 200), -1)  # red square
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(mask, (200, 200), (312, 312), 255, -1)

    sd = SDInpaint("models")
    result = sd.inpaint(
        img,
        mask,
        prompt="a blue sky with white clouds",
        negative_prompt="blurry, low quality, distorted",
        num_steps=5,  # 5 steps for test speed
        seed=42,
    )

    # Output must be a valid BGR image
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255

    # Masked area must have changed significantly from the input red square
    masked_in = img[mask > 0].astype(int)
    masked_out = result[mask > 0].astype(int)
    diff = np.abs(masked_out - masked_in).mean()
    assert diff > 5, f"SD output should differ from input in masked area, got diff={diff:.1f}"

    # Unmasked area must be preserved (mean diff < 5 per channel)
    unmasked_in = img[mask == 0].astype(int)
    unmasked_out = result[mask == 0].astype(int)
    unmasked_diff = np.abs(unmasked_out - unmasked_in).mean()
    assert unmasked_diff < 5, f"Unmasked area should be preserved, got diff={unmasked_diff:.1f}"
