"""LaMa (Large Mask Inpainting) wrapper using ONNX Runtime.

The model is from the official `opencv/inpainting_lama` Hugging Face repo:
  https://huggingface.co/opencv/inpainting_lama

LaMa uses Fourier convolutions to handle very large masks and to reproduce
periodic structure (brick walls, window grids, etc.) that patch-propagation
methods (TELEA, NS) cannot. It is the standard modern quality upgrade for
object removal.

We use ONNX Runtime instead of cv2.dnn because OpenCV 5's new graph engine
emits a "Targets not supported" warning for this model and silently produces
incorrect (near-identity) outputs. ORT runs the same model correctly.

Crop-based inference strategy
------------------------------
Instead of resizing the entire image to 512×512 (which makes small masks
invisible to the model), we:

1. **Dilate** the mask to ensure full object coverage.
2. **Crop** a generous region around the mask bounding box at full resolution.
3. **Resize** only the crop to 512×512 (preserving mask detail).
4. **Inpaint** via ONNX Runtime, optionally iterating.
5. **Resize** the result back to crop dimensions.
6. **Blend** back into the original — replacing only masked pixels with a
   feathered transition at the boundary.

This gives small masks ~5-10× more pixels in the model's 512×512 input,
dramatically improving removal quality.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


_MODEL_FILENAME = "inpainting_lama_2025jan.onnx"
_INPUT_SIZE = 512  # model expects 512x512 inputs

# Crop-based inference defaults
_DEFAULT_DILATE_PX = 10
_DEFAULT_MIN_PAD = 64       # minimum padding around mask bbox (px)
_DEFAULT_ITERATIONS = 1
_FEATHER_KERNEL = 7         # Gaussian blur kernel for mask boundary feathering
_MAX_MASK_RATIO = 0.35      # mask must cover <35% of crop for good context
_DARKNESS_THRESHOLD = 15    # mean pixel value below this = "black square" failure


class LaMa:
    """Lazy-loaded, thread-safe singleton wrapper around the LaMa ONNX model."""

    _instance: "LaMa | None" = None
    _lock = Lock()

    def __init__(self, model_path: Path | str) -> None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = use all available cores
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def get(cls, model_dir: Path | str | None = None) -> "LaMa":
        """Get or construct the LaMa singleton for the given model directory."""
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            if model_dir is None:
                from app.config import get_settings
                model_dir = get_settings().model_dir
            model_dir = Path(model_dir)
            model_path = model_dir / _MODEL_FILENAME
            if not model_path.exists():
                raise ModelNotFoundError(f"LaMa model not found: {model_path}")
            cls._instance = LaMa(model_path)
            return cls._instance

    @staticmethod
    def clear_cache() -> None:
        """Reset the singleton (used by tests)."""
        with LaMa._lock:
            LaMa._instance = None

    # ------------------------------------------------------------------
    # Pre/post processing helpers (public for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _padding_to_multiple_of_eight(h: int, w: int) -> tuple[int, int]:
        """How much padding (rows, cols) to add so H, W are multiples of 8.

        Kept for backwards compatibility with earlier tests; the model
        itself does internal resizing, so this is unused in practice.
        """
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        return pad_h, pad_w

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 [0,255] -> 4D float32 blob in NCHW, scaled by 1/255."""
        return cv2.dnn.blobFromImage(
            img_bgr, scalefactor=1.0 / 255.0, size=(_INPUT_SIZE, _INPUT_SIZE),
            mean=(0, 0, 0), swapRB=False, crop=False,
        )

    @staticmethod
    def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """uint8 single-channel mask -> 4D float32 binarized blob."""
        blob = cv2.dnn.blobFromImage(
            mask, scalefactor=1.0, size=(_INPUT_SIZE, _INPUT_SIZE),
            mean=(0,), swapRB=False, crop=False,
        )
        return (blob > 0).astype(np.float32)

    @staticmethod
    def _postprocess(out_nchw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """NCHW -> HWC uint8, resized to target dimensions.

        The model output is in display range (0..~255) — we just clip
        defensively and convert. Resize back to the original input size.
        """
        chw = out_nchw[0]
        hwc = chw.transpose(1, 2, 0)
        hwc_u8 = np.clip(hwc, 0, 255).astype(np.uint8)
        if hwc_u8.shape[0] != target_h or hwc_u8.shape[1] != target_w:
            hwc_u8 = cv2.resize(hwc_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return hwc_u8

    # ------------------------------------------------------------------
    # Crop geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_crop_bbox(
        mask: np.ndarray,
        h: int,
        w: int,
        min_pad: int = _DEFAULT_MIN_PAD,
        max_ratio: float = _MAX_MASK_RATIO,
    ) -> tuple[int, int, int, int]:
        """Compute crop bounding box (y1, x1, y2, x2) around non-zero mask pixels.

        Padding starts at ``max(mask_dim // 2, min_pad)`` and is expanded if
        the mask would cover more than ``max_ratio`` of the crop area — this
        ensures the model always has enough surrounding context. Clamped to
        image bounds.
        """
        ys, xs = np.where(mask > 0)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        mask_h = y_max - y_min + 1
        mask_w = x_max - x_min + 1
        pad_y = max(mask_h // 2, min_pad)
        pad_x = max(mask_w // 2, min_pad)

        # Expand padding so the mask covers < max_ratio of the crop.
        # mask_area / crop_area = (mask_h * mask_w) / ((mask_h + 2*pad_y) * (mask_w + 2*pad_x))
        ratio = (mask_h * mask_w) / ((mask_h + 2 * pad_y) * (mask_w + 2 * pad_x))
        if ratio > max_ratio:
            needed = int((mask_h * mask_w / max_ratio) ** 0.5)
            needed = max(needed, mask_h + 2 * min_pad, mask_w + 2 * min_pad)
            # Set padding so each dimension has at least (needed - mask_dim) / 2
            pad_y = max(pad_y, (needed - mask_h) // 2)
            pad_x = max(pad_x, (needed - mask_w) // 2)

        cy1 = max(0, y_min - pad_y)
        cy2 = min(h, y_max + pad_y + 1)
        cx1 = max(0, x_min - pad_x)
        cx2 = min(w, x_max + pad_x + 1)
        return cy1, cx1, cy2, cx2

    @staticmethod
    def _dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
        """Dilate a binary mask by ``dilate_px`` pixels using an elliptical kernel."""
        if dilate_px <= 0:
            return mask.copy()
        ksize = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.dilate(mask, kernel)

    @staticmethod
    def _feather_blend(
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
        feather_ksize: int = _FEATHER_KERNEL,
    ) -> np.ndarray:
        """Blend inpainted result into original using a feathered mask boundary.

        Inside the mask: use inpainted pixels. Outside: keep original.
        At the boundary: Gaussian-weighted transition for seamless join.
        """
        m = mask.astype(np.float32) / 255.0
        if feather_ksize > 1:
            m = cv2.GaussianBlur(m, (feather_ksize, feather_ksize), 0)
            m = np.clip(m, 0.0, 1.0)
        m3 = m[:, :, np.newaxis]  # broadcast across 3 channels
        blended = original.astype(np.float32) * (1.0 - m3) + inpainted.astype(np.float32) * m3
        return blended.astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        dilate_px: int = _DEFAULT_DILATE_PX,
        iterations: int = _DEFAULT_ITERATIONS,
    ) -> np.ndarray:
        """Inpaint img_bgr using the given mask (non-zero = to inpaint).

        Uses crop-based inference: crops a generous region around the mask
        at full resolution, resizes just the crop to 512×512 for the model,
        then blends the result back into the original image.

        Includes three safeguards against the "black square" failure:
        1. Context-ratio enforcement: crop is expanded if mask > 35% of crop
        2. NaN guard: NaN values from ORT are replaced with 0 before clipping
        3. TELEA fallback: if LaMa output in masked area is too dark,
           cv2.inpaint (TELEA) is used on the same crop

        Args:
            img_bgr: BGR uint8 image (H, W, 3).
            mask:    uint8 single-channel mask (H, W), non-zero = hole.
            dilate_px: Dilate the mask by this many pixels before inference
                to ensure full object coverage (default 10).
            iterations: Number of LaMa passes. Each pass feeds the previous
                output as input. Default 1; 2-3 can improve stubborn removals.

        Returns:
            BGR uint8 image of the same shape as img_bgr.
        """
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]

        # Empty mask = no work
        if not mask.any():
            return img_bgr.copy()

        # 1. Dilate mask for better coverage
        work_mask = self._dilate_mask(mask, dilate_px)

        # 2. Compute generous crop around the mask (context-ratio enforced)
        cy1, cx1, cy2, cx2 = self._compute_crop_bbox(work_mask, h, w)
        crop_img = img_bgr[cy1:cy2, cx1:cx2]
        crop_mask = work_mask[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop_img.shape[:2]

        # 3. Preprocess: resize crop + mask to model's 512x512
        img_blob = self._preprocess(crop_img)
        mask_blob = self._preprocess_mask(crop_mask)

        # 4. Run inference (optionally iterative) with NaN guard
        result_u8 = self._run_lama(img_blob, mask_blob, iterations)

        # 5. Resize result back to crop dimensions BEFORE quality check
        if result_u8.shape[0] != crop_h or result_u8.shape[1] != crop_w:
            result_u8 = cv2.resize(
                result_u8, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR,
            )

        # 6. Quality check: detect "black square" failure
        # A real LaMa failure produces near-uniform black (low mean AND low std).
        # A legitimately dark region has varied texture (std > 0) — don't
        # fall back in that case.
        masked_pixels = result_u8[crop_mask > 0]
        if masked_pixels.size > 0:
            is_too_dark = masked_pixels.mean() < _DARKNESS_THRESHOLD
            is_uniform = masked_pixels.std() < 2.0
            if is_too_dark and is_uniform:
                # LaMa produced a uniform black fill — fall back to TELEA.
                result_u8 = cv2.inpaint(
                    crop_img, crop_mask, 5.0, cv2.INPAINT_TELEA,
                )

        # 7. Blend back: feathered transition at mask boundary
        blended_crop = self._feather_blend(crop_img, result_u8, crop_mask)

        result = img_bgr.copy()
        result[cy1:cy2, cx1:cx2] = blended_crop
        return result

    def _run_lama(
        self,
        img_blob: np.ndarray,
        mask_blob: np.ndarray,
        iterations: int,
    ) -> np.ndarray:
        """Run LaMa inference with NaN guard. Returns HWC uint8."""
        out = self._session.run(None, {"image": img_blob, "mask": mask_blob})[0]
        result_hwc = out[0].transpose(1, 2, 0)
        # Guard against NaN — certain edge-case inputs produce NaN from ORT,
        # which astype(uint8) converts to 0 (black square).
        result_hwc = np.nan_to_num(result_hwc, nan=0.0, posinf=255.0, neginf=0.0)
        result_u8 = np.clip(result_hwc, 0, 255).astype(np.uint8)

        for _i in range(1, iterations):
            img_blob = self._preprocess(result_u8)
            out = self._session.run(None, {"image": img_blob, "mask": mask_blob})[0]
            result_hwc = out[0].transpose(1, 2, 0)
            result_hwc = np.nan_to_num(result_hwc, nan=0.0, posinf=255.0, neginf=0.0)
            result_u8 = np.clip(result_hwc, 0, 255).astype(np.uint8)

        return result_u8
