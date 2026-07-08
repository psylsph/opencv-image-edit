"""MobileSAM (Segment Anything with TinyViT) wrapper using ONNX Runtime.

Two ONNX models:
  - mobile_sam.encoder.onnx  (image -> image_embeddings, ~28MB)
  - mobile_sam_mask_decoder.onnx  from NVIDIA NanoSAM (embeddings + points
    -> low_res_masks + iou_predictions, ~16MB). This is the
    MobileSAM-trained decoder, NOT the SAM-H decoder (which gives
    near-uniform foreground and is unusable with the TinyViT encoder).

Why this combo matters: the original MobileSAM repo only ships a single
combined ONNX (encoder + decoder fused). Splitting them is straightforward
via NVIDIA's nanosam.tools.export_sam_mask_decoder_onnx. The vietanhdev
HuggingFace repo packages a SAM-H decoder alongside the MobileSAM encoder,
which produces logits so biased that threshold 0.5 marks 99% of the image
as foreground — those are incompatible architectures.

The NanoSAM decoder's multi-mask output (1, 4, 256, 256) contains 3
ambiguity candidates + 1 final combined mask. For single-point prompts
the 4th ("everything") mask always wins on IoU but is useless, so we pick
the best of candidates 0..2 by IoU, with a fallback to candidate 3 if all
have IoU < 0.5.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError


_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_MODEL_SIZE = 1024
_MASK_THRESHOLD = 0.5  # probability threshold for binary mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MobileSAM:
    """Lazy-loaded, thread-safe singleton wrapper around the MobileSAM encoder + decoder."""

    _instance: "MobileSAM | None" = None
    _lock = Lock()

    def __init__(self, encoder_path: Path | str, decoder_path: Path | str) -> None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = use all available cores
        self._encoder = ort.InferenceSession(
            str(encoder_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._decoder = ort.InferenceSession(
            str(decoder_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def get(cls, model_dir: Path | str | None = None) -> "MobileSAM":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            if model_dir is None:
                from app.config import get_settings
                model_dir = get_settings().model_dir
            model_dir = Path(model_dir)
            enc = model_dir / "mobile_sam.encoder.onnx"
            dec = model_dir / "mobile_sam_mask_decoder.onnx"
            if not enc.exists():
                raise ModelNotFoundError(f"MobileSAM encoder not found: {enc}")
            if not dec.exists():
                raise ModelNotFoundError(f"MobileSAM mask decoder not found: {dec}")
            cls._instance = MobileSAM(enc, dec)
            return cls._instance

    def encode(self, img_bgr: np.ndarray) -> np.ndarray:
        """Encode a BGR image (any size) to image embeddings.

        Returns:
            np.ndarray of shape (1, 256, 64, 64) float32.
        """
        # BGR -> RGB, resize to 1024x1024, normalize
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = cv2.resize(rgb, (_MODEL_SIZE, _MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (rgb - _MEAN) / _STD  # HWC, normalized
        out = self._encoder.run(None, {"input_image": x})[0]
        return out.astype(np.float32)

    def decode(
        self,
        embeddings: np.ndarray,
        point_xy: tuple[int, int],
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, float]:
        """Run the decoder for a single foreground point.

        Args:
            embeddings: (1, 256, 64, 64) float32 from encode().
            point_xy: (x, y) pixel coordinates in the original image space.
            original_size: (H, W) of the original image.

        Returns:
            (mask, score): mask is (H, W) uint8 with values 0 or 255, score is IoU.
        """
        h, w = original_size
        # Convert point from original-image space to 1024x1024 space
        px = point_xy[0] * _MODEL_SIZE / w
        py = point_xy[1] * _MODEL_SIZE / h
        point_coords = np.array([[[px, py]]], dtype=np.float32)  # (1, 1, 2)
        point_labels = np.array([[1]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)

        out_names = {o.name: o for o in self._decoder.get_outputs()}
        result = dict(zip(out_names.keys(), self._decoder.run(
            None,
            {
                "image_embeddings": embeddings,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
            },
        )))
        # NanoSAM decoder outputs: low_res_masks (1, 4, 256, 256) logits
        # + iou_predictions (1, 4) scores
        low_res = result["low_res_masks"]      # (1, 4, 256, 256) logits
        iou = result["iou_predictions"][0]     # (4,)

        # Pick the best of the 3 ambiguity masks (candidates 0..2).
        # Candidate 3 is the "combined" final output which tends to over-segment
        # for single-point prompts; only fall back to it if all 3 are bad.
        best_ambiguity = int(np.argmax(iou[:3]))
        if iou[best_ambiguity] < 0.5:
            best_idx = 3
        else:
            best_idx = best_ambiguity
        best_score = float(iou[best_idx])

        # Resize low-res (256x256) logit mask to original image size, then
        # sigmoid + threshold. Resizing in logit space preserves the relative
        # magnitude across the mask (standard SAM practice).
        low_res_mask = low_res[0, best_idx]  # (256, 256) logits
        resized_logits = cv2.resize(
            low_res_mask, (w, h), interpolation=cv2.INTER_LINEAR
        )
        probs = _sigmoid(resized_logits)
        mask = (probs > _MASK_THRESHOLD).astype(np.uint8) * 255
        return mask, best_score

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instance = None


def segment_with_point(
    img_bgr: np.ndarray,
    point_xy: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """End-to-end: encode image + decode with single foreground point.

    Returns (mask_uint8_0_or_255, score). For v1 we run encoder + decoder
    on every call; future versions can cache the embedding per-image.
    """
    sam = MobileSAM.get()
    h, w = img_bgr.shape[:2]
    embeddings = sam.encode(img_bgr)
    mask, score = sam.decode(embeddings, point_xy, (h, w))
    return mask, score
