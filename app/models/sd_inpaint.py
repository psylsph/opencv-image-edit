"""Stable Diffusion 1.5 Inpainting via ONNX Runtime — no torch/diffusers.

Generative inpainting: actually *invents* realistic content in the masked
area based on a text prompt. Dramatically better quality than LaMa for
complex fills, but much slower (~30-60s on CPU).

Architecture:
  1. CLIP text encoder → text embeddings (conditioning)
  2. VAE encoder → image latents
  3. UNet (9-channel inpainting) → noise prediction in DDIM loop
  4. VAE decoder → final image

All inference via ONNX Runtime. Scheduler (DDIM) and tokenizer (CLIP BPE)
implemented in pure numpy / Rust.

Model source: modularai/stable-diffusion-1.5-onnx (fp32 ONNX)
  - text_encoder/model.onnx (~469MB)
  - unet/model.onnx + model.onnx_data (~3.3GB total)
  - vae_encoder/model.onnx (~130MB)
  - vae_decoder/model.onnx (~188MB)
  - tokenizer/vocab.json, merges.txt (~1MB)
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from app.exceptions import ModelNotFoundError
from app.models.sd_scheduler import DDIMScheduler
from app.models.sd_tokenizer import CLIPTokenizer

logger = logging.getLogger(__name__)

_WIDTH = 512
_HEIGHT = 512
_LATENT_CHANNELS = 4
_VAE_SCALE_FACTOR = 0.18215
_MAX_TOKENS = 77
_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 7.5
_DEFAULT_NEGATIVE = "low quality, blurry, deformed, artifacts"


class SDInpaint:
    """Lazy-loaded, thread-safe singleton for SD 1.5 inpainting via ONNX."""

    _instance: "SDInpaint | None" = None
    _lock = Lock()

    def __init__(self, model_dir: Path | str) -> None:
        model_dir = Path(model_dir)
        sd_dir = model_dir / "sd-inpainting"

        # Verify all required files exist
        required = {
            "text_encoder": sd_dir / "text_encoder" / "model.onnx",
            "unet": sd_dir / "unet" / "model.onnx",
            "unet_data": sd_dir / "unet" / "model.onnx_data",
            "vae_encoder": sd_dir / "vae_encoder" / "model.onnx",
            "vae_decoder": sd_dir / "vae_decoder" / "model.onnx",
        }
        for name, path in required.items():
            if not path.exists():
                raise ModelNotFoundError(f"SD model file missing: {name} at {path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0

        logger.info("Loading SD inpainting models (this takes ~10s)...")

        # Load all 4 ONNX models
        self._text_encoder = ort.InferenceSession(
            str(required["text_encoder"]), sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._unet = ort.InferenceSession(
            str(required["unet"]), sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._vae_encoder = ort.InferenceSession(
            str(required["vae_encoder"]), sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._vae_decoder = ort.InferenceSession(
            str(required["vae_decoder"]), sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self._tokenizer = CLIPTokenizer(sd_dir / "tokenizer")

        # DDIM scheduler
        self._scheduler = DDIMScheduler()

        logger.info("SD inpainting models loaded successfully.")

    @classmethod
    def get(cls, model_dir: Path | str | None = None) -> "SDInpaint":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            if model_dir is None:
                from app.config import get_settings
                model_dir = get_settings().model_dir
            cls._instance = SDInpaint(model_dir)
            return cls._instance

    @staticmethod
    def clear_cache() -> None:
        with SDInpaint._lock:
            SDInpaint._instance = None

    @staticmethod
    def is_available(model_dir: Path | str | None = None) -> bool:
        """Check if SD model files exist without loading them."""
        if model_dir is None:
            from app.config import get_settings
            model_dir = get_settings().model_dir
        sd_dir = Path(model_dir) / "sd-inpainting"
        return (sd_dir / "unet" / "model.onnx").exists()

    # ------------------------------------------------------------------
    # Inference steps
    # ------------------------------------------------------------------

    def _encode_text(self, prompt: str, negative_prompt: str) -> np.ndarray:
        """Encode prompt → text embeddings (2, 77, 768).

        Row 0: unconditional (negative prompt)
        Row 1: conditional (prompt)
        """
        tokens_cond = np.array([self._tokenizer.tokenize(prompt)], dtype=np.int32)
        tokens_uncond = np.array(
            [self._tokenizer.tokenize(negative_prompt)], dtype=np.int32
        )

        input_name = self._text_encoder.get_inputs()[0].name
        emb_cond = self._text_encoder.run(None, {input_name: tokens_cond})[0]
        emb_uncond = self._text_encoder.run(None, {input_name: tokens_uncond})[0]

        return np.concatenate([emb_uncond, emb_cond], axis=0)

    def _encode_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """Encode a 512x512 RGB image → VAE latent (1, 4, 64, 64)."""
        # Normalize to [-1, 1]
        x = img_rgb.astype(np.float32) / 127.5 - 1.0
        # HWC -> CHW
        x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]
        input_name = self._vae_encoder.get_inputs()[0].name
        latent = self._vae_encoder.run(None, {input_name: x})[0]
        return latent * _VAE_SCALE_FACTOR

    def _decode_latent(self, latent: np.ndarray) -> np.ndarray:
        """Decode VAE latent (1, 4, 64, 64) → RGB image (512, 512, 3)."""
        input_name = self._vae_decoder.get_inputs()[0].name
        decoded = self._vae_decoder.run(
            None, {input_name: latent / _VAE_SCALE_FACTOR}
        )[0]
        # CHW -> HWC, denormalize to [0, 255]
        img = decoded[0].transpose(1, 2, 0)
        img = (img / 2 + 0.5).clip(0, 1)
        return (img * 255).astype(np.uint8)

    def _run_unet(
        self,
        latents: np.ndarray,
        t: int,
        text_embeds: np.ndarray,
    ) -> np.ndarray:
        """Run UNet for a single denoising step (4-channel base model).

        For legacy inpainting we don't feed mask/masked_image to the UNet.
        Instead we denoise the full image and blend with the original in
        latent space at each step.
        """
        # Duplicate latents for classifier-free guidance
        unet_input = np.concatenate([latents, latents], axis=0)
        timestep = np.array([t, t], dtype=np.int64)

        # Run UNet
        inputs = {inp.name: val for inp, val in zip(
            self._unet.get_inputs(),
            [unet_input, timestep, text_embeds],
        )}
        noise_pred = self._unet.run(None, inputs)[0]

        # Classifier-free guidance
        noise_uncond, noise_cond = noise_pred[0:1], noise_pred[1:2]
        return noise_uncond + _DEFAULT_GUIDANCE * (noise_cond - noise_uncond)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        prompt: str = "",
        negative_prompt: str = _DEFAULT_NEGATIVE,
        num_steps: int = _DEFAULT_STEPS,
        seed: int = 42,
    ) -> np.ndarray:
        """Generative inpainting: fill masked region using SD.

        Args:
            img_bgr: BGR uint8 image (H, W, 3).
            mask: uint8 single-channel mask (H, W), non-zero = to fill.
            prompt: Text description of what to generate in the masked area.
            negative_prompt: What to avoid.
            num_steps: DDIM steps (20 is good; 30 slightly better).
            seed: Random seed for reproducibility.

        Returns:
            BGR uint8 image, same shape as input.
        """
        h, w = img_bgr.shape[:2]
        rng = np.random.RandomState(seed)

        # 1. Resize to 512x512 working resolution
        img_512 = cv2.resize(img_bgr, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(mask, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_NEAREST)

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)

        # 2. Encode text
        logger.info("SD: encoding text prompt...")
        text_embeds = self._encode_text(prompt, negative_prompt)

        # 3. Encode image to latent space
        logger.info("SD: encoding image...")
        image_latent = self._encode_image(img_rgb)

        # 4. Prepare mask at latent resolution (64x64)
        mask_norm = (mask_512 > 0).astype(np.float32)
        mask_latent = cv2.resize(
            mask_norm, (_WIDTH // 8, _HEIGHT // 8), interpolation=cv2.INTER_NEAREST
        )[np.newaxis, np.newaxis, :, :]

        # 5. Initialize noisy latents
        latent_shape = (1, _LATENT_CHANNELS, _HEIGHT // 8, _WIDTH // 8)
        init_noise = rng.randn(*latent_shape).astype(np.float32)

        # 6. Set up DDIM scheduler
        self._scheduler.set_timesteps(num_steps)
        timesteps = self._scheduler.timesteps

        # Apply forward diffusion: add noise to image latents
        latents = self._scheduler.add_noise(
            image_latent, init_noise,
            np.array([timesteps[0]], dtype=np.int64),
        )

        # 7. DDIM denoising loop (legacy inpainting: 4-channel UNet)
        for i, t in enumerate(timesteps):
            noise_pred = self._run_unet(latents, int(t), text_embeds)
            latents = self._scheduler.step(noise_pred, int(t), latents)

            # Legacy inpainting: blend with original image latents at each step.
            # In masked region: keep denoised latents.
            # In unmasked region: restore original image latents.
            latents = latents * mask_latent + image_latent * (1.0 - mask_latent)

            if (i + 1) % 5 == 0:
                logger.info("SD: step %d/%d", i + 1, num_steps)

        # 8. Decode final latent to image
        logger.info("SD: decoding latent...")
        result_rgb = self._decode_latent(latents)

        # RGB -> BGR
        result_512 = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # 9. Resize back to original dimensions
        result = cv2.resize(result_512, (w, h), interpolation=cv2.INTER_LINEAR)

        # 10. Blend: only replace masked pixels with feathered transition
        m = mask.astype(np.float32) / 255.0
        m = cv2.GaussianBlur(m, (7, 7), 0)
        m = np.clip(m, 0.0, 1.0)[:, :, np.newaxis]
        blended = img_bgr.astype(np.float32) * (1.0 - m) + result.astype(np.float32) * m

        return blended.astype(np.uint8)
