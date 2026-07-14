"""DDIM scheduler for Stable Diffusion inpainting — pure numpy, no torch.

Implements the DDIM (Denoising Diffusion Implicit Models) sampler with the
same configuration as the diffusers DDIMScheduler used by SD 1.5:

  beta_start = 0.00085
  beta_end = 0.012
  beta_schedule = "scaled_linear"
  num_train_timesteps = 1000

The scheduler supports:
  - set_timesteps(num_inference_steps): selects a subset of timesteps
  - add_noise(latents, noise, timestep): forward diffusion
  - step(model_output, timestep, sample): reverse diffusion (denoise)
"""

from __future__ import annotations

import numpy as np


class DDIMScheduler:
    """DDIM scheduler matching diffusers' DDIMScheduler for SD 1.5."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
                ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)

        self.final_alpha_cumprod = (
            np.array([1.0], dtype=np.float32) if set_alpha_to_one else self.alphas_cumprod[0:1]
        )

        self.initial_alpha_cumprod = self.alphas_cumprod
        self.timesteps = np.arange(0, num_train_timesteps, dtype=np.int64)
        self.clip_sample = clip_sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Select a subset of timesteps for the inference loop."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        )

    def _get_variance(self, timestep: int, prev_timestep: int) -> float:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod[0]
        )
        beta_prod_t = 1.0 - alpha_prod_t
        return (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * beta_prod_t

    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Forward diffusion: add noise to samples at given timesteps."""
        alphas_cumprod = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_alpha_prod = alphas_cumprod**0.5
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod) ** 0.5
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Reverse diffusion: denoise sample by one step."""
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod[0]
        )

        beta_prod_t = 1.0 - alpha_prod_t

        # Predict the original sample x_0
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # Clip
        if self.clip_sample:
            pred_original_sample = np.clip(pred_original_sample, -1.0, 1.0)

        # Compute the direction pointing to x_t
        pred_sample_direction = (1.0 - alpha_prod_t_prev) ** 0.5 * model_output

        # Compute x_{t-1}
        return alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
