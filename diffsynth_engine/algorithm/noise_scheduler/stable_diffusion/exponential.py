import torch
import math

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.linear import ScaledLinearScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class ExponentialScheduler(ScaledLinearScheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, num_inference_steps: int):
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(
            math.log(self.sigma_max), math.log(self.sigma_min), num_inference_steps, device=self.device
        ).exp()
        timesteps = self.sigma_to_t(sigmas)
        sigmas = append_zero(sigmas)
        return sigmas, timesteps
