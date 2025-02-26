import torch

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.linear import ScaledLinearScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class KarrasScheduler(ScaledLinearScheduler):
    def __init__(self):
        super().__init__()
        self.rho = 7.0
        self.device = "cpu"

    def schedule(self, num_inference_steps: int):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, num_inference_steps, device=self.device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        timesteps = self.sigma_to_t(sigmas)
        sigmas = append_zero(sigmas).to(self.device)
        return sigmas, timesteps
