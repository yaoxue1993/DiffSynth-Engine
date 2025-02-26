import torch
import numpy as np
import scipy.stats as stats

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.linear import ScaledLinearScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class BetaScheduler(ScaledLinearScheduler):
    """
    Implemented based on: https://arxiv.org/abs/2407.12173
    """

    def __init__(self):
        super().__init__()
        self.alpha = 0.6
        self.beta = 0.6

    def schedule(self, num_inference_steps: int):
        timesteps = 1 - np.linspace(0, 1, num_inference_steps)
        timesteps = [stats.beta.ppf(x, self.alpha, self.beta) for x in timesteps]
        sigmas = [self.sigma_min + (x * (self.sigma_max - self.sigma_min)) for x in timesteps]
        sigmas = torch.FloatTensor(sigmas).to(self.device)
        timesteps = self.sigma_to_t(sigmas)
        sigmas = append_zero(sigmas)
        return sigmas, timesteps
