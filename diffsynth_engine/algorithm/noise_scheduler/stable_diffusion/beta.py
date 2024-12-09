import torch
from .stable_diffusion import StableDiffusionScheduler
from ..base_scheduler import append_zero
import numpy as np
import scipy.stats as stats

from diffsynth_engine.algorithm.noise_scheduler.stable_diffusion.stable_diffusion import StableDiffusionScheduler
from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero


class BetaScheduler(StableDiffusionScheduler):
    """
    Implemented based on: https://arxiv.org/abs/2407.12173
    """

    def __init__(self):
        super().__init__()
        self.alpha = 0.6
        self.beta = 0.6

    def schedule(self, num_inference_steps: int):
        timesteps = 1 - np.linspace(0, 1, num_inference_steps)
        timesteps = [stats.beta.ppf(x, self.alpha, self.beta) * (self.num_train_steps - 1) for x in timesteps] 
        sigmas = [self.sigma_min + (x * (self.sigma_max-self.sigma_min)) for x in timesteps]
        sigmas = torch.FloatTensor(sigmas).to(self.device)
        sigmas = append_zero(sigmas)
        timesteps = torch.FloatTensor(timesteps).to(self.device)
        return sigmas, timesteps
