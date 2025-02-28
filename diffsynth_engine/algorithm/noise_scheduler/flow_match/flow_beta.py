import torch
import numpy as np
import scipy.stats as stats

from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero
from diffsynth_engine.algorithm.noise_scheduler.flow_match.recifited_flow import RecifitedFlowScheduler


class FlowBetaScheduler(RecifitedFlowScheduler):
    def __init__(self):
        super().__init__()
        self.alpha = 0.6
        self.beta = 0.6

    def schedule(self, num_inference_steps: int, mu: float | None = None, sigmas: torch.Tensor | None = None):
        pseudo_timestep_range = 10000
        inner_sigmas = torch.arange(1, pseudo_timestep_range + 1, 1) / pseudo_timestep_range
        inner_sigmas = self._time_shift(mu, 1.0, inner_sigmas)
        sigma_min = inner_sigmas[0]
        sigma_max = inner_sigmas[-1]

        timesteps = 1 - np.linspace(0, 1, num_inference_steps)
        timesteps = [stats.beta.ppf(x, self.alpha, self.beta) for x in timesteps]
        sigmas = [sigma_min + (x * (sigma_max - sigma_min)) for x in timesteps]
        sigmas = torch.FloatTensor(sigmas)
        timesteps = self._sigma_to_t(sigmas)
        sigmas = append_zero(sigmas)
        return sigmas, timesteps
