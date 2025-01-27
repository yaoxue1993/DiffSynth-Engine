import torch
import math

from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero, BaseScheduler


class RecifitedFlowScheduler(BaseScheduler):
    def __init__(self, shift=1.0, num_train_timesteps=1000, use_dynamic_shifting=False):
        self.pseudo_timestep_range = 10000
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = 1 / num_train_timesteps
        self.sigma_max = 1
        self.use_dynamic_shifting = use_dynamic_shifting
        if not self.use_dynamic_shifting:
            # SD3/SD3.5
            self.sigma_min = self.shift * self.sigma_min / (1 + (self.shift - 1) * self.sigma_min)
            # self.sigma_max = 1

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _t_to_sigma(self, t):
        return t / self.num_train_timesteps

    def _time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def schedule(self, num_inference_steps: int, mu: float | None = None, sigmas: torch.Tensor | None = None):
        if sigmas is None:
            timesteps = torch.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )
            sigmas = timesteps / self.num_train_timesteps
        if self.use_dynamic_shifting:
            # FLUX
            sigmas = self._time_shift(mu, 1.0, sigmas)
        else:
            # SD3/SD3.5
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = append_zero(sigmas)

        return sigmas, timesteps
