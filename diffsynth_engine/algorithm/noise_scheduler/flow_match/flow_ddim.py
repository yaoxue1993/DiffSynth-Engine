import torch

from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import append_zero
from diffsynth_engine.algorithm.noise_scheduler.flow_match.recifited_flow import RecifitedFlowScheduler


class FlowDDIMScheduler(RecifitedFlowScheduler):
    def __init__(self, shift=1.0, num_train_timesteps=1000, use_dynamic_shifting=False):
        super().__init__(shift, num_train_timesteps, use_dynamic_shifting)
        self.pseudo_timestep_range = 10000

    def schedule(self, num_inference_steps: int, mu: float | None = None, sigmas: torch.Tensor | None = None):
        inner_sigmas = torch.arange(1, self.pseudo_timestep_range + 1, 1) / self.pseudo_timestep_range
        inner_sigmas = self._time_shift(mu, 1.0, inner_sigmas)
        sigmas = []
        ss = max(len(inner_sigmas) // num_inference_steps, 1)
        for i in range(1, len(inner_sigmas), ss):
            sigmas.append(float(inner_sigmas[i]))
        sigmas = sigmas[::-1]
        sigmas = torch.FloatTensor(sigmas)

        timesteps = self._sigma_to_t(sigmas)
        sigmas = append_zero(sigmas)

        return sigmas, timesteps
