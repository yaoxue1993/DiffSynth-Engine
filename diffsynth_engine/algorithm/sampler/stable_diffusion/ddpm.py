import torch
from .epsilon import EpsilonSampler


class DDPMSampler(EpsilonSampler):
    def _step_function(self, x, sigma, sigma_prev, noise):
        alpha_cumprod = 1 / ((sigma * sigma) + 1)
        alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
        alpha = alpha_cumprod / alpha_cumprod_prev

        mu = (1.0 / alpha) ** 0.5 * (x - (1 - alpha) * noise / (1 - alpha_cumprod) ** 0.5)
        if sigma_prev > 0:
            # Caution: this randn tensor needs to be controlled by `torch.manual_seed`.
            mu += ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)) ** 0.5 * torch.randn_like(x)
        return mu

    def step(self, latents, model_outputs, i):
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_outputs, latents)
        latents = self._step_function(
            latents / (1.0 + sigma**2.0) ** 0.5, sigma, sigma_next, (latents - denoised) / sigma
        )

        latents *= (1.0 + sigma_next**2.0) ** 0.5

        return self._unscaling(self.sigmas[i + 1], latents)

    def step2(self, latents, model_outputs, i):
        return self._step_function(latents, self.sigmas[i], self.sigmas[i + 1], model_outputs)
