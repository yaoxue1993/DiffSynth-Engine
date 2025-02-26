import torch

from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler


class EulerAncestralSampler(EpsilonSampler):
    def initialize(self, init_latents, timesteps, sigmas, mask):
        super().initialize(init_latents, timesteps, sigmas, mask)
        self.eta = 1.0

    def _get_ancestral_step(self, sigma_from, sigma_to, eta=1.0):
        sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        return sigma_down, sigma_up

    def step(self, latents, model_outputs, i):
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_outputs, latents)
        sigma_down, sigma_up = self._get_ancestral_step(sigma, sigma_next, eta=1.0)
        d = (latents - denoised) / sigma
        dt = sigma_down - sigma
        latents = latents + d * dt
        if sigma_next > 0:
            # Caution: this randn tensor needs to be controlled by `torch.manual_seed`.
            latents = latents + torch.randn_like(latents) * sigma_up

        return self._unscaling(self.sigmas[i + 1], latents)
