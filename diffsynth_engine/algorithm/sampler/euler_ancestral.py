class EulerAncestralScheduler:
    def __init__(self):
        pass

    def set_timesteps(self, sigmas, timesteps):
        self.sigmas = sigmas
        self.timesteps = timesteps

    def _unscaling(self, sigma, latents):
        return latents / ((sigma ** 2 + 1) ** 0.5)
    
    def _scaling(self, sigma, latents):
        return latents * ((sigma ** 2 + 1) ** 0.5)
    
    def _to_denoised(self, sigma, model_output, latents):
        return latents - model_output * sigma
    
    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up

    def step(self, latents, model_output, current_step):
        sigma = self.sigmas[current_step]
        sigma_next = self.sigmas[current_step + 1]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_output, latents)
        sigma_down, sigma_up = self.get_ancestral_step(sigma, sigma_next, eta=1.0)
        d = (latents - denoised) / sigma
        dt = sigma_down - sigma
        latents = latents + d * dt
        if sigma_next > 0:
            # Caution: this randn tensor needs to be controlled by `torch.manual_seed`.
            latents = latents + torch.randn_like(latents) * sigma_up

        if current_step + 1 < len(self.sigmas):
            latents = self._unscaling(self.sigmas[current_step + 1], latents)
        return latents