class DDPMScheduler:
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
    
    def step_function(self, x, sigma, sigma_prev, noise):
        alpha_cumprod = 1 / ((sigma * sigma) + 1)
        alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        mu = (1.0 / alpha) ** 0.5 * (x - (1 - alpha) * noise / (1 - alpha_cumprod) ** 0.5)
        if sigma_prev > 0:
            # Caution: this randn tensor needs to be controlled by `torch.manual_seed`.
            mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)) ** 0.5 * torch.randn_like(x)
        return mu

    def step(self, latents, model_output, current_step):
        sigma = self.sigmas[current_step]
        sigma_next = self.sigmas[current_step + 1]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_output, latents)
        latents = self.step_function(latents / (1.0 + sigma ** 2.0) ** 0.5, sigma, sigma_next, (latents - denoised) / sigma)
        if sigma_next > 0:
            latents *= (1.0 + sigma_next ** 2.0) ** 0.5

        if current_step + 1 < len(self.sigmas):
            latents = self._unscaling(self.sigmas[current_step + 1], latents)
        return latents