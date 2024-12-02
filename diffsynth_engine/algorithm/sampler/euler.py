class EulerScheduler:
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

    def step(self, latents, model_output, current_step):
        sigma = self.sigmas[current_step]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_output, latents)
        d = (latents - denoised) / sigma
        dt = self.sigmas[current_step + 1] - sigma
        latents = latents + d * dt

        if current_step + 1 < len(self.sigmas):
            latents = self._unscaling(self.sigmas[current_step + 1], latents)
        return latents
