class FlowMatchEulerSampler:
    def initialize(self, latents, timesteps, sigmas):        
        self.sigmas = sigmas
        self.timesteps = timesteps

    def step(self, latents, model_outputs, i):
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]

        latents = latents + model_outputs * (sigma_next - sigma)
        return latents
