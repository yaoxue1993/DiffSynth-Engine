from diffsynth_engine.algorithm.sampler.base import EpsilonSampler
class EulerSampler(EpsilonSampler):
    def initialize(self, latents, timesteps, sigmas):        
        self.sigmas = sigmas
        self.timesteps = timesteps    

    def step(self, latents, model_outputs, i):
        sigma = self.sigmas[i]
        latents = self._scaling(sigma, latents)

        denoised = self._to_denoised(sigma, model_outputs, latents)
        d = (latents - denoised) / sigma
        dt = self.sigmas[i + 1] - sigma
        latents = latents + d * dt
        return self._unscaling(self.sigmas[i + 1], latents)
