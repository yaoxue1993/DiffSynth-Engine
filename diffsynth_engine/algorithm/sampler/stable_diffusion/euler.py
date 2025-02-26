from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler


class EulerSampler(EpsilonSampler):
    def step(self, latents, model_outputs, i):
        sigma = self.sigmas[i]
        latents = self._scaling(latents, sigma)
        denoised = self._to_denoised(sigma, model_outputs, latents)
        d = (latents - denoised) / sigma
        dt = self.sigmas[i + 1] - sigma
        latents = latents + d * dt
        return self._unscaling(latents, self.sigmas[i + 1])
