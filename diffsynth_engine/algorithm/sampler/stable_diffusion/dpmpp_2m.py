from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler


class DPMSolverPlusPlus2MSampler(EpsilonSampler):
    """
    DPM Solver++ 2M sampler
    """

    def initialize(self, init_latents, timesteps, sigmas, mask):
        super().initialize(init_latents, timesteps, sigmas, mask)
        self.old_denoised = None

    def step(self, latents, model_outputs, i):
        s_prev, s, s_next = self.sigmas[i - 1], self.sigmas[i], self.sigmas[i + 1]
        t_prev, t, t_next = self._sigma_to_t(s_prev), self._sigma_to_t(s), self._sigma_to_t(s_next)
        h = t_next - t
        x = self._scaling(latents, s)
        denoised = self._to_denoised(s, model_outputs, x)
        if self.old_denoised is None or s_next == 0:
            self.old_denoised = denoised
            return (s_next / s) * x - (-h).expm1() * denoised
        h_last = t - t_prev
        r = h_last / h
        denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * self.old_denoised
        x = (s_next / s) * x - (-h).expm1() * denoised_d
        return self._unscaling(x, s_next)

    def _sigma_to_t(self, sigma):
        return sigma.log().neg()
