from diffsynth_engine.algorithm.sampler.base import EpsilonSampler

class DPMSolverPlusPlus2MSampler(EpsilonSampler):
    """
    DPM Solver++ 2M sampler
    """
    def __init__(self):
        self.prev_denoised = None

    def initialize(self, latents, timesteps, sigmas):
        self.timesteps = timesteps
        self.sigmas = sigmas

    def step(self, latents, model_outputs, current_step):
        s_prev, s, s_next = self.sigmas[current_step - 1], self.sigmas[current_step], self.sigmas[current_step + 1]
        t_prev, t, t_next = self._sigma_to_t(s_prev), self._sigma_to_t(s), self._sigma_to_t(s_next)
        h = t_next - t
        x = self._scaling(latents, s)
        denoised = self._to_denoised(s, model_outputs, x)        
        if self.prev_denoised is None or s_next == 0:
            self.prev_denoised = denoised                    
            return (s_next / s) * x - (-h).expm1() * denoised
        h_last = t - t_prev
        r = h_last / h
        denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * self.prev_denoised
        x = (s_next / s) * x - (-h).expm1() * denoised_d
        return self._unscaling(x, s_next)

    def _sigma_to_t(self, sigma):
        return sigma.log().neg()
