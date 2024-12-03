from diffsynth_engine.algorithm.sampler.base import EpsilonSampler
from diffsynth_engine.algorithm.sampler.brownian_tree import BrownianTreeNoiseSampler

class DPMSolverPlusPlus2MSDESampler(EpsilonSampler):
    """
    DPM Solver++ 2M SDE sampler
    """
    def __init__(self, solver_type='mid_point'):
        if solver_type not in {'heun', 'midpoint'}:
            raise ValueError('solver_type must be \'heun\' or \'midpoint\'')
        self.solver_type = solver_type
        self.eta = 1.0
        self.s_noise = 1.0

    def initialize(self, latents, timesteps, sigmas):
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        self.noise_sampler = BrownianTreeNoiseSampler(latents, sigma_min, sigma_max)
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.old_denoised = None    
        self.h_last = None

    def step(self, latents, model_outputs, i):
        x = self._scaling(latents, self.sigmas[i])
        denoised = self._to_denoised(self.sigmas[i], model_outputs, x)        

        if self.sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -self.sigmas[i].log(), -self.sigmas[i + 1].log()
            h = s - t
            eta_h = self.eta * h

            x = self.sigmas[i + 1] / self.sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if self.old_denoised is not None:
                r = self.h_last / h
                if self.solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - self.old_denoised)
                elif self.solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - self.old_denoised)

            if self.eta:
                x = x + self.noise_sampler(self.sigmas[i], self.sigmas[i + 1]) * self.sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * self.s_noise

        self.old_denoised = denoised
        self.h_last = h
        return self._unscaling(x, self.sigmas[i + 1])