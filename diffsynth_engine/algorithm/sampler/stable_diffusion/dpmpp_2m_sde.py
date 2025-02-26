from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler
from diffsynth_engine.algorithm.sampler.stable_diffusion.brownian_tree import BrownianTreeNoiseSampler


class DPMSolverPlusPlus2MSDESampler(EpsilonSampler):
    """
    DPM Solver++ 2M SDE sampler
    """

    def initialize(self, init_latents, timesteps, sigmas, mask):
        super().initialize(init_latents, timesteps, sigmas, mask)
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        self.noise_sampler = BrownianTreeNoiseSampler(init_latents, sigma_min, sigma_max)
        self.old_denoised = None
        self.h_last = None
        self.eta = 1.0
        self.s_noise = 1.0
        self.solver_type = "heun"  # {'heun', 'midpoint'}

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
                if self.solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - self.old_denoised)
                elif self.solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - self.old_denoised)

            if self.eta:
                x = (
                    x
                    + self.noise_sampler(self.sigmas[i], self.sigmas[i + 1])
                    * self.sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * self.s_noise
                )

        self.old_denoised = denoised
        self.h_last = h
        return self._unscaling(x, self.sigmas[i + 1])
