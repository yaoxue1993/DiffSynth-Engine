import torch

from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler
from diffsynth_engine.algorithm.sampler.stable_diffusion.brownian_tree import BrownianTreeNoiseSampler


class DPMSolverPlusPlus3MSDESampler(EpsilonSampler):
    def initialize(self, init_latents, timesteps, sigmas, mask):
        super().initialize(init_latents, timesteps, sigmas, mask)
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        self.noise_sampler = BrownianTreeNoiseSampler(init_latents, sigma_min, sigma_max)
        self.denoised_1 = None
        self.denoised_2 = None
        self.h_1 = None
        self.h_2 = None
        self.eta = 1.0
        self.s_noise = 1.0

    def step(self, latents, model_outputs, i):
        x = self._scaling(latents, self.sigmas[i])
        denoised = self._to_denoised(self.sigmas[i], model_outputs, x)
        if self.sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -self.sigmas[i].log(), -self.sigmas[i + 1].log()
            h = s - t
            h_eta = h * (self.eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if self.h_2 is not None:
                r0 = self.h_1 / h
                r1 = self.h_2 / h
                d1_0 = (denoised - self.denoised_1) / r0
                d1_1 = (self.denoised_1 - self.denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif self.h_1 is not None:
                r = self.h_1 / h
                d = (denoised - self.denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if self.eta:
                x = (
                    x
                    + self.noise_sampler(self.sigmas[i], self.sigmas[i + 1])
                    * self.sigmas[i + 1]
                    * (-2 * h * self.eta).expm1().neg().sqrt()
                    * self.s_noise
                )

        self.denoised_1, self.denoised_2 = denoised, self.denoised_1
        self.h_1, self.h_2 = h, self.h_1
        return self._unscaling(x, self.sigmas[i + 1])
