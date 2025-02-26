import torch

from diffsynth_engine.algorithm.sampler.stable_diffusion.epsilon import EpsilonSampler


class DEISSampler(EpsilonSampler):
    """
    According to the implementation of the webui forge, deis_mode only supports tab and rhoab.
    """

    def initialize(self, init_latents, timesteps, sigmas, mask):
        super().initialize(init_latents, timesteps, sigmas, mask)
        self.max_order = 3
        self.sigmas = sigmas
        self.timesteps = timesteps
        self.lower_order_nums = 0
        self.coeff_list = get_deis_coeff_list(self.sigmas, self.max_order)
        self.coeff_buffer = []

    def step(self, latents, model_outputs, i):
        s, s_next = self.sigmas[i], self.sigmas[i + 1]
        denoised = latents - model_outputs * s

        d = (latents - denoised) / s
        order = min(self.max_order, i + 1)
        if self.sigmas[i + 1] <= 0:
            order = 1
        if order == 1:
            x_next = latents + (s_next - s) * d
        elif order == 2:
            coeff, coeff_prev1 = self.coeff_list[i]
            x_next = latents + coeff * d + coeff_prev1 * self.coeff_buffer[-1]
        elif order == 3:
            coeff, coeff_prev1, coeff_prev2 = self.coeff_list[i]
            x_next = latents + coeff * d + coeff_prev1 * self.coeff_buffer[-1] + coeff_prev2 * self.coeff_buffer[-2]
        elif order == 4:
            coeff, coeff_prev1, coeff_prev2, coeff_prev3 = self.coeff_list[i]
            x_next = (
                latents
                + coeff * d
                + coeff_prev1 * self.coeff_buffer[-1]
                + coeff_prev2 * self.coeff_buffer[-2]
                + coeff_prev3 * self.coeff_buffer[-3]
            )

        if len(self.coeff_buffer) == self.max_order - 1:
            for k in range(self.max_order - 2):
                self.coeff_buffer[k] = self.coeff_buffer[k + 1]
            self.coeff_buffer[-1] = d
        else:
            self.coeff_buffer.append(d)
        return x_next


# Taken from: https://github.com/zju-pi/diff-sampler/blob/main/gits-main/solver_utils.py
# under Apache 2 license
# A pytorch reimplementation of DEIS (https://github.com/qsh-zh/deis).
#############################
### Utils for DEIS solver ###
#############################
# ----------------------------------------------------------------------------
# Transfer from the input time (sigma) used in EDM to that (t) used in DEIS.


def vp_sigma_inv(beta_d, beta_min, sigma):
    return ((beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min) / beta_d


def edm2t(edm_steps, epsilon_s=1e-3, sigma_min=0.002, sigma_max=80):
    vp_beta_d = (
        2
        * (torch.log(torch.tensor(sigma_min) ** 2 + 1) / epsilon_s - torch.log(torch.tensor(sigma_max) ** 2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = torch.log(torch.tensor(sigma_max) ** 2 + 1) - 0.5 * vp_beta_d
    t_steps = vp_sigma_inv(vp_beta_d, vp_beta_min, edm_steps)
    return t_steps, vp_beta_min, vp_beta_d + vp_beta_min


def cal_poly(prev_t, j, taus):
    poly = 1
    for k in range(prev_t.shape[0]):
        if k == j:
            continue
        poly *= (taus - prev_t[k]) / (prev_t[j] - prev_t[k])
    return poly


def t2alpha_fn(beta_0, beta_1, t):
    return torch.exp(-0.5 * t**2 * (beta_1 - beta_0) - t * beta_0)


def cal_intergrand(beta_0, beta_1, taus):
    with torch.inference_mode(mode=False):
        taus = taus.clone()
        beta_0 = beta_0.clone()
        beta_1 = beta_1.clone()
        with torch.enable_grad():
            taus.requires_grad_(True)
            alpha = t2alpha_fn(beta_0, beta_1, taus)
            log_alpha = alpha.log()
            log_alpha.sum().backward()
            d_log_alpha_dtau = taus.grad
    integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(alpha * (1 - alpha))
    return integrand


def get_deis_coeff_list(t_steps, max_order, N=10000):
    t_steps, beta_0, beta_1 = edm2t(t_steps)
    C = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        order = min(i + 1, max_order)
        if order == 1:
            C.append([])
        else:
            taus = torch.linspace(t_cur, t_next, N).to(t_next.device)
            dtau = (t_next - t_cur) / N
            prev_t = t_steps[[i - k for k in range(order)]]
            coeff_temp = []
            integrand = cal_intergrand(beta_0, beta_1, taus)
            for j in range(order):
                poly = cal_poly(prev_t, j, taus)
                coeff_temp.append(torch.sum(integrand * poly) * dtau)
            C.append(coeff_temp)
    return C
