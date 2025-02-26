import torch

from diffsynth_engine.algorithm.noise_scheduler.base_scheduler import BaseScheduler, append_zero


def linear_beta_schedule(beta_start: float = 0.00085, beta_end: float = 0.0120, num_train_steps: int = 1000):
    """
    DDPM Schedule
    """
    return torch.linspace(beta_start, beta_end, num_train_steps)


def scaled_linear_beta_schedule(beta_start: float = 0.00085, beta_end: float = 0.0120, num_train_steps: int = 1000):
    """
    Stable Diffusion Schedule
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_steps) ** 2


class ScaledLinearScheduler(BaseScheduler):
    def __init__(self):
        self.device = "cpu"
        self.num_train_steps = 1000
        self.beta_start = 0.00085
        self.beta_end = 0.0120
        self.sigmas = self.get_sigmas()
        self.log_sigmas = self.sigmas.log()

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self):
        # Stable Diffusion Sigmas
        # len(sigmas) == 1000, sigma_min=sigmas[0] == 0.0292, sigma_max=sigmas[-1] == 14.6146
        betas = scaled_linear_beta_schedule(
            beta_start=self.beta_start, beta_end=self.beta_end, num_train_steps=self.num_train_steps
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return sigmas

    def sigma_to_t(self, sigma):
        """
        找到sigma.log()在self.log_sigmas中的位置(low和high), 进行加权插值得到t
        """
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        """
        对t进行floor和ceil, 得到low_idx和high_idx, 计算对应位置的log_sigma, 进行加权插值并exp得到sigma
        """
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def schedule(self, num_inference_steps: int):
        """
        Uniformly sample timesteps for inference
        """
        timesteps = torch.linspace(self.num_train_steps - 1, 0, num_inference_steps, device=self.sigmas.device)
        sigmas = append_zero(self.t_to_sigma(timesteps))
        return sigmas, timesteps
