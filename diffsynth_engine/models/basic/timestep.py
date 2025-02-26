import torch
import torch.nn as nn
import math


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalTimesteps(nn.Module):
    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, device: str, dtype: torch.dtype
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbeddings(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.time_proj = TemporalTimesteps(
            num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, device=device, dtype=dtype
        )
        self.timestep_embedder = nn.Sequential(
            nn.Linear(dim_in, dim_out, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out, device=device, dtype=dtype),
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb
