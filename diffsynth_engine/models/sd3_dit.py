import torch
import torch.nn as nn

from diffsynth_engine.models.svd_unet import TemporalTimesteps


class TimestepEmbeddings(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.time_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )

    def forward(self, timestep, dtype):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, single=False):
        super().__init__()
        self.single = single
        self.linear = nn.Linear(dim, dim * (2 if single else 6))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# TODO: add SD3DiT
