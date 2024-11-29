import torch
from typing import Dict

from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder


class FluxVAEEncoder(VAEEncoder):
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=16,
            scaling_factor=0.3611,
            shift_factor=0.1159,
            use_quant_conv=False,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        return super().from_state_dict(state_dict,
                                       device=device,
                                       dtype=dtype,
                                       latent_channels=16,
                                       scaling_factor=0.3611,
                                       shift_factor=0.1159,
                                       use_quant_conv=False)


class FluxVAEDecoder(VAEDecoder):
    def __init__(self, device: str, dtype: torch.dtype = torch.float32):
        super().__init__(
            latent_channels=16,
            scaling_factor=0.3611,
            shift_factor=0.1159,
            use_post_quant_conv=False,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        return super().from_state_dict(state_dict,
                                       device=device,
                                       dtype=dtype,
                                       latent_channels=16,
                                       scaling_factor=0.3611,
                                       shift_factor=0.1159,
                                       use_post_quant_conv=False)
