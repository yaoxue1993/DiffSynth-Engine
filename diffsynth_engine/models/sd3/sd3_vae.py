import torch
from typing import Dict

from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder


class SD3VAEEncoder(VAEEncoder):
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float16):
        super().__init__(
            latent_channels=16,
            scaling_factor=1.5305,
            shift_factor=0.0609,
            use_quant_conv=False,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_state_dict(cls,
                        state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0',
                        dtype: torch.dtype = torch.float16
                        ):
        return super().from_state_dict(state_dict,
                                       latent_channels=16,
                                       scaling_factor=1.5305,
                                       shift_factor=0.0609,
                                       use_quant_conv=False,
                                       device=device,
                                       dtype=dtype)


class SD3VAEDecoder(VAEDecoder):
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float16):
        super().__init__(
            latent_channels=16,
            scaling_factor=1.5305,
            shift_factor=0.0609,
            use_post_quant_conv=False,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_state_dict(cls,
                        state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0',
                        dtype: torch.dtype = torch.float16
                        ):
        return super().from_state_dict(state_dict,
                                       latent_channels=16,
                                       scaling_factor=1.5305,
                                       shift_factor=0.0609,
                                       use_post_quant_conv=False,
                                       device=device,
                                       dtype=dtype)
