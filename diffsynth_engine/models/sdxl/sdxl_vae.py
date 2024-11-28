import torch
from typing import Dict

from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder


class SDXLVAEEncoder(VAEEncoder):
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float16):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.13025,
            shift_factor=0,
            use_quant_conv=True,
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
                                       latent_channels=4,
                                       scaling_factor=0.13025,
                                       shift_factor=0,
                                       use_quant_conv=True,
                                       device=device,
                                       dtype=dtype)


class SDXLVAEDecoder(VAEDecoder):
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float16):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.13025,
            shift_factor=0,
            use_post_quant_conv=True,
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
                                       latent_channels=4,
                                       scaling_factor=0.13025,
                                       shift_factor=0,
                                       use_post_quant_conv=True,
                                       device=device,
                                       dtype=dtype)
