from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder
import torch

class SDVAEEncoder(VAEEncoder):
    def __init__(self, device:str='cuda:0', dtype:torch.dtype=torch.float16):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.18215,
            shift_factor=None,
            use_quant_conv=True,
            device=device,
            dtype=dtype
        )


class SDVAEDecoder(VAEDecoder):
    def __init__(self, device:str='cuda:0', dtype:torch.dtype=torch.float16):
        super().__init__(
            latent_channels=4,
            scaling_factor=0.18215,
            shift_factor=None,
            use_post_quant_conv=True,
            device=device,
            dtype=dtype
        )
