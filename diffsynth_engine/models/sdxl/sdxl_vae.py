from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder


class SDXLVAEEncoder(VAEEncoder):
    def __init__(self):
        super().__init__(
            latent_channels=4,  
            scaling_factor=0.13025,   
            shift_factor=None,
            use_quant_conv=True            
        )


class SDXLVAEDecoder(VAEDecoder):
    def __init__(self):
        super().__init__(
            latent_channels=4,  
            scaling_factor=0.13025,   
            shift_factor=None,
            use_post_quant_conv=True            
        )
