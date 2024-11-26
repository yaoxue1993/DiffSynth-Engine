from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder


class FluxVAEEncoder(VAEEncoder):
    def __init__(self):
        super().__init__(
            latent_channels=16,
            scaling_factor=0.3611,
            shift_factor=0.1159,       
            use_quant_conv=False
        )


class FluxVAEDecoder(VAEDecoder):
    def __init__(self):
        super().__init__(
            latent_channels=16,
            scaling_factor=0.3611,
            shift_factor=0.1159,
            use_post_quant_conv=False
        )
