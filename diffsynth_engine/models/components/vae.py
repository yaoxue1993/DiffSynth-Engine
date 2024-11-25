import torch
import torch.nn as nn
from einops import rearrange

from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.models.basic.unet_helper import ResnetBlock, UpSampler, DownSampler
from diffsynth_engine.models.basic.tiler import TileWorker


class VAEAttentionBlock(nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True
            )
            for d in range(num_layers)
        ])

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class VAEDecoder(nn.Module):
    def __init__(self, 
        latent_channels:int=4,
        scaling_factor:float=0.18215,
        shift_factor:float=0,
        use_post_quant_conv:bool=True
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_post_quant_conv = use_post_quant_conv
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock2D
            ResnetBlock(256, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
        ])

        self.conv_norm_out = nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def _tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        original_dtype = sample.dtype
        sample = sample.to(dtype=next(iter(self.parameters())).dtype)
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self._tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. pre-process
        sample = (sample + self.shift_factor) / self.scaling_factor
        if self.use_post_quant_conv:
            sample = self.post_quant_conv(sample)
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.to(original_dtype)

        return hidden_states


class VAEEncoder(nn.Module):
    def __init__(self, 
        latent_channels:int=4,
        scaling_factor:float=0.18215,
        shift_factor:float=0,
        use_quant_conv:bool=True
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_quant_conv = use_quant_conv
        if use_quant_conv:
            self.quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        self.conv_norm_out = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)

    def _tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        original_dtype = sample.dtype
        sample = sample.to(dtype=next(iter(self.parameters())).dtype)
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self._tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. pre-process
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if self.use_quant_conv:
            hidden_states = self.quant_conv(hidden_states)
        hidden_states = hidden_states[:, :self.latent_channels]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor
        hidden_states = hidden_states.to(original_dtype)
        return hidden_states

    def encode_video(self, sample, batch_size=8):
        B = sample.shape[0]
        hidden_states = []

        for i in range(0, sample.shape[2], batch_size):

            j = min(i + batch_size, sample.shape[2])
            sample_batch = rearrange(sample[:,:,i:j], "B C T H W -> (B T) C H W")

            hidden_states_batch = self(sample_batch)
            hidden_states_batch = rearrange(hidden_states_batch, "(B T) C H W -> B C T H W", B=B)

            hidden_states.append(hidden_states_batch)

        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states
