import torch
import torch.nn as nn
import math

from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.models.basic.tiler import TileWorker

    
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * nn.functional.gelu(gate)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim):
        super().__init__()

        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = Attention(q_dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = Attention(q_dim=dim, kv_dim=cross_attention_dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
        self.act_fn = GEGLU(dim, dim * 4)
        self.ff = nn.Linear(dim * 4, dim)


    def forward(self, hidden_states, encoder_hidden_states, ipadapter_kwargs=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, ipadapter_kwargs=ipadapter_kwargs)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.act_fn(norm_hidden_states)
        ff_output = self.ff(ff_output)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DownSampler(nn.Module):
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            hidden_states = nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class AttentionBlock(nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, cross_attention_dim=None, norm_num_groups=32, eps=1e-5, need_proj_out=True):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim
            )
            for d in range(num_layers)
        ])
        self.need_proj_out = need_proj_out
        if need_proj_out:
            self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
            self,
            hidden_states, time_emb, text_emb, res_stack,
            cross_frame_attention=False,
            tiled=False, tile_size=64, tile_stride=32,
            ipadapter_kwargs_list={},
            **kwargs
    ):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        if cross_frame_attention:
            hidden_states = hidden_states.reshape(1, batch * height * width, inner_dim)
            encoder_hidden_states = text_emb.mean(dim=0, keepdim=True)
        else:
            encoder_hidden_states = text_emb
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                encoder_hidden_states = encoder_hidden_states.repeat(hidden_states.shape[0], 1, 1)

        if tiled:
            tile_size = min(tile_size, min(height, width))
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, inner_dim, height, width)
            def block_tile_forward(x):
                b, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
                x = block(x, encoder_hidden_states)
                x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
                return x
            for block in self.transformer_blocks:
                hidden_states = TileWorker().tiled_forward(
                    block_tile_forward,
                    hidden_states,
                    tile_size,
                    tile_stride,
                    tile_device=hidden_states.device,
                    tile_dtype=hidden_states.dtype
                )
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            for block_id, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    ipadapter_kwargs=ipadapter_kwargs_list.get(block_id, None)
                )
        if cross_frame_attention:
            hidden_states = hidden_states.reshape(batch, height * width, inner_dim)

        if self.need_proj_out:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        return hidden_states, time_emb, text_emb, res_stack


class PushBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_stack.append(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class PopBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_hidden_states = res_stack.pop()
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        return hidden_states, time_emb, text_emb, res_stack
