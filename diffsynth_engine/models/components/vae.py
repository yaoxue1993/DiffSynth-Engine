import os
import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange

from diffsynth_engine.models.basic.attention import Attention
from diffsynth_engine.models.basic.unet_helper import ResnetBlock, UpSampler, DownSampler
from diffsynth_engine.models.basic.tiler import TileWorker
from diffsynth_engine.models.base import PreTrainedModel, StateDictConverter
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class VAEStateDictConverter(StateDictConverter):
    def __init__(self, has_encoder: bool = False, has_decoder: bool = False):
        self.has_encoder = has_encoder
        self.has_decoder = has_decoder

    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = {
            # decoder
            "first_stage_model.decoder.conv_in.bias": "decoder.conv_in.bias",
            "first_stage_model.decoder.conv_in.weight": "decoder.conv_in.weight",
            "first_stage_model.decoder.conv_out.bias": "decoder.conv_out.bias",
            "first_stage_model.decoder.conv_out.weight": "decoder.conv_out.weight",
            "first_stage_model.decoder.mid.attn_1.k.bias": "decoder.blocks.1.transformer_blocks.0.to_k.bias",
            "first_stage_model.decoder.mid.attn_1.k.weight": "decoder.blocks.1.transformer_blocks.0.to_k.weight",
            "first_stage_model.decoder.mid.attn_1.norm.bias": "decoder.blocks.1.norm.bias",
            "first_stage_model.decoder.mid.attn_1.norm.weight": "decoder.blocks.1.norm.weight",
            "first_stage_model.decoder.mid.attn_1.proj_out.bias": "decoder.blocks.1.transformer_blocks.0.to_out.bias",
            "first_stage_model.decoder.mid.attn_1.proj_out.weight": "decoder.blocks.1.transformer_blocks.0.to_out.weight",
            "first_stage_model.decoder.mid.attn_1.q.bias": "decoder.blocks.1.transformer_blocks.0.to_q.bias",
            "first_stage_model.decoder.mid.attn_1.q.weight": "decoder.blocks.1.transformer_blocks.0.to_q.weight",
            "first_stage_model.decoder.mid.attn_1.v.bias": "decoder.blocks.1.transformer_blocks.0.to_v.bias",
            "first_stage_model.decoder.mid.attn_1.v.weight": "decoder.blocks.1.transformer_blocks.0.to_v.weight",
            "first_stage_model.decoder.mid.block_1.conv1.bias": "decoder.blocks.0.conv1.bias",
            "first_stage_model.decoder.mid.block_1.conv1.weight": "decoder.blocks.0.conv1.weight",
            "first_stage_model.decoder.mid.block_1.conv2.bias": "decoder.blocks.0.conv2.bias",
            "first_stage_model.decoder.mid.block_1.conv2.weight": "decoder.blocks.0.conv2.weight",
            "first_stage_model.decoder.mid.block_1.norm1.bias": "decoder.blocks.0.norm1.bias",
            "first_stage_model.decoder.mid.block_1.norm1.weight": "decoder.blocks.0.norm1.weight",
            "first_stage_model.decoder.mid.block_1.norm2.bias": "decoder.blocks.0.norm2.bias",
            "first_stage_model.decoder.mid.block_1.norm2.weight": "decoder.blocks.0.norm2.weight",
            "first_stage_model.decoder.mid.block_2.conv1.bias": "decoder.blocks.2.conv1.bias",
            "first_stage_model.decoder.mid.block_2.conv1.weight": "decoder.blocks.2.conv1.weight",
            "first_stage_model.decoder.mid.block_2.conv2.bias": "decoder.blocks.2.conv2.bias",
            "first_stage_model.decoder.mid.block_2.conv2.weight": "decoder.blocks.2.conv2.weight",
            "first_stage_model.decoder.mid.block_2.norm1.bias": "decoder.blocks.2.norm1.bias",
            "first_stage_model.decoder.mid.block_2.norm1.weight": "decoder.blocks.2.norm1.weight",
            "first_stage_model.decoder.mid.block_2.norm2.bias": "decoder.blocks.2.norm2.bias",
            "first_stage_model.decoder.mid.block_2.norm2.weight": "decoder.blocks.2.norm2.weight",
            "first_stage_model.decoder.norm_out.bias": "decoder.conv_norm_out.bias",
            "first_stage_model.decoder.norm_out.weight": "decoder.conv_norm_out.weight",
            "first_stage_model.decoder.up.0.block.0.conv1.bias": "decoder.blocks.15.conv1.bias",
            "first_stage_model.decoder.up.0.block.0.conv1.weight": "decoder.blocks.15.conv1.weight",
            "first_stage_model.decoder.up.0.block.0.conv2.bias": "decoder.blocks.15.conv2.bias",
            "first_stage_model.decoder.up.0.block.0.conv2.weight": "decoder.blocks.15.conv2.weight",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.bias": "decoder.blocks.15.conv_shortcut.bias",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.weight": "decoder.blocks.15.conv_shortcut.weight",
            "first_stage_model.decoder.up.0.block.0.norm1.bias": "decoder.blocks.15.norm1.bias",
            "first_stage_model.decoder.up.0.block.0.norm1.weight": "decoder.blocks.15.norm1.weight",
            "first_stage_model.decoder.up.0.block.0.norm2.bias": "decoder.blocks.15.norm2.bias",
            "first_stage_model.decoder.up.0.block.0.norm2.weight": "decoder.blocks.15.norm2.weight",
            "first_stage_model.decoder.up.0.block.1.conv1.bias": "decoder.blocks.16.conv1.bias",
            "first_stage_model.decoder.up.0.block.1.conv1.weight": "decoder.blocks.16.conv1.weight",
            "first_stage_model.decoder.up.0.block.1.conv2.bias": "decoder.blocks.16.conv2.bias",
            "first_stage_model.decoder.up.0.block.1.conv2.weight": "decoder.blocks.16.conv2.weight",
            "first_stage_model.decoder.up.0.block.1.norm1.bias": "decoder.blocks.16.norm1.bias",
            "first_stage_model.decoder.up.0.block.1.norm1.weight": "decoder.blocks.16.norm1.weight",
            "first_stage_model.decoder.up.0.block.1.norm2.bias": "decoder.blocks.16.norm2.bias",
            "first_stage_model.decoder.up.0.block.1.norm2.weight": "decoder.blocks.16.norm2.weight",
            "first_stage_model.decoder.up.0.block.2.conv1.bias": "decoder.blocks.17.conv1.bias",
            "first_stage_model.decoder.up.0.block.2.conv1.weight": "decoder.blocks.17.conv1.weight",
            "first_stage_model.decoder.up.0.block.2.conv2.bias": "decoder.blocks.17.conv2.bias",
            "first_stage_model.decoder.up.0.block.2.conv2.weight": "decoder.blocks.17.conv2.weight",
            "first_stage_model.decoder.up.0.block.2.norm1.bias": "decoder.blocks.17.norm1.bias",
            "first_stage_model.decoder.up.0.block.2.norm1.weight": "decoder.blocks.17.norm1.weight",
            "first_stage_model.decoder.up.0.block.2.norm2.bias": "decoder.blocks.17.norm2.bias",
            "first_stage_model.decoder.up.0.block.2.norm2.weight": "decoder.blocks.17.norm2.weight",
            "first_stage_model.decoder.up.1.block.0.conv1.bias": "decoder.blocks.11.conv1.bias",
            "first_stage_model.decoder.up.1.block.0.conv1.weight": "decoder.blocks.11.conv1.weight",
            "first_stage_model.decoder.up.1.block.0.conv2.bias": "decoder.blocks.11.conv2.bias",
            "first_stage_model.decoder.up.1.block.0.conv2.weight": "decoder.blocks.11.conv2.weight",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.bias": "decoder.blocks.11.conv_shortcut.bias",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.weight": "decoder.blocks.11.conv_shortcut.weight",
            "first_stage_model.decoder.up.1.block.0.norm1.bias": "decoder.blocks.11.norm1.bias",
            "first_stage_model.decoder.up.1.block.0.norm1.weight": "decoder.blocks.11.norm1.weight",
            "first_stage_model.decoder.up.1.block.0.norm2.bias": "decoder.blocks.11.norm2.bias",
            "first_stage_model.decoder.up.1.block.0.norm2.weight": "decoder.blocks.11.norm2.weight",
            "first_stage_model.decoder.up.1.block.1.conv1.bias": "decoder.blocks.12.conv1.bias",
            "first_stage_model.decoder.up.1.block.1.conv1.weight": "decoder.blocks.12.conv1.weight",
            "first_stage_model.decoder.up.1.block.1.conv2.bias": "decoder.blocks.12.conv2.bias",
            "first_stage_model.decoder.up.1.block.1.conv2.weight": "decoder.blocks.12.conv2.weight",
            "first_stage_model.decoder.up.1.block.1.norm1.bias": "decoder.blocks.12.norm1.bias",
            "first_stage_model.decoder.up.1.block.1.norm1.weight": "decoder.blocks.12.norm1.weight",
            "first_stage_model.decoder.up.1.block.1.norm2.bias": "decoder.blocks.12.norm2.bias",
            "first_stage_model.decoder.up.1.block.1.norm2.weight": "decoder.blocks.12.norm2.weight",
            "first_stage_model.decoder.up.1.block.2.conv1.bias": "decoder.blocks.13.conv1.bias",
            "first_stage_model.decoder.up.1.block.2.conv1.weight": "decoder.blocks.13.conv1.weight",
            "first_stage_model.decoder.up.1.block.2.conv2.bias": "decoder.blocks.13.conv2.bias",
            "first_stage_model.decoder.up.1.block.2.conv2.weight": "decoder.blocks.13.conv2.weight",
            "first_stage_model.decoder.up.1.block.2.norm1.bias": "decoder.blocks.13.norm1.bias",
            "first_stage_model.decoder.up.1.block.2.norm1.weight": "decoder.blocks.13.norm1.weight",
            "first_stage_model.decoder.up.1.block.2.norm2.bias": "decoder.blocks.13.norm2.bias",
            "first_stage_model.decoder.up.1.block.2.norm2.weight": "decoder.blocks.13.norm2.weight",
            "first_stage_model.decoder.up.1.upsample.conv.bias": "decoder.blocks.14.conv.bias",
            "first_stage_model.decoder.up.1.upsample.conv.weight": "decoder.blocks.14.conv.weight",
            "first_stage_model.decoder.up.2.block.0.conv1.bias": "decoder.blocks.7.conv1.bias",
            "first_stage_model.decoder.up.2.block.0.conv1.weight": "decoder.blocks.7.conv1.weight",
            "first_stage_model.decoder.up.2.block.0.conv2.bias": "decoder.blocks.7.conv2.bias",
            "first_stage_model.decoder.up.2.block.0.conv2.weight": "decoder.blocks.7.conv2.weight",
            "first_stage_model.decoder.up.2.block.0.norm1.bias": "decoder.blocks.7.norm1.bias",
            "first_stage_model.decoder.up.2.block.0.norm1.weight": "decoder.blocks.7.norm1.weight",
            "first_stage_model.decoder.up.2.block.0.norm2.bias": "decoder.blocks.7.norm2.bias",
            "first_stage_model.decoder.up.2.block.0.norm2.weight": "decoder.blocks.7.norm2.weight",
            "first_stage_model.decoder.up.2.block.1.conv1.bias": "decoder.blocks.8.conv1.bias",
            "first_stage_model.decoder.up.2.block.1.conv1.weight": "decoder.blocks.8.conv1.weight",
            "first_stage_model.decoder.up.2.block.1.conv2.bias": "decoder.blocks.8.conv2.bias",
            "first_stage_model.decoder.up.2.block.1.conv2.weight": "decoder.blocks.8.conv2.weight",
            "first_stage_model.decoder.up.2.block.1.norm1.bias": "decoder.blocks.8.norm1.bias",
            "first_stage_model.decoder.up.2.block.1.norm1.weight": "decoder.blocks.8.norm1.weight",
            "first_stage_model.decoder.up.2.block.1.norm2.bias": "decoder.blocks.8.norm2.bias",
            "first_stage_model.decoder.up.2.block.1.norm2.weight": "decoder.blocks.8.norm2.weight",
            "first_stage_model.decoder.up.2.block.2.conv1.bias": "decoder.blocks.9.conv1.bias",
            "first_stage_model.decoder.up.2.block.2.conv1.weight": "decoder.blocks.9.conv1.weight",
            "first_stage_model.decoder.up.2.block.2.conv2.bias": "decoder.blocks.9.conv2.bias",
            "first_stage_model.decoder.up.2.block.2.conv2.weight": "decoder.blocks.9.conv2.weight",
            "first_stage_model.decoder.up.2.block.2.norm1.bias": "decoder.blocks.9.norm1.bias",
            "first_stage_model.decoder.up.2.block.2.norm1.weight": "decoder.blocks.9.norm1.weight",
            "first_stage_model.decoder.up.2.block.2.norm2.bias": "decoder.blocks.9.norm2.bias",
            "first_stage_model.decoder.up.2.block.2.norm2.weight": "decoder.blocks.9.norm2.weight",
            "first_stage_model.decoder.up.2.upsample.conv.bias": "decoder.blocks.10.conv.bias",
            "first_stage_model.decoder.up.2.upsample.conv.weight": "decoder.blocks.10.conv.weight",
            "first_stage_model.decoder.up.3.block.0.conv1.bias": "decoder.blocks.3.conv1.bias",
            "first_stage_model.decoder.up.3.block.0.conv1.weight": "decoder.blocks.3.conv1.weight",
            "first_stage_model.decoder.up.3.block.0.conv2.bias": "decoder.blocks.3.conv2.bias",
            "first_stage_model.decoder.up.3.block.0.conv2.weight": "decoder.blocks.3.conv2.weight",
            "first_stage_model.decoder.up.3.block.0.norm1.bias": "decoder.blocks.3.norm1.bias",
            "first_stage_model.decoder.up.3.block.0.norm1.weight": "decoder.blocks.3.norm1.weight",
            "first_stage_model.decoder.up.3.block.0.norm2.bias": "decoder.blocks.3.norm2.bias",
            "first_stage_model.decoder.up.3.block.0.norm2.weight": "decoder.blocks.3.norm2.weight",
            "first_stage_model.decoder.up.3.block.1.conv1.bias": "decoder.blocks.4.conv1.bias",
            "first_stage_model.decoder.up.3.block.1.conv1.weight": "decoder.blocks.4.conv1.weight",
            "first_stage_model.decoder.up.3.block.1.conv2.bias": "decoder.blocks.4.conv2.bias",
            "first_stage_model.decoder.up.3.block.1.conv2.weight": "decoder.blocks.4.conv2.weight",
            "first_stage_model.decoder.up.3.block.1.norm1.bias": "decoder.blocks.4.norm1.bias",
            "first_stage_model.decoder.up.3.block.1.norm1.weight": "decoder.blocks.4.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.norm2.bias": "decoder.blocks.4.norm2.bias",
            "first_stage_model.decoder.up.3.block.1.norm2.weight": "decoder.blocks.4.norm2.weight",
            "first_stage_model.decoder.up.3.block.2.conv1.bias": "decoder.blocks.5.conv1.bias",
            "first_stage_model.decoder.up.3.block.2.conv1.weight": "decoder.blocks.5.conv1.weight",
            "first_stage_model.decoder.up.3.block.2.conv2.bias": "decoder.blocks.5.conv2.bias",
            "first_stage_model.decoder.up.3.block.2.conv2.weight": "decoder.blocks.5.conv2.weight",
            "first_stage_model.decoder.up.3.block.2.norm1.bias": "decoder.blocks.5.norm1.bias",
            "first_stage_model.decoder.up.3.block.2.norm1.weight": "decoder.blocks.5.norm1.weight",
            "first_stage_model.decoder.up.3.block.2.norm2.bias": "decoder.blocks.5.norm2.bias",
            "first_stage_model.decoder.up.3.block.2.norm2.weight": "decoder.blocks.5.norm2.weight",
            "first_stage_model.decoder.up.3.upsample.conv.bias": "decoder.blocks.6.conv.bias",
            "first_stage_model.decoder.up.3.upsample.conv.weight": "decoder.blocks.6.conv.weight",
            "first_stage_model.post_quant_conv.bias": "decoder.post_quant_conv.bias",
            "first_stage_model.post_quant_conv.weight": "decoder.post_quant_conv.weight",
            # encoder
            "first_stage_model.encoder.conv_in.bias": "encoder.conv_in.bias",
            "first_stage_model.encoder.conv_in.weight": "encoder.conv_in.weight",
            "first_stage_model.encoder.conv_out.bias": "encoder.conv_out.bias",
            "first_stage_model.encoder.conv_out.weight": "encoder.conv_out.weight",
            "first_stage_model.encoder.down.0.block.0.conv1.bias": "encoder.blocks.0.conv1.bias",
            "first_stage_model.encoder.down.0.block.0.conv1.weight": "encoder.blocks.0.conv1.weight",
            "first_stage_model.encoder.down.0.block.0.conv2.bias": "encoder.blocks.0.conv2.bias",
            "first_stage_model.encoder.down.0.block.0.conv2.weight": "encoder.blocks.0.conv2.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.bias": "encoder.blocks.0.norm1.bias",
            "first_stage_model.encoder.down.0.block.0.norm1.weight": "encoder.blocks.0.norm1.weight",
            "first_stage_model.encoder.down.0.block.0.norm2.bias": "encoder.blocks.0.norm2.bias",
            "first_stage_model.encoder.down.0.block.0.norm2.weight": "encoder.blocks.0.norm2.weight",
            "first_stage_model.encoder.down.0.block.1.conv1.bias": "encoder.blocks.1.conv1.bias",
            "first_stage_model.encoder.down.0.block.1.conv1.weight": "encoder.blocks.1.conv1.weight",
            "first_stage_model.encoder.down.0.block.1.conv2.bias": "encoder.blocks.1.conv2.bias",
            "first_stage_model.encoder.down.0.block.1.conv2.weight": "encoder.blocks.1.conv2.weight",
            "first_stage_model.encoder.down.0.block.1.norm1.bias": "encoder.blocks.1.norm1.bias",
            "first_stage_model.encoder.down.0.block.1.norm1.weight": "encoder.blocks.1.norm1.weight",
            "first_stage_model.encoder.down.0.block.1.norm2.bias": "encoder.blocks.1.norm2.bias",
            "first_stage_model.encoder.down.0.block.1.norm2.weight": "encoder.blocks.1.norm2.weight",
            "first_stage_model.encoder.down.0.downsample.conv.bias": "encoder.blocks.2.conv.bias",
            "first_stage_model.encoder.down.0.downsample.conv.weight": "encoder.blocks.2.conv.weight",
            "first_stage_model.encoder.down.1.block.0.conv1.bias": "encoder.blocks.3.conv1.bias",
            "first_stage_model.encoder.down.1.block.0.conv1.weight": "encoder.blocks.3.conv1.weight",
            "first_stage_model.encoder.down.1.block.0.conv2.bias": "encoder.blocks.3.conv2.bias",
            "first_stage_model.encoder.down.1.block.0.conv2.weight": "encoder.blocks.3.conv2.weight",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.bias": "encoder.blocks.3.conv_shortcut.bias",
            "first_stage_model.encoder.down.1.block.0.nin_shortcut.weight": "encoder.blocks.3.conv_shortcut.weight",
            "first_stage_model.encoder.down.1.block.0.norm1.bias": "encoder.blocks.3.norm1.bias",
            "first_stage_model.encoder.down.1.block.0.norm1.weight": "encoder.blocks.3.norm1.weight",
            "first_stage_model.encoder.down.1.block.0.norm2.bias": "encoder.blocks.3.norm2.bias",
            "first_stage_model.encoder.down.1.block.0.norm2.weight": "encoder.blocks.3.norm2.weight",
            "first_stage_model.encoder.down.1.block.1.conv1.bias": "encoder.blocks.4.conv1.bias",
            "first_stage_model.encoder.down.1.block.1.conv1.weight": "encoder.blocks.4.conv1.weight",
            "first_stage_model.encoder.down.1.block.1.conv2.bias": "encoder.blocks.4.conv2.bias",
            "first_stage_model.encoder.down.1.block.1.conv2.weight": "encoder.blocks.4.conv2.weight",
            "first_stage_model.encoder.down.1.block.1.norm1.bias": "encoder.blocks.4.norm1.bias",
            "first_stage_model.encoder.down.1.block.1.norm1.weight": "encoder.blocks.4.norm1.weight",
            "first_stage_model.encoder.down.1.block.1.norm2.bias": "encoder.blocks.4.norm2.bias",
            "first_stage_model.encoder.down.1.block.1.norm2.weight": "encoder.blocks.4.norm2.weight",
            "first_stage_model.encoder.down.1.downsample.conv.bias": "encoder.blocks.5.conv.bias",
            "first_stage_model.encoder.down.1.downsample.conv.weight": "encoder.blocks.5.conv.weight",
            "first_stage_model.encoder.down.2.block.0.conv1.bias": "encoder.blocks.6.conv1.bias",
            "first_stage_model.encoder.down.2.block.0.conv1.weight": "encoder.blocks.6.conv1.weight",
            "first_stage_model.encoder.down.2.block.0.conv2.bias": "encoder.blocks.6.conv2.bias",
            "first_stage_model.encoder.down.2.block.0.conv2.weight": "encoder.blocks.6.conv2.weight",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.bias": "encoder.blocks.6.conv_shortcut.bias",
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight": "encoder.blocks.6.conv_shortcut.weight",
            "first_stage_model.encoder.down.2.block.0.norm1.bias": "encoder.blocks.6.norm1.bias",
            "first_stage_model.encoder.down.2.block.0.norm1.weight": "encoder.blocks.6.norm1.weight",
            "first_stage_model.encoder.down.2.block.0.norm2.bias": "encoder.blocks.6.norm2.bias",
            "first_stage_model.encoder.down.2.block.0.norm2.weight": "encoder.blocks.6.norm2.weight",
            "first_stage_model.encoder.down.2.block.1.conv1.bias": "encoder.blocks.7.conv1.bias",
            "first_stage_model.encoder.down.2.block.1.conv1.weight": "encoder.blocks.7.conv1.weight",
            "first_stage_model.encoder.down.2.block.1.conv2.bias": "encoder.blocks.7.conv2.bias",
            "first_stage_model.encoder.down.2.block.1.conv2.weight": "encoder.blocks.7.conv2.weight",
            "first_stage_model.encoder.down.2.block.1.norm1.bias": "encoder.blocks.7.norm1.bias",
            "first_stage_model.encoder.down.2.block.1.norm1.weight": "encoder.blocks.7.norm1.weight",
            "first_stage_model.encoder.down.2.block.1.norm2.bias": "encoder.blocks.7.norm2.bias",
            "first_stage_model.encoder.down.2.block.1.norm2.weight": "encoder.blocks.7.norm2.weight",
            "first_stage_model.encoder.down.2.downsample.conv.bias": "encoder.blocks.8.conv.bias",
            "first_stage_model.encoder.down.2.downsample.conv.weight": "encoder.blocks.8.conv.weight",
            "first_stage_model.encoder.down.3.block.0.conv1.bias": "encoder.blocks.9.conv1.bias",
            "first_stage_model.encoder.down.3.block.0.conv1.weight": "encoder.blocks.9.conv1.weight",
            "first_stage_model.encoder.down.3.block.0.conv2.bias": "encoder.blocks.9.conv2.bias",
            "first_stage_model.encoder.down.3.block.0.conv2.weight": "encoder.blocks.9.conv2.weight",
            "first_stage_model.encoder.down.3.block.0.norm1.bias": "encoder.blocks.9.norm1.bias",
            "first_stage_model.encoder.down.3.block.0.norm1.weight": "encoder.blocks.9.norm1.weight",
            "first_stage_model.encoder.down.3.block.0.norm2.bias": "encoder.blocks.9.norm2.bias",
            "first_stage_model.encoder.down.3.block.0.norm2.weight": "encoder.blocks.9.norm2.weight",
            "first_stage_model.encoder.down.3.block.1.conv1.bias": "encoder.blocks.10.conv1.bias",
            "first_stage_model.encoder.down.3.block.1.conv1.weight": "encoder.blocks.10.conv1.weight",
            "first_stage_model.encoder.down.3.block.1.conv2.bias": "encoder.blocks.10.conv2.bias",
            "first_stage_model.encoder.down.3.block.1.conv2.weight": "encoder.blocks.10.conv2.weight",
            "first_stage_model.encoder.down.3.block.1.norm1.bias": "encoder.blocks.10.norm1.bias",
            "first_stage_model.encoder.down.3.block.1.norm1.weight": "encoder.blocks.10.norm1.weight",
            "first_stage_model.encoder.down.3.block.1.norm2.bias": "encoder.blocks.10.norm2.bias",
            "first_stage_model.encoder.down.3.block.1.norm2.weight": "encoder.blocks.10.norm2.weight",
            "first_stage_model.encoder.mid.attn_1.k.bias": "encoder.blocks.12.transformer_blocks.0.to_k.bias",
            "first_stage_model.encoder.mid.attn_1.k.weight": "encoder.blocks.12.transformer_blocks.0.to_k.weight",
            "first_stage_model.encoder.mid.attn_1.norm.bias": "encoder.blocks.12.norm.bias",
            "first_stage_model.encoder.mid.attn_1.norm.weight": "encoder.blocks.12.norm.weight",
            "first_stage_model.encoder.mid.attn_1.proj_out.bias": "encoder.blocks.12.transformer_blocks.0.to_out.bias",
            "first_stage_model.encoder.mid.attn_1.proj_out.weight": "encoder.blocks.12.transformer_blocks.0.to_out.weight",
            "first_stage_model.encoder.mid.attn_1.q.bias": "encoder.blocks.12.transformer_blocks.0.to_q.bias",
            "first_stage_model.encoder.mid.attn_1.q.weight": "encoder.blocks.12.transformer_blocks.0.to_q.weight",
            "first_stage_model.encoder.mid.attn_1.v.bias": "encoder.blocks.12.transformer_blocks.0.to_v.bias",
            "first_stage_model.encoder.mid.attn_1.v.weight": "encoder.blocks.12.transformer_blocks.0.to_v.weight",
            "first_stage_model.encoder.mid.block_1.conv1.bias": "encoder.blocks.11.conv1.bias",
            "first_stage_model.encoder.mid.block_1.conv1.weight": "encoder.blocks.11.conv1.weight",
            "first_stage_model.encoder.mid.block_1.conv2.bias": "encoder.blocks.11.conv2.bias",
            "first_stage_model.encoder.mid.block_1.conv2.weight": "encoder.blocks.11.conv2.weight",
            "first_stage_model.encoder.mid.block_1.norm1.bias": "encoder.blocks.11.norm1.bias",
            "first_stage_model.encoder.mid.block_1.norm1.weight": "encoder.blocks.11.norm1.weight",
            "first_stage_model.encoder.mid.block_1.norm2.bias": "encoder.blocks.11.norm2.bias",
            "first_stage_model.encoder.mid.block_1.norm2.weight": "encoder.blocks.11.norm2.weight",
            "first_stage_model.encoder.mid.block_2.conv1.bias": "encoder.blocks.13.conv1.bias",
            "first_stage_model.encoder.mid.block_2.conv1.weight": "encoder.blocks.13.conv1.weight",
            "first_stage_model.encoder.mid.block_2.conv2.bias": "encoder.blocks.13.conv2.bias",
            "first_stage_model.encoder.mid.block_2.conv2.weight": "encoder.blocks.13.conv2.weight",
            "first_stage_model.encoder.mid.block_2.norm1.bias": "encoder.blocks.13.norm1.bias",
            "first_stage_model.encoder.mid.block_2.norm1.weight": "encoder.blocks.13.norm1.weight",
            "first_stage_model.encoder.mid.block_2.norm2.bias": "encoder.blocks.13.norm2.bias",
            "first_stage_model.encoder.mid.block_2.norm2.weight": "encoder.blocks.13.norm2.weight",
            "first_stage_model.encoder.norm_out.bias": "encoder.conv_norm_out.bias",
            "first_stage_model.encoder.norm_out.weight": "encoder.conv_norm_out.weight",
            "first_stage_model.quant_conv.bias": "encoder.quant_conv.bias",
            "first_stage_model.quant_conv.weight": "encoder.quant_conv.weight"
        }
        new_state_dict = {}
        for key, param in state_dict.items():
            if key not in rename_dict:
                continue
            new_key = rename_dict[key]
            if "transformer_blocks" in new_key:
                param = param.squeeze()
            new_state_dict[new_key] = param
        return new_state_dict

    def _filter(self, state_dict: Dict[str, torch.Tensor]):
        new_state_dict = {}
        for key, param in state_dict.items():
            if self.has_encoder and self.has_decoder:
                new_state_dict[key] = param
            elif self.has_encoder and key.startswith("encoder."):
                new_state_dict[key[len("encoder."):]] = param
            elif self.has_decoder and key.startswith("decoder."):
                new_state_dict[key[len("decoder."):]] = param
        return new_state_dict

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.has_decoder or self.has_encoder, "Either decoder or encoder must be present"
        if "first_stage_model.decoder.conv_in.weight" in state_dict or \
                "first_stage_model.encoder.conv_in.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return self._filter(state_dict)


class VAEAttentionBlock(nn.Module):
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5,
                 device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True,
                                 device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True,
                use_xformers=True,
                device=device,
                dtype=dtype
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


class VAEDecoder(PreTrainedModel):
    converter = VAEStateDictConverter(has_decoder=True)

    def __init__(self,
                 latent_channels: int = 4,
                 scaling_factor: float = 0.18215,
                 shift_factor: float = 0,
                 use_post_quant_conv: bool = True,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_post_quant_conv = use_post_quant_conv
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1,
                                             device=device, dtype=dtype)
        self.conv_in = nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            UpSampler(512, device=device, dtype=dtype),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            UpSampler(512, device=device, dtype=dtype),
            # UpDecoderBlock2D
            ResnetBlock(512, 256, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
            UpSampler(256, device=device, dtype=dtype),
            # UpDecoderBlock2D
            ResnetBlock(256, 128, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
        ])

        self.conv_norm_out = nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6, device=device, dtype=dtype)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1, device=device, dtype=dtype)

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
        sample = sample / self.scaling_factor + self.shift_factor
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

    @classmethod
    def from_state_dict(cls,
                        state_dict: Dict[str, torch.Tensor],
                        device: str,
                        dtype: torch.dtype,
                        latent_channels: int = 4,
                        scaling_factor: float = 0.18215,
                        shift_factor: float = 0,
                        use_post_quant_conv: bool = True,
                        ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(
                cls,
                latent_channels=latent_channels,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                use_post_quant_conv=use_post_quant_conv,
                device=device,
                dtype=dtype
            )
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str | os.PathLike, **kwargs):
        raise NotImplementedError()


class VAEEncoder(PreTrainedModel):
    converter = VAEStateDictConverter(has_encoder=True)

    def __init__(self,
                 latent_channels: int = 4,
                 scaling_factor: float = 0.18215,
                 shift_factor: float = 0,
                 use_quant_conv: bool = True,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.use_quant_conv = use_quant_conv
        if use_quant_conv:
            self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1,
                                        device=device, dtype=dtype)
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(128, 128, eps=1e-6, device=device, dtype=dtype),
            DownSampler(128, padding=0, extra_padding=True, device=device, dtype=dtype),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(256, 256, eps=1e-6, device=device, dtype=dtype),
            DownSampler(256, padding=0, extra_padding=True, device=device, dtype=dtype),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            DownSampler(512, padding=0, extra_padding=True, device=device, dtype=dtype),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, device=device, dtype=dtype),
            ResnetBlock(512, 512, eps=1e-6, device=device, dtype=dtype),
        ])

        self.conv_norm_out = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, device=device, dtype=dtype)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 2 * latent_channels, kernel_size=3, padding=1, device=device, dtype=dtype)

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
            sample_batch = rearrange(sample[:, :, i:j], "B C T H W -> (B T) C H W")

            hidden_states_batch = self(sample_batch)
            hidden_states_batch = rearrange(hidden_states_batch, "(B T) C H W -> B C T H W", B=B)

            hidden_states.append(hidden_states_batch)

        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states

    @classmethod
    def from_state_dict(cls,
                        state_dict: Dict[str, torch.Tensor],
                        device: str,
                        dtype: torch.dtype,
                        latent_channels: int = 4,
                        scaling_factor: float = 0.18215,
                        shift_factor: float = 0,
                        use_quant_conv: bool = True,
                        ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(
                cls,
                latent_channels=latent_channels,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                use_quant_conv=use_quant_conv,
                device=device,
                dtype=dtype
            )
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str | os.PathLike, **kwargs):
        raise NotImplementedError()


class VAE(PreTrainedModel):
    converter = VAEStateDictConverter(has_encoder=True, has_decoder=True)

    def __init__(self,
                 latent_channels: int = 4,
                 scaling_factor: float = 0.18215,
                 shift_factor: float = 0,
                 use_quant_conv: bool = True,
                 use_post_quant_conv: bool = True,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__()
        self.encoder = VAEEncoder(latent_channels=latent_channels, scaling_factor=scaling_factor,
                                  shift_factor=shift_factor, use_quant_conv=use_quant_conv, device=device, dtype=dtype)
        self.decoder = VAEDecoder(latent_channels=latent_channels, scaling_factor=scaling_factor,
                                  shift_factor=shift_factor, use_post_quant_conv=use_post_quant_conv,
                                  device=device, dtype=dtype)

    def encode(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        return self.encoder(sample, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, **kwargs)

    def decode(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        return self.decoder(sample, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, **kwargs)

    @classmethod
    def from_state_dict(cls,
                        state_dict: Dict[str, torch.Tensor],
                        device: str,
                        dtype: torch.dtype,
                        latent_channels: int = 4,
                        scaling_factor: float = 0.18215,
                        shift_factor: float = 0,
                        use_quant_conv: bool = True,
                        use_post_quant_conv: bool = True,
                        ):
        with no_init_weights():
            model = torch.nn.utils.skip_init(
                cls,
                latent_channels=latent_channels,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                use_quant_conv=use_quant_conv,
                use_post_quant_conv=use_post_quant_conv,
                device=device,
                dtype=dtype
            )
        model.load_state_dict(state_dict)
        return model
