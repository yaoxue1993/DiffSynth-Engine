import torch
from typing import Dict

from diffsynth_engine.models.components.vae import VAEDecoder, VAEEncoder, VAEStateDictConverter
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class FluxVAEStateDictConverter(VAEStateDictConverter):
    def _from_civitai(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rename_dict = {
            # decoder
            "decoder.conv_in.bias": "decoder.conv_in.bias",
            "decoder.conv_in.weight": "decoder.conv_in.weight",
            "decoder.conv_out.bias": "decoder.conv_out.bias",
            "decoder.conv_out.weight": "decoder.conv_out.weight",
            "decoder.mid.attn_1.k.bias": "decoder.blocks.1.transformer_blocks.0.to_k.bias",
            "decoder.mid.attn_1.k.weight": "decoder.blocks.1.transformer_blocks.0.to_k.weight",
            "decoder.mid.attn_1.norm.bias": "decoder.blocks.1.norm.bias",
            "decoder.mid.attn_1.norm.weight": "decoder.blocks.1.norm.weight",
            "decoder.mid.attn_1.proj_out.bias": "decoder.blocks.1.transformer_blocks.0.to_out.bias",
            "decoder.mid.attn_1.proj_out.weight": "decoder.blocks.1.transformer_blocks.0.to_out.weight",
            "decoder.mid.attn_1.q.bias": "decoder.blocks.1.transformer_blocks.0.to_q.bias",
            "decoder.mid.attn_1.q.weight": "decoder.blocks.1.transformer_blocks.0.to_q.weight",
            "decoder.mid.attn_1.v.bias": "decoder.blocks.1.transformer_blocks.0.to_v.bias",
            "decoder.mid.attn_1.v.weight": "decoder.blocks.1.transformer_blocks.0.to_v.weight",
            "decoder.mid.block_1.conv1.bias": "decoder.blocks.0.conv1.bias",
            "decoder.mid.block_1.conv1.weight": "decoder.blocks.0.conv1.weight",
            "decoder.mid.block_1.conv2.bias": "decoder.blocks.0.conv2.bias",
            "decoder.mid.block_1.conv2.weight": "decoder.blocks.0.conv2.weight",
            "decoder.mid.block_1.norm1.bias": "decoder.blocks.0.norm1.bias",
            "decoder.mid.block_1.norm1.weight": "decoder.blocks.0.norm1.weight",
            "decoder.mid.block_1.norm2.bias": "decoder.blocks.0.norm2.bias",
            "decoder.mid.block_1.norm2.weight": "decoder.blocks.0.norm2.weight",
            "decoder.mid.block_2.conv1.bias": "decoder.blocks.2.conv1.bias",
            "decoder.mid.block_2.conv1.weight": "decoder.blocks.2.conv1.weight",
            "decoder.mid.block_2.conv2.bias": "decoder.blocks.2.conv2.bias",
            "decoder.mid.block_2.conv2.weight": "decoder.blocks.2.conv2.weight",
            "decoder.mid.block_2.norm1.bias": "decoder.blocks.2.norm1.bias",
            "decoder.mid.block_2.norm1.weight": "decoder.blocks.2.norm1.weight",
            "decoder.mid.block_2.norm2.bias": "decoder.blocks.2.norm2.bias",
            "decoder.mid.block_2.norm2.weight": "decoder.blocks.2.norm2.weight",
            "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
            "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
            "decoder.up.0.block.0.conv1.bias": "decoder.blocks.15.conv1.bias",
            "decoder.up.0.block.0.conv1.weight": "decoder.blocks.15.conv1.weight",
            "decoder.up.0.block.0.conv2.bias": "decoder.blocks.15.conv2.bias",
            "decoder.up.0.block.0.conv2.weight": "decoder.blocks.15.conv2.weight",
            "decoder.up.0.block.0.nin_shortcut.bias": "decoder.blocks.15.conv_shortcut.bias",
            "decoder.up.0.block.0.nin_shortcut.weight": "decoder.blocks.15.conv_shortcut.weight",
            "decoder.up.0.block.0.norm1.bias": "decoder.blocks.15.norm1.bias",
            "decoder.up.0.block.0.norm1.weight": "decoder.blocks.15.norm1.weight",
            "decoder.up.0.block.0.norm2.bias": "decoder.blocks.15.norm2.bias",
            "decoder.up.0.block.0.norm2.weight": "decoder.blocks.15.norm2.weight",
            "decoder.up.0.block.1.conv1.bias": "decoder.blocks.16.conv1.bias",
            "decoder.up.0.block.1.conv1.weight": "decoder.blocks.16.conv1.weight",
            "decoder.up.0.block.1.conv2.bias": "decoder.blocks.16.conv2.bias",
            "decoder.up.0.block.1.conv2.weight": "decoder.blocks.16.conv2.weight",
            "decoder.up.0.block.1.norm1.bias": "decoder.blocks.16.norm1.bias",
            "decoder.up.0.block.1.norm1.weight": "decoder.blocks.16.norm1.weight",
            "decoder.up.0.block.1.norm2.bias": "decoder.blocks.16.norm2.bias",
            "decoder.up.0.block.1.norm2.weight": "decoder.blocks.16.norm2.weight",
            "decoder.up.0.block.2.conv1.bias": "decoder.blocks.17.conv1.bias",
            "decoder.up.0.block.2.conv1.weight": "decoder.blocks.17.conv1.weight",
            "decoder.up.0.block.2.conv2.bias": "decoder.blocks.17.conv2.bias",
            "decoder.up.0.block.2.conv2.weight": "decoder.blocks.17.conv2.weight",
            "decoder.up.0.block.2.norm1.bias": "decoder.blocks.17.norm1.bias",
            "decoder.up.0.block.2.norm1.weight": "decoder.blocks.17.norm1.weight",
            "decoder.up.0.block.2.norm2.bias": "decoder.blocks.17.norm2.bias",
            "decoder.up.0.block.2.norm2.weight": "decoder.blocks.17.norm2.weight",
            "decoder.up.1.block.0.conv1.bias": "decoder.blocks.11.conv1.bias",
            "decoder.up.1.block.0.conv1.weight": "decoder.blocks.11.conv1.weight",
            "decoder.up.1.block.0.conv2.bias": "decoder.blocks.11.conv2.bias",
            "decoder.up.1.block.0.conv2.weight": "decoder.blocks.11.conv2.weight",
            "decoder.up.1.block.0.nin_shortcut.bias": "decoder.blocks.11.conv_shortcut.bias",
            "decoder.up.1.block.0.nin_shortcut.weight": "decoder.blocks.11.conv_shortcut.weight",
            "decoder.up.1.block.0.norm1.bias": "decoder.blocks.11.norm1.bias",
            "decoder.up.1.block.0.norm1.weight": "decoder.blocks.11.norm1.weight",
            "decoder.up.1.block.0.norm2.bias": "decoder.blocks.11.norm2.bias",
            "decoder.up.1.block.0.norm2.weight": "decoder.blocks.11.norm2.weight",
            "decoder.up.1.block.1.conv1.bias": "decoder.blocks.12.conv1.bias",
            "decoder.up.1.block.1.conv1.weight": "decoder.blocks.12.conv1.weight",
            "decoder.up.1.block.1.conv2.bias": "decoder.blocks.12.conv2.bias",
            "decoder.up.1.block.1.conv2.weight": "decoder.blocks.12.conv2.weight",
            "decoder.up.1.block.1.norm1.bias": "decoder.blocks.12.norm1.bias",
            "decoder.up.1.block.1.norm1.weight": "decoder.blocks.12.norm1.weight",
            "decoder.up.1.block.1.norm2.bias": "decoder.blocks.12.norm2.bias",
            "decoder.up.1.block.1.norm2.weight": "decoder.blocks.12.norm2.weight",
            "decoder.up.1.block.2.conv1.bias": "decoder.blocks.13.conv1.bias",
            "decoder.up.1.block.2.conv1.weight": "decoder.blocks.13.conv1.weight",
            "decoder.up.1.block.2.conv2.bias": "decoder.blocks.13.conv2.bias",
            "decoder.up.1.block.2.conv2.weight": "decoder.blocks.13.conv2.weight",
            "decoder.up.1.block.2.norm1.bias": "decoder.blocks.13.norm1.bias",
            "decoder.up.1.block.2.norm1.weight": "decoder.blocks.13.norm1.weight",
            "decoder.up.1.block.2.norm2.bias": "decoder.blocks.13.norm2.bias",
            "decoder.up.1.block.2.norm2.weight": "decoder.blocks.13.norm2.weight",
            "decoder.up.1.upsample.conv.bias": "decoder.blocks.14.conv.bias",
            "decoder.up.1.upsample.conv.weight": "decoder.blocks.14.conv.weight",
            "decoder.up.2.block.0.conv1.bias": "decoder.blocks.7.conv1.bias",
            "decoder.up.2.block.0.conv1.weight": "decoder.blocks.7.conv1.weight",
            "decoder.up.2.block.0.conv2.bias": "decoder.blocks.7.conv2.bias",
            "decoder.up.2.block.0.conv2.weight": "decoder.blocks.7.conv2.weight",
            "decoder.up.2.block.0.norm1.bias": "decoder.blocks.7.norm1.bias",
            "decoder.up.2.block.0.norm1.weight": "decoder.blocks.7.norm1.weight",
            "decoder.up.2.block.0.norm2.bias": "decoder.blocks.7.norm2.bias",
            "decoder.up.2.block.0.norm2.weight": "decoder.blocks.7.norm2.weight",
            "decoder.up.2.block.1.conv1.bias": "decoder.blocks.8.conv1.bias",
            "decoder.up.2.block.1.conv1.weight": "decoder.blocks.8.conv1.weight",
            "decoder.up.2.block.1.conv2.bias": "decoder.blocks.8.conv2.bias",
            "decoder.up.2.block.1.conv2.weight": "decoder.blocks.8.conv2.weight",
            "decoder.up.2.block.1.norm1.bias": "decoder.blocks.8.norm1.bias",
            "decoder.up.2.block.1.norm1.weight": "decoder.blocks.8.norm1.weight",
            "decoder.up.2.block.1.norm2.bias": "decoder.blocks.8.norm2.bias",
            "decoder.up.2.block.1.norm2.weight": "decoder.blocks.8.norm2.weight",
            "decoder.up.2.block.2.conv1.bias": "decoder.blocks.9.conv1.bias",
            "decoder.up.2.block.2.conv1.weight": "decoder.blocks.9.conv1.weight",
            "decoder.up.2.block.2.conv2.bias": "decoder.blocks.9.conv2.bias",
            "decoder.up.2.block.2.conv2.weight": "decoder.blocks.9.conv2.weight",
            "decoder.up.2.block.2.norm1.bias": "decoder.blocks.9.norm1.bias",
            "decoder.up.2.block.2.norm1.weight": "decoder.blocks.9.norm1.weight",
            "decoder.up.2.block.2.norm2.bias": "decoder.blocks.9.norm2.bias",
            "decoder.up.2.block.2.norm2.weight": "decoder.blocks.9.norm2.weight",
            "decoder.up.2.upsample.conv.bias": "decoder.blocks.10.conv.bias",
            "decoder.up.2.upsample.conv.weight": "decoder.blocks.10.conv.weight",
            "decoder.up.3.block.0.conv1.bias": "decoder.blocks.3.conv1.bias",
            "decoder.up.3.block.0.conv1.weight": "decoder.blocks.3.conv1.weight",
            "decoder.up.3.block.0.conv2.bias": "decoder.blocks.3.conv2.bias",
            "decoder.up.3.block.0.conv2.weight": "decoder.blocks.3.conv2.weight",
            "decoder.up.3.block.0.norm1.bias": "decoder.blocks.3.norm1.bias",
            "decoder.up.3.block.0.norm1.weight": "decoder.blocks.3.norm1.weight",
            "decoder.up.3.block.0.norm2.bias": "decoder.blocks.3.norm2.bias",
            "decoder.up.3.block.0.norm2.weight": "decoder.blocks.3.norm2.weight",
            "decoder.up.3.block.1.conv1.bias": "decoder.blocks.4.conv1.bias",
            "decoder.up.3.block.1.conv1.weight": "decoder.blocks.4.conv1.weight",
            "decoder.up.3.block.1.conv2.bias": "decoder.blocks.4.conv2.bias",
            "decoder.up.3.block.1.conv2.weight": "decoder.blocks.4.conv2.weight",
            "decoder.up.3.block.1.norm1.bias": "decoder.blocks.4.norm1.bias",
            "decoder.up.3.block.1.norm1.weight": "decoder.blocks.4.norm1.weight",
            "decoder.up.3.block.1.norm2.bias": "decoder.blocks.4.norm2.bias",
            "decoder.up.3.block.1.norm2.weight": "decoder.blocks.4.norm2.weight",
            "decoder.up.3.block.2.conv1.bias": "decoder.blocks.5.conv1.bias",
            "decoder.up.3.block.2.conv1.weight": "decoder.blocks.5.conv1.weight",
            "decoder.up.3.block.2.conv2.bias": "decoder.blocks.5.conv2.bias",
            "decoder.up.3.block.2.conv2.weight": "decoder.blocks.5.conv2.weight",
            "decoder.up.3.block.2.norm1.bias": "decoder.blocks.5.norm1.bias",
            "decoder.up.3.block.2.norm1.weight": "decoder.blocks.5.norm1.weight",
            "decoder.up.3.block.2.norm2.bias": "decoder.blocks.5.norm2.bias",
            "decoder.up.3.block.2.norm2.weight": "decoder.blocks.5.norm2.weight",
            "decoder.up.3.upsample.conv.bias": "decoder.blocks.6.conv.bias",
            "decoder.up.3.upsample.conv.weight": "decoder.blocks.6.conv.weight",
            # encoder
            "encoder.conv_in.bias": "encoder.conv_in.bias",
            "encoder.conv_in.weight": "encoder.conv_in.weight",
            "encoder.conv_out.bias": "encoder.conv_out.bias",
            "encoder.conv_out.weight": "encoder.conv_out.weight",
            "encoder.down.0.block.0.conv1.bias": "encoder.blocks.0.conv1.bias",
            "encoder.down.0.block.0.conv1.weight": "encoder.blocks.0.conv1.weight",
            "encoder.down.0.block.0.conv2.bias": "encoder.blocks.0.conv2.bias",
            "encoder.down.0.block.0.conv2.weight": "encoder.blocks.0.conv2.weight",
            "encoder.down.0.block.0.norm1.bias": "encoder.blocks.0.norm1.bias",
            "encoder.down.0.block.0.norm1.weight": "encoder.blocks.0.norm1.weight",
            "encoder.down.0.block.0.norm2.bias": "encoder.blocks.0.norm2.bias",
            "encoder.down.0.block.0.norm2.weight": "encoder.blocks.0.norm2.weight",
            "encoder.down.0.block.1.conv1.bias": "encoder.blocks.1.conv1.bias",
            "encoder.down.0.block.1.conv1.weight": "encoder.blocks.1.conv1.weight",
            "encoder.down.0.block.1.conv2.bias": "encoder.blocks.1.conv2.bias",
            "encoder.down.0.block.1.conv2.weight": "encoder.blocks.1.conv2.weight",
            "encoder.down.0.block.1.norm1.bias": "encoder.blocks.1.norm1.bias",
            "encoder.down.0.block.1.norm1.weight": "encoder.blocks.1.norm1.weight",
            "encoder.down.0.block.1.norm2.bias": "encoder.blocks.1.norm2.bias",
            "encoder.down.0.block.1.norm2.weight": "encoder.blocks.1.norm2.weight",
            "encoder.down.0.downsample.conv.bias": "encoder.blocks.2.conv.bias",
            "encoder.down.0.downsample.conv.weight": "encoder.blocks.2.conv.weight",
            "encoder.down.1.block.0.conv1.bias": "encoder.blocks.3.conv1.bias",
            "encoder.down.1.block.0.conv1.weight": "encoder.blocks.3.conv1.weight",
            "encoder.down.1.block.0.conv2.bias": "encoder.blocks.3.conv2.bias",
            "encoder.down.1.block.0.conv2.weight": "encoder.blocks.3.conv2.weight",
            "encoder.down.1.block.0.nin_shortcut.bias": "encoder.blocks.3.conv_shortcut.bias",
            "encoder.down.1.block.0.nin_shortcut.weight": "encoder.blocks.3.conv_shortcut.weight",
            "encoder.down.1.block.0.norm1.bias": "encoder.blocks.3.norm1.bias",
            "encoder.down.1.block.0.norm1.weight": "encoder.blocks.3.norm1.weight",
            "encoder.down.1.block.0.norm2.bias": "encoder.blocks.3.norm2.bias",
            "encoder.down.1.block.0.norm2.weight": "encoder.blocks.3.norm2.weight",
            "encoder.down.1.block.1.conv1.bias": "encoder.blocks.4.conv1.bias",
            "encoder.down.1.block.1.conv1.weight": "encoder.blocks.4.conv1.weight",
            "encoder.down.1.block.1.conv2.bias": "encoder.blocks.4.conv2.bias",
            "encoder.down.1.block.1.conv2.weight": "encoder.blocks.4.conv2.weight",
            "encoder.down.1.block.1.norm1.bias": "encoder.blocks.4.norm1.bias",
            "encoder.down.1.block.1.norm1.weight": "encoder.blocks.4.norm1.weight",
            "encoder.down.1.block.1.norm2.bias": "encoder.blocks.4.norm2.bias",
            "encoder.down.1.block.1.norm2.weight": "encoder.blocks.4.norm2.weight",
            "encoder.down.1.downsample.conv.bias": "encoder.blocks.5.conv.bias",
            "encoder.down.1.downsample.conv.weight": "encoder.blocks.5.conv.weight",
            "encoder.down.2.block.0.conv1.bias": "encoder.blocks.6.conv1.bias",
            "encoder.down.2.block.0.conv1.weight": "encoder.blocks.6.conv1.weight",
            "encoder.down.2.block.0.conv2.bias": "encoder.blocks.6.conv2.bias",
            "encoder.down.2.block.0.conv2.weight": "encoder.blocks.6.conv2.weight",
            "encoder.down.2.block.0.nin_shortcut.bias": "encoder.blocks.6.conv_shortcut.bias",
            "encoder.down.2.block.0.nin_shortcut.weight": "encoder.blocks.6.conv_shortcut.weight",
            "encoder.down.2.block.0.norm1.bias": "encoder.blocks.6.norm1.bias",
            "encoder.down.2.block.0.norm1.weight": "encoder.blocks.6.norm1.weight",
            "encoder.down.2.block.0.norm2.bias": "encoder.blocks.6.norm2.bias",
            "encoder.down.2.block.0.norm2.weight": "encoder.blocks.6.norm2.weight",
            "encoder.down.2.block.1.conv1.bias": "encoder.blocks.7.conv1.bias",
            "encoder.down.2.block.1.conv1.weight": "encoder.blocks.7.conv1.weight",
            "encoder.down.2.block.1.conv2.bias": "encoder.blocks.7.conv2.bias",
            "encoder.down.2.block.1.conv2.weight": "encoder.blocks.7.conv2.weight",
            "encoder.down.2.block.1.norm1.bias": "encoder.blocks.7.norm1.bias",
            "encoder.down.2.block.1.norm1.weight": "encoder.blocks.7.norm1.weight",
            "encoder.down.2.block.1.norm2.bias": "encoder.blocks.7.norm2.bias",
            "encoder.down.2.block.1.norm2.weight": "encoder.blocks.7.norm2.weight",
            "encoder.down.2.downsample.conv.bias": "encoder.blocks.8.conv.bias",
            "encoder.down.2.downsample.conv.weight": "encoder.blocks.8.conv.weight",
            "encoder.down.3.block.0.conv1.bias": "encoder.blocks.9.conv1.bias",
            "encoder.down.3.block.0.conv1.weight": "encoder.blocks.9.conv1.weight",
            "encoder.down.3.block.0.conv2.bias": "encoder.blocks.9.conv2.bias",
            "encoder.down.3.block.0.conv2.weight": "encoder.blocks.9.conv2.weight",
            "encoder.down.3.block.0.norm1.bias": "encoder.blocks.9.norm1.bias",
            "encoder.down.3.block.0.norm1.weight": "encoder.blocks.9.norm1.weight",
            "encoder.down.3.block.0.norm2.bias": "encoder.blocks.9.norm2.bias",
            "encoder.down.3.block.0.norm2.weight": "encoder.blocks.9.norm2.weight",
            "encoder.down.3.block.1.conv1.bias": "encoder.blocks.10.conv1.bias",
            "encoder.down.3.block.1.conv1.weight": "encoder.blocks.10.conv1.weight",
            "encoder.down.3.block.1.conv2.bias": "encoder.blocks.10.conv2.bias",
            "encoder.down.3.block.1.conv2.weight": "encoder.blocks.10.conv2.weight",
            "encoder.down.3.block.1.norm1.bias": "encoder.blocks.10.norm1.bias",
            "encoder.down.3.block.1.norm1.weight": "encoder.blocks.10.norm1.weight",
            "encoder.down.3.block.1.norm2.bias": "encoder.blocks.10.norm2.bias",
            "encoder.down.3.block.1.norm2.weight": "encoder.blocks.10.norm2.weight",
            "encoder.mid.attn_1.k.bias": "encoder.blocks.12.transformer_blocks.0.to_k.bias",
            "encoder.mid.attn_1.k.weight": "encoder.blocks.12.transformer_blocks.0.to_k.weight",
            "encoder.mid.attn_1.norm.bias": "encoder.blocks.12.norm.bias",
            "encoder.mid.attn_1.norm.weight": "encoder.blocks.12.norm.weight",
            "encoder.mid.attn_1.proj_out.bias": "encoder.blocks.12.transformer_blocks.0.to_out.bias",
            "encoder.mid.attn_1.proj_out.weight": "encoder.blocks.12.transformer_blocks.0.to_out.weight",
            "encoder.mid.attn_1.q.bias": "encoder.blocks.12.transformer_blocks.0.to_q.bias",
            "encoder.mid.attn_1.q.weight": "encoder.blocks.12.transformer_blocks.0.to_q.weight",
            "encoder.mid.attn_1.v.bias": "encoder.blocks.12.transformer_blocks.0.to_v.bias",
            "encoder.mid.attn_1.v.weight": "encoder.blocks.12.transformer_blocks.0.to_v.weight",
            "encoder.mid.block_1.conv1.bias": "encoder.blocks.11.conv1.bias",
            "encoder.mid.block_1.conv1.weight": "encoder.blocks.11.conv1.weight",
            "encoder.mid.block_1.conv2.bias": "encoder.blocks.11.conv2.bias",
            "encoder.mid.block_1.conv2.weight": "encoder.blocks.11.conv2.weight",
            "encoder.mid.block_1.norm1.bias": "encoder.blocks.11.norm1.bias",
            "encoder.mid.block_1.norm1.weight": "encoder.blocks.11.norm1.weight",
            "encoder.mid.block_1.norm2.bias": "encoder.blocks.11.norm2.bias",
            "encoder.mid.block_1.norm2.weight": "encoder.blocks.11.norm2.weight",
            "encoder.mid.block_2.conv1.bias": "encoder.blocks.13.conv1.bias",
            "encoder.mid.block_2.conv1.weight": "encoder.blocks.13.conv1.weight",
            "encoder.mid.block_2.conv2.bias": "encoder.blocks.13.conv2.bias",
            "encoder.mid.block_2.conv2.weight": "encoder.blocks.13.conv2.weight",
            "encoder.mid.block_2.norm1.bias": "encoder.blocks.13.norm1.bias",
            "encoder.mid.block_2.norm1.weight": "encoder.blocks.13.norm1.weight",
            "encoder.mid.block_2.norm2.bias": "encoder.blocks.13.norm2.bias",
            "encoder.mid.block_2.norm2.weight": "encoder.blocks.13.norm2.weight",
            "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
            "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
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

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.has_decoder or self.has_encoder, "Either decoder or encoder must be present"
        if "decoder.conv_in.weight" in state_dict or "encoder.conv_in.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info(f"use civitai format state dict")
        else:
            logger.info("use diffsynth format state dict")
        return self._filter(state_dict)


class FluxVAEEncoder(VAEEncoder):
    converter = FluxVAEStateDictConverter(has_encoder=True)

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
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
        return model


class FluxVAEDecoder(VAEDecoder):
    converter = FluxVAEStateDictConverter(has_decoder=True)

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
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
        return model
