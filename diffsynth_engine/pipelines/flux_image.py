import re
import os
import torch
import math
from typing import Callable, Dict, List, Tuple
from types import ModuleType
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.models.flux import FluxTextEncoder1, FluxTextEncoder2, FluxVAEDecoder, FluxVAEEncoder, FluxDiT
from diffsynth_engine.models.basic.tiler import FastTileWorker
from diffsynth_engine.models.basic.lora import LoRAContext, LoRALinear, LoRAConv2d
from diffsynth_engine.models.base import LoRAStateDictConverter
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils import logging
from diffsynth_engine.conf.keymap import (
    flux_civitai_dit_rename_dict, flux_civitai_dit_suffix_rename_dict,
    flux_civitai_clip_rename_dict, flux_civitai_clip_attn_rename_dict
)

logger = logging.get_logger(__name__)

class FluxLoRAConverter(LoRAStateDictConverter):
    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:            
        dit_dict = {}
        te_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            if "lora_unet" in key:
                key = key.replace("lora_unet_", "")                
                key = re.sub(r'(\d+)_', r'\1.', key)
                key = re.sub(r'_(\d+)', r'.\1', key)
                key = key.replace("modulation_lin", "modulation.lin")
                key = key.replace("mod_lin", "mod.lin")
                key = key.replace("attn_qkv", "attn.qkv")
                key = key.replace("attn_proj", "attn.proj")
                key = key.replace(".alpha", ".weight")
                names = key.split(".")
                if key in flux_civitai_dit_rename_dict:
                    rename = flux_civitai_dit_rename_dict[key]
                    if key.startswith("final_layer.adaLN_modulation.1."):
                        param = torch.concat([param[3072:], param[:3072]], dim=0)
                elif names[0] == "double_blocks":
                    rename = f"blocks.{names[1]}." + flux_civitai_dit_suffix_rename_dict[".".join(names[2:])]
                elif names[0] == "single_blocks":
                    if ".".join(names[2:]) in flux_civitai_dit_suffix_rename_dict:
                        rename = f"single_blocks.{names[1]}." + flux_civitai_dit_suffix_rename_dict[".".join(names[2:])]
                    else:
                        raise ValueError(f"Unsupported key: {key}")
                lora_args = {}
                lora_args["alpha"] = param
                lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args['up'].shape[1]                
                rename = rename.replace(".weight", "")
                dit_dict[rename] = lora_args
            elif "lora_te" in key:
                name = key.replace("lora_te1", "text_encoder")
                name = name.replace("text_model_encoder_layers", "text_model.encoder.layers")
                name = name.replace(".alpha", ".weight")
                rename = ""
                if name in flux_civitai_clip_rename_dict:
                    if name == "text_model.embeddings.position_embedding.weight":
                        param = param.reshape((1, param.shape[0], param.shape[1]))
                    rename = flux_civitai_clip_rename_dict[name]
                elif name.startswith("text_model.encoder.layers."):
                    names = name.split(".")
                    layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                    rename = ".".join(["encoders", layer_id, flux_civitai_clip_attn_rename_dict[layer_type], tail])
                else:
                    raise ValueError(f"Unsupported key: {key}")
                lora_args = {}
                lora_args['alpha'] = param
                lora_args['up'] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args['up'].shape[1]
                rename = rename.replace(".weight", "")                
                te_dict[rename] = lora_args
            else:
                raise ValueError(f"Unsupported key: {key}")
        return {"dit": dit_dict, "text_encoder_1": te_dict}


    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            key = key.replace(".alpha", ".weight")
            key = key.replace("transformer.", "")
            if "single_transformer_blocks" in key: # transformer.single_transformer_blocks.0.attn.to_k.weight
                key = key.replace("single_transformer_blocks", "single_blocks") # single_transformer_blocks.0.attn.to_k.weight
                key = key.replace("norm.linear", "modulation.lin")
                key = key.replace("proj_out", "linear2")
                key = key.replace("attn", "linear1")   # linear1 = [to_q, to_k, to_v, mlp]
                key = key.replace("proj_mlp", "linear1.mlp")
            elif "transformer_blocks" in key:
                key = key.replace("transformer_blocks", "double_blocks")
                key = key.replace("attn.add_k_proj", "txt_attn.qkv.to_k")
                key = key.replace("attn.add_q_proj", "txt_attn.qkv.to_q")
                key = key.replace("attn.add_v_proj", "txt_attn.qkv.to_v")
                key = key.replace("attn.to_add_out", "txt_attn.qkv.to_out")
                key = key.replace("attn.to_k", "img_attn.qkv.to_k")
                key = key.replace("attn.to_q", "img_attn.qkv.to_q")
                key = key.replace("attn.to_v", "img_attn.qkv.to_v")
                key = key.replace("attn.to_out", "img_attn.qkv.to_out")
                key = key.replace("ff.net", "img_mlp")
                key = key.replace("ff_context.net", "txt_mlp")
                key = key.replace("norm1.linear", "img_mod.lin")
                key = key.replace("norm1_context.linear", "txt_mod.lin")
                key = key.replace(".0.proj", ".0")
            elif "context_embedder" in key:
                key = key.replace("context_embedder", "txt_in")
            elif "x_embedder" in key and "_x_embedder" not in key:
                key = key.replace("x_embedder", "img_in")
            elif "time_text_embed" in key:
                key = key.replace("time_text_embed.", "")
                key = key.replace("timestep_embedder", "time_in")
                key = key.replace("guidance_embedder", "guidance_in")
                key = key.replace("text_embedder", "vector_in")
                key = key.replace("linear_1", "in_layer")
                key = key.replace("linear_2", "out_layer")
            elif "norm_out.linear" in key:
                key = key.replace("norm_out.linear", "final_layer.adaLN_modulation.1")
            elif "proj_out" in key:
                key = key.replace("proj_out", "final_layer.linear")
            else:
                raise ValueError(f"Unsupported key: {key}")
            lora_args = {}
            lora_args['alpha'] = param
            lora_args['up'] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
            lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
            lora_args["rank"] = lora_args['up'].shape[1]
            key = key.replace(".weight", "")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        elif key.startswith("transformer"):
            return self._from_diffusers(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxImagePipeline(BasePipeline):
    lora_converter = FluxLoRAConverter()
    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: T5TokenizerFast,
                 text_encoder_1: FluxTextEncoder1,
                 text_encoder_2: FluxTextEncoder2,
                 dit: FluxDiT,
                 vae_decoder: FluxVAEDecoder,
                 vae_encoder: FluxVAEEncoder,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.dit = dit
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']

    @classmethod
    def from_pretrained(cls, pretrained_model_paths: str | os.PathLike | List[str | os.PathLike],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.bfloat16,
                        cpu_offload: bool = False) -> "FluxImagePipeline":
        """
        Init pipeline from one or several .safetensors files, assume there is no key conflict.
        """
        loaded_state_dict = {}
        if isinstance(pretrained_model_paths, str):
            pretrained_model_paths = [pretrained_model_paths]

        for path in pretrained_model_paths:
            assert os.path.isfile(path) and path.endswith(".safetensors"), \
                f"{path} is not a .safetensors file"
            logger.info(f"loading state dict from {path} ...")
            state_dict = load_file(path, device="cpu")
            loaded_state_dict.update(state_dict)

        init_device = "cpu" if cpu_offload else device
        tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
        tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)
        with LoRAContext():
            dit = FluxDiT.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
            text_encoder_1 = FluxTextEncoder1.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)        
        text_encoder_2 = FluxTextEncoder2.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)        
        vae_decoder = FluxVAEDecoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)
        vae_encoder = FluxVAEEncoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)

        pipe = cls(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            dit=dit,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            device=device,
            dtype=dtype,
        )
        if cpu_offload:
            pipe.enable_cpu_offload()
        return pipe

    def patch_lora(self, lora_list: List[Tuple[str, float]], fused: bool = False, save_original_weight: bool = True):
        for (lora_path, lora_scale) in lora_list:
            state_dict = load_file(lora_path, device="cpu")
            lora_state_dict = self.lora_converter.convert(state_dict)
            for model_name, state_dict in lora_state_dict.items():
                model = getattr(self, model_name)
                for key, param in state_dict.items():
                    module = model.get_submodule(key)
                    if not isinstance(module, (LoRALinear, LoRAConv2d)):
                        raise ValueError(f"Unsupported lora key: {key}")
                    lora_args = {
                        "name": key,
                        "scale": lora_scale,
                        "rank": param['rank'],
                        "alpha": param['alpha'],
                        "up": param['up'],
                        "down": param['down'],
                        "device": self.device,
                        "dtype": self.dtype,
                        "save_original_weight": save_original_weight
                    }
                    if fused:
                        module.add_frozen_lora(**lora_args)
                    else:
                        module.add_lora(**lora_args)        
    
    def unpatch_lora(self):
        for key, module in self.dit.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()
        for key, module in self.text_encoder_1.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.bfloat16) -> "FluxImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.dit

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def encode_prompt(self, prompt, clip_sequence_length=77, t5_sequence_length=512):
        input_ids = self.tokenizer(prompt, max_length=clip_sequence_length)["input_ids"].to(device=self.device)
        _, pooled_prompt_emb = self.text_encoder_1(input_ids)

        input_ids = self.tokenizer_2(prompt, max_length=t5_sequence_length)["input_ids"].to(device=self.device)
        prompt_emb = self.text_encoder_2(input_ids)

        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=self.device,
                                                                               dtype=prompt_emb.dtype)
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}

    def prepare_extra_input(self, latents=None, guidance=1.0):
        latent_image_ids = self.dit.prepare_image_ids(latents)
        guidance = torch.tensor([guidance] * latents.shape[0], device=latents.device, dtype=latents.dtype)
        return {"image_ids": latent_image_ids, "guidance": guidance}

    def predict_noise(self,
                      hidden_states: torch.Tensor,
                      timestep: torch.Tensor,
                      prompt_emb: torch.Tensor,
                      pooled_prompt_emb: torch.Tensor,
                      guidance: torch.Tensor,
                      text_ids: torch.Tensor,
                      image_ids: torch.Tensor | None = None,
                      tiled: bool = False,
                      tile_size: int = 128,
                      tile_stride: int = 64,
                      ):
        if tiled:
            def flux_forward_fn(hl, hr, wl, wr):
                return self.predict_noise(
                    hidden_states=hidden_states[:, :, hl: hr, wl: wr],
                    timestep=timestep,
                    prompt_emb=prompt_emb,
                    pooled_prompt_emb=pooled_prompt_emb,
                    guidance=guidance,
                    text_ids=text_ids,
                    image_ids=None,
                    tiled=False,
                )

            return FastTileWorker().tiled_forward(
                flux_forward_fn,
                hidden_states,
                tile_size=tile_size,
                tile_stride=tile_stride,
                tile_device=hidden_states.device,
                tile_dtype=hidden_states.dtype
            )

        if image_ids is None:
            image_ids = self.dit.prepare_image_ids(hidden_states)

        # warning: keep the order of time_embedding + guidance_embedding + pooled_text_embedding
        # addition of floating point numbers does not meet commutative law
        conditioning = self.dit.time_embedder(timestep, hidden_states.dtype)
        if self.dit.guidance_embedder is not None:
            guidance = guidance * 1000
            conditioning += self.dit.guidance_embedder(guidance, hidden_states.dtype)
        conditioning += self.dit.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.dit.context_embedder(prompt_emb)
        image_rotary_emb = self.dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        height, width = hidden_states.shape[-2:]
        hidden_states = self.dit.patchify(hidden_states)
        hidden_states = self.dit.x_embedder(hidden_states)

        # Joint Blocks
        for block_id, block in enumerate(self.dit.blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for block_id, block in enumerate(self.dit.single_blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        hidden_states = self.dit.final_norm_out(hidden_states, conditioning)
        hidden_states = self.dit.final_proj_out(hidden_states)
        hidden_states = self.dit.unpatchify(hidden_states, height, width)

        return hidden_states

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            cfg_scale: float = 1.0,
            embedded_guidance: float = 3.5,
            input_image: Image.Image | None = None,
            denoising_strength: float = 1.0,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 30,
            t5_sequence_length: int = 512,
            tiled: bool = False,
            tile_size: int = 128,
            tile_stride: int = 64,
            seed: int | None = None,
            progress_bar_cmd: Callable = tqdm,
            progress_bar_st: ModuleType | None = None,
    ):
        latents = self.generate_noise((1, 16, height // 8, width // 8), seed=seed,
                                      device="cpu", dtype=self.dtype).to(device=self.device)

        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)
        mu = calculate_shift(image_seq_len)
        # Prepare scheduler
        if input_image is not None:
            # eg. num_inference_steps = 20, denoising_strength = 0.6, total_steps = 33, t_start = 13
            total_steps = max(int(num_inference_steps / denoising_strength), num_inference_steps)
            sigmas = torch.linspace(1.0, 1 / total_steps, total_steps)
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps, mu=mu, sigmas=sigmas)
            t_start = max(total_steps - num_inference_steps, 0)
            sigma_start, sigmas = sigmas[t_start], sigmas[t_start:]
            timesteps = timesteps[t_start:]

            self.load_models_to_device(['vae_encoder'])
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image)
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps, mu=mu, sigmas=sigmas)
        sigmas, timesteps = sigmas.to(device=self.device), timesteps.to(self.device)

        # Initialize sampler
        self.sampler.initialize(latents=latents, timesteps=timesteps, sigmas=sigmas)

        # Encode prompts
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        positive_prompt_emb = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        if cfg_scale != 1.0:
            negative_prompt_emb = self.encode_prompt(negative_prompt, t5_sequence_length=t5_sequence_length)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Denoise
        self.load_models_to_device(['dit'])
        for i, timestep in enumerate(progress_bar_cmd(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)

            # Classifier-free guidance
            positive_noise_pred = self.predict_noise(
                hidden_states=latents, timestep=timestep,
                **positive_prompt_emb, **extra_input,
            )
            if cfg_scale != 1.0:
                negative_noise_pred = self.predict_noise(
                    hidden_states=latents, timestep=timestep,
                    **negative_prompt_emb, **extra_input,
                )
                noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            else:
                noise_pred = positive_noise_pred

            # Iterate
            latents = self.sampler.step(latents, noise_pred, i)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(i / len(timesteps))

        # Decode image
        self.load_models_to_device(['vae_decoder'])
        vae_output = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(vae_output)

        # Offload all models
        self.load_models_to_device([])
        return image
