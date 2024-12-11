import os
import torch
import logging
from typing import Callable, Dict, Union, Optional, List
from types import ModuleType
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.models.basic.timestep import TemporalTimesteps
from diffsynth_engine.models.sdxl import SDXLTextEncoder, SDXLTextEncoder2, SDXLVAEDecoder, SDXLVAEEncoder, SDXLUNet
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler
from diffsynth_engine.algorithm.sampler import EulerSampler
from diffsynth_engine.utils.prompt import tokenize_long_prompt
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH, SDXL_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


# TODO: add controlnet/ipadapter/kolors
class SDXLImagePipeline(BasePipeline):

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: CLIPTokenizer,
                 text_encoder: SDXLTextEncoder,
                 text_encoder_2: SDXLTextEncoder2,
                 unet: SDXLUNet,
                 vae_decoder: SDXLVAEDecoder,
                 vae_encoder: SDXLVAEEncoder,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = ScaledLinearScheduler()
        self.sampler = EulerSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.add_time_proj = TemporalTimesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0,
                                               device=device, dtype=dtype)        
        self.model_names = ['text_encoder', 'text_encoder_2', 'unet', 'vae_decoder', 'vae_encoder']

    @classmethod
    def from_pretrained(cls, pretrained_model_paths: str | os.PathLike | List[str | os.PathLike],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16,
                        cpu_offload: bool = False) -> "SDXLImagePipeline":
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
        tokenizer = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)
        tokenizer_2 = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_2_CONF_PATH)
        text_encoder = SDXLTextEncoder.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        text_encoder_2 = SDXLTextEncoder2.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        unet = SDXLUNet.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        vae_decoder = SDXLVAEDecoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)
        vae_encoder = SDXLVAEEncoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)

        pipe = cls(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            device=device,
            dtype=dtype,
        )
        if cpu_offload:
            pipe.enable_cpu_offload()
        return pipe

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16) -> "SDXLImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.unet

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def encode_prompt(self, prompt, clip_skip):
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(self.device)
        prompt_emb_1 = self.text_encoder(input_ids, clip_skip=clip_skip)

        input_ids_2 = tokenize_long_prompt(self.tokenizer_2, prompt).to(self.device)
        add_text_embeds, prompt_emb_2 = self.text_encoder_2(input_ids_2, clip_skip=clip_skip)

        # Merge
        if prompt_emb_1.shape[0] != prompt_emb_2.shape[0]:
            max_batch_size = min(prompt_emb_1.shape[0], prompt_emb_2.shape[0])
            prompt_emb_1 = prompt_emb_1[: max_batch_size]
            prompt_emb_2 = prompt_emb_2[: max_batch_size]
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)

        # For very long prompt, we only use the first 77 tokens to compute `add_text_embeds`.
        add_text_embeds = add_text_embeds[0:1]
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0] * prompt_emb.shape[1], -1))

        return prompt_emb, add_text_embeds

    def prepare_extra_input(self, latents):
        height, width = latents.shape[2] * 8, latents.shape[3] * 8
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device).repeat(latents.shape[0])
        return add_time_id

    def prepare_add_embeds(self, add_text_embeds, add_time_id, dtype):
        time_embeds = self.add_time_proj(add_time_id)
        time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
        add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(dtype)
        return add_embeds

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            cfg_scale: float = 7.5,
            clip_skip: int = 2,
            input_image: Optional[Image.Image] = None,
            denoising_strength: float = 1.0,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 20,
            tiled: bool = False,
            tile_size: int = 64,
            tile_stride: int = 32,
            seed: int | None = None,
            progress_bar_cmd: Callable = tqdm,
            progress_bar_st: ModuleType | None = None,
    ):        
        latents = self.generate_noise((1, 4, height // 8, width // 8), seed=seed, device=self.device,
                                    dtype=self.dtype)

        # Prepare scheduler
        if input_image is not None:    
            # eg. num_inference_steps = 20, denoising_strength = 0.6, total_steps = 33, t_start = 13
            total_steps = max(int(num_inference_steps / denoising_strength), num_inference_steps)
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)                        
            self.load_models_to_device(['vae_encoder'])
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image)
            t_start = max(total_steps - num_inference_steps, 0)
            sigma_start, sigmas = sigmas[t_start], sigmas[t_start:]            
            timesteps = timesteps[t_start:]
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)            

        # Initialize sampler
        self.sampler.initialize(latents=latents, timesteps=timesteps, sigmas=sigmas)            

        # Encode prompts
        self.load_models_to_device(['text_encoder', 'text_encoder_2'])
        positive_prompt_emb, positive_add_text_embeds = self.encode_prompt(prompt, clip_skip=clip_skip)
        negative_prompt_emb, negative_add_text_embeds = self.encode_prompt(negative_prompt, clip_skip=clip_skip)

        # Prepare extra input
        add_time_id = self.prepare_extra_input(latents)
    
        # Denoise
        self.load_models_to_device(['unet'])
        for i, timestep in enumerate(progress_bar_cmd(timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            # Classifier-free guidance
            y = self.prepare_add_embeds(positive_add_text_embeds, add_time_id, self.dtype)
            positive_noise_pred = self.unet(
                x=latents, timestep=timestep, y=y,
                context=positive_prompt_emb,
                device=self.device,
            )

            if cfg_scale != 1.0:
                y = self.prepare_add_embeds(negative_add_text_embeds, add_time_id, self.dtype)
                negative_noise_pred = self.unet(
                    x=latents, timestep=timestep, y=y,
                    context=negative_prompt_emb,
                    device=self.device,
                )
                noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            else:
                noise_pred = positive_noise_pred

            # Denoise
            latents = self.sampler.step(latents, noise_pred, i)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(i / len(timesteps))

        # Decode image
        self.load_models_to_device(['vae_decoder'])
        vae_output = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(vae_output)

        # offload all models
        self.load_models_to_device([])
        return image
