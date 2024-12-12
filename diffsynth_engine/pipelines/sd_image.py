import os
import torch
import logging
from typing import Callable, Dict, Union, Optional, List
from types import ModuleType
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.models.basic.timestep import TemporalTimesteps
from diffsynth_engine.models.sd import SDTextEncoder, SDVAEDecoder, SDVAEEncoder, SDUNet
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler
from diffsynth_engine.algorithm.sampler import EulerSampler
from diffsynth_engine.utils.prompt import tokenize_long_prompt
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class SDImagePipeline(BasePipeline):

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 text_encoder: SDTextEncoder,
                 unet: SDUNet,
                 vae_decoder: SDVAEDecoder,
                 vae_encoder: SDVAEEncoder,
                 batch_cfg: bool = True,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16
                 ):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = ScaledLinearScheduler()
        self.sampler = EulerSampler()
        # models
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.batch_cfg = batch_cfg
        self.model_names = ['text_encoder','unet', 'vae_decoder', 'vae_encoder']

    @classmethod
    def from_pretrained(cls, pretrained_model_paths: str | os.PathLike | List[str | os.PathLike],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16,
                        cpu_offload: bool = False,
                        batch_cfg: bool = True) -> "SDImagePipeline":
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
        text_encoder = SDTextEncoder.from_state_dict(
            loaded_state_dict, device=init_device, dtype=dtype)
        unet = SDUNet.from_state_dict(
            loaded_state_dict, device=init_device, dtype=dtype)
        vae_decoder = SDVAEDecoder.from_state_dict(
            loaded_state_dict, device=init_device, dtype=torch.float32)
        vae_encoder = SDVAEEncoder.from_state_dict(
            loaded_state_dict, device=init_device, dtype=torch.float32)

        pipe = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            batch_cfg=batch_cfg,
            device=device,
            dtype=dtype,
        )
        if cpu_offload:
            pipe.enable_cpu_offload()
        return pipe

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16) -> "SDImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.unet

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        vae_dtype = self.vae_decoder.conv_in.weight.dtype
        image = self.vae_decoder(latent.to(vae_dtype), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def encode_prompt(self, prompt, clip_skip):
        input_ids = tokenize_long_prompt(
            self.tokenizer, prompt).to(self.device)
        prompt_emb = self.text_encoder(input_ids, clip_skip=clip_skip)
        return prompt_emb

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool = True
    ):
        if cfg_scale < 1.0:
            return self.predict_noise(latents, timestep, positive_prompt_emb)
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(latents, timestep, positive_prompt_emb)
            negative_noise_pred = self.predict_noise(latents, timestep, negative_prompt_emb)
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)            
            positive_noise_pred, negative_noise_pred = self.predict_noise(latents, timestep, prompt_emb).chunk(2)
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, timestep, prompt_emb):
        noise_pred = self.unet(
            x=latents, timestep=timestep,
            context=prompt_emb,
            device=self.device,
        )
        return noise_pred

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            cfg_scale: float = 7.5,
            clip_skip: int = 1,
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
        
        latents = self.generate_noise((1, 4, height // 8, width // 8), seed=seed, device=self.device, dtype=self.dtype)
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
            sigmas, timesteps = self.noise_scheduler.schedule(
                num_inference_steps)
            # k-diffusion
            # if you have any questions about this, please ask @dizhipeng.dzp for more details
            latents = latents * sigmas[0] / ((sigmas[0] ** 2 + 1) ** 0.5)

        # Initialize sampler
        self.sampler.initialize(
            latents=latents, timesteps=timesteps, sigmas=sigmas)

        # Encode prompts
        self.load_models_to_device(['text_encoder', 'text_encoder_2'])
        positive_prompt_emb = self.encode_prompt(prompt, clip_skip=clip_skip)
        negative_prompt_emb = self.encode_prompt(negative_prompt, clip_skip=clip_skip)

        # Denoise
        self.load_models_to_device(['unet'])
        for i, timestep in enumerate(progress_bar_cmd(timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                latents=latents, 
                timestep=timestep,
                positive_prompt_emb=positive_prompt_emb, 
                negative_prompt_emb=negative_prompt_emb,
                cfg_scale=cfg_scale, 
                batch_cfg=self.batch_cfg
            )
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
