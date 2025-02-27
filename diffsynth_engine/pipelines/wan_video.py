from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.models.wan.wan_dit import WanDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.models.wan.wan_image_encoder import WanImageEncoder
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_modelscope_model
from diffsynth_engine.models.basic.lora import LoRAContext
from .base import BasePipeline
from einops import rearrange
from dataclasses import dataclass
from typing import Dict, Optional
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class WanModelConfig:
    model_path: Optional[str] = None
    vae_path: Optional[str] = None    
    t5_path: Optional[str] = None    
    image_encoder_path: Optional[str] = None

    vae_dtype: torch.dtype = torch.float32
    dit_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    image_encoder_dtype: torch.dtype = torch.bfloat16


class WanVideoPipeline(BasePipeline):
    def __init__(self, 
        tokenizer: WanT5Tokenizer,
        text_encoder: WanTextEncoder,
        dit: WanDiT,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
        batch_cfg: bool = False,        
        device="cuda", 
        dtype=torch.float16, 
    ):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = RecifitedFlowScheduler(shift=5, sigma_min=0.0)        
        self.sampler = FlowMatchEulerSampler()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae
        self.image_encoder = image_encoder
        self.batch_cfg = batch_cfg

        self.model_names = ['text_encoder', 'dit', 'vae']
    
    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = [u[:v] for u, v in zip(prompt_emb, seq_lens)]
        return prompt_emb
        
    def encode_image(self, image, num_frames, height, width):
        with torch.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = self.vae.encode([torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)], device=self.device)[0]
            y = torch.concat([msk, y])
        return {"clip_fea": clip_context, "y": [y]}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
       
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None):
        with torch.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, progress_callback=progress_callback)
        return frames

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        image_emb: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(latents=latents, image_emb=image_emb, timestep=timestep, context=positive_prompt_emb)
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents=latents, image_emb=image_emb, timestep=timestep, context=positive_prompt_emb
            )
            negative_noise_pred = self.predict_noise(
                latents=latents, image_emb=image_emb, timestep=timestep, context=negative_prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)            
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents=latents, image_emb=image_emb, timestep=timestep, context=prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, image_emb, timestep, context):        
        seq_len = latents.shape[2] * latents.shape[3] * latents.shape[4] // 4 # b c f h w -> b (f*h*w / (patch_size * patch_size)) c 
        noise_pred = self.dit(
            x=latents,
            timestep=timestep,
            context=context,
            seq_len=seq_len,
            clip_fea=image_emb.get("clip_fea", None),
            y=image_emb.get("y", None),
        )
        return noise_pred
    
    def prepare_latents(self, latents, input_video, denoising_strength, num_inference_steps, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        if input_video is not None:
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            noise = latents                        
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=latents.dtype, device=latents.device)            
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)
            init_latents = latents.clone()

        return init_latents, latents, sigmas, timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        progress_callback=None, # def progress_callback(current, total, status) 
    ):
        assert height % 8 == 0 and width % 8 == 0, "height and width must be divisible by 8"
        assert (num_frames - 1) % 4 == 0, "num_frames is not 4X+1"

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device='cpu', dtype=torch.float32).to(self.device)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(noise, input_video, denoising_strength, num_inference_steps, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)        

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = None if cfg_scale <= 1.0 else self.encode_prompt(negative_prompt)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Denoise
        self.load_models_to_device(["dit"])
        with torch.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):        
            for i, timestep in enumerate(tqdm(timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.dtype, device=self.device)
                # Classifier-free guidance
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    positive_prompt_emb=prompt_emb_posi,
                    negative_prompt_emb=prompt_emb_nega,
                    image_emb=image_emb,
                    cfg_scale=cfg_scale,                
                    batch_cfg=self.batch_cfg
                )
                # Scheduler
                latents = self.sampler.step(latents, noise_pred, i)
                if progress_callback is not None:
                    progress_callback(i + 1, len(timesteps), "DENOISING")                

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, progress_callback=progress_callback)
        frames = self.tensor2video(frames[0])
        return frames

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_paths: str | WanModelConfig ,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        batch_cfg: bool = False,
        cpu_offload: bool = True,
    ) -> "WanVideoPipeline":
        if isinstance(pretrained_model_paths, str):
            model_config = WanModelConfig(model_path=pretrained_model_paths)
        else:
            model_config = pretrained_model_paths

        if model_config.vae_path is None:
            model_config.vae_path = fetch_modelscope_model("muse/wan2.1-vae", path="vae.safetensors")
            model_config.vae_dtype = dtype
        if model_config.t5_path is None:
            model_config.t5_path = fetch_modelscope_model("muse/wan2.1-umt5", path="umt5.safetensors")
            model_config.t5_dtype = dtype
        if model_config.model_path is None:
            model_config.model_path = fetch_modelscope_model("muse/wan2.1-1.3b", path="dit.safetensors")
            model_config.dit_dtype = dtype

        assert os.path.isfile(model_config.model_path), f"{model_config.model_path} is not a file"        
        assert os.path.isfile(model_config.vae_path), f"{model_config.vae_path} is not a file"
        assert os.path.isfile(model_config.t5_path), f"{model_config.t5_path} is not a file"

        logger.info(f"loading state dict from {model_config.model_path} ...")
        dit_state_dict = load_file(model_config.model_path, device="cpu")

        logger.info(f"loading state dict from {model_config.t5_path} ...")
        t5_state_dict = load_file(model_config.t5_path, device="cpu")        

        logger.info(f"loading state dict from {model_config.vae_path} ...")
        vae_state_dict = load_file(model_config.vae_path, device="cpu")

        init_device = "cpu" if cpu_offload else device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean='whitespace')
        text_encoder = WanTextEncoder.from_state_dict(t5_state_dict, device=init_device, dtype=model_config.t5_dtype)
        vae = WanVideoVAE.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)
        image_encoder = None        
        if model_config.image_encoder_path is not None:
            logger.info(f"loading state dict from {model_config.image_encoder_path} ...")
            image_encoder_state_dict = load_file(model_config.image_encoder_path, device="cpu")
            image_encoder = WanImageEncoder.from_state_dict(image_encoder_state_dict, device=init_device, dtype=model_config.image_encoder_dtype)
        
    
        with LoRAContext():
            dit = WanDiT.from_state_dict(dit_state_dict, device=init_device, dtype=model_config.dit_dtype)

        pipe = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            image_encoder=image_encoder,
            batch_cfg=batch_cfg,
        )
        if cpu_offload:
            pipe.enable_cpu_offload()
        return pipe