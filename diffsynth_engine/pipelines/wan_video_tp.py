import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Callable, Optional
from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.models.wan.wan_dit import WanDiT
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_image_encoder import WanImageEncoder
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from .base import BasePipeline

class WanVideoVAEDecodeProcedure(nn.Moudle):
    def __init__(self, device, dtype):
        self.vae = WanVideoVAE(device=device, dtype=dtype)  # only use decoder
        self.device = device
        self.dtype = dtype

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None):
        latents = latents.to(dtype=self.dtype, device=self.device)
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, progress_callback=progress_callback)
        frames = [frame.to(dtype=self.dtype, device=self.device) for frame in frames]
        return torch.stack(frames)
    
    def forward(self, latents:torch.Tensor, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None):
        return self.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, progress_callback=progress_callback)


class WanVideoImageEncodeProcedure(nn.Module):
    def __init__(self, device, dtype):
        self.image_encoder = WanImageEncoder(device=device, dtype=dtype)
        self.vae = WanVideoVAE(device=device, dtype=dtype) # only use encoder
        self.device = device
        self.dtype = dtype

    def encode_image(self, image:torch.Tensor, num_frames:int, height:int, width:int):    
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device, dtype=self.dtype)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae.encode([torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device, self.dtype)], dim=1)], device=self.device)[0]
        y = torch.concat([msk, y])
        return clip_context, y

    def forward(self, image: torch.Tensor, num_frames: int, height: int, width: int):
        return self.encode_image(image, num_frames, height, width)

    
class WanVideoTextEncodeProcedure(nn.Module):
    def __init__(self, device, dtype):
        self.tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean='whitespace')
        self.text_encoder = WanTextEncoder(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def encode_prompt(self, prompt: Optional[str]):
        if prompt is None:
            return None
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)        
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = prompt_emb.masked_fill(mask.unsqueeze(-1).expand_as(prompt_emb)==0, 0)
        return prompt_emb
    
    def forward(self, prompt:str, negative_prompt:Optional[str] = None):
        prompt_emb_posi = self.encode_prompt(prompt)        
        prompt_emb_nega = self.encode_prompt(negative_prompt)
        return prompt_emb_posi, prompt_emb_nega

class WanVideoDenoiseProcedure(nn.Module):
    def __init__(self, device, dtype):
        self.dit = WanDiT(device=device, dtype=dtype)
        self.noise_scheduler = RecifitedFlowScheduler(shift=5.0, sigma_min=0.001, sigma_max=0.999)
        self.sampler = FlowMatchEulerSampler()
        self.device = device
        self.dtype = dtype
    

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        image_clip_feature: torch.Tensor,
        image_y: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(latents=latents, image_clip_feature=image_clip_feature, image_y=image_y, timestep=timestep, context=positive_prompt_emb)
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents=latents, image_clip_feature=image_clip_feature, image_y=image_y, timestep=timestep, context=positive_prompt_emb
            )
            negative_noise_pred = self.predict_noise(
                latents=latents, image_clip_feature=image_clip_feature, image_y=image_y, timestep=timestep, context=negative_prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)            
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents=latents, image_clip_feature=image_clip_feature, image_y=image_y, timestep=timestep, context=prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, image_clip_feature, image_y, timestep, context):        
        latents = latents.to(dtype=self.dtype, device=self.device)        
        noise_pred = self.dit(
            x=latents,
            timestep=timestep,
            context=context,
            clip_feature=image_clip_feature,
            y=image_y,
        )
        return noise_pred

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
    
        
    def forward(
        self, 
        seed: int,            
        height: int,
        width: int,
        num_frames: int,            
        num_inference_steps: int,
        init_latents: Optional[torch.Tensor],        
        denoising_strength: float,        
        prompt_emb_posi: torch.Tensor, 
        prompt_emb_nega: torch.Tensor, 
        image_clip_feature: torch.Tensor,
        image_y: torch.Tensor,
        cfg_scale: float, 
        batch_cfg: bool, 
        progress_callback: Callable
    ):    
        timesteps = self.noise_scheduler.get_timesteps(num_frames)
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device='cpu', dtype=torch.float32).to(self.device)        
        sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)                
        if init_latents is None:
            latents = noise
        else:
            t_start = max(num_inference_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]
            latents = self.sampler.add_noise(init_latents, noise, sigma_start)
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype, device=self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=prompt_emb_posi,
                negative_prompt_emb=prompt_emb_nega,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                cfg_scale=cfg_scale,                
                batch_cfg=batch_cfg
            )
            # Scheduler
            latents = self.sampler.step(latents, noise_pred, i)
            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps), "DENOISING")                
        return latents
    
    
class WanVideoDistributePipeline(BasePipeline):
    def __init__(self, batch_cfg, device, dtype):
        super().__init__()
        self.image_encode_procedure = WanVideoImageEncodeProcedure(device=device, dtype=dtype)
        self.text_encode_procedure = WanVideoTextEncodeProcedure(device=device, dtype=dtype)
        self.denoise_procedure = WanVideoDenoiseProcedure(device=device, dtype=dtype)
        self.vae_decode_procedure = WanVideoVAEDecodeProcedure(device=device, dtype=dtype)
        self.batch_cfg = batch_cfg

    def __call__(self,         
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
        tile_stride=(18, 16)
    ):  
        image_clip_feature, image_y = self.image_encode_procedure(
            input_image=input_image, 
            num_frames=num_frames, 
            height=height, 
            width=width
        )
        prompt_emb_posi, prompt_emb_nega = self.text_encode_procedure(
            prompt=prompt, 
            negative_prompt=negative_prompt
        )
        latents = self.denoise_procedure(
            seed=seed, 
            height=height, 
            width=width, 
            input_video=input_video, 
            denoising_strength=denoising_strength, 
            num_frames=num_frames, 
            num_inference_steps=num_inference_steps, 
            prompt_emb_posi=prompt_emb_posi, 
            prompt_emb_nega=prompt_emb_nega, 
            image_clip_feature=image_clip_feature, 
            image_y=image_y, 
            cfg_scale=cfg_scale, 
            batch_cfg=self.batch_cfg
        )
        frames = self.vae_decode_procedure(
            latents=latents, 
            tiled=tiled, 
            tile_size=tile_size, 
            tile_stride=tile_stride
        )
        return frames