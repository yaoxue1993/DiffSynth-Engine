from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.models.wan.wan_dit import WanDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.models.wan.wan_image_encoder import WanImageEncoder

from .base import BasePipeline
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from typing import Dict
import torch
import numpy as np



class WanVideoPipeline(BasePipeline):
    def __init__(self, 
        tokenizer,
        text_encoder: WanTextEncoder,
        dit: WanDiT,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
        batch_cfg: bool = True,        
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
        return self.prompter.encode_prompt(prompt)
        
    
    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
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
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        image_emb: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool = True,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(latents, timestep, positive_prompt_emb)
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents, timestep, positive_prompt_emb
            )
            negative_noise_pred = self.predict_noise(
                latents, timestep, negative_prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)            
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents, timestep, prompt_emb
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, image_emb, timestep, prompt_emb):        
        seq_len = latents.shape[2] * latents.shape[3] * latents.shape[4] // 4 # b c f h w -> b (f*h*w / (patch_size * patch_size)) c 
        noise_pred = self.dit(
            x=latents,
            timestep=timestep,
            context=prompt_emb,
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
    ):

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device='cpu', dtype=torch.float32).to(self.device)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(noise, input_video, denoising_strength, num_inference_steps, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)        

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Denoise
        self.load_models_to_device(["dit"])
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

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        return frames