import os
import torch
import numpy as np
from typing import Dict, List
from PIL import Image, ImageOps
from einops import repeat
from dataclasses import dataclass
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ModelConfig:
    pass


class BasePipeline:
    def __init__(self, device="cuda:0", dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.cpu_offload = False
        self.model_names = []

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_config: str | os.PathLike | ModelConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        cpu_offload: bool = False,
    ) -> "BasePipeline":
        raise NotImplementedError()

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str = "cuda:0", dtype: torch.dtype = torch.float16
    ) -> "BasePipeline":
        raise NotImplementedError()

    @staticmethod
    def preprocess_image(image: Image.Image) -> torch.Tensor:
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]

        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def preprocess_images(images: List[Image.Image]) -> List[torch.Tensor]:
        return [BasePipeline.preprocess_image(image) for image in images]

    @staticmethod
    def vae_output_to_image(vae_output: torch.Tensor) -> Image.Image:
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    @staticmethod
    def generate_noise(shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        vae_dtype = self.vae_decoder.conv_in.weight.dtype
        image = self.vae_decoder(latent.to(vae_dtype), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def prepare_mask(
        self, input_image: Image.Image, mask_image: Image.Image, vae_scale_factor: int = 8, latent_channels=4
    ) -> torch.Tensor:
        height, width = mask_image.size
        # mask
        mask = torch.Tensor(np.array(mask_image) / 255).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(height // vae_scale_factor, width // vae_scale_factor))
        mask = repeat(mask, "b 1 h w -> b c h w", c=latent_channels)
        mask = mask.to(self.device, self.dtype)
        # overlay_image
        overlay_image = Image.new("RGBa", (width, height))
        overlay_image.paste(input_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask_image.convert("L")))
        overlay_image = overlay_image.convert("RGBA")
        # mask: [1, 4, H, W]  = mask_image[1, 1, H//8, W//8] * 4
        # overlay_image: [1, 4, H, W]  = mask_image[1, 1, H, W] + input_image[1, 3, H, W]
        return mask, overlay_image

    def prepare_latents(self, latents, input_image, denoising_strength, num_inference_steps):
        # Prepare scheduler
        if input_image is not None:
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            self.load_models_to_device(["vae_encoder"])
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)
            # k-diffusion
            # if you have any questions about this, please ask @dizhipeng.dzp for more details
            latents = latents * sigmas[0] / ((sigmas[0] ** 2 + 1) ** 0.5)
            init_latents = latents.clone()
        sigmas, timesteps = sigmas.to(device=self.device), timesteps.to(self.device)
        return init_latents, latents, sigmas, timesteps

    def eval(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.eval()
        return self

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, load_model_names: List[str] | None = None):
        load_model_names = load_model_names if load_model_names else []
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload unnecessary models to cpu
        for model_name in self.model_names:
            if model_name not in load_model_names:
                model = getattr(self, model_name)
                if model is not None:
                    model.cpu()
        # load the needed models to device
        for model_name in load_model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()
