import os
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from PIL import Image

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class BasePipeline:

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.cpu_offload = False
        self.model_names = []

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike],
                        device: str = "cuda", torch_dtype: torch.dtype = torch.float16) -> "BasePipeline":
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, "torch.Tensor"],
                        device: str = "cuda", torch_dtype: torch.dtype = torch.float16) -> "BasePipeline":
        raise NotImplementedError()

    @staticmethod
    def preprocess_image(image: Image.Image) -> "torch.Tensor":
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def preprocess_images(images: List[Image.Image]) -> List["torch.Tensor"]:
        return [BasePipeline.preprocess_image(image) for image in images]

    @staticmethod
    def vae_output_to_image(vae_output: "torch.Tensor") -> Image.Image:
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    @staticmethod
    def generate_noise(shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def eval(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.eval()

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, load_model_names: Optional[List[str]] = None):
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
