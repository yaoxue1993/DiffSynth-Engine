import unittest
import os
import torch
import numpy as np
from safetensors.torch import load_file
from PIL import Image

from diffsynth_engine.models.sdxl import SDXLVAEEncoder, SDXLVAEDecoder
from diffsynth_engine.utils.env import DIFFSYNTH_CACHE
from diffsynth_engine.utils.constants import TEST_ASSETS_PATH

_sdxl_model_path = os.path.join(DIFFSYNTH_CACHE, "sdxl", "sd_xl_base_1.0.safetensors")
_input_image = os.path.join(TEST_ASSETS_PATH, "wukong_1024_1024.png")


class TestSDXLVAE(unittest.TestCase):

    def test_encode(self):
        loaded_state_dict = load_file(_sdxl_model_path)
        encoder = SDXLVAEEncoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()

        loaded_state_dict = load_file(os.path.join(TEST_ASSETS_PATH, "test_sdxl_vae.safetensors"))
        expected = loaded_state_dict["encoded"]
        with Image.open(_input_image).convert("RGB") as image:
            image_tensor = torch.tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            result = encoder(image_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result, atol=1e-6))

    def test_decode(self):
        loaded_state_dict = load_file(_sdxl_model_path)
        decoder = SDXLVAEDecoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()

        loaded_state_dict = load_file(os.path.join(TEST_ASSETS_PATH, "test_sdxl_vae.safetensors"))
        latent_tensor, expected = loaded_state_dict["encoded"], loaded_state_dict["decoded"]
        latent_tensor = latent_tensor.to('cuda:0')
        with torch.no_grad():
            result = decoder(latent_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result, atol=1e-6))
