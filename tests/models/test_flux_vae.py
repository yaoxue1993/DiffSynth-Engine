import unittest
import os
import torch
import numpy as np
from safetensors.torch import load_file
from PIL import Image

from diffsynth_engine.models.flux import FluxVAEEncoder, FluxVAEDecoder
from diffsynth_engine.constants import MODEL_CACHE_PATH, TEST_ASSETS_PATH

_vae_model_path = os.path.join(MODEL_CACHE_PATH, "flux.1-dev", "ae.safetensors")
_input_image = os.path.join(TEST_ASSETS_PATH, "wukong_1024_1024.png")


class TestFluxVAE(unittest.TestCase):

    def setUp(self):
        loaded_state_dict = load_file(_vae_model_path)
        self.encoder = FluxVAEEncoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()
        self.decoder = FluxVAEDecoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()

    def test_encode(self):
        loaded_state_dict = load_file(os.path.join(TEST_ASSETS_PATH, "test_flux_vae.safetensors"))
        expected = loaded_state_dict["encoded"]
        with Image.open(_input_image).convert("RGB") as image:
            image_tensor = torch.tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to('cuda:0')
        result = self.encoder(image_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result))

    def test_decode(self):
        loaded_state_dict = load_file(os.path.join(TEST_ASSETS_PATH, "test_flux_vae.safetensors"))
        letent_tensor, expected = loaded_state_dict["encoded"], loaded_state_dict["decoded"]
        latent_tensor = torch.randn(1, 16, 128, 128).to('cuda:0')
        result = self.decoder(latent_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result))
