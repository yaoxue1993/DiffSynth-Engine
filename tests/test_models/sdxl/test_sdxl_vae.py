import torch
import numpy as np
from safetensors.torch import load_file
from PIL import Image

from diffsynth_engine.models.sdxl import SDXLVAEEncoder, SDXLVAEDecoder
from tests.common.test_case import TestCase


class TestSDXLVAE(TestCase):

    def setUp(self):
        super().setUp()
        # TODO: add sdxl model
        self._sdxl_model_path = None
        self._input_image = self.testdata_dir / "input" / "wukong_1024_1024.png"

    def test_encode(self):
        loaded_state_dict = load_file(self._sdxl_model_path)
        encoder = SDXLVAEEncoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()

        loaded_state_dict = load_file(self.testdata_dir / "expect" / "sdxl" / "test_sdxl_vae.safetensors")
        expected = loaded_state_dict["encoded"]
        with Image.open(self._input_image).convert("RGB") as image:
            image_tensor = torch.tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            result = encoder(image_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result, atol=1e-6))

    def test_decode(self):
        loaded_state_dict = load_file(self._sdxl_model_path)
        decoder = SDXLVAEDecoder.from_state_dict(loaded_state_dict, device='cuda:0', dtype=torch.float32).eval()

        loaded_state_dict = load_file(self.testdata_dir / "expect" / "sdxl" / "test_sdxl_vae.safetensors")
        latent_tensor, expected = loaded_state_dict["encoded"], loaded_state_dict["decoded"]
        latent_tensor = latent_tensor.to('cuda:0')
        with torch.no_grad():
            result = decoder(latent_tensor).cpu()
        self.assertTrue(torch.allclose(expected, result, atol=1e-6))
