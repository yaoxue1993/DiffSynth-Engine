from ..common.test_case import ImageTestCase

from diffsynth_engine.pipelines import FluxImagePipeline
from diffsynth_engine import fetch_modelscope_model

class TestFLUXImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = fetch_modelscope_model("muse/flux-with-vae", revision="20240902173035", path="flux_with_vae.safetensors")
        cls.pipe = FluxImagePipeline.from_pretrained(model_path).eval()

    def test_txt2img(self):
        image = self.pipe(
            prompt="A cat holding a sign that says hello world",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_txt2img.png", threshold=0.99)
    
    def test_inpainting(self):
        image = self.pipe(
            prompt="a beautiful girl with green hair",
            input_image=self.get_input_image("test_image.png"),
            mask_image=self.get_input_image("mask_image.png"),
            denoising_strength=0.8,
            width=1024,
            height=1024,
            num_inference_steps=20,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_inpainting.png", threshold=0.99)\

    def test_fused_lora(self):
        lora_model_path = fetch_modelscope_model("MAILAND/Merjic-Maria", revision="12", path="12.safetensors")
        self.pipe.patch_loras([(lora_model_path, 0.8)], fused=True)
        image = self.pipe(
            prompt="1 girl, maria",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.pipe.unpatch_loras()
        self.assertImageEqualAndSaveFailed(image, "flux/flux_lora.png", threshold=0.99)
    
    def test_unfused_lora(self):
        lora_model_path = fetch_modelscope_model("MAILAND/Merjic-Maria", revision="12", path="12.safetensors")
        self.pipe.patch_loras([(lora_model_path, 0.8)])
        image = self.pipe(
            prompt="1 girl, maria",
            width=1024,
            height=1024,
            num_inference_steps=50,
            seed=42,
        )
        self.pipe.unpatch_loras()
        self.assertImageEqualAndSaveFailed(image, "flux/flux_lora.png", threshold=0.98)