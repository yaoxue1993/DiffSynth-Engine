from ..common.test_case import ImageTestCase

from diffsynth_engine.pipelines import FluxImagePipeline


class TestFLUXImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        clip_l_path = cls.download_model("modelscope://muse/flux_clip_l?revision=20241209&endpoint=www.modelscope.cn")
        t5xxl_path = cls.download_model(
            "modelscope://muse/google_t5_v1_1_xxl?revision=20241024105236&endpoint=www.modelscope.cn")
        flux_with_vae_path = cls.download_model(
            "modelscope://muse/flux-with-vae?revision=20240902173035&endpoint=www.modelscope.cn")
        model_paths = [clip_l_path, t5xxl_path, flux_with_vae_path]
        cls.pipe = FluxImagePipeline.from_pretrained(model_paths).eval()

    def test_txt2img(self):
        image = self.pipe(
            prompt="A cat holding a sign that says hello world",
            width=1024,
            height=1024,
            embedded_guidance=3.5,
            num_inference_steps=50,
            seed=42,
        )
        self.assertImageEqualAndSaveFailed(image, "flux/flux_txt2img.png", threshold=0.99)
