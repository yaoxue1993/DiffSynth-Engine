from ..common.test_case import ImageTestCase

from diffsynth_engine.pipelines.sdxl_image import SDXLImagePipeline
class TestSDXLImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = cls.download_model("modelscope://muse/sd_xl_base_1.0?revision=20240425120250")
        cls.pipe = SDXLImagePipeline.from_pretrained(model_path)

    def test_txt2img(self):
        image = self.pipe(
            prompt="a beautiful girl",
            width=1024,
            height=1024,
            num_inference_steps=20,
            seed=42,
            clip_skip=2,
        )

        self.assertImageEqualAndSaveFailed(image, "sdxl/sdxl_txt2img.png", threshold=0.99)
