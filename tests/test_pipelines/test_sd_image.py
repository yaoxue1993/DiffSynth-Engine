from ..common.test_case import ImageTestCase

from diffsynth_engine.pipelines.sd_image import SDImagePipeline
class TestSDImage(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        model_path = cls.download_model("modelscope://muse/v1-5-pruned-emaonly?revision=20240118200020")
        cls.pipe = SDImagePipeline.from_pretrained(model_path)

    def test_txt2img(self):
        image = self.pipe(
            prompt="beautiful girl",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=42
        )
        self.assertImageEqualAndSaveFailed(image, "sd/sd_txt2img.png", threshold=0.999)
