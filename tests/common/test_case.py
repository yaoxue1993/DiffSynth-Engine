import unittest
import os
import torch
from pathlib import Path
from PIL import Image

from diffsynth_engine.utils.download import download_model
from tests.common.utils import make_deterministic, compute_normalized_ssim

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCase(unittest.TestCase):
    testdata_dir = Path(TEST_ROOT) / "data"

    def setUp(self):
        self.seed = 42
        make_deterministic(self.seed)

    @classmethod
    def download_model(cls, path: str) -> str:
        return download_model(path)

    @classmethod
    def get_device_name(cls) -> str:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            return device_name
        else:
            return "cpu"


class ImageTestCase(TestCase):
    def get_expect_image(self, name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"expect/{name}")

    def get_input_image(self, name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"input/{name}")

    def assertImageEqual(self, input_image: Image.Image, expect_image: Image.Image, threshold=0.965):
        self.assertGreaterEqual(compute_normalized_ssim(input_image, expect_image), threshold)

    def assertImageEqualAndSaveFailed(self, input_image: Image.Image, expect_image_path: str, threshold=0.965):
        """
        比较input_image和testdata/expect/{name}的SSIM相似度，如果失败则保存input_image到当前工作目录
        """
        try:
            expect_image = self.get_expect_image(expect_image_path)
            self.assertImageEqual(input_image, expect_image, threshold=threshold)
        except Exception as e:
            name = expect_image_path.split("/")[-1]
            input_image.save(f"{name}")
            raise e
