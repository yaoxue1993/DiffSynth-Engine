import unittest
import os
import torch
from pathlib import Path
from PIL import Image
from typing import Dict
from safetensors.torch import load_file

from diffsynth_engine.utils.download import download_model
from tests.common.utils import make_deterministic, compute_normalized_ssim

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test flags
RUN_EXTRA_TEST = os.environ.get("RUN_EXTRA_TEST", "")


class TestCase(unittest.TestCase):
    testdata_dir = Path(TEST_ROOT) / "data"

    def setUp(self):
        self.seed = 42
        make_deterministic(self.seed)

    @staticmethod
    def download_model(path: str) -> str:
        return download_model(path)

    @staticmethod
    def get_device_name() -> str:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            return device_name
        else:
            return "cpu"

    @staticmethod
    def get_expect_tensor(name) -> Dict[str, torch.Tensor]:
        return load_file(ImageTestCase.testdata_dir / "expect" / f"{name}")

    @staticmethod
    def get_input_tensor(name) -> Dict[str, torch.Tensor]:
        return load_file(ImageTestCase.testdata_dir / "input" / f"{name}")

    def assertTensorEqual(self, input_tensor: torch.Tensor, expect_tensor: torch.Tensor, atol=1e-3, rtol=1e-3):
        # 计算绝对误差和相对误差
        abs_diff = torch.abs(input_tensor - expect_tensor)
        rel_diff = abs_diff / (torch.abs(expect_tensor))
        # 计算平均绝对误差和相对误差
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        # 计算最大绝对误差和相对误差
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()

        if not torch.allclose(input_tensor, expect_tensor, atol=atol, rtol=rtol):
            print(f"atol: {atol}\trtol: {rtol}")
            print(f"mean_abs_diff: {mean_abs_diff}\tmean_rel_diff: {mean_rel_diff}")
            print(f"max_abs_diff: {max_abs_diff}\tmax_rel_diff: {max_rel_diff}")

        self.assertTrue(torch.allclose(input_tensor, expect_tensor, atol=atol, rtol=rtol))


class ImageTestCase(TestCase):

    @staticmethod
    def get_expect_image(name) -> Image.Image:
        return Image.open(ImageTestCase.testdata_dir / f"expect/{name}")

    @staticmethod
    def get_input_image(name) -> Image.Image:
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
