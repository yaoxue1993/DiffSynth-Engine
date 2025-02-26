from ..common.test_case import VideoTestCase
import torch
from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine import fetch_modelscope_model
import os

class TestWanVideo(VideoTestCase):
    @classmethod
    def setUpClass(cls):
        config = WanModelConfig(
            model_path='wan/dit.safetensors',
            vae_path='wan/vae.safetensors',
            t5_path='wan/t5.safetensors'
        )
        cls.pipe = WanVideoPipeline.from_pretrained(config)

    def test_txt2img(self):
        video = self.pipe(
            prompt="A cat run on the street",
        )
        self.save_video(video, "test_txt2img.mp4")
    

