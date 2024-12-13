import unittest
import torch
from safetensors.torch import load_file, save_file

from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.models.sd import SDTextEncoder
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import ensure_directory_exists
from tests.common.test_case import TestCase, RUN_EXTRA_TEST
import os


class TestSDTextEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)

        cls.model_path = cls.download_model("modelscope://muse/v1-5-pruned-emaonly?revision=20240118200020")        
        loaded_state_dict = load_file(cls.model_path)
        cls.text_encoder = SDTextEncoder.from_state_dict(loaded_state_dict,
                                                              device='cuda:0', dtype=torch.float16).eval()
        cls.clip_skip = 1
        cls.texts = ["Hello, World!"]

    def test_encoder(self):
        text_ids = self.tokenizer(self.texts)["input_ids"].to(device='cuda:0')
        with torch.no_grad():
            embeds = self.text_encoder(text_ids).cpu()
        expected_tensors = self.get_expect_tensor("sd/sd_text_encoder.safetensors")
        expected_embeds = expected_tensors["embeds"]
        self.assertTensorEqual(embeds, expected_embeds)


    @unittest.skipUnless(RUN_EXTRA_TEST, "RUN_EXTRA_TEST is not set")
    def test_encoder_and_save_tensors(self):
        from transformers.models.clip import CLIPTextModel, CLIPTextConfig
        clip_l_config = {
            "bos_token_id": 0,
            "pad_token_id": 1,            
            "eos_token_id": 2,
            "num_attention_heads": 12,
            "hidden_size": 768,            
            "intermediate_size": 3072,
            "projection_dim": 768,
        }
        config = CLIPTextConfig(**clip_l_config)
        clip_l_model = CLIPTextModel(config).to(device='cuda:0', dtype=torch.float16).eval()

        model_path = self.download_model("modelscope://AI-ModelScope/stable-diffusion-v1-5?revision=master")
        model_path = os.path.join(model_path, "text_encoder/model.safetensors")
        loaded_state_dict = load_file(model_path)
        del loaded_state_dict['text_model.embeddings.position_ids']
        clip_l_model.load_state_dict(loaded_state_dict)

        text_ids = self.tokenizer(self.texts)["input_ids"].to(device='cuda:0')
        with torch.no_grad():
            model_output = clip_l_model(text_ids, output_hidden_states=True)
            expected_embeds = model_output.hidden_states[-self.clip_skip]
            embeds = self.text_encoder(text_ids, clip_skip=self.clip_skip)

        self.assertTensorEqual(embeds, expected_embeds)

        expect = {"embeds": embeds}
        save_path = self.testdata_dir / "expect/sd/sd_text_encoder.safetensors"
        ensure_directory_exists(save_path)
        save_file(expect, save_path)
