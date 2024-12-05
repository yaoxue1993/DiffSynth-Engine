import os
import torch
from typing import Callable, Dict, List
from types import ModuleType
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.models.sdxl import SDXLTextEncoder, SDXLTextEncoder2, SDXLVAEDecoder, SDXLVAEEncoder, SDXLUNet
from diffsynth_engine.models.basic.unet_helper import PushBlock, PopBlock
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer
from diffsynth_engine.algorithm.noise_scheduler import KarrasScheduler
from diffsynth_engine.algorithm.sampler import DPMSolverPlusPlus2MSampler
from diffsynth_engine.utils.prompt import tokenize_long_prompt
from diffsynth_engine.utils.constants import SDXL_TOKENIZER_CONF_PATH, SDXL_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


# TODO: add controlnet/ipadapter/kolors
class SDXLImagePipeline(BasePipeline):

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: CLIPTokenizer,
                 text_encoder: SDXLTextEncoder,
                 text_encoder_2: SDXLTextEncoder2,
                 unet: SDXLUNet,
                 vae_decoder: SDXLVAEDecoder,
                 vae_encoder: SDXLVAEEncoder,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.float16):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = KarrasScheduler()
        self.sampler = DPMSolverPlusPlus2MSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.model_names = ['text_encoder', 'text_encoder_2', 'unet', 'vae_decoder', 'vae_encoder']

    @classmethod
    def from_pretrained(cls, pretrained_model_paths: str | os.PathLike | List[str | os.PathLike],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16,
                        cpu_offload: bool = False) -> "SDXLImagePipeline":
        """
        Init pipeline from one or several .safetensors files, assume there is no key conflict.
        """
        loaded_state_dict = {}
        if isinstance(pretrained_model_paths, str):
            pretrained_model_paths = [pretrained_model_paths]

        for path in pretrained_model_paths:
            assert os.path.isfile(path) and path.endswith(".safetensors"), \
                f"{path} is not a .safetensors file"
            logger.info(f"loading state dict from {path} ...")
            state_dict = load_file(path, device="cpu")
            loaded_state_dict.update(state_dict)

        init_device = "cpu" if cpu_offload else device
        tokenizer = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONF_PATH)
        tokenizer_2 = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_2_CONF_PATH)
        text_encoder = SDXLTextEncoder.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        text_encoder_2 = SDXLTextEncoder2.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        unet = SDXLUNet.from_state_dict(loaded_state_dict, device=init_device, dtype=dtype)
        vae_decoder = SDXLVAEDecoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)
        vae_encoder = SDXLVAEEncoder.from_state_dict(loaded_state_dict, device=init_device, dtype=torch.float32)

        pipe = cls(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            device=device,
            dtype=dtype,
        )
        if cpu_offload:
            pipe.enable_cpu_offload()
        return pipe

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor],
                        device: str = 'cuda:0', dtype: torch.dtype = torch.float16) -> "SDXLImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.unet

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def encode_prompt(self, prompt, clip_skip=1, clip_skip_2=2):
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(self.device)
        prompt_emb_1 = self.text_encoder(input_ids, clip_skip=clip_skip)

        input_ids_2 = tokenize_long_prompt(self.tokenizer_2, prompt).to(self.device)
        add_text_embeds, prompt_emb_2 = self.text_encoder_2(input_ids_2, clip_skip=clip_skip_2)

        # Merge
        if prompt_emb_1.shape[0] != prompt_emb_2.shape[0]:
            max_batch_size = min(prompt_emb_1.shape[0], prompt_emb_2.shape[0])
            prompt_emb_1 = prompt_emb_1[: max_batch_size]
            prompt_emb_2 = prompt_emb_2[: max_batch_size]
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)

        # For very long prompt, we only use the first 77 tokens to compute `add_text_embeds`.
        add_text_embeds = add_text_embeds[0:1]
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0] * prompt_emb.shape[1], -1))

        return {"encoder_hidden_states": prompt_emb, "add_text_embeds": add_text_embeds}

    def prepare_extra_input(self, latents=None):
        height, width = latents.shape[2] * 8, latents.shape[3] * 8
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device).repeat(latents.shape[0])
        return {"add_time_id": add_time_id}

    def predict_noise(self,
                      sample: torch.Tensor,
                      add_time_id: torch.Tensor,
                      add_text_embeds: torch.Tensor,
                      timestep: torch.Tensor,
                      encoder_hidden_states: torch.Tensor,
                      unet_batch_size: int = 1,
                      tiled: bool = False,
                      tile_size: int = 64,
                      tile_stride: int = 32,
                      device: str = 'cuda:0',
                      vram_limit_level: int = 0,  # TODO: define enum
                      ) -> torch.Tensor:
        # 1. time
        t_emb = self.unet.time_proj(timestep).to(sample.dtype)
        t_emb = self.unet.time_embedding(t_emb)

        time_embeds = self.unet.add_time_proj(add_time_id)
        time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
        add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(sample.dtype)
        add_embeds = self.unet.add_time_embedding(add_embeds)

        time_emb = t_emb + add_embeds

        # 2. pre-process
        hidden_states = self.unet.conv_in(sample)
        text_emb = encoder_hidden_states if self.unet.text_intermediate_proj is None else \
            self.unet.text_intermediate_proj(encoder_hidden_states)
        res_stack = [hidden_states]

        # 3. blocks
        for block_id, block in enumerate(self.unet.blocks):
            if isinstance(block, PushBlock):
                hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
                if vram_limit_level >= 1:
                    res_stack[-1] = res_stack[-1].cpu()
            elif isinstance(block, PopBlock):
                if vram_limit_level >= 1:
                    res_stack[-1] = res_stack[-1].to(device)
                hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
            else:
                hidden_states_input = hidden_states
                hidden_states_output = []
                for batch_id in range(0, sample.shape[0], unet_batch_size):
                    batch_id_ = min(batch_id + unet_batch_size, sample.shape[0])
                    hidden_states, _, _, _ = block(
                        hidden_states_input[batch_id: batch_id_],
                        time_emb[batch_id: batch_id_],
                        text_emb[batch_id: batch_id_],
                        res_stack,
                        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                    )
                    hidden_states_output.append(hidden_states)
                hidden_states = torch.concat(hidden_states_output, dim=0)

        # 4. output
        hidden_states = self.unet.conv_norm_out(hidden_states)
        hidden_states = self.unet.conv_act(hidden_states)
        hidden_states = self.unet.conv_out(hidden_states)

        return hidden_states

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            cfg_scale: float = 7.5,
            clip_skip: int = 1,
            clip_skip_2: int = 2,
            input_image: Image.Image | None = None,
            denoising_strength: float = 1.0,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 20,
            tiled: bool = False,
            tile_size: int = 64,
            tile_stride: int = 32,
            seed: int | None = None,
            progress_bar_cmd: Callable = tqdm,
            progress_bar_st: ModuleType | None = None,
    ):
        """
        Args:

            TODO: add details
        """

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare noise scheduler and sampler
        sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)

        # Prepare latent tensors
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 4, height // 8, width // 8), seed=seed, device=self.device,
                                        dtype=self.dtype)
            latents = self.sampler.step(latents, noise, 0)
        else:
            latents = self.generate_noise((1, 4, height // 8, width // 8), seed=seed, device=self.device,
                                          dtype=self.dtype)

        # Encode prompts
        self.load_models_to_device(['text_encoder', 'text_encoder_2'])
        positive_prompt_emb = self.encode_prompt(prompt, clip_skip=clip_skip, clip_skip_2=clip_skip_2)
        negative_prompt_emb = self.encode_prompt(negative_prompt, clip_skip=clip_skip, clip_skip_2=clip_skip_2)

        # Prepare extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(['unet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            positive_noise_pred = self.predict_noise(
                sample=latents, timestep=timestep, **extra_input,
                **positive_prompt_emb, **tiler_kwargs,
                device=self.device,
            )

            if cfg_scale != 1.0:
                negative_noise_pred = self.predict_noise(
                    sample=latents, timestep=timestep, **extra_input,
                    **negative_prompt_emb, **tiler_kwargs,
                    device=self.device,
                )
                noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            else:
                noise_pred = positive_noise_pred

            # DDIM
            # TODO: and fix it
            latents = self.sampler.step(latents, noise_pred, progress_id)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(timesteps))

        # Decode image
        self.load_models_to_device(['vae_decoder'])
        vae_output = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(vae_output)

        # offload all models
        self.load_models_to_device([])
        return image
