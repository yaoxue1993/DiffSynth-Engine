import os
import torch
from typing import Callable, Dict, Union, Optional
from types import ModuleType
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.models.flux import FluxTextEncoder1, FluxTextEncoder2, FluxVAEDecoder, FluxVAEEncoder, FluxDiT
from diffsynth_engine.models.basic.tiler import FastTileWorker
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.schedulers import FlowMatchScheduler
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH


class FluxImagePipeline(BasePipeline):

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: T5TokenizerFast,
                 text_encoder_1: FluxTextEncoder1,
                 text_encoder_2: FluxTextEncoder2,
                 dit: FluxDiT,
                 vae_decoder: FluxVAEDecoder,
                 vae_encoder: FluxVAEEncoder,
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.dit = dit
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike],
                        device: str = "cuda", torch_dtype: torch.dtype = torch.float16) -> "FluxImagePipeline":
        assert os.path.isfile(pretrained_model_path) and pretrained_model_path.endswith(".safetensors"), \
            "pretrained_model_path must be a .safetensors file"
        logger.info(f"initialing {cls.__name__} from {pretrained_model_path} ...")
        loaded_state_dict = load_file(pretrained_model_path, device="cpu")

        tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
        tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)
        text_encoder_1 = FluxTextEncoder1.from_state_dict(loaded_state_dict, device=device, dtype=torch_dtype)
        text_encoder_2 = FluxTextEncoder2.from_state_dict(loaded_state_dict, device=device, dtype=torch_dtype)
        dit = FluxDiT.from_state_dict(loaded_state_dict, device=device, dtype=torch_dtype)
        vae_decoder = FluxVAEDecoder.from_state_dict(loaded_state_dict, device=device, dtype=torch.float32)
        vae_encoder = FluxVAEEncoder.from_state_dict(loaded_state_dict, device=device, dtype=torch.float32)

        pipe = cls(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            dit=dit,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            device=device,
            torch_dtype=torch_dtype,
        )
        return pipe

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, "torch.Tensor"],
                        device: str = "cuda", torch_dtype: torch.dtype = torch.float16) -> "FluxImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.dit

    def encode_image(self, image: "torch.Tensor", tiled=False, tile_size=64, tile_stride=32) -> "torch.Tensor":
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: "torch.Tensor", tiled=False, tile_size=64, tile_stride=32) -> "torch.Tensor":
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def encode_prompt(self, prompt, clip_sequence_length=77, t5_sequence_length=512):
        input_ids = self.tokenizer_1(prompt, max_length=clip_sequence_length)["input_ids"].to(device=self.device)
        _, pooled_prompt_emb = self.text_encoder_1(input_ids)

        input_ids = self.tokenizer_2(prompt, max_length=t5_sequence_length)["input_ids"].to(device=self.device)
        prompt_emb = self.text_encoder_2(input_ids)

        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=self.device,
                                                                               dtype=prompt_emb.dtype)
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}

    def prepare_extra_input(self, latents=None, guidance=1.0):
        latent_image_ids = self.dit.prepare_image_ids(latents)
        guidance = torch.tensor([guidance] * latents.shape[0], device=latents.device, dtype=latents.dtype)
        return {"image_ids": latent_image_ids, "guidance": guidance}

    def predict_noise(self,
                      hidden_states: "torch.Tensor",
                      timestep: "torch.Tensor",
                      prompt_emb: "torch.Tensor",
                      pooled_prompt_emb: "torch.Tensor",
                      guidance: "torch.Tensor",
                      text_ids: "torch.Tensor",
                      image_ids: Optional["torch.Tensor"] = None,
                      tiled: bool = False,
                      tile_size: int = 128,
                      tile_stride: int = 64,
                      ):
        if tiled:
            def flux_forward_fn(hl, hr, wl, wr):
                return self.predict_noise(
                    hidden_states=hidden_states[:, :, hl: hr, wl: wr],
                    timestep=timestep,
                    prompt_emb=prompt_emb,
                    pooled_prompt_emb=pooled_prompt_emb,
                    guidance=guidance,
                    text_ids=text_ids,
                    image_ids=None,
                    tiled=False,
                )

            return FastTileWorker().tiled_forward(
                flux_forward_fn,
                hidden_states,
                tile_size=tile_size,
                tile_stride=tile_stride,
                tile_device=hidden_states.device,
                tile_dtype=hidden_states.dtype
            )

        if image_ids is None:
            image_ids = self.dit.prepare_image_ids(hidden_states)

        conditioning = self.dit.time_embedder(timestep, hidden_states.dtype) + self.dit.pooled_text_embedder(
            pooled_prompt_emb)
        if self.dit.guidance_embedder is not None:
            guidance = guidance * 1000
            conditioning = conditioning + self.dit.guidance_embedder(guidance, hidden_states.dtype)
        prompt_emb = self.dit.context_embedder(prompt_emb)
        image_rotary_emb = self.dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        height, width = hidden_states.shape[-2:]
        hidden_states = self.dit.patchify(hidden_states)
        hidden_states = self.dit.x_embedder(hidden_states)

        # Joint Blocks
        for block_id, block in enumerate(self.dit.blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for block_id, block in enumerate(self.dit.single_blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        hidden_states = self.dit.final_norm_out(hidden_states, conditioning)
        hidden_states = self.dit.final_proj_out(hidden_states)
        hidden_states = self.dit.unpatchify(hidden_states, height, width)

        return hidden_states

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            cfg_scale: float = 1.0,
            embedded_guidance: float = 3.5,
            input_image: Optional[Image.Image] = None,
            denoising_strength: float = 1.0,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 30,
            t5_sequence_length: int = 512,
            tiled: bool = False,
            tile_size: int = 128,
            tile_stride: int = 64,
            seed: Optional[int] = None,
            progress_bar_cmd: Callable = tqdm,
            progress_bar_st: Optional[ModuleType] = None,
    ):
        """
        Args:

            TODO: add details
        """

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device=self.device,
                                        dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device=self.device,
                                          dtype=self.torch_dtype)

        # Encode prompts
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        positive_prompt_emb = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        if cfg_scale != 1.0:
            negative_prompt_emb = self.encode_prompt(negative_prompt, t5_sequence_length=t5_sequence_length)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            positive_noise_pred = self.predict_noise(
                hidden_states=latents, timestep=timestep,
                **positive_prompt_emb, **tiler_kwargs, **extra_input,
            )
            if cfg_scale != 1.0:
                negative_noise_pred = self.predict_noise(
                    hidden_states=latents, timestep=timestep,
                    **negative_prompt_emb, **tiler_kwargs, **extra_input,
                )
                noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            else:
                noise_pred = positive_noise_pred

            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        self.load_models_to_device(['vae_decoder'])
        vae_output = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(vae_output)

        # Offload all models
        self.load_models_to_device([])
        return image
