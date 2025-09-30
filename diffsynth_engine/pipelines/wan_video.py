import torch
import torch.distributed as dist
from typing import Callable, List, Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.configs import WanPipelineConfig, WanStateDicts
from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.models.wan.wan_dit import WanDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.models.wan.wan_image_encoder import WanImageEncoder
from diffsynth_engine.models.basic.lora import LoRAContext
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.pipelines import BasePipeline, LoRAStateDictConverter
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.fp8_linear_optimized import enable_fp8_linear_optimized
from diffsynth_engine.utils.parallel import ParallelWrapper
from diffsynth_engine.utils import logging


logger = logging.get_logger(__name__)




class WanLoRAConverter(LoRAStateDictConverter):
    def _from_diffsynth(self, state_dict):
        dit_dict = {}
        for key, param in state_dict.items():
            lora_args = {}
            if ".lora_A.default.weight" not in key:
                continue

            lora_args["up"] = state_dict[key.replace(".lora_A.default.weight", ".lora_B.default.weight")]
            lora_args["down"] = param
            lora_args["rank"] = lora_args["up"].shape[1]
            if key.replace(".lora_A.default.weight", ".alpha") in state_dict:
                lora_args["alpha"] = state_dict[key.replace(".lora_A.default.weight", ".alpha")]
            else:
                lora_args["alpha"] = lora_args["rank"]
            key = key.replace(".lora_A.default.weight", "")
            dit_dict[key] = lora_args
        # Lightning LoRA should be applied to dit2 (low noise model)
        return {"dit2": dit_dict}

    def _from_civitai(self, state_dict):
        dit_dict = {}
        for key, param in state_dict.items():
            if ".lora_A.weight" not in key:
                continue

            lora_args = {}
            lora_args["up"] = state_dict[key.replace(".lora_A.weight", ".lora_B.weight")]
            lora_args["down"] = param
            lora_args["rank"] = lora_args["up"].shape[1]
            if key.replace(".lora_A.weight", ".alpha") in state_dict:
                lora_args["alpha"] = state_dict[key.replace(".lora_A.weight", ".alpha")]
            else:
                lora_args["alpha"] = lora_args["rank"]
            key = key.replace("diffusion_model.", "").replace(".lora_A.weight", "")
            dit_dict[key] = lora_args
        # Lightning LoRA should be applied to dit2 (low noise model)
        return {"dit2": dit_dict}

    def _from_fun(self, state_dict):
        dit_dict = {}
        for key, param in state_dict.items():
            if ".lora_down.weight" not in key:
                continue

            lora_args = {}
            lora_args["up"] = state_dict[key.replace(".lora_down.weight", ".lora_up.weight")]
            lora_args["down"] = param
            lora_args["rank"] = lora_args["up"].shape[1]
            if key.replace(".lora_down.weight", ".alpha") in state_dict:
                lora_args["alpha"] = state_dict[key.replace(".lora_down.weight", ".alpha")]
            else:
                lora_args["alpha"] = lora_args["rank"]
            key = key.replace("lora_unet_blocks_", "blocks.").replace(".lora_down.weight", "")
            key = key.replace("_self_attn_", ".self_attn.")
            key = key.replace("_cross_attn_", ".cross_attn.")
            key = key.replace("_ffn_", ".ffn.")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def _from_lightning(self, state_dict):
        """Convert Lightning LoRA format with .lora_up/.lora_down naming"""
        dit_dict = {}
        for key, param in state_dict.items():
            if ".lora_down.weight" not in key:
                continue

            lora_args = {}
            lora_args["up"] = state_dict[key.replace(".lora_down.weight", ".lora_up.weight")]
            lora_args["down"] = param
            lora_args["rank"] = lora_args["up"].shape[1]
            if key.replace(".lora_down.weight", ".alpha") in state_dict:
                lora_args["alpha"] = state_dict[key.replace(".lora_down.weight", ".alpha")]
            else:
                lora_args["alpha"] = lora_args["rank"]
            # Clean up the key: remove diffusion_model prefix and .lora_down.weight suffix
            clean_key = key.replace("diffusion_model.", "").replace(".lora_down.weight", "")
            dit_dict[clean_key] = lora_args
        # Lightning LoRA should be applied to dit2 (low noise model)
        return {"dit2": dit_dict}

    def convert(self, state_dict):
        if "lora_unet_blocks_0_cross_attn_k.lora_down.weight" in state_dict:
            state_dict = self._from_fun(state_dict)
            logger.info("use fun format state dict")
        elif "diffusion_model.blocks.0.cross_attn.k.lora_A.weight" in state_dict:
            state_dict = self._from_civitai(state_dict)
            logger.info("use civitai format state dict")
        elif "diffusion_model.blocks.0.cross_attn.k.lora_down.weight" in state_dict:
            state_dict = self._from_lightning(state_dict)
            logger.info("use lightning lora format state dict")
        else:
            state_dict = self._from_diffsynth(state_dict)
            logger.info("use diffsynth format state dict")
        return state_dict


class WanVideoPipeline(BasePipeline):
    lora_converter = WanLoRAConverter()

    def __init__(
        self,
        config: WanPipelineConfig,
        tokenizer: WanT5Tokenizer,
        text_encoder: WanTextEncoder,
        dit: WanDiT,
        dit2: WanDiT | None,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
    ):
        super().__init__(
            vae_tiled=config.vae_tiled,
            vae_tile_size=config.vae_tile_size,
            vae_tile_stride=config.vae_tile_stride,
            device=config.device,
            dtype=config.model_dtype,
        )
        self.config = config
        self.upsampling_factor = vae.upsampling_factor
        # sampler
        self.noise_scheduler = RecifitedFlowScheduler(
            shift=config.shift if config.shift is not None else 5.0,
            sigma_min=0.001,
            sigma_max=0.999,
        )
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dit = dit  # high noise model
        self.dit2 = dit2  # low noise model
        self.vae = vae
        self.image_encoder = image_encoder
        self.model_names = ["text_encoder", "dit", "dit2", "vae", "image_encoder"]

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        assert self.config.tp_degree is None or self.config.tp_degree == 1, (
            "load LoRA is not allowed when tensor parallel is enabled; "
            "set tp_degree=None or tp_degree=1 during pipeline initialization"
        )
        assert not (self.config.use_fsdp and fused), (
            "load fused LoRA is not allowed when fully sharded data parallel is enabled; "
            "either load LoRA with fused=False or set use_fsdp=False during pipeline initialization"
        )
        super().load_loras(lora_list, fused, save_original_weight)

    def load_loras_enhanced(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        """Enhanced LoRA loading that works with ParallelWrapper and sequence parallel"""
        # Check constraints
        assert self.config.tp_degree is None or self.config.tp_degree == 1, (
            "load LoRA is not allowed when tensor parallel is enabled; "
            "set tp_degree=None or tp_degree=1 during pipeline initialization"
        )
        assert not (self.config.use_fsdp and fused), (
            "load fused LoRA is not allowed when fully sharded data parallel is enabled; "
            "either load LoRA with fused=False or set use_fsdp=False during pipeline initialization"
        )

        # Check if this is running inside a ParallelWrapper worker
        import torch.distributed as dist
        is_parallel = hasattr(self.config, 'sp_ulysses_degree') and self.config.sp_ulysses_degree > 1

        if is_parallel and dist.is_available() and dist.is_initialized():
            # In sequence parallel mode, load LoRAs on all workers
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            logger.info(f"Loading LoRAs in sequence parallel mode (rank {rank}/{world_size})")

            # Ensure all workers have the same LoRA files
            for lora_path, lora_scale in lora_list:
                logger.info(f"[Rank {rank}] loading LoRA from {lora_path} with scale {lora_scale}")

                # Load LoRA state dict on each worker - use CPU to avoid device conflicts
                from diffsynth_engine.utils.loader import load_file
                state_dict = load_file(lora_path, device="cpu")
                lora_state_dict = self.lora_converter.convert(state_dict)

                # Apply LoRAs to local models with proper device placement
                for model_name, lora_params in lora_state_dict.items():
                    logger.info(f"[Rank {rank}] Processing LoRA for model: {model_name}")
                    # Lightning LoRA should only be applied to dit2 (low noise model)
                    if model_name == "dit2":
                        model = getattr(self, model_name, None)
                        logger.info(f"[Rank {rank}] dit2 model exists: {model is not None}")
                        if model is not None:
                            # Get the actual device this model is on
                            # In sequence parallel mode, use the configured device for the current rank
                            if hasattr(model, 'parameters'):
                                try:
                                    # Try to get device from first parameter
                                    model_device = next(model.parameters()).device
                                    # If device is CPU but we're in parallel mode, use current CUDA device
                                    if model_device.type == 'cpu' and torch.cuda.is_available():
                                        current_device = torch.cuda.current_device()
                                        model_device = torch.device(f'cuda:{current_device}')
                                        logger.info(f"[Rank {rank}] Model parameters on CPU, using current CUDA device: {model_device}")
                                except:
                                    model_device = self.device
                            else:
                                model_device = self.device

                            lora_args = []
                            for key, param in lora_params.items():
                                # Move LoRA parameters to the correct device
                                lora_args.append(
                                    {
                                        "name": key,
                                        "scale": lora_scale,
                                        "rank": param["rank"],
                                        "alpha": param["alpha"],
                                        "up": param["up"].to(model_device),
                                        "down": param["down"].to(model_device),
                                        "device": model_device,
                                        "dtype": self.dtype,
                                        "save_original_weight": save_original_weight,
                                    }
                                )
                            model.load_loras(lora_args, fused=fused)
                            logger.info(f"[Rank {rank}] Applied LoRA to {model_name} (low noise model) on device {model_device}")
                        else:
                            logger.warning(f"[Rank {rank}] dit2 model not found, skipping LoRA loading")
                    elif model_name == "dit":
                        logger.info(f"[Rank {rank}] Skipping LoRA for {model_name} (high noise model) - Lightning LoRA is for low noise only")

            # Synchronize all workers
            dist.barrier()
            logger.info(f"[Rank {rank}] LoRA loading synchronized across all workers")
        else:
            # Non-parallel mode, use standard method
            logger.info("Loading LoRAs in non-parallel mode")
            super().load_loras(lora_list, fused, save_original_weight)

    def unload_loras(self):
        self.dit.unload_loras()
        self.text_encoder.unload_loras()

    def get_default_fps(self) -> int:
        return self.config.fps

    def encode_prompt(self, prompt):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = prompt_emb.masked_fill(mask.unsqueeze(-1).expand_as(prompt_emb) == 0, 0)
        return prompt_emb

    def encode_clip_feature(self, images: Image.Image | List[Image.Image], height, width):
        if not images or not self.dit.has_clip_feature:
            return None

        self.load_models_to_device(["image_encoder"])
        if isinstance(images, Image.Image):
            images = [images]
        images = [self.preprocess_image(img.resize((width, height), Image.Resampling.LANCZOS)) for img in images]
        images = [img.to(device=self.device, dtype=self.config.image_encoder_dtype) for img in images]
        clip_context = self.image_encoder.encode_image(images).to(self.dtype)
        return clip_context

    def encode_vae_feature(self, images: Image.Image | List[Image.Image], num_frames, height, width):
        if not images or not self.dit.has_vae_feature:
            return None

        self.load_models_to_device(["vae"])
        if isinstance(images, Image.Image):
            images = [images]
        images = [self.preprocess_image(img.resize((width, height), Image.Resampling.LANCZOS)) for img in images]
        indices = torch.linspace(0, num_frames - 1, len(images), dtype=torch.long)
        msk = torch.zeros(
            1,
            num_frames,
            height // self.upsampling_factor,
            width // self.upsampling_factor,
            device=self.device,
            dtype=self.config.vae_dtype,
        )
        msk[:, indices] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // self.upsampling_factor, width // self.upsampling_factor)
        msk = msk.transpose(1, 2).squeeze(0)

        video = torch.zeros(3, num_frames, height, width).to(device=self.device, dtype=self.config.vae_dtype)
        video[:, indices] = torch.concat([img.transpose(0, 1) for img in images], dim=1).to(
            device=self.device, dtype=self.config.vae_dtype
        )
        y = self.vae.encode([video], device=self.device)[0]
        y = torch.concat([msk, y]).to(dtype=self.dtype)
        return y.unsqueeze(0)

    def encode_image_latents(self, images: Image.Image | List[Image.Image], height, width):
        if not images or not self.dit.fuse_image_latents:
            return

        self.load_models_to_device(["vae"])
        if isinstance(images, Image.Image):
            images = [images]
        frames = [self.preprocess_image(img.resize((width, height), Image.Resampling.LANCZOS)) for img in images]
        video = torch.stack(frames, dim=2).squeeze(0)
        latents = self.encode_video([video]).to(dtype=self.dtype, device=self.device)
        return latents

    def encode_video(self, videos: List[torch.Tensor]) -> torch.Tensor:
        videos = [video.to(dtype=self.config.vae_dtype, device=self.device) for video in videos]
        latents = self.vae.encode(
            videos,
            device=self.device,
            tiled=self.vae_tiled,
            tile_size=self.vae_tile_size,
            tile_stride=self.vae_tile_stride,
        )
        latents = latents.to(dtype=self.config.model_dtype, device=self.device)
        return latents

    def decode_video(self, latents: torch.Tensor, progress_callback=None) -> List[torch.Tensor]:
        latents = latents.to(dtype=self.config.vae_dtype, device=self.device)
        videos = self.vae.decode(
            latents,
            device=self.device,
            tiled=self.vae_tiled,
            tile_size=self.vae_tile_size,
            tile_stride=self.vae_tile_stride,
            progress_callback=progress_callback,
        )
        videos = [video.to(dtype=self.config.model_dtype, device=self.device) for video in videos]
        return videos

    def predict_noise_with_cfg(
        self,
        model: WanDiT,
        latents: torch.Tensor,
        image_clip_feature: torch.Tensor,
        image_y: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool,
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                model=model,
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=positive_prompt_emb,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=positive_prompt_emb,
            )
            negative_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=negative_prompt_emb,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            if image_y is not None:
                image_y = torch.cat([image_y, image_y], dim=0)
            if image_clip_feature is not None:
                image_clip_feature = torch.cat([image_clip_feature, image_clip_feature], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                model=model,
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=prompt_emb,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, model, latents, image_clip_feature, image_y, timestep, context):
        latents = latents.to(dtype=self.config.model_dtype, device=self.device)

        noise_pred = model(
            x=latents,
            timestep=timestep,
            context=context,
            clip_feature=image_clip_feature,
            y=image_y,
        )
        return noise_pred

    def prepare_latents(
        self,
        latents,
        input_video,
        denoising_strength,
        num_inference_steps,
    ):
        height, width = latents.shape[-2:]
        height, width = height * self.upsampling_factor, width * self.upsampling_factor
        if input_video is not None:  # video to video
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            self.load_models_to_device(["vae"])
            noise = latents
            frames = [
                self.preprocess_image(frame.resize((width, height), Image.Resampling.LANCZOS)) for frame in input_video
            ]
            video = torch.stack(frames, dim=2).squeeze(0)
            video = video.to(dtype=self.config.vae_dtype, device=self.device)
            latents = self.encode_video([video]).to(dtype=latents.dtype, device=latents.device)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)
            init_latents = latents.clone()

        return init_latents, latents, sigmas, timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image: Image.Image | List[Image.Image] | None = None,
        input_video: List[Image.Image] | None = None,
        denoising_strength=1.0,
        seed=None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=None,
        num_inference_steps=None,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        divisor = 32 if self.vae.z_dim == 48 else 16  # 32 for wan2.2 vae, 16 for wan2.1 vae
        assert height % divisor == 0 and width % divisor == 0, f"height and width must be divisible by {divisor}"
        assert (num_frames - 1) % 4 == 0, "num_frames must be 4X+1"
        cfg_scale = self.config.cfg_scale if cfg_scale is None else cfg_scale
        num_inference_steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps

        # Initialize noise
        if dist.is_initialized() and seed is None:
            raise ValueError("must provide a seed when parallelism is enabled")
        noise = self.generate_noise(
            (
                1,
                self.vae.z_dim,
                (num_frames - 1) // 4 + 1,
                height // self.upsampling_factor,
                width // self.upsampling_factor,
            ),
            seed=seed,
            device="cpu",
            dtype=torch.float32,
        ).to(self.device)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(
            noise,
            input_video,
            denoising_strength,
            num_inference_steps,
        )
        mask = torch.ones((1, 1, *latents.shape[2:]), dtype=latents.dtype, device=latents.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = self.encode_prompt(negative_prompt)

        # Encode image
        image_clip_feature = self.encode_clip_feature(input_image, height, width)
        image_y = self.encode_vae_feature(input_image, num_frames, height, width)
        image_latents = self.encode_image_latents(input_image, height, width)
        if image_latents is not None:
            latents[:, :, : image_latents.shape[2], :, :] = image_latents
            init_latents = latents.clone()
            mask[:, :, : image_latents.shape[2], :, :] = 0

        # Initialize sampler
        self.sampler.initialize(sigmas=sigmas)

        # Denoise
        hide_progress = dist.is_initialized() and dist.get_rank() != 0
        for i, timestep in enumerate(tqdm(timesteps, disable=hide_progress)):
            if timestep.item() / 1000 >= self.config.boundary:
                self.load_models_to_device(["dit"])
                model = self.dit
                cfg_scale_ = cfg_scale if isinstance(cfg_scale, float) else cfg_scale[1]
            else:
                self.load_models_to_device(["dit2"])
                model = self.dit2
                cfg_scale_ = cfg_scale if isinstance(cfg_scale, float) else cfg_scale[0]

            timestep = timestep * mask[:, :, :, ::2, ::2].flatten()  # seq_len
            timestep = timestep.to(dtype=self.dtype, device=self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                model=model,
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=prompt_emb_posi,
                negative_prompt_emb=prompt_emb_nega,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                cfg_scale=cfg_scale_,
                batch_cfg=self.config.batch_cfg,
            )
            # Scheduler
            latents = self.sampler.step(latents, noise_pred, i)
            latents = latents * mask + init_latents * (1 - mask)
            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps), "DENOISING")

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, progress_callback=progress_callback)
        frames = self.vae_output_to_image(frames)
        return frames

    @classmethod
    def from_pretrained(cls, model_path_or_config: WanPipelineConfig) -> "WanVideoPipeline":
        if isinstance(model_path_or_config, str):
            config = WanPipelineConfig(model_path=model_path_or_config)
        else:
            config = model_path_or_config

        dit_state_dict, dit2_state_dict = None, None
        if isinstance(config.model_path, list):
            high_noise_model_ckpt = [path for path in config.model_path if "high_noise_model" in path]
            low_noise_model_ckpt = [path for path in config.model_path if "low_noise_model" in path]
            if high_noise_model_ckpt and low_noise_model_ckpt:
                logger.info(f"loading high noise model state dict from {high_noise_model_ckpt} ...")
                dit_state_dict = cls.load_model_checkpoint(high_noise_model_ckpt, device="cpu", dtype=config.model_dtype)
                logger.info(f"loading low noise model state dict from {low_noise_model_ckpt} ...")
                dit2_state_dict = cls.load_model_checkpoint(low_noise_model_ckpt, device="cpu", dtype=config.model_dtype)
        if dit_state_dict is None:
            logger.info(f"loading dit state dict from {config.model_path} ...")
            dit_state_dict = cls.load_model_checkpoint(config.model_path, device="cpu", dtype=config.model_dtype)

        model_state_dict = dit_state_dict
        if dit2_state_dict is not None:
            model_state_dict = {
                "high_noise_model": dit_state_dict,
                "low_noise_model": dit2_state_dict,
            }

        if config.t5_path is None:
            config.t5_path = fetch_model("muse/wan2.1-umt5", path="umt5.safetensors")
        logger.info(f"loading t5 state dict from {config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(config.t5_path, device="cpu", dtype=config.t5_dtype)

        if config.vae_path is None:
            config.vae_path = (
                fetch_model("muse/wan2.2-vae", path="vae.safetensors")
                if cls._get_dit_type(model_state_dict) == "wan2.2-ti2v-5b"
                else fetch_model("muse/wan2.1-vae", path="vae.safetensors")
            )
        logger.info(f"loading vae state dict from {config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(config.vae_path, device="cpu", dtype=config.vae_dtype)

        image_encoder_state_dict = None
        if config.image_encoder_path is not None:
            logger.info(f"loading state dict from {config.image_encoder_path} ...")
            image_encoder_state_dict = cls.load_model_checkpoint(config.image_encoder_path, device="cpu", dtype=config.image_encoder_dtype)

        state_dicts = WanStateDicts(
            model=model_state_dict,
            t5=t5_state_dict,
            vae=vae_state_dict,
            image_encoder=image_encoder_state_dict,
        )
        return cls.from_state_dict(state_dicts, config)

    @classmethod
    def from_state_dict(cls, state_dicts: WanStateDicts, config: WanPipelineConfig) -> "WanVideoPipeline":
        if config.parallelism > 1:
            pipe = ParallelWrapper(
                cfg_degree=config.cfg_degree,
                sp_ulysses_degree=config.sp_ulysses_degree,
                sp_ring_degree=config.sp_ring_degree,
                tp_degree=config.tp_degree,
                use_fsdp=config.use_fsdp,
            )
            pipe.load_module(cls._from_state_dict, state_dicts=state_dicts, config=config)
        else:
            pipe = cls._from_state_dict(state_dicts, config)
        return pipe

    @classmethod
    def _from_state_dict(cls, state_dicts: WanStateDicts, config: WanPipelineConfig) -> "WanVideoPipeline":
        # default params from model config
        vae_type = cls._get_vae_type(state_dicts.vae)
        dit_type = cls._get_dit_type(state_dicts.model)
        vae_config: dict = WanVideoVAE.get_model_config(vae_type)
        dit_config: dict = WanDiT.get_model_config(dit_type)
        config.boundary = dit_config.pop("boundary", -1.0)
        config.shift = dit_config.pop("shift", 5.0)
        config.cfg_scale = dit_config.pop("cfg_scale", 5.0)
        config.num_inference_steps = dit_config.pop("num_inference_steps", 50)
        config.fps = dit_config.pop("fps", 16)

        init_device = "cpu" if config.offload_mode is not None else config.device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean="whitespace")
        text_encoder = WanTextEncoder.from_state_dict(state_dicts.t5, device=init_device, dtype=config.t5_dtype)
        vae = WanVideoVAE.from_state_dict(
            state_dicts.vae, config=vae_config, device=init_device, dtype=config.vae_dtype
        )

        image_encoder = None
        if state_dicts.image_encoder is not None:
            image_encoder = WanImageEncoder.from_state_dict(
                state_dicts.image_encoder,
                device=init_device,
                dtype=config.image_encoder_dtype,
            )

        dit_state_dict, dit2_state_dict = None, None
        if "high_noise_model" in state_dicts.model and "low_noise_model" in state_dicts.model:
            dit_state_dict = state_dicts.model["high_noise_model"]
            dit2_state_dict = state_dicts.model["low_noise_model"]
        else:
            dit_state_dict = state_dicts.model

        with LoRAContext():
            attn_kwargs = {
                "attn_impl": config.dit_attn_impl.value,
                "sparge_smooth_k": config.sparge_smooth_k,
                "sparge_cdfthreshd": config.sparge_cdfthreshd,
                "sparge_simthreshd1": config.sparge_simthreshd1,
                "sparge_pvthreshd": config.sparge_pvthreshd,
            }
            dit = WanDiT.from_state_dict(
                dit_state_dict,
                config=dit_config,
                device=init_device,
                dtype=config.model_dtype,
                attn_kwargs=attn_kwargs,
            )
            if config.use_fp8_linear_optimized:
                enable_fp8_linear_optimized(dit, low_memory_mode=config.fp8_low_memory_mode)
                logger.info(f"Enabled optimized FP8 linear layers for WAN DiT (low_memory={config.fp8_low_memory_mode})")
            elif config.use_fp8_linear:
                enable_fp8_linear(dit)

            dit2 = None
            if dit2_state_dict is not None:
                dit2 = WanDiT.from_state_dict(
                    dit2_state_dict,
                    config=dit_config,
                    device=init_device,
                    dtype=config.model_dtype,
                    attn_kwargs=attn_kwargs,
                )
                if config.use_fp8_linear_optimized:
                    enable_fp8_linear_optimized(dit2, low_memory_mode=config.fp8_low_memory_mode)
                    logger.info(f"Enabled optimized FP8 linear layers for WAN DiT2 (low_memory={config.fp8_low_memory_mode})")
                elif config.use_fp8_linear:
                    enable_fp8_linear(dit2)

        pipe = cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            dit2=dit2,
            vae=vae,
            image_encoder=image_encoder,
        )
        pipe.eval()

        if config.offload_mode is not None:
            pipe.enable_cpu_offload(config.offload_mode)

        if config.model_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["dit", "dit2"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.t5_dtype == torch.float8_e4m3fn:
            pipe.dtype = torch.bfloat16  # compute dtype
            pipe.enable_fp8_autocast(
                model_names=["text_encoder"], compute_dtype=pipe.dtype, use_fp8_linear=config.use_fp8_linear
            )

        if config.use_torch_compile:
            pipe.compile()
        return pipe

    @staticmethod
    def _get_dit_type(model_state_dict: Dict[str, torch.Tensor] | Dict[str, Dict[str, torch.Tensor]]) -> str:
        # determine wan dit type by model params
        dit_type = None
        if "high_noise_model" in model_state_dict and "low_noise_model" in model_state_dict:
            if model_state_dict["high_noise_model"]["patch_embedding.weight"].shape[1] == 36:
                dit_type = "wan2.2-i2v-a14b"
            elif model_state_dict["high_noise_model"]["patch_embedding.weight"].shape[1] == 16:
                dit_type = "wan2.2-t2v-a14b"
        elif model_state_dict["patch_embedding.weight"].shape[1] == 48:
            dit_type = "wan2.2-ti2v-5b"
        elif "img_emb.emb_pos" in model_state_dict:
            dit_type = "wan2.1-flf2v-14b"
        elif "img_emb.proj.0.weight" in model_state_dict:
            dit_type = "wan2.1-i2v-14b"
        elif "blocks.39.self_attn.norm_q.weight" in model_state_dict:
            dit_type = "wan2.1-t2v-14b"
        else:
            dit_type = "wan2.1-t2v-1.3b"
        return dit_type

    @staticmethod
    def _get_vae_type(vae_state_dict: Dict[str, torch.Tensor]) -> str:
        # determine wan vae type by model params
        vae_type = "wan2.1-vae"
        if vae_state_dict["encoder.conv1.weight"].shape[1] == 12:  # in_channels
            vae_type = "wan2.2-vae"
        return vae_type

    def compile(self):
        self.dit.compile_repeated_blocks(dynamic=True)
        if self.dit2 is not None:
            self.dit2.compile_repeated_blocks(dynamic=True)
