from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video
from diffsynth_engine.configs import AttnImpl

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import torch


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model(
            "Wan-AI/Wan2.2-T2V-A14B",
            revision="bf16",
            path=[
                "high_noise_model/diffusion_pytorch_model-00001-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00002-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00003-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00004-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00005-of-00006-bf16.safetensors",
                "high_noise_model/diffusion_pytorch_model-00006-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00001-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00002-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00003-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00004-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00005-of-00006-bf16.safetensors",
                "low_noise_model/diffusion_pytorch_model-00006-of-00006-bf16.safetensors",
            ],
        ),
        parallelism=8,
        offload_mode="cpu_offload",

    )
    config.cfg_degree = 1
    #config.tp_degree = 1
    config.sp_ulysses_degree = 8
    config.batch_cfg = False
    config.use_fsdp = False
    # FP8 优化选项 (根据需要选择一种)
    #config.use_fp8_linear = True                    # 使用标准 FP8
    config.use_fp8_linear_optimized = True           # 使用优化的 FP8 (推荐)
    config.fp8_low_memory_mode = True                # 低内存模式防止显存溢出
    config.model_dtype = torch.float8_e4m3fn
    config.t5_dtype = torch.float8_e4m3fn
    config.dit_attn_impl = AttnImpl.SAGE
    print(f"dit_attn_impl: {getattr(config, 'dit_attn_impl', '未知')}")
    pipe = WanVideoPipeline.from_pretrained(config)

    # 为不同噪声等级的DIT模型加载对应的T2V LoRA
    # dit (高噪声模型) 加载 high_noise LoRA
    high_noise_t2v_lora_path = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
    print(f"为高噪声DIT加载T2V LoRA: {high_noise_t2v_lora_path}")
    try:
        pipe.dit.load_loras([(high_noise_t2v_lora_path, 1.0)], fused=False)
    except Exception as e:
        print(f"高噪声T2V LoRA加载失败: {e}")

    # dit2 (低噪声模型) 加载 low_noise LoRA (如果存在)
    if pipe.dit2 is not None:
        low_noise_t2v_lora_path = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        print(f"为低噪声DIT加载T2V LoRA: {low_noise_t2v_lora_path}")
        pipe.dit2.load_loras([(low_noise_t2v_lora_path, 1.0)], fused=False)
    else:
        print("低噪声DIT模型不存在，跳过T2V LoRA加载")

    video = pipe(
        prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_frames=81,
        width=480,
        height=832,
        seed=42,
        cfg_scale=1.0
    )
    save_video(video, "wan_t2v.mp4", fps=pipe.get_default_fps())

    #del pipe
