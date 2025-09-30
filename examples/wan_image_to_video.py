from PIL import Image
import torch
import os

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video
from diffsynth_engine.configs import AttnImpl

# 设置CUDA架构
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model(
            "Wan-AI/Wan2.2-I2V-A14B",
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

    # 配置优化选项
    config.cfg_degree = 1
    config.sp_ulysses_degree = 8
    config.batch_cfg = False
    config.use_fsdp = False

    # FP8 优化选项
    config.use_fp8_linear_optimized = True           # 使用优化的 FP8
    config.fp8_low_memory_mode = True                # 低内存模式防止显存溢出
    config.model_dtype = torch.float8_e4m3fn
    config.t5_dtype = torch.float8_e4m3fn

    # 使用稳定的注意力实现
    config.dit_attn_impl = AttnImpl.SAGE

    print(f"dit_attn_impl: {getattr(config, 'dit_attn_impl', '未知')}")
    print(f"FP8 优化: {config.use_fp8_linear_optimized}")
    print(f"低内存模式: {config.fp8_low_memory_mode}")
    pipe = WanVideoPipeline.from_pretrained(config)

    # 为不同噪声等级的DIT模型加载对应的LoRA
    # dit (高噪声模型) 加载 high_noise LoRA
    high_noise_lora_path = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
    print(f"为高噪声DIT加载LoRA: {high_noise_lora_path}")
    pipe.dit.load_loras([(high_noise_lora_path, 1.0)], fused=False)

    # dit2 (低噪声模型) 加载 low_noise LoRA (如果存在)
    if pipe.dit2 is not None:
        low_noise_lora_path = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
        print(f"为低噪声DIT加载LoRA: {low_noise_lora_path}")
        pipe.dit2.load_loras([(low_noise_lora_path, 1.0)], fused=False)
    else:
        print("低噪声DIT模型不存在，跳过LoRA加载")

    # 加载输入图像
    input_image_path = "input/wan_i2v_input.jpg"
    print(f"加载输入图像: {input_image_path}")
    image = Image.open(input_image_path).convert("RGB")

    # 生成视频
    print("开始生成视频...")
    video = pipe(
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        negative_prompt="blur, noise, low quality, artifacts, distorted, deformed",
        input_image=image,
        num_frames=81,
        width=480,
        height=832,
        seed=42,
        num_inference_steps=8,  # 可以调整推理步数
        cfg_scale=1.0
    )

    # 保存视频
    output_path = "wan_i2v.mp4"
    print(f"保存视频到: {output_path}")
    save_video(video, output_path, fps=pipe.get_default_fps())

    print("视频生成完成！")

    # 清理内存
    del pipe
    torch.cuda.empty_cache()
