#!/usr/bin/env python3
"""
WAN Text-to-Video FastAPI Service
支持动态控制Lightning LoRA和推理参数
"""

import os
import torch
import asyncio
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from torch.profiler import profile, record_function, ProfilerActivity

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video
from diffsynth_engine.configs import AttnImpl

# 设置CUDA架构
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

app = FastAPI(
    title="WAN Text-to-Video API",
    description="基于WAN模型的文本转视频API服务，支持Lightning LoRA优化",
    version="1.0.0"
)

# 全局变量存储模型
pipe = None
lightning_lora_loaded = False

class T2VRequest(BaseModel):
    """文本转视频请求参数"""
    prompt: str = Field(..., description="视频生成提示词")
    negative_prompt: str = Field(
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        description="负面提示词"
    )
    num_inference_steps: int = Field(default=30, ge=1, le=100, description="推理步数")
    cfg_scale: float = Field(default=1.0, ge=0.1, le=20.0, description="CFG引导强度")
    num_frames: int = Field(default=81, ge=16, le=200, description="视频帧数")
    width: int = Field(default=480, ge=256, le=1024, description="视频宽度")
    height: int = Field(default=832, ge=256, le=1024, description="视频高度")
    seed: Optional[int] = Field(default=42, ge=0, description="随机种子")
    fast: bool = Field(default=False, description="是否使用Lightning LoRA快速模式")

class T2VResponse(BaseModel):
    """文本转视频响应"""
    message: str
    video_path: str
    config: dict

async def load_model():
    """加载WAN T2V模型"""
    global pipe

    print("正在初始化WAN T2V模型...")

    # 配置模型
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

    # 优化配置
    config.cfg_degree = 1
    config.sp_ulysses_degree = 8
    config.batch_cfg = False
    config.use_fsdp = False

    # FP8优化
    config.use_fp8_linear_optimized = True
    config.fp8_low_memory_mode = True
    config.model_dtype = torch.float8_e4m3fn
    config.t5_dtype = torch.float8_e4m3fn
    config.dit_attn_impl = AttnImpl.SAGE

    print("正在加载模型权重...")
    pipe = WanVideoPipeline.from_pretrained(config)

    print("WAN T2V模型加载完成！")

async def load_lightning_lora():
    """加载Lightning LoRA"""
    global lightning_lora_loaded

    if lightning_lora_loaded:
        return

    print("正在加载Lightning LoRA...")

    try:
        # 使用增强的LoRA加载方法，支持sequence parallel
        low_noise_t2v_lora_path = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        if os.path.exists(low_noise_t2v_lora_path):
            print(f"加载Lightning LoRA: {low_noise_t2v_lora_path}")

            # 检查是否有增强的LoRA加载方法
            # 使用正常的scale (1.0)
            if hasattr(pipe, 'load_loras_enhanced'):
                pipe.load_loras_enhanced([(low_noise_t2v_lora_path, 1.0)], fused=False)
                print("Lightning LoRA加载成功 (增强模式，scale=1.0)")
            else:
                pipe.load_loras([(low_noise_t2v_lora_path, 1.0)], fused=False)
                print("Lightning LoRA加载成功 (标准模式，scale=1.0)")
        else:
            print("Lightning LoRA文件不存在")

        lightning_lora_loaded = True
        print("Lightning LoRA加载完成！")

    except Exception as e:
        print(f"Lightning LoRA加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

async def unload_lightning_lora():
    """卸载Lightning LoRA"""
    global lightning_lora_loaded

    if not lightning_lora_loaded:
        return

    print("正在卸载Lightning LoRA...")

    try:
        pipe.unload_loras()
        lightning_lora_loaded = False
        print("Lightning LoRA卸载完成！")

    except Exception as e:
        print(f"Lightning LoRA卸载失败: {e}")

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    await load_model()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "WAN Text-to-Video API Service",
        "status": "running",
        "lightning_lora_loaded": lightning_lora_loaded
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "lightning_lora_loaded": lightning_lora_loaded
    }

@app.post("/generate", response_model=T2VResponse)
async def generate_video(request: T2VRequest):
    """生成视频"""
    global pipe, lightning_lora_loaded

    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 根据fast参数决定是否使用Lightning LoRA
        if request.fast and not lightning_lora_loaded:
            await load_lightning_lora()
        elif not request.fast and lightning_lora_loaded:
            await unload_lightning_lora()

        # 根据是否使用Lightning LoRA调整默认推理步数
        inference_steps = request.num_inference_steps

        print(f"开始生成视频 - 快速模式: {request.fast}, 推理步数: {inference_steps}")

        # 生成视频
        video = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            seed=request.seed,
            num_inference_steps=inference_steps,
            cfg_scale=request.cfg_scale
        )

        # 保存视频到临时文件
        temp_dir = tempfile.mkdtemp()
        video_filename = f"wan_t2v_{hash(request.prompt) % 10000}.mp4"
        video_path = os.path.join(temp_dir, video_filename)

        save_video(video, video_path, fps=pipe.get_default_fps())

        config_info = {
            "inference_steps": inference_steps,
            "cfg_scale": request.cfg_scale,
            "num_frames": request.num_frames,
            "resolution": f"{request.width}x{request.height}",
            "lightning_lora": request.fast,
            "seed": request.seed
        }

        return T2VResponse(
            message="视频生成成功",
            video_path=video_path,
            config=config_info
        )

    except Exception as e:
        print(f"生成视频时出错: {e}")
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")

@app.get("/download/{video_path:path}")
async def download_video(video_path: str):
    """下载生成的视频"""
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")

    return FileResponse(
        path=video_path,
        media_type='video/mp4',
        filename=os.path.basename(video_path)
    )

@app.post("/toggle_lightning_lora")
async def toggle_lightning_lora(enable: bool):
    """手动切换Lightning LoRA状态"""
    global lightning_lora_loaded

    try:
        if enable and not lightning_lora_loaded:
            await load_lightning_lora()
        elif not enable and lightning_lora_loaded:
            await unload_lightning_lora()

        return {
            "message": f"Lightning LoRA {'启用' if enable else '禁用'}成功",
            "lightning_lora_loaded": lightning_lora_loaded
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换Lightning LoRA失败: {str(e)}")

@app.get("/memory_analysis")
async def memory_analysis():
    """分析显存占用情况"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        analysis_result = {}

        # 基本显存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

                analysis_result[f"gpu_{i}"] = {
                    "allocated_gb": round(allocated, 2),
                    "cached_gb": round(cached, 2),
                    "max_allocated_gb": round(max_allocated, 2)
                }

        # 模型信息
        analysis_result["models"] = {
            "dit_type": str(type(pipe.dit)),
            "dit2_type": str(type(pipe.dit2)) if pipe.dit2 is not None else None,
            "dit2_exists": pipe.dit2 is not None,
            "lightning_lora_loaded": lightning_lora_loaded
        }

        # 配置信息
        analysis_result["config"] = {
            "parallelism": pipe.config.parallelism,
            "sp_ulysses_degree": pipe.config.sp_ulysses_degree,
            "use_fsdp": pipe.config.use_fsdp,
            "offload_mode": pipe.config.offload_mode,
            "model_dtype": str(pipe.config.model_dtype),
            "use_fp8_linear_optimized": pipe.config.use_fp8_linear_optimized
        }

        return analysis_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内存分析失败: {str(e)}")

@app.post("/profile_generation")
async def profile_generation(prompt: str = "测试视频生成"):
    """使用torch profiler分析一次视频生成的显存使用"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        profile_result = {}

        # 记录分析前的显存状态
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            profile_result["initial_memory_gb"] = round(initial_memory, 2)

        # 使用torch profiler进行分析
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            use_cuda=True
        ) as prof:
            with record_function("video_generation"):
                # 生成一个小视频用于分析
                video = pipe(
                    prompt=prompt,
                    negative_prompt="低质量",
                    num_frames=9,  # 减少帧数以快速分析
                    width=256,     # 减少分辨率
                    height=256,
                    seed=42,
                    num_inference_steps=2  # 减少步数
                )

        # 分析结果
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            profile_result["peak_memory_gb"] = round(peak_memory, 2)
            profile_result["memory_increase_gb"] = round(peak_memory - initial_memory, 2)

        # 获取关键统计信息
        key_stats = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
        profile_result["top_memory_operations"] = key_stats

        # 获取CUDA内核统计
        cuda_stats = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_memory_usage", row_limit=5
        )
        profile_result["cuda_kernel_stats"] = cuda_stats

        return {
            "message": "显存分析完成",
            "profile_data": profile_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling失败: {str(e)}")

if __name__ == "__main__":
    # 运行FastAPI服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
