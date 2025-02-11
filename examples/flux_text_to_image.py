import torch

from diffsynth_engine import fetch_modelscope_model
from diffsynth_engine.pipelines import FluxImagePipeline

model_path = fetch_modelscope_model("muse/flux-with-vae", revision="20240902173035", subpath="flux1-dev-with-vae.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0', dtype=torch.bfloat16,
                                         cpu_offload=True)
image = pipe(
    prompt="A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    embedded_guidance=3.5,
    num_inference_steps=50,
    seed=42
)
image.save("flux_text_to_image.png")
