import torch

from diffsynth_engine.utils.download import download_model
from diffsynth_engine.pipelines import SDXLImagePipeline

model_path = download_model("modelscope://muse/sd_xl_base_1.0?revision=20240425120250&endpoint=www.modelscope.cn")
pipe = SDXLImagePipeline.from_pretrained(model_path, device='cuda:0', dtype=torch.float16, cpu_offload=True)
image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=20,
    seed=42
)
image.save("sdxl_text_to_image.png")
