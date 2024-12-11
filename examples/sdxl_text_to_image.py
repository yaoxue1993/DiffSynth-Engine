import torch

from diffsynth_engine.utils.download import download_model
from diffsynth_engine.pipelines import SDXLImagePipeline

model_path = download_model("modelscope://muse/sd_xl_base_1.0?revision=20240425120250&endpoint=www.modelscope.cn")
pipe = SDXLImagePipeline.from_pretrained(model_path, device='cuda:0', dtype=torch.float16, cpu_offload=True)
image = pipe(prompt="a cat playing with a ball")
image.save("sdxl_text_to_image.png")
