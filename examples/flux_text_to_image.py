import torch

from diffsynth_engine.utils.download import download_model
from diffsynth_engine.pipelines import FluxImagePipeline

clip_l_path = download_model("modelscope://muse/flux_clip_l?revision=20241209&endpoint=www.modelscope.cn")
t5xxl_path = download_model("modelscope://muse/google_t5_v1_1_xxl?revision=20241024105236&endpoint=www.modelscope.cn")
flux_with_vae_path = download_model("modelscope://muse/flux-with-vae?revision=20240902173035&endpoint=www.modelscope.cn")
pretrained_model_paths = [clip_l_path, t5xxl_path, flux_with_vae_path]

pipe = FluxImagePipeline.from_pretrained(pretrained_model_paths, device='cuda:0', dtype=torch.bfloat16,
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
