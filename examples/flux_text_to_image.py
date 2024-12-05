import torch

from diffsynth_engine.utils.download import download_model
from diffsynth_engine.pipelines import FluxImagePipeline

clip_l_path = download_model("modelscope://muse/sd35_clip_l?revision=v1&endpoint=www.modelscope.cn")
clip_g_path = download_model("modelscope://muse/sd35_clip_g?revision=v1&endpoint=www.modelscope.cn")
t5xxl_path = download_model("modelscope://muse/google_t5_v1_1_xxl?revision=20241024105236&endpoint=www.modelscope.cn")
vae_path = download_model("modelscope://muse/flux_vae?revision=20241015120836&endpoint=www.modelscope.cn")
dit_path = download_model("modelscope://MusePublic/489_ckpt_FLUX_1?revision=2172&endpoint=www.modelscope.cn")
pretrained_model_paths = [clip_l_path, t5xxl_path, vae_path, dit_path]

pipe = FluxImagePipeline.from_pretrained(pretrained_model_paths, device='cuda:0', dtype=torch.bfloat16,
                                         cpu_offload=True)
image = pipe(prompt="a cat playing with a ball")
image.save("flux_text_to_image.png")
