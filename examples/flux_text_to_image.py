from diffsynth_engine import fetch_model, FluxImagePipeline, FluxModelConfig

#model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")

basepath = "/home/admin/workspace/aop_lab/app_source/flux"
config = FluxModelConfig(
    model_path=f'{basepath}/majicFlus_v1.safetensors',
    vae_path=f'{basepath}/ae.safetensors',
    clip_path=f'{basepath}/clip_l_bf16.safetensors',
    t5_path=f'{basepath}/t5xxl_v1_1_bf16.safetensors'    
)
pipe = FluxImagePipeline.from_pretrained(config, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")