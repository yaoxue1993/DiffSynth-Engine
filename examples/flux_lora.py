from diffsynth_engine import fetch_model, FluxImagePipeline, FluxModelConfig

#model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")

basepath = "/home/admin/workspace/aop_lab/app_source/flux"
config = FluxModelConfig(
    model_path=f'{basepath}/majicFlus_v1.safetensors',
    vae_path=f'{basepath}/ae.safetensors',
    clip_path=f'{basepath}/clip_l_bf16.safetensors',
    t5_path=f'{basepath}/t5xxl_v1_1_bf16.safetensors'    
)
pipe = FluxImagePipeline.from_pretrained(config, device='cuda:0')
pipe.patch_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a cat qipao")
image.save("image.png")