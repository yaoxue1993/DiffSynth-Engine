from diffsynth_engine import fetch_model, FluxImagePipeline, FluxModelConfig

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")


config = FluxModelConfig(model_path=model_path)
pipe = FluxImagePipeline.from_pretrained(config, device='cuda:0')
pipe.load_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a cat qipao")
image.save("image.png")