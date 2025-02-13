from diffsynth_engine import fetch_model, FluxImagePipeline, FluxModelConfig

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")

config = FluxModelConfig(model_path=model_path)
pipe = FluxImagePipeline.from_pretrained(config, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")