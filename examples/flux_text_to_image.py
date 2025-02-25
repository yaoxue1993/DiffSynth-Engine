from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")