from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
pipe.patch_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a cat")
image.save("image.png")