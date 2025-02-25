# DiffSynth-Engine
Diffsynth Engine is a high-performance diffusion inference engine designed for developers.

Key Features:
- Clean and Readable Code: Fully re-implements the Diffusion sampler and scheduler without relying on third-party libraries like k-diffusion, ldm, or sgm.

- Extensive Model Support: Compatible with multiple formats (e.g., CivitAI format) of base models and LoRA models , catering to diverse use cases.

- Flexible Memory Management: Supports various levels of model quantization (e.g., FP8, INT8) 
and offload strategies, enabling users to run large models (e.g., Flux.1 Dev) on limited GPU memory.

- High-Performance Inference: Optimizes the inference pipeline to achieve fast generation across various hardware environments.

- Platform Compatibility: Supports Windows, macOS (Apple Silicon), and Linux, ensuring a smooth experience across different operating systems.

## Quick Start
### Requirements

- Python 3.10+
- NVIDIA GPU with compute capability 8.6+ (e.g., RTX 50 Series, RTX 40 Series, RTX 30 Series) or Apple Silicon M-series.

### Installation

Install for PyPI
```python
pip3 install diffsynth-engine
```

Install for source
```python
git clone https://github.com/modelscope/diffsynth-engine.git && cd diffsynth-engine
pip3 install -e .
```

### Usage
Text to image
```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")
```
Text to image with LoRA
```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")

pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
pipe.load_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a girl, qipao")
image.save("image.png")
```

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contact


## Citation
If you use this codebase, or otherwise found our work valuable, please cite:

```bibtex
@misc{diffsynth-engine2025,
      title={DiffSynth-Engine: a high-performance diffusion inference engine},
      author={Zhipeng Di, Guoxuan Zhu, Zhongjie Duan, Zihao Chu, Yingda Chen, Weiyi Lu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/modelscope/diffsynth-engine}},
}
```
