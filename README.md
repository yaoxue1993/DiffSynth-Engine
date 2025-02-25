# DiffSynth-Engine

[![PyPI](https://img.shields.io/pypi/v/DiffSynth-Engine)](https://pypi.org/project/DiffSynth-Engine/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Engine.svg)](https://github.com/modelscope/DiffSynth-Engine/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Engine.svg)](https://GitHub.com/modelscope/DiffSynth-Engine/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Engine)](https://GitHub.com/modelscope/DiffSynth-Engine/commit/)

<p align="center">
<a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

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
If you have any questions or feedback, please scan the QR code or send email to muse@alibaba-inc.com.

<div style="display: flex; justify-content: space-between;">
    <img src="assets/weixin.jpg" alt="weixin" style="zoom: 33%;" />
    <img src="assets/dingtalk.png" alt="dingtalk" style="zoom: 60%;" />
</div>

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
