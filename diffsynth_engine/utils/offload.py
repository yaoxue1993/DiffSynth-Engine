import torch.nn as nn

from diffsynth_engine.models.basic.transformer_helper import RMSNorm
from diffsynth_engine.models.basic.relative_position_emb import RelativePositionEmbedding


SUPPORTED_OFFLOAD_MODULES = (
    nn.Embedding,
    nn.Linear,
    nn.LayerNorm,
    nn.Conv2d,
    nn.GroupNorm,
    RMSNorm,
    RelativePositionEmbedding,
)


def enable_sequential_cpu_offload(module: nn.Module, device: str = "cuda:0"):
    if isinstance(module, SUPPORTED_OFFLOAD_MODULES):
        add_cpu_offload_hook(module, device)
        return
    for submodule in module.children():
        enable_sequential_cpu_offload(submodule, device)


def add_cpu_offload_hook(module: nn.Module, device: str = "cuda:0"):
    def _forward_pre_hook(module: nn.Module, input):
        offload_params = {}
        for name, param in module.named_parameters():
            offload_params[name] = param.data
            param.data = param.data.to(device=device)
        setattr(module, "_offload_params", offload_params)

    def _forward_hook(module: nn.Module, input, output):
        offload_params = getattr(module, "_offload_params", {})
        for name, param in module.named_parameters():
            if name in offload_params:
                param.data = offload_params[name]

    if getattr(module, "_sequential_cpu_offload_enabled", False):
        return
    module.register_forward_pre_hook(_forward_pre_hook)
    module.register_forward_hook(_forward_hook)
    setattr(module, "_sequential_cpu_offload_enabled", True)
