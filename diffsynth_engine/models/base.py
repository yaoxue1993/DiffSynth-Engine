import os
import torch
import torch.nn as nn
from typing import Dict, Union
from safetensors.torch import load_file

from diffsynth_engine.models.utils import no_init_weights


class LoRAStateDictConverter:
    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"lora": lora_state_dict}


StateDictType = Dict[str, torch.Tensor]


class StateDictConverter:
    def convert(self, state_dict: StateDictType) -> StateDictType:
        return state_dict


class PreTrainedModel(nn.Module):
    converter = StateDictConverter()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        state_dict = self.converter.convert(state_dict)
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], device: str, dtype: torch.dtype, **kwargs):
        state_dict = load_file(pretrained_model_path, device=device)
        return cls.from_state_dict(state_dict, device=device, dtype=dtype, **kwargs)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype, **kwargs):
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype, non_blocking=True)
        return model


def split_suffix(name: str):
    suffix_list = [
        ".lora_up.weight",
        ".lora_down.weight",
        ".weight",
        ".bias",
        ".alpha",
    ]
    for suffix in suffix_list:
        if name.endswith(suffix):
            return name.replace(suffix, ""), suffix
    return name, ""
