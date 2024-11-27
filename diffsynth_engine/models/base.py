import torch
import torch.nn as nn
from typing import Dict, Union
import os
class StateDictConverter:
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return state_dict

class PreTrainedModel(nn.Module):
    converter = StateDictConverter()

    def load_state_dict(self,
        state_dict:Dict[str, torch.Tensor],
        strict:bool=True, 
        assign:bool=False
    ):
        state_dict = self.converter.convert(state_dict)
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model

