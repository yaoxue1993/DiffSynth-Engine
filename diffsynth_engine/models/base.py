import os
import torch.nn as nn
from typing import Dict, Union


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike]):
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, "torch.Tensor"]):
        raise NotImplementedError()
