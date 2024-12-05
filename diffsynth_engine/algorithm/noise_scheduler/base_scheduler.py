import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BaseScheduler:
    def schedule(self, num_inference_steps: int):
        raise NotImplementedError()
