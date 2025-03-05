from safetensors.torch import load_file
import torch

old = load_file("old.safetensors")
new = load_file("new.safetensors")
t1 = old["t"].to("cuda:0")
t2 = new["t"].to("cuda:0")

mse = torch.nn.MSELoss()

print(mse(t1, t2).item())