import torch
import torch.nn as nn
import math
import json
from typing import Tuple, Optional
from einops import rearrange
from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.constants import WAN_DIT_1_3B_T2V_CONFIG_FILE, WAN_DIT_14B_I2V_CONFIG_FILE, WAN_DIT_14B_T2V_CONFIG_FILE

from .attention import attention
from .rope import rope_apply, rope_params, sinusoidal_embedding_1d

def modulate(x, shift, scale):
    return (x * (1 + scale) + shift)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight
        
    
class SelfAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, window_size:Tuple[int, int] = (-1, -1), eps:float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, grid_size, freqs):
        q = self.norm_q(self.q(x)).view(x.shape[0], -1, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(x.shape[0], -1, self.num_heads, self.head_dim)
        v = self.v(x).view(x.shape[0], -1, self.num_heads, self.head_dim)
        
        x = attention(
            q=rope_apply(q, grid_size, freqs),
            k=rope_apply(k, grid_size, freqs),
            v=v,
            k_lens=torch.tensor([9900], dtype=torch.long, device=x.device), # TODO
            window_size=self.window_size
        ).flatten(2)
        return self.o(x)

class CrossAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, eps:float = 1e-6, has_image_input:bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y        
        q = self.norm_q(self.q(x)).view(x.shape[0], -1, self.num_heads, self.head_dim)        
        k = self.norm_k(self.k(ctx)).view(ctx.shape[0], -1, self.num_heads, self.head_dim)
        v = self.v(ctx).view(ctx.shape[0], -1, self.num_heads, self.head_dim)        
        x = attention(q, k, v).flatten(2)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img)).view(img.shape[0], -1, self.num_heads, self.head_dim)
            v_img = self.v_img(img).view(img.shape[0], -1, self.num_heads, self.head_dim)
            x = x + attention(q, k_img, v_img).flatten(2)
        return self.o(x)

class DiTBlock(nn.Module):
    def __init__(self, has_image_input:bool, dim:int, num_heads:int, ffn_dim:int, window_size:Tuple[int, int], eps:float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.window_size = window_size

        self.self_attn = SelfAttention(dim, num_heads, window_size, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, t_mod, grid_size, freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + t_mod).chunk(6, dim=1)
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), grid_size, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, in_dim),
            nn.GELU(), 
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim:int, out_dim:int, patch_size:Tuple[int, int, int], eps:float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x

                
class WanDiTStateDictConverter(StateDictConverter):
    def convert(self, state_dict):
        return state_dict    

class WanDiT(PreTrainedModel):
    converter = WanDiTStateDictConverter()

    def __init__(
        self,
        dim:int, 
        in_dim:int,
        ffn_dim:int,
        out_dim:int,
        text_dim:int,
        freq_dim:int,
        eps:float,
        patch_size:Tuple[int, int, int],
        window_size:Tuple[int, int],
        num_heads:int,
        num_layers:int,
        has_image_input:bool,
        device:str,
        dtype:torch.dtype
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input  
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, window_size, eps)
            for _ in range(num_layers)
        ])
        
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        if has_image_input:
            self.img_emb = MLP(1280, dim) # clip_feature_dim = 1280

    def patchify(self, x:torch.Tensor):
        x = self.patch_embedding(x)        
        grid_size = torch.stack([
            torch.tensor(u.shape[1:], dtype=torch.long, device=x.device) for u in x
        ])
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        return x, grid_size

    def unpatchify(self, x:torch.Tensor, grid_size:torch.Tensor):
        result = []
        for u, g in zip(x, grid_size):
            result.append(rearrange(u, '(f h w) (x y z c) -> c (f x) (h y) (w z)', f=g[0], h=g[1], w=g[2], x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]))
        return torch.stack(result)

    def forward(self, 
        x:torch.Tensor, 
        context:torch.Tensor, 
        timestep:torch.Tensor, 
        clip_feature:Optional[torch.Tensor] = None, 
        y:Optional[torch.Tensor] = None
    ):  
        self.freqs = self.freqs.to(x.device)                    
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)            
            clip_embdding = self.img_emb(clip_feature)            
            context = torch.cat([clip_embdding, context], dim=1)
        x, grid_size = self.patchify(x)
        for block in self.blocks:
            x = block(x, context, t_mod, grid_size, self.freqs)
        x = self.head(x, t)
        x = self.unpatchify(x, grid_size)
        return x
    
    @classmethod
    def from_state_dict(cls, state_dict, device, dtype, model_type='1.3b-t2v'):
        if model_type == '1.3b-t2v':
            config = json.load(open(WAN_DIT_1_3B_T2V_CONFIG_FILE, 'r'))
        elif model_type == '14b-t2v':
            config = json.load(open(WAN_DIT_14B_T2V_CONFIG_FILE, 'r'))
        elif model_type == '14b-i2v':
            config = json.load(open(WAN_DIT_14B_I2V_CONFIG_FILE, 'r'))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, **config, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
        return model