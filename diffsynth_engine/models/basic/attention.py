import torch
import torch.nn as nn
from einops import rearrange, repeat


def low_version_attention(query, key, value, attn_bias=None, scale=None):
    scale = 1 / query.shape[-1] ** 0.5 if scale is None else scale
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    return attn @ value


class Attention(nn.Module):
    def __init__(self,
                 q_dim,
                 num_heads,
                 head_dim,
                 kv_dim=None,
                 bias_q=False,
                 bias_kv=False,
                 bias_out=False,
                 scale=None,
                 use_xformers=False,
                 device: str = 'cuda:0',
                 dtype: torch.dtype = torch.float16
                 ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(q_dim, dim_inner, bias=bias_q, device=device, dtype=dtype)
        self.to_k = nn.Linear(kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype)
        self.to_v = nn.Linear(kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype)
        self.to_out = nn.Linear(dim_inner, q_dim, bias=bias_out, device=device, dtype=dtype)

        self.scale = scale
        self.use_xformers = use_xformers

    def sdpa_attn(self, hidden_states, encoder_hidden_states, attn_mask=None):
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        hidden_states = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        hidden_states = rearrange(hidden_states, "b n s d -> b s (n d)", n=self.num_heads)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def xformers_attn(self, hidden_states, encoder_hidden_states, attn_mask=None):
        import xformers.ops as xops
        bs = hidden_states.shape[0]
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "1 ... -> b ...", b=bs)

        hidden_states = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=self.scale)
        hidden_states = rearrange(hidden_states, "b s n d -> b s (n d)")
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def original_attn(self, hidden_states, encoder_hidden_states, attn_mask=None):
        bs = hidden_states.shape[0]
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        q = rearrange(q, "b s (n d) -> (b n) s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> (b n) s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> (b n) s d", n=self.num_heads)

        if attn_mask is not None:
            attn_mask = repeat(attn_mask, "1 n ... -> (b n) ...", b=bs)

        hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        hidden_states = rearrange(hidden_states, "(b n) s d -> b s (n d)", n=self.num_heads)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                attn_mask=None,
                ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.use_xformers:
            return self.xformers_attn(hidden_states, encoder_hidden_states, attn_mask)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # 检查是否支持sdpa            
            return self.sdpa_attn(hidden_states, encoder_hidden_states, attn_mask)
        return self.original_attn(hidden_states, encoder_hidden_states, attn_mask)
