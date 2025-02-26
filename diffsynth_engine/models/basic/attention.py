import torch
import torch.nn as nn
from einops import rearrange

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        q_dim,
        num_heads,
        head_dim,
        kv_dim=None,
        bias_q=False,
        bias_kv=False,
        bias_out=False,
        scale=None,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
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
        self.attn_implementation = self._get_actual_attn_implementation(attn_implementation)

    @staticmethod
    def _get_actual_attn_implementation(attn_implementation):
        supported_implementations = ("xformers", "sdpa", "eager")
        if attn_implementation not in supported_implementations:
            raise ValueError(
                f"attn_implementation must be one of {supported_implementations}, but got '{attn_implementation}'"
            )

        actual_implementation = "eager" if attn_implementation == "eager" else ""
        if attn_implementation == "xformers":
            try:
                from xformers.ops import memory_efficient_attention

                actual_implementation = "xformers"
            except ImportError:
                pass
        if not actual_implementation or attn_implementation == "sdpa":
            use_mps = torch.backends.mps.is_available()
            if hasattr(torch.nn.functional, "scaled_dot_product_attention") and not use_mps:
                actual_implementation = "sdpa"

        if actual_implementation != attn_implementation:
            warning_msg = (
                "xformers is not supported on this platform"
                if attn_implementation == "xformers"
                else "torch.nn.functional.scaled_dot_product_attention is not supported"
            )
            logger.warning(f"{warning_msg}, fallback to '{actual_implementation}' attention")
        return actual_implementation

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

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

        hidden_states = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=self.scale)
        hidden_states = rearrange(hidden_states, "b s n d -> b s (n d)")
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def eager_attn(self, hidden_states, encoder_hidden_states, attn_mask=None):
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        hidden_states = self._eager_attn(q, k, v, attn_bias=attn_mask, scale=self.scale)
        hidden_states = rearrange(hidden_states, "b n s d -> b s (n d)", n=self.num_heads)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    @staticmethod
    def _eager_attn(query, key, value, attn_bias=None, scale=None):
        scale = 1 / query.shape[-1] ** 0.5 if scale is None else scale
        query = query * scale
        attn = torch.matmul(query, key.transpose(-2, -1))
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        return attn @ value

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attn_mask=None,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if self.attn_implementation == "xformers":
            return self.xformers_attn(hidden_states, encoder_hidden_states, attn_mask)
        if self.attn_implementation == "sdpa":
            return self.sdpa_attn(hidden_states, encoder_hidden_states, attn_mask)
        return self.eager_attn(hidden_states, encoder_hidden_states, attn_mask)
