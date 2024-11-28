import torch
import torch.nn as nn
from typing import Optional

from diffsynth_engine.models.base import PreTrainedModel
from diffsynth_engine.models.basic.relative_position_emb import RelativePositionEmbedding
from diffsynth_engine.models.basic.transformer_helper import RMSNorm, NewGELUActivation
from diffsynth_engine.models.basic.attention import Attention    


class T5FeedForward(nn.Module):
    # T5DenseGatedActDense
    def __init__(self, d_model, d_ff, dropout_rate, device:str='cuda:0', dtype:torch.dtype=torch.float16):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.wo = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_gelu * hidden_linear)

        hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states
        
class T5EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim:int, 
                 num_heads:int, 
                 head_dim:int, 
                 d_ff:int, 
                 eps:float, 
                 dropout_rate:float = 0.0, 
                 device:str='cuda:0', 
                 dtype:torch.dtype=torch.float16):
        super().__init__()
        self.attn = Attention(
            q_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype
        )
        self.attn_norm = RMSNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.feed_forward = T5FeedForward(embed_dim, d_ff, dropout_rate, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None
    ):
        # Self Attention
        attn_output = self.attn(
            self.attn_norm(hidden_states),
            mask=attention_mask,
            position_bias=position_bias            
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        # Apply Feed Forward layer
        hidden_states = hidden_states + self.dropout(self.ffn_norm(self.feed_forward(hidden_states)))
        return hidden_states, position_bias

class T5EncoderModel(PreTrainedModel):
    def __init__(
        self,
        embed_dim: int = 4096,
        vocab_size: int = 32128,
        num_encoder_layers: int = 24,
        d_ff: int = 10240,
        num_heads: int = 64,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        # token_embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)

        # relative position embedding
        self.relative_position_embedding = RelativePositionEmbedding(
            relative_attention_num_buckets, relative_attention_max_distance, num_heads, device=device, dtype=dtype
        )

        # encoders
        self.encoders = nn.ModuleList([
            T5EncoderLayer(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                head_dim=embed_dim // num_heads, 
                d_ff=d_ff, 
                eps=eps, 
                dropout_rate=dropout_rate, 
                device=device, 
                dtype=dtype
            ) for i in range(num_encoder_layers)
        ])

        # final_layer_norm
        self.final_layer_norm = nn.LayerNorm(embed_dim, device=device, dtype=dtype)

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, input_ids:torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None):
        inputs_embeds = self.token_embedding(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        seq_len = hidden_states.shape[1]
        position_bias = self.relative_position_embedding(seq_len, seq_len)
        if attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
            attention_mask = causal_mask + position_bias
        else:
            attention_mask = position_bias
        for layer_module in self.encoders:
            hidden_states= layer_module(
                hidden_states, 
                attention_mask=attention_mask
            )
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states