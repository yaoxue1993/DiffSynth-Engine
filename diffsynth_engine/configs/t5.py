from pydantic import BaseModel


class T5Config(BaseModel):
    vocab_size: int = 32128
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True
    dense_act_fn: str = "gelu_new"
    is_decoder: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
