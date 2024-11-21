from pydantic import BaseModel


class T5Config(BaseModel):
    is_decoder: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
