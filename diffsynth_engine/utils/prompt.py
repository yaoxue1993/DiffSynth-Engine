from typing import Union, Optional

from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast


def tokenize_long_prompt(
    tokenizer: Union[CLIPTokenizer, T5TokenizerFast], prompt: str, max_length: Optional[int] = None
):
    return tokenizer(prompt)["input_ids"]
