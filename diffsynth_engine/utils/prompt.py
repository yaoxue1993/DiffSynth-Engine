from typing import Union, Optional

from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast


def tokenize_long_prompt(tokenizer: Union[CLIPTokenizer, T5TokenizerFast],
                         prompt: str,
                         max_length: Optional[int] = None):
    # Get model_max_length from self.tokenizer
    length = tokenizer.model_max_length if max_length is None else max_length

    # To avoid the warning. set self.tokenizer.model_max_length to +oo.
    tokenizer.model_max_length = 99999999

    # Tokenize it!
    input_ids = tokenizer(prompt)["input_ids"]

    # Determine the real length.
    max_length = (input_ids.shape[1] + length - 1) // length * length

    # Restore tokenizer.model_max_length
    tokenizer.model_max_length = length

    # Tokenize it again with fixed length.
    input_ids = tokenizer(
        prompt,
        max_length=max_length,
    )["input_ids"]

    # Reshape input_ids to fit the text encoder.
    num_sentence = input_ids.shape[1] // length
    input_ids = input_ids.reshape((num_sentence, length))

    return input_ids
