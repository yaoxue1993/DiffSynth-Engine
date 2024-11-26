import os


REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_PATH = os.path.join(REPO_PATH, "diffsynth_engine", "conf")

# tokenizers
FLUX_TOKENIZER_1_CONF_PATH = os.path.join(CONF_PATH, "flux", "tokenizer_1")
FLUX_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "flux", "tokenizer_2")
SDXL_TOKENIZER_CONF_PATH = os.path.join(CONF_PATH, "stable_diffusion_xl", "tokenizer")
SDXL_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "stable_diffusion_xl", "tokenizer_2")
