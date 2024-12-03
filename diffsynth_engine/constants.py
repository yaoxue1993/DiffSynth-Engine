import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# conf
CONF_PATH = os.path.join(PACKAGE_ROOT, "conf")
# tokenizers
FLUX_TOKENIZER_1_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "flux", "tokenizer_1")
FLUX_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "flux", "tokenizer_2")
SDXL_TOKENIZER_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "stable_diffusion_xl", "tokenizer")
SDXL_TOKENIZER_2_CONF_PATH = os.path.join(CONF_PATH, "tokenizers", "stable_diffusion_xl", "tokenizer_2")

# cache
MODEL_CACHE_PATH = os.path.join(REPO_ROOT, ".cache", "models")

# test assets
TEST_ASSETS_PATH = os.path.join(REPO_ROOT, "tests", "assets")
