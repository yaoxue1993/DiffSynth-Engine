from .sdxl_civitai import sdxl_civitai_unet_keymap, sdxl_civitai_te1_keymap, sdxl_civitai_te2_keymap
from .sd_civitai import sd_civitai_unet_keymap
from .flux_civitai import (
    flux_civitai_dit_rename_dict, flux_civitai_dit_suffix_rename_dict,
    flux_civitai_clip_rename_dict, flux_civitai_clip_attn_rename_dict
)
suffix_list = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".weight",
    ".bias",
    ".alpha",
]

def split_key(key):
    name = key
    suffix = ""
    for s in suffix_list:
        if s in key:
            name = key.replace(s, "")
            suffix = s
            break
    return name, suffix