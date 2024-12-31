from .sdxl_civitai import sdxl_civitai_unet_keymap, sdxl_civitai_te1_keymap, sdxl_civitai_te2_keymap
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