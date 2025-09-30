# cross-platform definitions and utilities
import torch
import gc
import platform
import logging

logger = logging.getLogger(__name__)


def get_gpu_architecture():
    """Get GPU architecture information."""
    if not torch.cuda.is_available():
        return None, None

    try:
        props = torch.cuda.get_device_properties(0)
        return props.major, props.minor
    except Exception:
        return None, None


def is_nvidia():
    """Check if using NVIDIA GPU."""
    return torch.cuda.is_available() and not torch.version.hip


def is_amd():
    """Check if using AMD GPU."""
    return torch.cuda.is_available() and torch.version.hip


def supports_fp8_hardware():
    """Check if hardware supports FP8 efficiently."""
    if not torch.cuda.is_available():
        return False

    try:
        props = torch.cuda.get_device_properties(0)

        # NVIDIA: Ada Lovelace (RTX 40xx) and Hopper (H100) support hardware FP8
        if is_nvidia():
            if props.major >= 9:  # Hopper
                return True
            if props.major == 8 and props.minor >= 9:  # Ada Lovelace
                return True
            return False

        # AMD: Only RDNA3 (gfx94x) supports FP8
        if is_amd():
            return "gfx94" in props.gcnArchName

        return False
    except Exception as e:
        logger.warning(f"Failed to check FP8 hardware support: {e}")
        return False


def get_torch_version():
    """Get PyTorch version as tuple."""
    try:
        version_str = torch.__version__.split('.')
        major = int(version_str[0])
        minor = int(version_str[1])
        return major, minor
    except Exception:
        return None, None


def supports_fp8_software():
    """Check if PyTorch version supports FP8."""
    major, minor = get_torch_version()
    if major is None:
        return False

    # FP8 support was added in PyTorch 2.3
    if major < 2:
        return False
    if major == 2 and minor < 3:
        return False

    return True


# data type
# AMD only supports float8_e4m3fnuz
# https://onnx.ai/onnx/technical/float8.html
if is_amd() and torch.cuda.is_available():
    try:
        props = torch.cuda.get_device_properties(0)
        if "gfx94" in props.gcnArchName:
            DTYPE_FP8 = torch.float8_e4m3fnuz
        else:
            DTYPE_FP8 = torch.float8_e4m3fn
    except Exception:
        DTYPE_FP8 = torch.float8_e4m3fn
else:
    DTYPE_FP8 = torch.float8_e4m3fn


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


def pin_memory(tensor: torch.Tensor):
    return tensor.pin_memory() if platform.system() == "Linux" else tensor
