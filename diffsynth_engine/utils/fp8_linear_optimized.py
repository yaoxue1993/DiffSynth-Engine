import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Union
from diffsynth_engine.utils.platform import (
    DTYPE_FP8,
    supports_fp8_hardware,
    supports_fp8_software
)
import logging

logger = logging.getLogger(__name__)


def supports_fp8_compute(device=None):
    """Check if the device supports efficient FP8 computation."""
    return supports_fp8_hardware() and supports_fp8_software()


class OptimizedFP8Linear(nn.Linear):
    """Optimized FP8 Linear layer with pre-computed scaling factors."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        # Force dtype to be a regular float type for initialization
        init_dtype = dtype
        if dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            init_dtype = torch.bfloat16  # Use bfloat16 for initialization

        super().__init__(in_features, out_features, bias, device, init_dtype)

        # Pre-computed scaling factors - use buffers to avoid FSDP issues
        self.scale_weight = None
        self.scale_input = None

        # Cache for fp8 weights (use weak reference to avoid memory issues)
        self._fp8_weight_cache = None
        self._fp8_weight_device = None
        self._cache_enabled = True

        # Performance flags
        self.fp8_enabled = supports_fp8_compute(device)
        self.use_precomputed_scales = True

    def _initialize_scales(self, input_tensor: torch.Tensor):
        """Initialize scaling factors based on weight and input statistics."""
        if self.scale_weight is not None:
            return

        device = input_tensor.device

        # Initialize weight scale based on weight statistics
        with torch.no_grad():
            weight_max = torch.max(torch.abs(self.weight)).item()
            fp8_max = 448.0
            if DTYPE_FP8 == torch.float8_e4m3fnuz:
                fp8_max = fp8_max / 2.0

            # Conservative scaling to avoid overflow
            weight_scale = max(1.0, weight_max / (fp8_max * 0.95))

            # Use register_buffer instead of Parameter to avoid FSDP issues
            self.register_buffer(
                'scale_weight',
                torch.tensor(weight_scale, device=device, dtype=torch.float32)
            )

            # For input scaling, we'll use dynamic scaling initially
            # but can be made static after calibration
            self.register_buffer(
                'scale_input',
                torch.ones((), device=device, dtype=torch.float32)
            )

    def _get_fp8_weight(self):
        """Get cached FP8 weight or compute it."""
        current_device = self.weight.device

        # Check if caching is enabled and cache is valid
        if (self._cache_enabled and
            self._fp8_weight_cache is not None and
            self._fp8_weight_device == current_device and
            self._fp8_weight_cache.device == current_device):
            return self._fp8_weight_cache

        # Compute FP8 weight
        with torch.no_grad():
            if self.scale_weight is not None:
                # Use pre-computed scaling
                scaled_weight = self.weight / self.scale_weight.to(
                    device=current_device, dtype=self.weight.dtype
                )
            else:
                scaled_weight = self.weight

            fp8_weight = scaled_weight.to(DTYPE_FP8).contiguous()

            # Cache only if enabled and not in training mode
            if self._cache_enabled and not self.training:
                self._fp8_weight_cache = fp8_weight
                self._fp8_weight_device = current_device

            return fp8_weight

    def _optimized_fp8_linear(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """Optimized FP8 linear computation with pre-computed scales."""
        try:
            # Only use FP8 computation when weight is FP8 and input is regular float
            # This matches ComfyUI's approach
            if (self.weight.dtype not in [torch.float8_e4m3fn] or
                input.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]):
                return None
            device = input.device
            input_dtype = input.dtype
            original_shape = input.shape

            # Initialize scales if needed
            self._initialize_scales(input)

            # Handle 2D/3D tensor shapes
            tensor_2d = len(input.shape) == 2
            if tensor_2d:
                input = input.unsqueeze(1)

            input_shape = input.shape
            if len(input_shape) != 3:
                return None

            # Get FP8 weight
            fp8_weight = self._get_fp8_weight()

            # Prepare scales
            scale_weight = self.scale_weight.to(device)
            scale_input = self.scale_input.to(device)

            # Convert input to FP8 with scaling (ComfyUI style)
            if self.use_precomputed_scales:
                # Use pre-computed input scaling (can be calibrated)
                input_scaled = (input / scale_input.to(input_dtype)).reshape(-1, input_shape[2])
                # Clamp before FP8 conversion (like ComfyUI)
                input_clamped = torch.clamp(input_scaled, min=-448, max=448)
                input_fp8 = input_clamped.to(DTYPE_FP8).contiguous()
            else:
                # Dynamic input scaling (fallback)
                input_reshaped = input.reshape(-1, input_shape[2])
                x_max = torch.max(torch.abs(input_reshaped), dim=-1, keepdim=True).values
                fp8_max = 448.0
                if DTYPE_FP8 == torch.float8_e4m3fnuz:
                    fp8_max = fp8_max / 2.0
                # Use clamp for dynamic scaling (like ComfyUI)
                dynamic_scale = torch.clamp(x_max / fp8_max, min=1.0).float()
                input_scaled_dynamic = input_reshaped / dynamic_scale
                input_clamped = torch.clamp(input_scaled_dynamic, min=-448, max=448)
                input_fp8 = input_clamped.to(DTYPE_FP8).contiguous()
                scale_input = dynamic_scale.squeeze(-1)

            # Perform scaled matrix multiplication
            if self.bias is not None:
                bias = self.bias.to(input_dtype)
                result = torch._scaled_mm(
                    input_fp8,
                    fp8_weight.t(),
                    out_dtype=input_dtype,
                    bias=bias,
                    scale_a=scale_input,
                    scale_b=scale_weight
                )
            else:
                result = torch._scaled_mm(
                    input_fp8,
                    fp8_weight.t(),
                    out_dtype=input_dtype,
                    scale_a=scale_input,
                    scale_b=scale_weight
                )

            if isinstance(result, tuple):
                result = result[0]

            # Reshape to original dimensions
            if tensor_2d:
                return result.reshape(original_shape[0], -1)
            else:
                return result.reshape(input_shape[0], input_shape[1], -1)

        except Exception as e:
            logger.debug(f"FP8 linear failed, falling back to standard: {e}")
            return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 optimization."""
        if not self.training and self.fp8_enabled:
            # Try optimized FP8 computation
            result = self._optimized_fp8_linear(input)
            if result is not None:
                return result

        # Fallback to standard linear
        return F.linear(input, self.weight, self.bias)

    def calibrate_input_scale(self, calibration_inputs: list):
        """Calibrate input scaling factor based on representative inputs."""
        if not calibration_inputs:
            return

        max_vals = []
        with torch.no_grad():
            for input_tensor in calibration_inputs:
                if len(input_tensor.shape) == 2:
                    input_tensor = input_tensor.unsqueeze(1)
                input_reshaped = input_tensor.reshape(-1, input_tensor.shape[-1])
                max_val = torch.max(torch.abs(input_reshaped)).item()
                max_vals.append(max_val)

        # Use 95th percentile for robust scaling
        max_vals.sort()
        percentile_95 = max_vals[int(len(max_vals) * 0.95)]

        fp8_max = 448.0
        if DTYPE_FP8 == torch.float8_e4m3fnuz:
            fp8_max = fp8_max / 2.0

        optimal_scale = max(1.0, percentile_95 / (fp8_max * 0.95))

        # Update scale_input buffer safely
        if self.scale_input is not None:
            self.scale_input.fill_(optimal_scale)
        self.use_precomputed_scales = True

        # Clear weight cache to force recomputation
        self._fp8_weight_cache = None

        logger.info(f"Calibrated input scale to {optimal_scale:.4f}")

    def clear_cache(self):
        """Clear FP8 weight cache to free memory."""
        self._fp8_weight_cache = None
        self._fp8_weight_device = None

    def disable_cache(self):
        """Disable weight caching to save memory."""
        self.clear_cache()
        self._cache_enabled = False

    def enable_cache(self):
        """Enable weight caching for better performance."""
        self._cache_enabled = True


def convert_linear_to_fp8(module: nn.Module, recursive: bool = True) -> nn.Module:
    """Convert Linear layers to support FP8 computation (ComfyUI style)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not hasattr(child, '_fp8_enabled'):
            # Add FP8 support to existing Linear layer instead of replacing
            # This avoids the uniform_ initialization issue
            _add_fp8_support(child)

        elif recursive:
            convert_linear_to_fp8(child, recursive=True)

    return module


def _add_fp8_support(linear_layer: nn.Linear):
    """Add FP8 computation support to an existing Linear layer."""
    # Add FP8-specific attributes without changing the layer type
    linear_layer.scale_weight = None
    linear_layer.scale_input = None
    linear_layer._fp8_weight_cache = None
    linear_layer._fp8_weight_device = None
    linear_layer._cache_enabled = True
    linear_layer.fp8_enabled = supports_fp8_compute()
    linear_layer.use_precomputed_scales = True
    linear_layer._fp8_enabled = True

    # Store original forward method
    linear_layer._original_forward = linear_layer.forward

    # Replace forward method with FP8-aware version
    def fp8_forward(self, input):
        if not self.training and self.fp8_enabled:
            # Try FP8 computation
            result = _fp8_linear_computation(self, input)
            if result is not None:
                return result

        # Fallback to original forward
        return self._original_forward(input)

    # Bind the new forward method
    linear_layer.forward = fp8_forward.__get__(linear_layer, type(linear_layer))


def _fp8_linear_computation(layer, input: torch.Tensor) -> Optional[torch.Tensor]:
    """FP8 linear computation for converted layers."""
    try:
        # Only use FP8 when weight is FP8 and input is regular float
        if (layer.weight.dtype not in [torch.float8_e4m3fn] or
            input.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]):
            return None

        # Use ComfyUI's approach
        device = input.device
        input_dtype = input.dtype
        original_shape = input.shape

        # Handle 2D/3D tensors
        tensor_2d = len(input.shape) == 2
        if tensor_2d:
            input = input.unsqueeze(1)

        input_shape = input.shape
        if len(input_shape) != 3:
            return None

        # Get weight and bias
        weight = layer.weight.t()  # Transpose like ComfyUI
        bias = layer.bias

        # Get or create scales
        scale_weight = getattr(layer, 'scale_weight', None)
        scale_input = getattr(layer, 'scale_input', None)

        if scale_weight is None:
            scale_weight = torch.ones((), device=device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(device)

        if scale_input is None:
            scale_input = torch.ones((), device=device, dtype=torch.float32)
            # Clamp input directly (ComfyUI style)
            input_clamped = torch.clamp(input, min=-448, max=448)
            input_fp8 = input_clamped.reshape(-1, input_shape[2]).to(layer.weight.dtype).contiguous()
        else:
            scale_input = scale_input.to(device)
            input_scaled = (input * (1.0 / scale_input).to(input_dtype))
            input_fp8 = input_scaled.reshape(-1, input_shape[2]).to(layer.weight.dtype).contiguous()

        # Perform scaled matrix multiplication
        if bias is not None:
            result = torch._scaled_mm(
                input_fp8, weight,
                out_dtype=input_dtype,
                bias=bias,
                scale_a=scale_input,
                scale_b=scale_weight
            )
        else:
            result = torch._scaled_mm(
                input_fp8, weight,
                out_dtype=input_dtype,
                scale_a=scale_input,
                scale_b=scale_weight
            )

        if isinstance(result, tuple):
            result = result[0]

        # Reshape back
        if tensor_2d:
            return result.reshape(original_shape[0], -1)
        else:
            return result.reshape(input_shape[0], input_shape[1], -1)

    except Exception as e:
        logger.debug(f"FP8 computation failed: {e}")
        return None


@contextmanager
def optimized_fp8_inference(enabled=True):
    """Context manager for optimized FP8 inference."""
    if not enabled:
        yield
        return

    # This context manager can be used to enable additional optimizations
    # For now, it's mainly for compatibility with existing code
    yield


def clear_fp8_cache(module: nn.Module):
    """Clear FP8 caches in all OptimizedFP8Linear layers."""
    for child in module.modules():
        if isinstance(child, OptimizedFP8Linear):
            child.clear_cache()


def disable_fp8_cache(module: nn.Module):
    """Disable FP8 caching in all OptimizedFP8Linear layers to save memory."""
    for child in module.modules():
        if isinstance(child, OptimizedFP8Linear):
            child.disable_cache()


def enable_fp8_cache(module: nn.Module):
    """Enable FP8 caching in all OptimizedFP8Linear layers for better performance."""
    for child in module.modules():
        if isinstance(child, OptimizedFP8Linear):
            child.enable_cache()


# Backward compatibility
def enable_fp8_linear_optimized(module: nn.Module, low_memory_mode: bool = False):
    """Enable optimized FP8 linear layers on a module.

    Args:
        module: The module to convert
        low_memory_mode: If True, disable caching to save memory
    """
    # Check if FP8 is supported before conversion
    if not supports_fp8_compute():
        logger.warning("FP8 compute not supported on this hardware/software, skipping optimization")
        return

    convert_linear_to_fp8(module, recursive=True)

    if low_memory_mode:
        disable_fp8_cache(module)
        logger.info("Enabled optimized FP8 with low memory mode")
    else:
        logger.info("Enabled optimized FP8 with caching")

    setattr(module, "fp8_linear_optimized_enabled", True)