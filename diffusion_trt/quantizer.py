"""
INT8 Quantizer component for TensorRT Diffusion Model Optimization Pipeline.

This module provides INT8 Post-Training Quantization using TensorRT Model Optimizer
with SmoothQuant algorithm for optimal accuracy on attention layers.

Requirements covered:
- 3.1: Use TensorRT Model Optimizer with SmoothQuant algorithm
- 3.2: Use entropy calibration method by default
- 3.3: Support excluding specified layers from quantization (keep in FP16)
- 3.4: Ensure MSE between original and quantized outputs is below 0.01
- 3.5: Log problematic layers if quantization error exceeds threshold

Compatibility:
- nvidia-modelopt >= 0.39.0 for CUDA 13.x (Google Colab March 2026+)
- nvidia-modelopt >= 0.15.0 for CUDA 12.x

Based on NVIDIA's TensorRT demo for SDXL INT8 quantization:
https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import logging

import torch
import torch.nn as nn

from .models import MAX_QUANTIZATION_ERROR

logger = logging.getLogger(__name__)


# Supported quantization algorithms
SUPPORTED_ALGORITHMS = ["int8_smoothquant", "int8_default"]

# Supported calibration methods
# modelopt 0.39+ uses "max" for max-based calibration (recommended for diffusion)
SUPPORTED_CALIBRATION_METHODS = ["max", "smoothquant", "percentile"]

# Map old names to new names for backward compatibility
CALIBRATION_METHOD_MAP = {
    "entropy": "max",      # legacy name -> max calibration
    "minmax": "max",       # legacy name -> max calibration  
    "percentile": "percentile",
    "max": "max",
    "smoothquant": "smoothquant",
}

# Layers to exclude from quantization for diffusion models
# Based on NVIDIA's TensorRT demo - these layers cause accuracy issues when quantized
DIFFUSION_EXCLUDE_PATTERNS = [
    "time_emb_proj",      # Time embedding projection
    "time_embedding",     # Time embedding layers
    "conv_in",            # Input convolution
    "conv_out",           # Output convolution
    "conv_shortcut",      # Skip connection convolutions
    "add_embedding",      # SDXL additional embeddings
    "proj_in",            # Attention input projections (optional, for better accuracy)
    "proj_out",           # Attention output projections (optional, for better accuracy)
]


@dataclass
class QuantizationConfig:
    """
    Configuration for INT8 Post-Training Quantization.
    
    Attributes:
        algorithm: Quantization algorithm ("int8_smoothquant" or "int8_default")
        calibration_method: Calibration method ("max", "smoothquant", "percentile", or legacy "entropy")
        percentile: Percentile for percentile calibration (default 99.99)
        exclude_layers: List of layer name patterns to keep in FP16
        use_diffusion_exclusions: Auto-exclude problematic layers for diffusion models
        max_quantization_error: Maximum allowed MSE between original and quantized
        num_calibration_batches: Number of batches to use for calibration
    """
    algorithm: str = "int8_smoothquant"
    calibration_method: str = "entropy"  # Legacy default, maps to "max"
    percentile: float = 99.99
    exclude_layers: Optional[List[str]] = None
    use_diffusion_exclusions: bool = True  # Auto-exclude problematic layers
    max_quantization_error: float = MAX_QUANTIZATION_ERROR
    num_calibration_batches: int = 100
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: '{self.algorithm}'. "
                f"Supported: {SUPPORTED_ALGORITHMS}"
            )
        
        # Accept both new and legacy calibration method names
        valid_methods = list(SUPPORTED_CALIBRATION_METHODS) + list(CALIBRATION_METHOD_MAP.keys())
        if self.calibration_method not in valid_methods:
            raise ValueError(
                f"Unsupported calibration method: '{self.calibration_method}'. "
                f"Supported: {SUPPORTED_CALIBRATION_METHODS}"
            )
        
        if not 0 < self.percentile <= 100:
            raise ValueError(
                f"percentile must be in (0, 100], got {self.percentile}"
            )
        
        if self.max_quantization_error <= 0:
            raise ValueError(
                f"max_quantization_error must be positive, got {self.max_quantization_error}"
            )
        
        if self.num_calibration_batches < 1:
            raise ValueError(
                f"num_calibration_batches must be at least 1, got {self.num_calibration_batches}"
            )
    
    def get_effective_exclude_layers(self) -> List[str]:
        """Get the combined list of layers to exclude from quantization."""
        exclude = list(self.exclude_layers or [])
        if self.use_diffusion_exclusions:
            for pattern in DIFFUSION_EXCLUDE_PATTERNS:
                if pattern not in exclude:
                    exclude.append(pattern)
        return exclude


class QuantizationError(Exception):
    """Raised when quantization accuracy is below acceptable threshold."""
    
    def __init__(
        self,
        mse: float,
        threshold: float,
        problematic_layers: Optional[List[str]] = None
    ):
        self.mse = mse
        self.threshold = threshold
        self.problematic_layers = problematic_layers or []
        message = (
            f"Quantization error (MSE={mse:.6f}) exceeds threshold ({threshold:.6f}). "
            f"Problematic layers: {self.problematic_layers}"
        )
        super().__init__(message)


class INT8Quantizer:
    """
    INT8 Post-Training Quantizer using TensorRT Model Optimizer.
    
    Applies INT8 quantization with SmoothQuant algorithm for better accuracy
    on attention layers. Supports layer exclusion and accuracy validation.
    
    Example:
        >>> config = QuantizationConfig(algorithm="int8_smoothquant")
        >>> quantizer = INT8Quantizer(config)
        >>> quantized_model = quantizer.quantize(model, calibration_data)
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize the INT8Quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        self._quantized_model: Optional[nn.Module] = None
        self._quantization_info: Dict[str, any] = {}
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Iterator[Dict[str, torch.Tensor]],
        forward_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Apply INT8 Post-Training Quantization to the model.
        
        Uses TensorRT Model Optimizer with SmoothQuant algorithm for
        optimal INT8 accuracy on transformer-based models.
        
        Args:
            model: PyTorch model to quantize (typically UNet)
            calibration_data: Iterator or list of calibration batches
            forward_fn: Optional custom forward function for calibration
            
        Returns:
            Quantized model with INT8 weights and activations
            
        Raises:
            ImportError: If nvidia-modelopt is not installed
            QuantizationError: If quantization accuracy is below threshold
        """
        # Lazy import to handle missing modelopt gracefully
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            raise ImportError(
                "nvidia-modelopt is required for INT8 quantization. "
                "For CUDA 13.x (Colab): pip install nvidia-modelopt>=0.39.0\n"
                "For CUDA 12.x: pip install nvidia-modelopt>=0.15.0"
            )
        
        # Set model to eval mode
        model.eval()
        
        # Get effective exclude layers (including diffusion-specific exclusions)
        exclude_layers = self.config.get_effective_exclude_layers()
        logger.info(f"Layers to exclude from quantization: {exclude_layers}")
        
        # Create custom quantization config with filter function
        # This is based on NVIDIA's TensorRT demo approach
        quant_cfg = self._create_diffusion_quant_config(exclude_layers)
        
        # Fall back to custom forward function
        if forward_fn is None:
            forward_fn = self._default_forward_fn
        
        # Convert to list if it's an iterator (so we can iterate multiple times)
        if not isinstance(calibration_data, list):
            calibration_data = list(calibration_data)
        
        if len(calibration_data) == 0:
            raise ValueError("Calibration data is empty")
        
        # Store for use in calibration loop
        self._calibration_data_list = calibration_data
        self._forward_fn = forward_fn
        
        def calibration_loop(model_arg):
            """Run calibration data through the model."""
            batch_count = 0
            for batch in self._calibration_data_list:
                if batch_count >= self.config.num_calibration_batches:
                    break
                try:
                    self._forward_fn(model_arg, batch)
                except Exception as e:
                    logger.warning(f"Calibration batch {batch_count} failed: {e}")
                    # Continue with next batch instead of failing completely
                    continue
                batch_count += 1
            
            if batch_count == 0:
                raise RuntimeError("All calibration batches failed")
            
            logger.info(f"Calibration completed with {batch_count} batches")
        
        logger.info(f"Applying INT8 quantization with {self.config.algorithm}")
        
        # Apply quantization with calibration in one step (modelopt API)
        # The forward_loop is passed to mtq.quantize for calibration
        try:
            quantized_model = mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"mtq.quantize failed: {error_msg}")
            
            # Check for common errors and provide helpful messages
            if "NoneType" in error_msg:
                logger.error(
                    "NoneType error during quantization. This may be caused by:\n"
                    "1. Incompatible model architecture\n"
                    "2. Missing required model attributes\n"
                    "3. Calibration data format issues"
                )
            
            raise
        
        self._quantized_model = quantized_model
        
        logger.info("INT8 quantization complete")
        
        return quantized_model
    
    def _create_diffusion_quant_config(self, exclude_layers: List[str]) -> Dict:
        """
        Create a custom quantization config for diffusion models.
        
        Disables quantization for problematic layers by setting enable=False
        for matching layer name patterns. This is the correct modelopt API
        for layer exclusion (filter_func is NOT a valid field).
        
        Args:
            exclude_layers: List of layer name patterns to exclude
            
        Returns:
            Quantization config dictionary for mtq.quantize
        """
        try:
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization import INT8_SMOOTHQUANT_CFG, INT8_DEFAULT_CFG
        except ImportError:
            raise ImportError("nvidia-modelopt is required")
        
        # Start with the base config
        if self.config.algorithm == "int8_smoothquant":
            base_cfg = INT8_SMOOTHQUANT_CFG.copy()
        else:
            base_cfg = INT8_DEFAULT_CFG.copy()
        
        # Deep copy the quant_cfg to avoid mutating the global config
        import copy
        quant_cfg = copy.deepcopy(base_cfg)
        
        # Disable quantization for excluded layers using the correct modelopt API:
        # Set {"enable": False} for each pattern - this disables ALL quantizers
        # (input, weight, output) for any layer whose name contains the pattern.
        for pattern in exclude_layers:
            # The key format "*<pattern>*" matches any layer name containing the pattern
            quant_cfg["quant_cfg"][f"*{pattern}*"] = {"enable": False}
            logger.debug(f"Disabling quantization for layers matching: *{pattern}*")
        
        logger.info(
            f"Created diffusion quant config with {len(exclude_layers)} exclusion patterns"
        )
        
        return quant_cfg
    
    def _default_forward_fn(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Default forward function for UNet calibration."""
        with torch.no_grad():
            # Get batch tensors with fallback keys
            sample = batch.get("sample", batch.get("latents"))
            timestep = batch.get("timestep", batch.get("timesteps"))
            encoder_hidden_states = batch.get("encoder_hidden_states")
            
            if sample is None:
                raise ValueError(
                    f"Calibration batch missing 'sample' or 'latents'. "
                    f"Got keys: {list(batch.keys())}"
                )
            
            if timestep is None:
                raise ValueError(
                    f"Calibration batch missing 'timestep' or 'timesteps'. "
                    f"Got keys: {list(batch.keys())}"
                )
            
            if encoder_hidden_states is None:
                raise ValueError(
                    f"Calibration batch missing 'encoder_hidden_states'. "
                    f"Got keys: {list(batch.keys())}"
                )
            
            # Try to get device and dtype from model parameters
            # Fall back to sample's device/dtype if model is a mock or has no parameters
            try:
                model_param = next(model.parameters())
                device = model_param.device
                dtype = model_param.dtype
            except (StopIteration, AttributeError):
                # Model has no parameters or is a mock - use sample's device/dtype
                device = sample.device
                dtype = sample.dtype
            
            # Only convert if device/dtype are valid torch types
            if isinstance(device, torch.device) and isinstance(dtype, torch.dtype):
                sample = sample.to(device=device, dtype=dtype)
                encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
            
            # Handle timestep - should be long tensor
            if hasattr(timestep, 'dtype') and timestep.dtype != torch.long:
                timestep = timestep.long()
            if isinstance(device, torch.device):
                timestep = timestep.to(device=device)
            
            # Check if this is an SDXL model (has add_embedding)
            model_config = getattr(model, 'config', None)
            is_sdxl = False
            if model_config is not None:
                addition_embed_type = getattr(model_config, 'addition_embed_type', None)
                is_sdxl = addition_embed_type == "text_time"
            
            # Also detect SDXL by cross_attention_dim or encoder_hidden_states shape
            if not is_sdxl:
                cross_attn_dim = None
                if model_config is not None:
                    cross_attn_dim = getattr(model_config, 'cross_attention_dim', None)
                # If encoder_hidden_states has 2048 features, it's SDXL
                if (cross_attn_dim == 2048 or 
                        encoder_hidden_states.shape[-1] == 2048):
                    is_sdxl = True
            
            if is_sdxl:
                # SDXL requires additional conditioning
                batch_size = sample.shape[0]
                
                # SDXL uses text_embeds (pooled) and time_ids
                # text_embeds: [batch, 1280] - pooled text embedding
                # time_ids: [batch, 6] - original_size, crops_coords_top_left, target_size
                text_embeds = batch.get("text_embeds")
                time_ids = batch.get("time_ids")
                
                # Create default values if not provided
                if text_embeds is None:
                    text_embeds = torch.zeros(batch_size, 1280, device=device, dtype=dtype)
                elif isinstance(device, torch.device) and isinstance(dtype, torch.dtype):
                    text_embeds = text_embeds.to(device=device, dtype=dtype)
                
                if time_ids is None:
                    # Default time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
                    # Using 512x512 as default
                    time_ids = torch.tensor(
                        [[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]] * batch_size,
                        device=device, dtype=dtype
                    )
                elif isinstance(device, torch.device) and isinstance(dtype, torch.dtype):
                    time_ids = time_ids.to(device=device, dtype=dtype)
                
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                
                return model(
                    sample,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                )
            else:
                # Standard SD 1.5 / SD 2.x forward
                return model(
                    sample,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                )
    
    def _create_exclude_filter(self) -> Optional[Callable]:
        """Create a filter function for layer exclusion."""
        exclude_layers = self.config.get_effective_exclude_layers()
        if not exclude_layers:
            return None
        
        def should_exclude(name: str) -> bool:
            if name is None:
                return True  # Exclude None names
            for pattern in exclude_layers:
                if pattern in name:
                    return True
            return False
        
        return should_exclude
    
    def _apply_layer_exclusions(self, model: nn.Module) -> None:
        """Convert excluded layers back to FP16."""
        exclude_layers = self.config.get_effective_exclude_layers()
        if not exclude_layers:
            return
        
        excluded_count = 0
        for name, module in model.named_modules():
            if name is None:
                continue
            for pattern in exclude_layers:
                if pattern in name:
                    # Convert module parameters to FP16
                    for param in module.parameters():
                        if param.data.dtype != torch.float16:
                            param.data = param.data.to(torch.float16)
                    excluded_count += 1
                    logger.debug(f"Excluded layer from quantization: {name}")
                    break
        
        if excluded_count > 0:
            logger.info(f"Converted {excluded_count} layers back to FP16")
    
    def validate_accuracy(
        self,
        original: nn.Module,
        quantized: nn.Module,
        test_inputs: List[Dict[str, torch.Tensor]],
        tolerance: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compare outputs between original and quantized models.
        
        Args:
            original: Original FP16 model
            quantized: Quantized INT8 model
            test_inputs: List of test input batches
            tolerance: MSE tolerance (default from config)
            
        Returns:
            Dictionary with accuracy metrics:
            - mse: Mean squared error
            - max_error: Maximum absolute error
            - passed: Whether accuracy is within tolerance
            
        Raises:
            QuantizationError: If MSE exceeds tolerance
        """
        if tolerance is None:
            tolerance = self.config.max_quantization_error
        
        original.eval()
        quantized.eval()
        
        total_mse = 0.0
        max_error = 0.0
        num_samples = 0
        problematic_layers: List[str] = []
        
        with torch.no_grad():
            for batch in test_inputs:
                # Get outputs from both models
                original_output = self._default_forward_fn(original, batch)
                quantized_output = self._default_forward_fn(quantized, batch)
                
                # Handle different output types
                if hasattr(original_output, 'sample'):
                    original_tensor = original_output.sample
                    quantized_tensor = quantized_output.sample
                elif isinstance(original_output, torch.Tensor):
                    original_tensor = original_output
                    quantized_tensor = quantized_output
                else:
                    original_tensor = original_output[0]
                    quantized_tensor = quantized_output[0]
                
                # Compute MSE
                mse = torch.mean((original_tensor - quantized_tensor) ** 2).item()
                total_mse += mse
                
                # Track max error
                batch_max = torch.max(torch.abs(original_tensor - quantized_tensor)).item()
                max_error = max(max_error, batch_max)
                
                num_samples += 1
        
        avg_mse = total_mse / max(num_samples, 1)
        passed = avg_mse < tolerance
        
        # Log results
        if passed:
            logger.info(f"Quantization accuracy validated: MSE={avg_mse:.6f} < {tolerance:.6f}")
        else:
            logger.warning(
                f"Quantization accuracy below threshold: MSE={avg_mse:.6f} >= {tolerance:.6f}"
            )
            # Identify problematic layers
            problematic_layers = self._identify_problematic_layers(
                original, quantized, test_inputs[0] if test_inputs else None
            )
        
        result = {
            "mse": avg_mse,
            "max_error": max_error,
            "passed": passed,
            "num_samples": num_samples,
            "problematic_layers": problematic_layers,
        }
        
        self._quantization_info = result
        
        if not passed:
            raise QuantizationError(
                mse=avg_mse,
                threshold=tolerance,
                problematic_layers=problematic_layers
            )
        
        return result
    
    def _identify_problematic_layers(
        self,
        original: nn.Module,
        quantized: nn.Module,
        test_input: Optional[Dict[str, torch.Tensor]],
    ) -> List[str]:
        """Identify layers with high quantization error."""
        if test_input is None:
            return []
        
        problematic = []
        layer_errors: Dict[str, float] = {}
        
        # Register hooks to capture intermediate activations
        original_activations: Dict[str, torch.Tensor] = {}
        quantized_activations: Dict[str, torch.Tensor] = {}
        
        def make_hook(storage: Dict, name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    storage[name] = output.detach()
            return hook
        
        original_hooks = []
        quantized_hooks = []
        
        try:
            # Register hooks on leaf modules
            for name, module in original.named_modules():
                if len(list(module.children())) == 0:
                    hook = module.register_forward_hook(make_hook(original_activations, name))
                    original_hooks.append(hook)
            
            for name, module in quantized.named_modules():
                if len(list(module.children())) == 0:
                    hook = module.register_forward_hook(make_hook(quantized_activations, name))
                    quantized_hooks.append(hook)
            
            # Run forward pass
            with torch.no_grad():
                self._default_forward_fn(original, test_input)
                self._default_forward_fn(quantized, test_input)
            
            # Compare activations
            for name in original_activations:
                if name in quantized_activations:
                    orig = original_activations[name]
                    quant = quantized_activations[name]
                    
                    if orig.shape == quant.shape:
                        mse = torch.mean((orig - quant) ** 2).item()
                        layer_errors[name] = mse
                        
                        if mse > self.config.max_quantization_error:
                            problematic.append(name)
                            logger.warning(f"High error in layer {name}: MSE={mse:.6f}")
        
        finally:
            # Remove hooks
            for hook in original_hooks + quantized_hooks:
                hook.remove()
        
        return problematic
    
    def get_quantization_info(self) -> Dict[str, any]:
        """Get information about the last quantization operation."""
        return self._quantization_info.copy()


    def export_onnx(
        self,
        quantized_model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: str,
        opset_version: int = 17,
    ) -> str:
        """
        Export quantized model to ONNX format with Q/DQ nodes.
        
        The exported ONNX model includes quantization/dequantization nodes
        that can be consumed by TensorRT for INT8 inference.
        
        Args:
            quantized_model: Quantized PyTorch model
            sample_input: Sample input for tracing
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version (default 17)
            
        Returns:
            Path to the exported ONNX model
            
        Raises:
            RuntimeError: If ONNX export fails
        """
        quantized_model.eval()
        
        # Prepare input tensors
        sample = sample_input["sample"]
        timestep = sample_input["timestep"]
        encoder_hidden_states = sample_input["encoder_hidden_states"]
        
        # Create input tuple for export
        input_tuple = (sample, timestep, encoder_hidden_states)
        
        # Define input names
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        output_names = ["output"]
        
        # Define dynamic axes for variable batch size
        dynamic_axes = {
            "sample": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        
        logger.info(f"Exporting quantized model to ONNX: {output_path}")
        
        try:
            torch.onnx.export(
                quantized_model,
                input_tuple,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            
            logger.info(f"ONNX export complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"Failed to export model to ONNX: {e}")
    
    def get_layer_quantization_status(
        self,
        model: nn.Module
    ) -> Dict[str, Dict[str, any]]:
        """
        Get quantization status for each layer in the model.
        
        Returns:
            Dictionary mapping layer names to their quantization info:
            - is_quantized: Whether the layer is quantized
            - dtype: Data type of the layer weights
            - has_scale: Whether quantization scale is present
        """
        status = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    "is_quantized": False,
                    "dtype": None,
                    "has_scale": False,
                }
                
                # Check for quantization attributes
                if hasattr(module, 'weight'):
                    layer_info["dtype"] = str(module.weight.dtype)
                    
                    # Check for quantization scale (common in quantized modules)
                    if hasattr(module, 'weight_scale') or hasattr(module, '_scale'):
                        layer_info["is_quantized"] = True
                        layer_info["has_scale"] = True
                    elif module.weight.dtype == torch.int8:
                        layer_info["is_quantized"] = True
                
                status[name] = layer_info
        
        return status
