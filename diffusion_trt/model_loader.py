"""
Model loading component for the TensorRT Diffusion Model Optimization Pipeline.

This module provides VRAM-aware model loading for diffusion models from HuggingFace Hub,
with memory optimizations targeting NVIDIA T4 GPUs on Google Colab's free tier.

Supported models:
- stabilityai/sdxl-turbo
- runwayml/stable-diffusion-v1-5
"""

from dataclasses import dataclass, field
from typing import Optional, List
import gc
import time

import torch
from diffusers import DiffusionPipeline

from .models import T4_VRAM_LIMIT_GB


# Supported model identifiers
SUPPORTED_MODELS: List[str] = [
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5",
]

# Maximum VRAM for model weights (leaving room for activations and cache)
MAX_MODEL_WEIGHTS_VRAM_GB: float = 10.0


class OutOfMemoryError(Exception):
    """
    Raised when VRAM usage exceeds the T4 GPU limit.
    
    Attributes:
        current_usage_gb: Current VRAM usage in GB when error occurred
        limit_gb: VRAM limit that was exceeded
        message: Detailed error message
    """
    
    def __init__(
        self, 
        current_usage_gb: float, 
        limit_gb: float = T4_VRAM_LIMIT_GB,
        operation: str = "operation"
    ):
        self.current_usage_gb = current_usage_gb
        self.limit_gb = limit_gb
        self.message = (
            f"VRAM usage ({current_usage_gb:.2f}GB) exceeds limit ({limit_gb:.2f}GB) "
            f"during {operation}. Consider clearing caches or reducing model size."
        )
        super().__init__(self.message)


@dataclass
class ModelConfig:
    """
    Configuration for loading a diffusion model.
    
    Attributes:
        model_id: HuggingFace model identifier (e.g., "stabilityai/sdxl-turbo")
        dtype: Data type for model weights (default: torch.float16)
        variant: Model variant to load (default: "fp16")
        device: Target device for model (default: "cuda")
        enable_attention_slicing: Enable attention slicing for memory optimization
        enable_vae_tiling: Enable VAE tiling for memory optimization
    """
    model_id: str
    dtype: torch.dtype = torch.float16
    variant: Optional[str] = "fp16"
    device: str = "cuda"
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_id not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: '{self.model_id}'. "
                f"Supported models: {SUPPORTED_MODELS}"
            )
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(
                f"Invalid device: '{self.device}'. Must be 'cuda' or 'cpu'."
            )


class ModelLoader:
    """
    VRAM-aware model loader for diffusion pipelines.
    
    Loads diffusion models from HuggingFace Hub with memory optimizations
    targeting NVIDIA T4 GPUs. Applies attention slicing and VAE tiling
    by default to reduce VRAM usage.
    
    Example:
        >>> config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        >>> loader = ModelLoader()
        >>> pipeline = loader.load(config)
        >>> print(f"VRAM usage: {loader.get_vram_usage():.2f} GB")
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize the ModelLoader.
        
        Args:
            max_retries: Maximum number of retry attempts for model download
        """
        self._max_retries = max_retries
        self._loaded_pipeline: Optional[DiffusionPipeline] = None
    
    def load(self, config: ModelConfig) -> DiffusionPipeline:
        """
        Load a diffusion pipeline with memory optimizations.
        
        Loads the specified model with FP16 precision and applies
        attention slicing and VAE tiling optimizations by default.
        
        Args:
            config: Model configuration specifying model_id and options
            
        Returns:
            Loaded DiffusionPipeline with memory optimizations applied
            
        Raises:
            ValueError: If model_id is not in the supported list
            OutOfMemoryError: If VRAM usage exceeds 15.6GB during loading
            RuntimeError: If model download fails after max retries
        """
        # Validate model is supported
        if config.model_id not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: '{config.model_id}'. "
                f"Supported models: {SUPPORTED_MODELS}"
            )
        
        # Clear any existing cached memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load model with exponential backoff retry
        pipeline = self._load_with_retry(config)
        
        # Apply memory optimizations
        if config.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        
        if config.enable_vae_tiling:
            pipeline.enable_vae_tiling()
        
        # Move to target device
        if config.device == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to(config.device)
        
        # Check VRAM usage after loading
        current_vram = self.get_vram_usage()
        if current_vram > T4_VRAM_LIMIT_GB:
            # Clean up and raise error
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
            raise OutOfMemoryError(
                current_usage_gb=current_vram,
                limit_gb=T4_VRAM_LIMIT_GB,
                operation="model loading"
            )
        
        self._loaded_pipeline = pipeline
        return pipeline
    
    def _load_with_retry(self, config: ModelConfig) -> DiffusionPipeline:
        """
        Load model with exponential backoff retry logic.
        
        Args:
            config: Model configuration
            
        Returns:
            Loaded DiffusionPipeline
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(self._max_retries):
            try:
                # Build load kwargs
                load_kwargs = {
                    "torch_dtype": config.dtype,
                    "use_safetensors": True,
                }
                
                # Add variant if specified
                if config.variant:
                    load_kwargs["variant"] = config.variant
                
                pipeline = DiffusionPipeline.from_pretrained(
                    config.model_id,
                    **load_kwargs
                )
                return pipeline
                
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        raise RuntimeError(
            f"Failed to load model '{config.model_id}' after {self._max_retries} attempts. "
            f"Last error: {last_exception}"
        )
    
    def extract_unet(self, pipeline: DiffusionPipeline) -> torch.nn.Module:
        """
        Extract the UNet component from a diffusion pipeline for TensorRT optimization.
        
        The UNet is the primary component that benefits from TensorRT optimization
        as it performs the iterative denoising during inference.
        
        Args:
            pipeline: Loaded diffusion pipeline
            
        Returns:
            UNet module extracted from the pipeline
            
        Raises:
            AttributeError: If pipeline does not have a UNet component
        """
        if not hasattr(pipeline, 'unet'):
            raise AttributeError(
                f"Pipeline of type '{type(pipeline).__name__}' does not have a 'unet' attribute. "
                "Cannot extract UNet for TensorRT optimization."
            )
        
        unet = pipeline.unet
        
        # Ensure UNet is in eval mode for optimization
        unet.eval()
        
        return unet
    
    def get_vram_usage(self) -> float:
        """
        Get current VRAM usage in GB.
        
        Returns:
            Current VRAM usage in gigabytes. Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Get allocated memory (memory currently in use by tensors)
        allocated_bytes = torch.cuda.memory_allocated()
        
        # Convert to GB
        allocated_gb = allocated_bytes / (1024 ** 3)
        
        return allocated_gb
    
    def get_vram_reserved(self) -> float:
        """
        Get reserved VRAM in GB (includes cached memory).
        
        Returns:
            Reserved VRAM in gigabytes. Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0
        
        reserved_bytes = torch.cuda.memory_reserved()
        reserved_gb = reserved_bytes / (1024 ** 3)
        
        return reserved_gb
    
    def get_vram_info(self) -> dict:
        """
        Get detailed VRAM information.
        
        Returns:
            Dictionary with VRAM statistics:
            - allocated_gb: Memory currently in use by tensors
            - reserved_gb: Memory reserved by PyTorch allocator
            - total_gb: Total GPU memory
            - free_gb: Estimated free memory
        """
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "free_gb": 0.0,
            }
        
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        
        allocated_gb = allocated_bytes / (1024 ** 3)
        reserved_gb = reserved_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = total_gb - allocated_gb
        
        return {
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "total_gb": total_gb,
            "free_gb": free_gb,
        }
    
    def clear_memory(self) -> None:
        """
        Clear cached GPU memory and run garbage collection.
        
        Useful for freeing up VRAM before loading a new model
        or when approaching memory limits.
        """
        self._loaded_pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
