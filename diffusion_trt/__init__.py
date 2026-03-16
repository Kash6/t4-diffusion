"""
TensorRT Diffusion Model Optimization Pipeline.

A TensorRT-based optimization pipeline for diffusion models targeting NVIDIA T4 GPUs
on Google Colab's free tier. Combines INT8 Post-Training Quantization via TensorRT
Model Optimizer with DeepCache/X-Slim style feature caching to achieve 2-3x speedups
while staying within the 15.6GB VRAM constraint.

Target models: SDXL-Turbo (4 steps) and SD 1.5 (20 steps)
Target hardware: NVIDIA T4 GPU (sm_75, 15.6GB VRAM, INT8 Tensor Cores)
"""

__version__ = "0.1.0"
__author__ = "Diffusion TRT Team"

# Data models
from .models import (
    BenchmarkMetrics,
    CacheEntry,
    OptimizationResult,
    MAX_QUANTIZATION_ERROR,
    T4_VRAM_LIMIT_GB,
)

# Model loading component
from .model_loader import (
    ModelLoader,
    ModelConfig,
    OutOfMemoryError,
    SUPPORTED_MODELS,
)

# Core components will be imported here as they are implemented
# from .calibration import CalibrationEngine, CalibrationConfig
# from .quantizer import INT8Quantizer, QuantizationConfig
# from .trt_builder import TensorRTBuilder, TRTConfig
# from .cache_manager import CacheManager, CacheConfig
# from .pipeline import OptimizedPipeline, PipelineConfig

__all__ = [
    "__version__",
    # Data models
    "OptimizationResult",
    "BenchmarkMetrics",
    "CacheEntry",
    "T4_VRAM_LIMIT_GB",
    "MAX_QUANTIZATION_ERROR",
    # Model loading
    "ModelLoader",
    "ModelConfig",
    "OutOfMemoryError",
    "SUPPORTED_MODELS",
    # Components will be added as implemented
]
