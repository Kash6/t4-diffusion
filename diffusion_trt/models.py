"""
Data models for the TensorRT Diffusion Model Optimization Pipeline.

This module contains dataclasses for optimization results, benchmark metrics,
and cache entries with validation rules and computed properties.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import torch


# T4 GPU VRAM limit in GB
T4_VRAM_LIMIT_GB = 15.6

# Maximum acceptable quantization error (MSE)
MAX_QUANTIZATION_ERROR = 0.01


@dataclass
class OptimizationResult:
    """
    Results from optimizing a diffusion model.
    
    Captures performance metrics, memory usage, and configuration
    for a completed optimization run.
    
    Attributes:
        model_id: HuggingFace model identifier (e.g., "stabilityai/sdxl-turbo")
        original_latency_ms: Baseline latency before optimization in milliseconds
        optimized_latency_ms: Latency after optimization in milliseconds
        speedup: Ratio of original to optimized latency (must be >= 1.0)
        vram_usage_gb: Peak VRAM usage during inference in GB (must be <= 15.6)
        quantization_error: MSE between FP16 and INT8 outputs (should be < 0.01)
        cache_hit_rate: Fraction of cache hits during inference (0.0 to 1.0)
        engine_path: Path to serialized TensorRT engine file
        timestamp: When the optimization was completed
        config: Configuration dictionary used for optimization
    
    Validation Rules:
        - speedup must be >= 1.0 (no slowdown)
        - vram_usage_gb must be <= 15.6 (T4 limit)
        - quantization_error should be < 0.01 for acceptable quality
    """
    model_id: str
    original_latency_ms: float
    optimized_latency_ms: float
    speedup: float
    vram_usage_gb: float
    quantization_error: Optional[float]
    cache_hit_rate: float
    engine_path: Optional[str]
    timestamp: datetime
    config: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate optimization result fields after initialization."""
        # Validate speedup >= 1.0 (no slowdown)
        if self.speedup < 1.0:
            raise ValueError(
                f"speedup must be >= 1.0 (no slowdown), got {self.speedup}"
            )
        
        # Validate VRAM usage within T4 limit
        if self.vram_usage_gb > T4_VRAM_LIMIT_GB:
            raise ValueError(
                f"vram_usage_gb must be <= {T4_VRAM_LIMIT_GB} (T4 limit), "
                f"got {self.vram_usage_gb}"
            )
        
        # Validate cache hit rate is in valid range
        if not 0.0 <= self.cache_hit_rate <= 1.0:
            raise ValueError(
                f"cache_hit_rate must be between 0.0 and 1.0, "
                f"got {self.cache_hit_rate}"
            )
        
        # Validate latencies are positive
        if self.original_latency_ms <= 0:
            raise ValueError(
                f"original_latency_ms must be positive, "
                f"got {self.original_latency_ms}"
            )
        if self.optimized_latency_ms <= 0:
            raise ValueError(
                f"optimized_latency_ms must be positive, "
                f"got {self.optimized_latency_ms}"
            )
    
    @property
    def has_acceptable_quality(self) -> bool:
        """Check if quantization error is within acceptable bounds."""
        if self.quantization_error is None:
            return True
        return self.quantization_error < MAX_QUANTIZATION_ERROR
    
    @property
    def latency_reduction_ms(self) -> float:
        """Calculate absolute latency reduction in milliseconds."""
        return self.original_latency_ms - self.optimized_latency_ms
    
    @property
    def latency_reduction_percent(self) -> float:
        """Calculate percentage latency reduction."""
        return (self.latency_reduction_ms / self.original_latency_ms) * 100


@dataclass
class BenchmarkMetrics:
    """
    Benchmark metrics from running inference performance tests.
    
    Captures detailed timing statistics, throughput, and memory usage
    from a benchmark run.
    
    Attributes:
        latency_mean_ms: Average latency across all runs in milliseconds
        latency_std_ms: Standard deviation of latency in milliseconds
        latency_p50_ms: 50th percentile (median) latency in milliseconds
        latency_p95_ms: 95th percentile latency in milliseconds
        latency_p99_ms: 99th percentile latency in milliseconds
        throughput_images_per_sec: Number of images generated per second
        vram_peak_gb: Peak VRAM usage during benchmark in GB
        vram_allocated_gb: Allocated VRAM at end of benchmark in GB
        cache_hit_rate: Fraction of cache hits during inference (0.0 to 1.0)
        num_runs: Number of benchmark runs (excluding warmup)
        warmup_runs: Number of warmup runs before measurement
    
    Validation Rules:
        - latency_mean_ms must be positive
        - latency_std_ms must be non-negative
        - num_runs must be >= 1
        - throughput_images_per_sec = 1000 / latency_mean_ms
    """
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_images_per_sec: float
    vram_peak_gb: float
    vram_allocated_gb: float
    cache_hit_rate: float
    num_runs: int
    warmup_runs: int
    
    def __post_init__(self) -> None:
        """Validate benchmark metrics fields after initialization."""
        # Validate latency_mean_ms is positive
        if self.latency_mean_ms <= 0:
            raise ValueError(
                f"latency_mean_ms must be positive, got {self.latency_mean_ms}"
            )
        
        # Validate latency_std_ms is non-negative
        if self.latency_std_ms < 0:
            raise ValueError(
                f"latency_std_ms must be non-negative, got {self.latency_std_ms}"
            )
        
        # Validate num_runs >= 1
        if self.num_runs < 1:
            raise ValueError(
                f"num_runs must be >= 1, got {self.num_runs}"
            )
        
        # Validate warmup_runs is non-negative
        if self.warmup_runs < 0:
            raise ValueError(
                f"warmup_runs must be non-negative, got {self.warmup_runs}"
            )
        
        # Validate cache hit rate is in valid range
        if not 0.0 <= self.cache_hit_rate <= 1.0:
            raise ValueError(
                f"cache_hit_rate must be between 0.0 and 1.0, "
                f"got {self.cache_hit_rate}"
            )
        
        # Validate VRAM values are non-negative
        if self.vram_peak_gb < 0:
            raise ValueError(
                f"vram_peak_gb must be non-negative, got {self.vram_peak_gb}"
            )
        if self.vram_allocated_gb < 0:
            raise ValueError(
                f"vram_allocated_gb must be non-negative, got {self.vram_allocated_gb}"
            )
        
        # Validate percentile ordering
        if not (self.latency_p50_ms <= self.latency_p95_ms <= self.latency_p99_ms):
            raise ValueError(
                f"Percentiles must be ordered: p50 <= p95 <= p99, "
                f"got p50={self.latency_p50_ms}, p95={self.latency_p95_ms}, "
                f"p99={self.latency_p99_ms}"
            )
    
    @property
    def latency_cv(self) -> float:
        """
        Coefficient of variation for latency.
        
        CV = std / mean, measures relative variability.
        Returns 0 if mean is 0 to avoid division by zero.
        """
        if self.latency_mean_ms > 0:
            return self.latency_std_ms / self.latency_mean_ms
        return 0.0
    
    @property
    def expected_throughput(self) -> float:
        """
        Calculate expected throughput from mean latency.
        
        throughput = 1000 / latency_mean_ms (images per second)
        """
        return 1000.0 / self.latency_mean_ms
    
    @property
    def is_vram_compliant(self) -> bool:
        """Check if VRAM usage is within T4 limits."""
        return self.vram_peak_gb <= T4_VRAM_LIMIT_GB


@dataclass
class CacheEntry:
    """
    Entry in the feature cache for DeepCache/X-Slim style caching.
    
    Stores intermediate UNet features that can be reused at similar
    timesteps to skip redundant computations.
    
    Attributes:
        timestep: Diffusion timestep when features were computed
        block_idx: Index of the UNet block that produced these features
        features: Cached feature tensor (must be on CUDA device)
        created_at: Inference step when this entry was created
        access_count: Number of times this entry has been accessed
        size_bytes: Size of the features tensor in bytes (computed automatically)
    
    Validation Rules:
        - timestep must be in valid range [0, num_inference_steps)
        - block_idx must be valid UNet block index (non-negative)
        - features tensor must be on CUDA device
    """
    timestep: int
    block_idx: int
    features: torch.Tensor
    created_at: int
    access_count: int = 0
    size_bytes: int = field(default=0, init=True)
    
    def __post_init__(self) -> None:
        """Validate cache entry fields and compute size_bytes."""
        # Validate timestep is non-negative
        if self.timestep < 0:
            raise ValueError(
                f"timestep must be non-negative, got {self.timestep}"
            )
        
        # Validate block_idx is non-negative
        if self.block_idx < 0:
            raise ValueError(
                f"block_idx must be non-negative, got {self.block_idx}"
            )
        
        # Validate created_at is non-negative
        if self.created_at < 0:
            raise ValueError(
                f"created_at must be non-negative, got {self.created_at}"
            )
        
        # Validate access_count is non-negative
        if self.access_count < 0:
            raise ValueError(
                f"access_count must be non-negative, got {self.access_count}"
            )
        
        # Validate features tensor is on CUDA device
        if not self.features.is_cuda:
            raise ValueError(
                f"features tensor must be on CUDA device, "
                f"got device: {self.features.device}"
            )
        
        # Compute size_bytes from tensor dimensions
        self.size_bytes = self.features.numel() * self.features.element_size()
    
    def record_access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
    
    @property
    def size_mb(self) -> float:
        """Return size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Return size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)
