"""
VRAM monitoring utilities for the TensorRT Diffusion Optimization Pipeline.

Provides context manager and utilities for tracking GPU memory usage,
enforcing VRAM limits, and managing GPU memory.
"""

import gc
from typing import Optional

import torch

# Import T4 VRAM limit from models module
from diffusion_trt.models import T4_VRAM_LIMIT_GB


def get_vram_usage() -> float:
    """
    Get current VRAM usage in GB.
    
    Returns:
        Current GPU memory allocated in gigabytes.
        Returns 0.0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_peak_vram() -> float:
    """
    Get peak VRAM usage since last reset in GB.
    
    Returns:
        Peak GPU memory allocated in gigabytes.
        Returns 0.0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def clear_cache() -> None:
    """
    Clear GPU memory cache and run garbage collection.
    
    This frees up unused cached memory on the GPU and triggers
    Python garbage collection to release any unreferenced tensors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VRAMMonitor:
    """
    Context manager for monitoring and enforcing VRAM usage limits.
    
    Tracks VRAM usage during operations and optionally enforces
    a maximum VRAM limit (default: T4's 15.6GB).
    
    Attributes:
        limit_gb: Maximum allowed VRAM in GB (default: 15.6)
        enforce_limit: Whether to raise OutOfMemoryError if limit exceeded
        peak_gb: Peak VRAM usage during the monitored scope
        start_gb: VRAM usage when entering the context
        end_gb: VRAM usage when exiting the context
    
    Example:
        >>> with VRAMMonitor() as monitor:
        ...     # Perform GPU operations
        ...     images = pipeline(prompts)
        >>> print(f"Peak VRAM: {monitor.peak_gb:.2f} GB")
        >>> assert monitor.peak_gb <= 15.6, "Exceeded T4 VRAM limit!"
    
    Example with limit enforcement:
        >>> with VRAMMonitor(enforce_limit=True) as monitor:
        ...     # Will raise OutOfMemoryError if VRAM exceeds 15.6GB
        ...     large_tensor = torch.randn(10000, 10000, device='cuda')
    """
    
    def __init__(
        self,
        limit_gb: float = T4_VRAM_LIMIT_GB,
        enforce_limit: bool = False
    ) -> None:
        """
        Initialize the VRAM monitor.
        
        Args:
            limit_gb: Maximum allowed VRAM in GB (default: 15.6 for T4)
            enforce_limit: If True, raises OutOfMemoryError when limit exceeded
        """
        self.limit_gb = limit_gb
        self.enforce_limit = enforce_limit
        self.peak_gb: float = 0.0
        self.start_gb: float = 0.0
        self.end_gb: float = 0.0
        self._initial_peak: float = 0.0
    
    def __enter__(self) -> "VRAMMonitor":
        """
        Enter the monitoring context.
        
        Resets peak memory tracking and records starting VRAM usage.
        
        Returns:
            Self for use in with statement.
        """
        if torch.cuda.is_available():
            # Record initial state
            self.start_gb = get_vram_usage()
            self._initial_peak = get_peak_vram()
            # Reset peak memory stats to track only this scope
            torch.cuda.reset_peak_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the monitoring context.
        
        Records final VRAM usage and peak usage during the scope.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        
        Returns:
            False to propagate any exceptions.
        """
        if torch.cuda.is_available():
            self.end_gb = get_vram_usage()
            self.peak_gb = get_peak_vram()
        return False
    
    def get_vram_usage(self) -> float:
        """
        Get current VRAM usage in GB.
        
        Returns:
            Current GPU memory allocated in gigabytes.
        """
        return get_vram_usage()
    
    def get_peak_vram(self) -> float:
        """
        Get peak VRAM usage since entering the context in GB.
        
        Returns:
            Peak GPU memory allocated in gigabytes since __enter__.
        """
        return get_peak_vram()
    
    def check_limit(self) -> None:
        """
        Check if current VRAM usage exceeds the limit.
        
        Raises:
            torch.cuda.OutOfMemoryError: If enforce_limit is True and
                current VRAM usage exceeds limit_gb.
        """
        if self.enforce_limit:
            current = get_vram_usage()
            if current > self.limit_gb:
                raise torch.cuda.OutOfMemoryError(
                    f"VRAM usage ({current:.2f} GB) exceeds limit "
                    f"({self.limit_gb:.2f} GB)"
                )
    
    def clear_cache(self) -> None:
        """
        Clear GPU memory cache and run garbage collection.
        
        Useful for freeing up memory when approaching VRAM limits.
        """
        clear_cache()
    
    @property
    def delta_gb(self) -> float:
        """
        Get the change in VRAM usage during the monitored scope.
        
        Returns:
            Difference between end and start VRAM usage in GB.
        """
        return self.end_gb - self.start_gb
    
    @property
    def is_within_limit(self) -> bool:
        """
        Check if peak VRAM usage stayed within the limit.
        
        Returns:
            True if peak_gb <= limit_gb, False otherwise.
        """
        return self.peak_gb <= self.limit_gb


__all__ = [
    "VRAMMonitor",
    "get_vram_usage",
    "get_peak_vram",
    "clear_cache",
]
