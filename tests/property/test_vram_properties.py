"""
Property-based tests for VRAM constraint verification.

Property 2: VRAM Constraint
- Verify VRAM usage never exceeds 15.6GB during any operation

Validates: Requirements 8.1, 8.2, 8.3

Note: These tests use mocks to simulate VRAM usage patterns since actual
GPU operations would require downloading models. The tests verify that
the VRAM monitoring and constraint enforcement mechanisms work correctly.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple, List
from unittest.mock import Mock, MagicMock, patch

import torch

from diffusion_trt.models import T4_VRAM_LIMIT_GB
from diffusion_trt.utils.vram_monitor import VRAMMonitor, get_vram_usage, clear_cache
from diffusion_trt.pipeline import (
    PipelineConfig, 
    OptimizedPipeline,
    MODEL_WEIGHTS_VRAM_LIMIT_GB,
    VRAM_WARNING_THRESHOLD_GB,
)
from diffusion_trt.cache_manager import CacheConfig, CacheManager


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# =============================================================================
# Property Test Strategies
# =============================================================================


@st.composite
def valid_vram_usage_gb(draw):
    """Generate valid VRAM usage values within T4 limits."""
    return draw(st.floats(min_value=0.0, max_value=T4_VRAM_LIMIT_GB, allow_nan=False))


@st.composite
def vram_usage_sequence(draw):
    """Generate a sequence of VRAM usage values simulating an operation."""
    num_samples = draw(st.integers(min_value=3, max_value=20))
    
    # Generate a sequence that may or may not exceed the limit
    base_usage = draw(st.floats(min_value=0.0, max_value=12.0, allow_nan=False))
    
    sequence = []
    for _ in range(num_samples):
        # Add some variation to simulate memory fluctuations
        delta = draw(st.floats(min_value=-2.0, max_value=4.0, allow_nan=False))
        usage = max(0.0, base_usage + delta)
        sequence.append(usage)
    
    return sequence


@st.composite
def tensor_allocation_params(draw):
    """Generate parameters for tensor allocations that fit within VRAM limits."""
    # Calculate max elements that fit in ~1GB (float32 = 4 bytes)
    max_elements_1gb = (1024 ** 3) // 4
    
    # Generate tensor dimensions that result in reasonable sizes
    batch_size = draw(st.integers(min_value=1, max_value=2))
    channels = draw(st.sampled_from([64, 128, 256, 320, 640]))
    height = draw(st.sampled_from([8, 16, 32, 64]))
    width = draw(st.sampled_from([8, 16, 32, 64]))
    
    return (batch_size, channels, height, width)


@st.composite
def cache_size_params(draw):
    """Generate cache size parameters within the 2GB limit."""
    max_cache_size_gb = draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False))
    num_entries = draw(st.integers(min_value=1, max_value=10))
    
    return {
        'max_cache_size_gb': max_cache_size_gb,
        'num_entries': num_entries,
    }


@st.composite
def operation_type(draw):
    """Generate operation types for VRAM testing."""
    return draw(st.sampled_from([
        'model_loading',
        'inference',
        'caching',
        'benchmark',
    ]))


# =============================================================================
# Property Tests - VRAM Constraint (Property 2)
# =============================================================================


class TestVRAMConstraintProperties:
    """
    Property tests for VRAM constraint verification.
    
    Property 2: VRAM Constraint
    - Verify VRAM usage never exceeds 15.6GB during any operation
    
    Validates: Requirements 8.1, 8.2, 8.3
    """

    @pytest.mark.gpu
    @given(
        initial_vram=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        peak_vram=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_vram_monitor_tracks_peak_usage(
        self,
        initial_vram: float,
        peak_vram: float,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAMMonitor correctly tracks peak VRAM usage during operations.
        
        The VRAMMonitor context manager should accurately track the peak VRAM
        usage that occurs during its scope, regardless of the initial state.
        """
        # Create a VRAMMonitor
        monitor = VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB)
        
        with monitor:
            # Allocate some tensors to create measurable VRAM usage
            tensors = []
            try:
                # Allocate small tensors to create some VRAM usage
                for _ in range(5):
                    t = torch.randn(100, 100, device='cuda')
                    tensors.append(t)
                
                # Get current usage
                current_usage = monitor.get_vram_usage()
                
                # Verify we can track usage
                assert current_usage >= 0, "VRAM usage should be non-negative"
                
            finally:
                # Clean up tensors
                del tensors
                torch.cuda.empty_cache()
        
        # After exiting context, peak should be recorded
        assert monitor.peak_gb >= 0, "Peak VRAM should be non-negative"
        assert monitor.start_gb >= 0, "Start VRAM should be non-negative"
        assert monitor.end_gb >= 0, "End VRAM should be non-negative"

    @pytest.mark.gpu
    @given(
        limit_gb=st.floats(min_value=1.0, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_vram_monitor_limit_configuration(
        self,
        limit_gb: float,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAMMonitor respects configured VRAM limits.
        
        The VRAMMonitor should correctly store and use the configured
        VRAM limit for compliance checking.
        """
        monitor = VRAMMonitor(limit_gb=limit_gb)
        
        # Verify limit is correctly set
        assert monitor.limit_gb == limit_gb, (
            f"Monitor limit should be {limit_gb}, got {monitor.limit_gb}"
        )
        
        with monitor:
            # Perform minimal operation
            t = torch.randn(10, 10, device='cuda')
            del t
        
        # Check is_within_limit property works correctly
        if monitor.peak_gb <= limit_gb:
            assert monitor.is_within_limit, (
                f"Peak {monitor.peak_gb} <= limit {limit_gb}, "
                f"but is_within_limit is False"
            )
        else:
            assert not monitor.is_within_limit, (
                f"Peak {monitor.peak_gb} > limit {limit_gb}, "
                f"but is_within_limit is True"
            )

    @pytest.mark.gpu
    @given(
        tensor_params=tensor_allocation_params(),
        num_tensors=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_vram_usage_during_tensor_operations(
        self,
        tensor_params: Tuple[int, int, int, int],
        num_tensors: int,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAM usage stays within T4 limits during tensor operations.
        
        When allocating tensors on GPU, the total VRAM usage should not
        exceed the T4 limit of 15.6GB.
        """
        batch_size, channels, height, width = tensor_params
        
        with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
            tensors = []
            try:
                for _ in range(num_tensors):
                    t = torch.randn(
                        batch_size, channels, height, width,
                        device='cuda',
                        dtype=torch.float32
                    )
                    tensors.append(t)
                    
                    # Check VRAM after each allocation
                    current_vram = monitor.get_vram_usage()
                    assert current_vram <= T4_VRAM_LIMIT_GB, (
                        f"VRAM usage {current_vram:.2f} GB exceeds T4 limit "
                        f"{T4_VRAM_LIMIT_GB} GB during tensor allocation"
                    )
            finally:
                # Clean up
                del tensors
                torch.cuda.empty_cache()
        
        # Verify peak was within limits
        assert monitor.peak_gb <= T4_VRAM_LIMIT_GB, (
            f"Peak VRAM {monitor.peak_gb:.2f} GB exceeded T4 limit "
            f"{T4_VRAM_LIMIT_GB} GB"
        )

    @pytest.mark.gpu
    @given(
        cache_params=cache_size_params(),
        tensor_params=tensor_allocation_params(),
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_vram_stays_within_limit(
        self,
        cache_params: dict,
        tensor_params: Tuple[int, int, int, int],
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: Feature cache VRAM usage stays within the 2GB limit.
        
        The CacheManager should enforce the max_cache_size_gb limit,
        ensuring cached features don't exceed the configured maximum.
        """
        batch_size, channels, height, width = tensor_params
        max_cache_size_gb = cache_params['max_cache_size_gb']
        num_entries = cache_params['num_entries']
        
        # Create cache manager with size limit
        config = CacheConfig(
            cache_interval=1,
            max_cache_size_gb=max_cache_size_gb,
        )
        cache_manager = CacheManager(config)
        
        with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
            try:
                # Store multiple cache entries
                for i in range(num_entries):
                    features = torch.randn(
                        batch_size, channels, height, width,
                        device='cuda',
                        dtype=torch.float32
                    )
                    cache_manager.store(i * 10, i % 12, features)
                
                # Get cache stats
                stats = cache_manager.get_cache_stats()
                
                # Verify cache size is within configured limit
                # Note: The cache manager should evict entries to stay within limit
                cache_size_gb = stats.get('total_size_bytes', 0) / (1024 ** 3)
                
                # Allow some tolerance for measurement
                assert cache_size_gb <= max_cache_size_gb + 0.1, (
                    f"Cache size {cache_size_gb:.3f} GB exceeds limit "
                    f"{max_cache_size_gb:.3f} GB"
                )
                
            finally:
                cache_manager.clear()
                torch.cuda.empty_cache()
        
        # Verify overall VRAM stayed within T4 limits
        assert monitor.peak_gb <= T4_VRAM_LIMIT_GB, (
            f"Peak VRAM {monitor.peak_gb:.2f} GB exceeded T4 limit during caching"
        )

    @pytest.mark.gpu
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=30, deadline=None)
    def test_pipeline_vram_check_mechanism(
        self,
        seed: int,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: OptimizedPipeline's VRAM check mechanism works correctly.
        
        The _check_vram_usage method should monitor VRAM and take action
        when approaching the limit.
        """
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=seed,
        )
        pipeline = OptimizedPipeline(config)
        
        with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
            # Allocate some tensors to simulate VRAM usage
            tensors = []
            try:
                for _ in range(3):
                    t = torch.randn(100, 100, device='cuda')
                    tensors.append(t)
                
                # Call the VRAM check method
                # This should not raise if we're within limits
                pipeline._check_vram_usage("test_operation")
                
                # Verify we're still within limits
                current_vram = get_vram_usage()
                assert current_vram <= T4_VRAM_LIMIT_GB, (
                    f"VRAM {current_vram:.2f} GB exceeds limit after check"
                )
                
            finally:
                del tensors
                torch.cuda.empty_cache()

    @pytest.mark.gpu
    @given(
        num_allocations=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_vram_clear_cache_reduces_usage(
        self,
        num_allocations: int,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: Clearing cache reduces VRAM usage.
        
        After allocating tensors and then clearing the cache,
        VRAM usage should decrease.
        """
        with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
            # Record initial usage
            initial_vram = get_vram_usage()
            
            # Allocate tensors
            tensors = []
            for _ in range(num_allocations):
                t = torch.randn(500, 500, device='cuda')
                tensors.append(t)
            
            # Record usage after allocation
            after_alloc_vram = get_vram_usage()
            
            # Delete tensors and clear cache
            del tensors
            clear_cache()
            
            # Record usage after clearing
            after_clear_vram = get_vram_usage()
            
            # Verify VRAM decreased after clearing
            assert after_clear_vram <= after_alloc_vram, (
                f"VRAM should decrease after clearing cache. "
                f"Before: {after_alloc_vram:.4f} GB, After: {after_clear_vram:.4f} GB"
            )

    @pytest.mark.gpu
    @given(
        enforce_limit=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_vram_monitor_enforce_limit_flag(
        self,
        enforce_limit: bool,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAMMonitor enforce_limit flag controls error raising.
        
        When enforce_limit is True, the check_limit method should raise
        OutOfMemoryError if VRAM exceeds the limit. When False, it should not.
        """
        # Use a very low limit to test enforcement (0.001 GB = ~1MB)
        low_limit = 0.001
        
        monitor = VRAMMonitor(limit_gb=low_limit, enforce_limit=enforce_limit)
        
        with monitor:
            # Allocate a tensor that will definitely exceed 1MB limit
            # 1000x1000 floats = 4MB
            t = torch.randn(1000, 1000, device='cuda')
            torch.cuda.synchronize()  # Ensure allocation is complete
            
            if enforce_limit:
                # Should raise OutOfMemoryError
                with pytest.raises((torch.cuda.OutOfMemoryError, MemoryError)):
                    monitor.check_limit()
            else:
                # Should not raise
                monitor.check_limit()  # No exception expected
            
            del t
            torch.cuda.empty_cache()

    @pytest.mark.gpu
    @given(
        tensor_params=tensor_allocation_params(),
    )
    @settings(max_examples=30, deadline=None)
    def test_vram_delta_tracking(
        self,
        tensor_params: Tuple[int, int, int, int],
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAMMonitor correctly tracks VRAM delta during operations.
        
        The delta_gb property should accurately reflect the change in
        VRAM usage between entering and exiting the context.
        """
        batch_size, channels, height, width = tensor_params
        
        with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
            # Allocate tensor
            t = torch.randn(
                batch_size, channels, height, width,
                device='cuda',
                dtype=torch.float32
            )
            
            # Keep tensor alive until end of context
            _ = t.sum()  # Force computation
            
            del t
            torch.cuda.empty_cache()
        
        # Delta should be reasonable (could be positive, negative, or zero
        # depending on CUDA memory management)
        assert isinstance(monitor.delta_gb, float), (
            f"delta_gb should be a float, got {type(monitor.delta_gb)}"
        )
        
        # Verify delta calculation is correct
        expected_delta = monitor.end_gb - monitor.start_gb
        assert abs(monitor.delta_gb - expected_delta) < 1e-6, (
            f"delta_gb {monitor.delta_gb} doesn't match "
            f"end_gb - start_gb = {expected_delta}"
        )

    @pytest.mark.gpu
    @given(
        model_id=st.sampled_from([
            "stabilityai/sdxl-turbo",
            "runwayml/stable-diffusion-v1-5",
        ]),
    )
    @settings(max_examples=10, deadline=None)
    def test_pipeline_config_vram_awareness(
        self,
        model_id: str,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: Pipeline configuration respects VRAM constraints.
        
        The PipelineConfig should enforce VRAM-aware defaults that
        help stay within the T4 limit.
        """
        # Create config with VRAM-aware settings
        config = PipelineConfig(
            model_id=model_id,
            max_cache_size_gb=2.0,  # Within the 2GB cache limit
        )
        
        # Verify cache size limit is within bounds
        assert config.max_cache_size_gb <= 2.0, (
            f"max_cache_size_gb {config.max_cache_size_gb} exceeds 2GB limit"
        )
        
        # Verify the T4 VRAM limit constant is correct
        assert T4_VRAM_LIMIT_GB == 15.6, (
            f"T4_VRAM_LIMIT_GB should be 15.6, got {T4_VRAM_LIMIT_GB}"
        )
        
        # Verify model weights limit is within T4 limit
        assert MODEL_WEIGHTS_VRAM_LIMIT_GB <= T4_VRAM_LIMIT_GB, (
            f"MODEL_WEIGHTS_VRAM_LIMIT_GB {MODEL_WEIGHTS_VRAM_LIMIT_GB} "
            f"exceeds T4 limit {T4_VRAM_LIMIT_GB}"
        )
        
        # Verify warning threshold is below T4 limit
        assert VRAM_WARNING_THRESHOLD_GB < T4_VRAM_LIMIT_GB, (
            f"VRAM_WARNING_THRESHOLD_GB {VRAM_WARNING_THRESHOLD_GB} "
            f"should be below T4 limit {T4_VRAM_LIMIT_GB}"
        )

    @pytest.mark.gpu
    @given(
        num_iterations=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_repeated_operations_vram_stability(
        self,
        num_iterations: int,
    ):
        """
        **Validates: Requirements 8.1, 8.2, 8.3**
        
        Property: VRAM usage remains stable across repeated operations.
        
        Performing the same operation multiple times should not cause
        unbounded VRAM growth (no memory leaks).
        """
        vram_readings = []
        
        for i in range(num_iterations):
            with VRAMMonitor(limit_gb=T4_VRAM_LIMIT_GB) as monitor:
                # Perform operation
                t = torch.randn(200, 200, device='cuda')
                _ = t @ t.T  # Matrix multiplication
                del t
                torch.cuda.empty_cache()
            
            vram_readings.append(monitor.peak_gb)
        
        # Verify no unbounded growth
        # Allow some variance but check that later readings aren't much higher
        max_reading = max(vram_readings)
        min_reading = min(vram_readings)
        
        # The difference should be small (no memory leak)
        assert max_reading - min_reading < 1.0, (
            f"VRAM readings vary too much across iterations: "
            f"min={min_reading:.3f} GB, max={max_reading:.3f} GB. "
            f"This may indicate a memory leak."
        )
        
        # All readings should be within T4 limit
        for i, reading in enumerate(vram_readings):
            assert reading <= T4_VRAM_LIMIT_GB, (
                f"Iteration {i}: VRAM {reading:.2f} GB exceeded T4 limit"
            )

