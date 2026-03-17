"""
Unit tests for the VRAMMonitor context manager and VRAM utilities.

Tests cover:
- get_vram_usage() method returning current usage in GB
- Peak VRAM tracking during operations
- VRAM limit enforcement (15.6GB for T4)
- clear_cache() functionality
"""

import pytest
import torch

from diffusion_trt.utils.vram_monitor import (
    VRAMMonitor,
    get_vram_usage,
    get_peak_vram,
    clear_cache,
)
from diffusion_trt.models import T4_VRAM_LIMIT_GB


# Mark for CUDA-dependent tests
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@requires_cuda
class TestGetVramUsage:
    """Tests for the get_vram_usage function."""
    
    def test_returns_float(self):
        """get_vram_usage should return a float value."""
        usage = get_vram_usage()
        assert isinstance(usage, float)
    
    def test_returns_non_negative(self):
        """get_vram_usage should return a non-negative value."""
        usage = get_vram_usage()
        assert usage >= 0.0
    
    def test_increases_with_allocation(self):
        """VRAM usage should increase when allocating tensors."""
        initial = get_vram_usage()
        # Allocate ~100MB tensor
        tensor = torch.randn(25 * 1024 * 1024, device='cuda')  # 100MB (float32)
        after = get_vram_usage()
        assert after > initial
        del tensor
        torch.cuda.empty_cache()


@requires_cuda
class TestGetPeakVram:
    """Tests for the get_peak_vram function."""
    
    def test_returns_float(self):
        """get_peak_vram should return a float value."""
        peak = get_peak_vram()
        assert isinstance(peak, float)
    
    def test_returns_non_negative(self):
        """get_peak_vram should return a non-negative value."""
        peak = get_peak_vram()
        assert peak >= 0.0
    
    def test_tracks_peak_allocation(self):
        """Peak VRAM should track maximum allocation."""
        torch.cuda.reset_peak_memory_stats()
        # Allocate and deallocate
        tensor = torch.randn(25 * 1024 * 1024, device='cuda')  # 100MB
        peak_during = get_peak_vram()
        del tensor
        torch.cuda.empty_cache()
        peak_after = get_peak_vram()
        # Peak should remain at the high water mark
        assert peak_after >= peak_during - 0.01  # Allow small tolerance


@requires_cuda
class TestClearCache:
    """Tests for the clear_cache function."""
    
    def test_clears_without_error(self):
        """clear_cache should execute without raising errors."""
        # Allocate some memory first
        tensor = torch.randn(10 * 1024 * 1024, device='cuda')
        del tensor
        # Should not raise
        clear_cache()
    
    def test_reduces_cached_memory(self):
        """clear_cache should reduce cached GPU memory."""
        # Allocate and deallocate to create cached memory
        tensor = torch.randn(25 * 1024 * 1024, device='cuda')
        del tensor
        # Get reserved memory before clearing
        reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)
        clear_cache()
        reserved_after = torch.cuda.memory_reserved() / (1024 ** 3)
        # Reserved memory should decrease or stay same
        assert reserved_after <= reserved_before


@requires_cuda
class TestVRAMMonitorContextManager:
    """Tests for VRAMMonitor as a context manager."""
    
    def test_enters_and_exits(self):
        """VRAMMonitor should work as a context manager."""
        with VRAMMonitor() as monitor:
            assert monitor is not None
    
    def test_tracks_start_usage(self):
        """VRAMMonitor should record starting VRAM usage."""
        with VRAMMonitor() as monitor:
            pass
        assert isinstance(monitor.start_gb, float)
        assert monitor.start_gb >= 0.0
    
    def test_tracks_end_usage(self):
        """VRAMMonitor should record ending VRAM usage."""
        with VRAMMonitor() as monitor:
            pass
        assert isinstance(monitor.end_gb, float)
        assert monitor.end_gb >= 0.0
    
    def test_tracks_peak_usage(self):
        """VRAMMonitor should track peak VRAM during scope."""
        with VRAMMonitor() as monitor:
            # Allocate memory to create a peak
            tensor = torch.randn(25 * 1024 * 1024, device='cuda')
            del tensor
        assert isinstance(monitor.peak_gb, float)
        assert monitor.peak_gb >= 0.0
    
    def test_peak_captures_temporary_allocation(self):
        """Peak should capture memory even if freed before exit."""
        with VRAMMonitor() as monitor:
            # Allocate ~100MB
            tensor = torch.randn(25 * 1024 * 1024, device='cuda')
            del tensor
            torch.cuda.empty_cache()
        # Peak should be at least 0.09 GB (allowing for overhead)
        assert monitor.peak_gb >= 0.09
    
    def test_delta_gb_property(self):
        """delta_gb should return difference between end and start."""
        with VRAMMonitor() as monitor:
            pass
        expected_delta = monitor.end_gb - monitor.start_gb
        assert abs(monitor.delta_gb - expected_delta) < 0.001


@requires_cuda
class TestVRAMMonitorLimitEnforcement:
    """Tests for VRAM limit enforcement."""
    
    def test_default_limit_is_t4(self):
        """Default limit should be T4 VRAM limit (15.6GB)."""
        monitor = VRAMMonitor()
        assert monitor.limit_gb == T4_VRAM_LIMIT_GB
        assert monitor.limit_gb == 15.6
    
    def test_custom_limit(self):
        """Should accept custom VRAM limit."""
        monitor = VRAMMonitor(limit_gb=8.0)
        assert monitor.limit_gb == 8.0
    
    def test_enforce_limit_default_false(self):
        """enforce_limit should default to False."""
        monitor = VRAMMonitor()
        assert monitor.enforce_limit is False
    
    def test_is_within_limit_true(self):
        """is_within_limit should return True when under limit."""
        with VRAMMonitor(limit_gb=100.0) as monitor:
            pass
        assert monitor.is_within_limit is True
    
    def test_check_limit_no_error_when_under(self):
        """check_limit should not raise when under limit."""
        with VRAMMonitor(limit_gb=100.0, enforce_limit=True) as monitor:
            monitor.check_limit()  # Should not raise


@requires_cuda
class TestVRAMMonitorMethods:
    """Tests for VRAMMonitor instance methods."""
    
    def test_get_vram_usage_method(self):
        """Instance get_vram_usage should return current usage."""
        with VRAMMonitor() as monitor:
            usage = monitor.get_vram_usage()
            assert isinstance(usage, float)
            assert usage >= 0.0
    
    def test_get_peak_vram_method(self):
        """Instance get_peak_vram should return peak usage."""
        with VRAMMonitor() as monitor:
            peak = monitor.get_peak_vram()
            assert isinstance(peak, float)
            assert peak >= 0.0
    
    def test_clear_cache_method(self):
        """Instance clear_cache should free GPU memory."""
        with VRAMMonitor() as monitor:
            tensor = torch.randn(10 * 1024 * 1024, device='cuda')
            del tensor
            monitor.clear_cache()  # Should not raise


class TestVRAMMonitorWithoutCUDA:
    """Tests for VRAMMonitor behavior when CUDA is not available."""
    
    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Test only runs when CUDA is not available"
    )
    def test_get_vram_usage_returns_zero(self):
        """get_vram_usage should return 0.0 without CUDA."""
        usage = get_vram_usage()
        assert usage == 0.0
    
    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Test only runs when CUDA is not available"
    )
    def test_get_peak_vram_returns_zero(self):
        """get_peak_vram should return 0.0 without CUDA."""
        peak = get_peak_vram()
        assert peak == 0.0


class TestVRAMMonitorBasicFunctionality:
    """Tests that run regardless of CUDA availability."""
    
    def test_vram_monitor_instantiation(self):
        """VRAMMonitor should instantiate with default parameters."""
        monitor = VRAMMonitor()
        assert monitor.limit_gb == T4_VRAM_LIMIT_GB
        assert monitor.enforce_limit is False
        assert monitor.peak_gb == 0.0
        assert monitor.start_gb == 0.0
        assert monitor.end_gb == 0.0
    
    def test_vram_monitor_custom_limit(self):
        """VRAMMonitor should accept custom limit."""
        monitor = VRAMMonitor(limit_gb=8.0, enforce_limit=True)
        assert monitor.limit_gb == 8.0
        assert monitor.enforce_limit is True
    
    def test_context_manager_returns_self(self):
        """Context manager should return self on enter."""
        monitor = VRAMMonitor()
        with monitor as m:
            assert m is monitor
    
    def test_delta_gb_calculation(self):
        """delta_gb should calculate correctly."""
        monitor = VRAMMonitor()
        monitor.start_gb = 1.0
        monitor.end_gb = 2.5
        assert monitor.delta_gb == 1.5
    
    def test_is_within_limit_calculation(self):
        """is_within_limit should compare peak to limit."""
        monitor = VRAMMonitor(limit_gb=10.0)
        monitor.peak_gb = 5.0
        assert monitor.is_within_limit is True
        
        monitor.peak_gb = 15.0
        assert monitor.is_within_limit is False
    
    def test_clear_cache_runs_without_error(self):
        """clear_cache should run without error even without CUDA."""
        clear_cache()  # Should not raise
    
    def test_get_vram_usage_returns_float(self):
        """get_vram_usage should always return a float."""
        usage = get_vram_usage()
        assert isinstance(usage, float)
    
    def test_get_peak_vram_returns_float(self):
        """get_peak_vram should always return a float."""
        peak = get_peak_vram()
        assert isinstance(peak, float)
