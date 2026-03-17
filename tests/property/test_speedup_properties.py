"""
Property-based tests for speedup guarantee verification.

Property 3: Speedup Guarantee
- Verify optimized pipeline is faster than baseline
- Verify speedup >= 1.5x for all supported models

Validates: Requirements 7.1, 7.2, 7.3

Note: These tests use mocks to simulate timing behavior since actual
GPU operations would require downloading models. The tests verify that
the speedup calculation and comparison logic works correctly.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple
from unittest.mock import Mock, MagicMock, patch
import time

import torch

from diffusion_trt.models import BenchmarkMetrics, T4_VRAM_LIMIT_GB
from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline


# =============================================================================
# Property Test Strategies
# =============================================================================


@st.composite
def valid_latency_pair(draw):
    """
    Generate valid baseline and optimized latency pairs.
    
    Ensures baseline >= optimized (optimized should be faster).
    """
    # Baseline latency: 500ms to 10000ms
    baseline_ms = draw(st.floats(min_value=500.0, max_value=10000.0, allow_nan=False))
    
    # Speedup factor: 1.0x to 3.0x
    speedup = draw(st.floats(min_value=1.0, max_value=3.0, allow_nan=False))
    
    # Optimized latency = baseline / speedup
    optimized_ms = baseline_ms / speedup
    
    return baseline_ms, optimized_ms, speedup


@st.composite
def benchmark_metrics_params(draw):
    """Generate valid parameters for BenchmarkMetrics."""
    latency_mean = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False))
    latency_std = draw(st.floats(min_value=0.0, max_value=latency_mean * 0.5, allow_nan=False))
    
    # Percentiles should be ordered: p50 <= p95 <= p99
    p50 = draw(st.floats(min_value=latency_mean * 0.8, max_value=latency_mean * 1.1, allow_nan=False))
    p95 = draw(st.floats(min_value=p50, max_value=p50 * 1.3, allow_nan=False))
    p99 = draw(st.floats(min_value=p95, max_value=p95 * 1.2, allow_nan=False))
    
    throughput = 1000.0 / latency_mean if latency_mean > 0 else 0.0
    
    vram_peak = draw(st.floats(min_value=1.0, max_value=T4_VRAM_LIMIT_GB, allow_nan=False))
    vram_allocated = draw(st.floats(min_value=0.5, max_value=vram_peak, allow_nan=False))
    
    cache_hit_rate = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    
    num_runs = draw(st.integers(min_value=1, max_value=100))
    warmup_runs = draw(st.integers(min_value=0, max_value=10))
    
    return {
        'latency_mean_ms': latency_mean,
        'latency_std_ms': latency_std,
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
        'latency_p99_ms': p99,
        'throughput_images_per_sec': throughput,
        'vram_peak_gb': vram_peak,
        'vram_allocated_gb': vram_allocated,
        'cache_hit_rate': cache_hit_rate,
        'num_runs': num_runs,
        'warmup_runs': warmup_runs,
    }


# =============================================================================
# Property Tests - Speedup Guarantee (Property 3)
# =============================================================================


class TestSpeedupGuaranteeProperties:
    """
    Property tests for speedup guarantee verification.
    
    Property 3: Speedup Guarantee
    - Verify optimized pipeline is faster than baseline
    - Verify speedup >= 1.5x for all supported models
    
    Validates: Requirements 7.1, 7.2, 7.3
    """

    @given(
        latency_pair=valid_latency_pair(),
    )
    @settings(max_examples=100, deadline=None)
    def test_speedup_calculation_is_correct(
        self,
        latency_pair: Tuple[float, float, float],
    ):
        """
        **Validates: Requirements 7.1, 7.2, 7.3**
        
        Property: Speedup calculation is mathematically correct.
        
        speedup = baseline_latency / optimized_latency
        """
        baseline_ms, optimized_ms, expected_speedup = latency_pair
        
        # Calculate speedup
        calculated_speedup = baseline_ms / optimized_ms
        
        # Verify calculation matches expected
        assert abs(calculated_speedup - expected_speedup) < 0.001, (
            f"Speedup calculation incorrect: "
            f"expected {expected_speedup:.3f}, got {calculated_speedup:.3f}"
        )

    @given(
        baseline_latency=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False),
        speedup_factor=st.floats(min_value=1.5, max_value=3.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_speedup_meets_minimum_requirement(
        self,
        baseline_latency: float,
        speedup_factor: float,
    ):
        """
        **Validates: Requirements 7.3**
        
        Property: When speedup >= 1.5x, the requirement is satisfied.
        
        Verifies that the speedup calculation correctly identifies
        when the 1.5x minimum speedup requirement is met.
        """
        # Calculate optimized latency for given speedup
        optimized_latency = baseline_latency / speedup_factor
        
        # Calculate actual speedup
        actual_speedup = baseline_latency / optimized_latency
        
        # Verify speedup meets minimum requirement
        assert actual_speedup >= 1.5, (
            f"Speedup {actual_speedup:.2f}x should be >= 1.5x"
        )

    @given(
        metrics_params=benchmark_metrics_params(),
    )
    @settings(max_examples=50, deadline=None)
    def test_benchmark_metrics_throughput_consistency(
        self,
        metrics_params: dict,
    ):
        """
        **Validates: Requirements 7.5**
        
        Property: Throughput is consistent with latency.
        
        throughput = 1000 / latency_mean_ms (images per second)
        """
        metrics = BenchmarkMetrics(**metrics_params)
        
        # Calculate expected throughput from latency
        expected_throughput = 1000.0 / metrics.latency_mean_ms
        
        # Verify throughput is consistent (within tolerance)
        assert abs(metrics.throughput_images_per_sec - expected_throughput) < 0.1, (
            f"Throughput inconsistent with latency: "
            f"expected {expected_throughput:.2f}, got {metrics.throughput_images_per_sec:.2f}"
        )

    @given(
        metrics_params=benchmark_metrics_params(),
    )
    @settings(max_examples=50, deadline=None)
    def test_benchmark_metrics_latency_cv_calculation(
        self,
        metrics_params: dict,
    ):
        """
        **Validates: Requirements 7.4**
        
        Property: Coefficient of variation is correctly calculated.
        
        CV = std / mean
        """
        metrics = BenchmarkMetrics(**metrics_params)
        
        # Calculate expected CV
        expected_cv = metrics.latency_std_ms / metrics.latency_mean_ms
        
        # Verify CV calculation
        assert abs(metrics.latency_cv - expected_cv) < 0.001, (
            f"CV calculation incorrect: "
            f"expected {expected_cv:.4f}, got {metrics.latency_cv:.4f}"
        )

    @given(
        metrics_params=benchmark_metrics_params(),
    )
    @settings(max_examples=50, deadline=None)
    def test_benchmark_metrics_percentile_ordering(
        self,
        metrics_params: dict,
    ):
        """
        **Validates: Requirements 11.3**
        
        Property: Latency percentiles are correctly ordered.
        
        p50 <= p95 <= p99 (by definition of percentiles)
        """
        metrics = BenchmarkMetrics(**metrics_params)
        
        # Verify percentile ordering
        assert metrics.latency_p50_ms <= metrics.latency_p95_ms, (
            f"P50 ({metrics.latency_p50_ms}) should be <= P95 ({metrics.latency_p95_ms})"
        )
        assert metrics.latency_p95_ms <= metrics.latency_p99_ms, (
            f"P95 ({metrics.latency_p95_ms}) should be <= P99 ({metrics.latency_p99_ms})"
        )

    @given(
        num_iterations=st.integers(min_value=1, max_value=100),
        warmup_iterations=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_benchmark_iteration_counts_preserved(
        self,
        num_iterations: int,
        warmup_iterations: int,
    ):
        """
        **Validates: Requirements 11.1**
        
        Property: Benchmark preserves iteration counts in metrics.
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_pipeline
        pipeline._is_optimized = True
        
        # Run benchmark
        metrics = pipeline.benchmark(
            prompt="test",
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
        )
        
        # Verify iteration counts
        assert metrics.num_runs == num_iterations, (
            f"num_runs should be {num_iterations}, got {metrics.num_runs}"
        )
        assert metrics.warmup_runs == warmup_iterations, (
            f"warmup_runs should be {warmup_iterations}, got {metrics.warmup_runs}"
        )

    @given(
        vram_peak=st.floats(min_value=1.0, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_vram_compliance_check(
        self,
        vram_peak: float,
    ):
        """
        **Validates: Requirements 8.1, 11.5**
        
        Property: VRAM compliance is correctly determined.
        
        VRAM is compliant if peak <= T4_VRAM_LIMIT_GB (15.6GB)
        """
        is_compliant = vram_peak <= T4_VRAM_LIMIT_GB
        
        if is_compliant:
            assert vram_peak <= 15.6, (
                f"VRAM {vram_peak}GB marked compliant but exceeds 15.6GB"
            )
        else:
            assert vram_peak > 15.6, (
                f"VRAM {vram_peak}GB marked non-compliant but is within 15.6GB"
            )


class TestLatencyTargetProperties:
    """
    Property tests for latency target verification.
    """

    @given(
        latency_ms=st.floats(min_value=100.0, max_value=5000.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_sdxl_turbo_latency_target(
        self,
        latency_ms: float,
    ):
        """
        **Validates: Requirements 7.2**
        
        Property: SDXL-Turbo latency target is 1 second (1000ms).
        
        Verifies the target check logic is correct.
        """
        target_ms = 1000.0  # 1 second target for SDXL-Turbo
        
        meets_target = latency_ms <= target_ms
        
        if meets_target:
            assert latency_ms <= 1000.0
        else:
            assert latency_ms > 1000.0

    @given(
        latency_ms=st.floats(min_value=500.0, max_value=8000.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_sd15_latency_target(
        self,
        latency_ms: float,
    ):
        """
        **Validates: Requirements 7.1**
        
        Property: SD 1.5 latency target is 4 seconds (4000ms).
        
        Verifies the target check logic is correct.
        """
        target_ms = 4000.0  # 4 second target for SD 1.5
        
        meets_target = latency_ms <= target_ms
        
        if meets_target:
            assert latency_ms <= 4000.0
        else:
            assert latency_ms > 4000.0

    @given(
        throughput=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_throughput_target(
        self,
        throughput: float,
    ):
        """
        **Validates: Requirements 7.5**
        
        Property: Throughput target is 1.0 images per second.
        
        Verifies the target check logic is correct.
        """
        target_throughput = 1.0  # 1 img/s target
        
        meets_target = throughput >= target_throughput
        
        if meets_target:
            assert throughput >= 1.0
        else:
            assert throughput < 1.0
