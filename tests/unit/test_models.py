"""
Unit tests for data models in diffusion_trt.models.

Tests validation rules and property calculations for:
- OptimizationResult
- BenchmarkMetrics
- CacheEntry

Validates: Requirements 7.4, 11.3
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

import torch

from diffusion_trt.models import (
    OptimizationResult,
    BenchmarkMetrics,
    CacheEntry,
    T4_VRAM_LIMIT_GB,
    MAX_QUANTIZATION_ERROR,
)


# =============================================================================
# OptimizationResult Tests
# =============================================================================


class TestOptimizationResultValidConstruction:
    """Test valid construction of OptimizationResult."""

    @pytest.mark.unit
    def test_valid_optimization_result(self):
        """Test creating a valid OptimizationResult with all fields."""
        result = OptimizationResult(
            model_id="stabilityai/sdxl-turbo",
            original_latency_ms=2000.0,
            optimized_latency_ms=800.0,
            speedup=2.5,
            vram_usage_gb=12.0,
            quantization_error=0.005,
            cache_hit_rate=0.75,
            engine_path="/path/to/engine.trt",
            timestamp=datetime.now(),
            config={"enable_int8": True},
        )
        assert result.model_id == "stabilityai/sdxl-turbo"
        assert result.speedup == 2.5
        assert result.vram_usage_gb == 12.0

    @pytest.mark.unit
    def test_valid_with_none_quantization_error(self):
        """Test OptimizationResult with None quantization_error."""
        result = OptimizationResult(
            model_id="runwayml/stable-diffusion-v1-5",
            original_latency_ms=8000.0,
            optimized_latency_ms=4000.0,
            speedup=2.0,
            vram_usage_gb=10.0,
            quantization_error=None,
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.quantization_error is None
        assert result.engine_path is None

    @pytest.mark.unit
    def test_valid_with_boundary_values(self):
        """Test OptimizationResult with boundary values."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=100.0,
            optimized_latency_ms=100.0,
            speedup=1.0,  # Minimum valid speedup
            vram_usage_gb=T4_VRAM_LIMIT_GB,  # Maximum valid VRAM
            quantization_error=0.0,
            cache_hit_rate=0.0,  # Minimum cache hit rate
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.speedup == 1.0
        assert result.vram_usage_gb == T4_VRAM_LIMIT_GB
        assert result.cache_hit_rate == 0.0

    @pytest.mark.unit
    def test_valid_with_max_cache_hit_rate(self):
        """Test OptimizationResult with maximum cache hit rate."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=1000.0,
            optimized_latency_ms=500.0,
            speedup=2.0,
            vram_usage_gb=8.0,
            quantization_error=0.001,
            cache_hit_rate=1.0,  # Maximum cache hit rate
            engine_path="/engine.trt",
            timestamp=datetime.now(),
            config={"cache_interval": 2},
        )
        assert result.cache_hit_rate == 1.0


class TestOptimizationResultValidation:
    """Test validation errors for invalid OptimizationResult inputs."""

    @pytest.mark.unit
    def test_invalid_speedup_below_one(self):
        """Test that speedup < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="speedup must be >= 1.0"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=1000.0,
                optimized_latency_ms=2000.0,
                speedup=0.5,  # Invalid: slowdown
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=0.5,
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_vram_exceeds_limit(self):
        """Test that vram_usage_gb > 15.6 raises ValueError."""
        with pytest.raises(ValueError, match="vram_usage_gb must be <="):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=1000.0,
                optimized_latency_ms=500.0,
                speedup=2.0,
                vram_usage_gb=16.0,  # Invalid: exceeds T4 limit
                quantization_error=0.005,
                cache_hit_rate=0.5,
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_cache_hit_rate_negative(self):
        """Test that negative cache_hit_rate raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be between"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=1000.0,
                optimized_latency_ms=500.0,
                speedup=2.0,
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=-0.1,  # Invalid: negative
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_cache_hit_rate_above_one(self):
        """Test that cache_hit_rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be between"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=1000.0,
                optimized_latency_ms=500.0,
                speedup=2.0,
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=1.5,  # Invalid: above 1.0
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_original_latency_zero(self):
        """Test that original_latency_ms <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="original_latency_ms must be positive"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=0.0,  # Invalid: zero
                optimized_latency_ms=500.0,
                speedup=2.0,
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=0.5,
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_original_latency_negative(self):
        """Test that negative original_latency_ms raises ValueError."""
        with pytest.raises(ValueError, match="original_latency_ms must be positive"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=-100.0,  # Invalid: negative
                optimized_latency_ms=500.0,
                speedup=2.0,
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=0.5,
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )

    @pytest.mark.unit
    def test_invalid_optimized_latency_zero(self):
        """Test that optimized_latency_ms <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="optimized_latency_ms must be positive"):
            OptimizationResult(
                model_id="test-model",
                original_latency_ms=1000.0,
                optimized_latency_ms=0.0,  # Invalid: zero
                speedup=2.0,
                vram_usage_gb=10.0,
                quantization_error=0.005,
                cache_hit_rate=0.5,
                engine_path=None,
                timestamp=datetime.now(),
                config={},
            )


class TestOptimizationResultProperties:
    """Test computed properties of OptimizationResult."""

    @pytest.mark.unit
    def test_has_acceptable_quality_true(self):
        """Test has_acceptable_quality returns True when error < threshold."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=1000.0,
            optimized_latency_ms=500.0,
            speedup=2.0,
            vram_usage_gb=10.0,
            quantization_error=0.005,  # Below MAX_QUANTIZATION_ERROR (0.01)
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.has_acceptable_quality is True

    @pytest.mark.unit
    def test_has_acceptable_quality_false(self):
        """Test has_acceptable_quality returns False when error >= threshold."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=1000.0,
            optimized_latency_ms=500.0,
            speedup=2.0,
            vram_usage_gb=10.0,
            quantization_error=0.02,  # Above MAX_QUANTIZATION_ERROR (0.01)
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.has_acceptable_quality is False

    @pytest.mark.unit
    def test_has_acceptable_quality_none_error(self):
        """Test has_acceptable_quality returns True when error is None."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=1000.0,
            optimized_latency_ms=500.0,
            speedup=2.0,
            vram_usage_gb=10.0,
            quantization_error=None,
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.has_acceptable_quality is True

    @pytest.mark.unit
    def test_latency_reduction_ms(self):
        """Test latency_reduction_ms calculation."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=2000.0,
            optimized_latency_ms=800.0,
            speedup=2.5,
            vram_usage_gb=10.0,
            quantization_error=0.005,
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.latency_reduction_ms == 1200.0

    @pytest.mark.unit
    def test_latency_reduction_percent(self):
        """Test latency_reduction_percent calculation."""
        result = OptimizationResult(
            model_id="test-model",
            original_latency_ms=2000.0,
            optimized_latency_ms=800.0,
            speedup=2.5,
            vram_usage_gb=10.0,
            quantization_error=0.005,
            cache_hit_rate=0.5,
            engine_path=None,
            timestamp=datetime.now(),
            config={},
        )
        assert result.latency_reduction_percent == 60.0


# =============================================================================
# BenchmarkMetrics Tests
# =============================================================================


class TestBenchmarkMetricsValidConstruction:
    """Test valid construction of BenchmarkMetrics."""

    @pytest.mark.unit
    def test_valid_benchmark_metrics(self):
        """Test creating a valid BenchmarkMetrics with all fields."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=250.0,
            latency_std_ms=15.0,
            latency_p50_ms=245.0,
            latency_p95_ms=280.0,
            latency_p99_ms=300.0,
            throughput_images_per_sec=4.0,
            vram_peak_gb=12.0,
            vram_allocated_gb=10.0,
            cache_hit_rate=0.75,
            num_runs=10,
            warmup_runs=3,
        )
        assert metrics.latency_mean_ms == 250.0
        assert metrics.num_runs == 10

    @pytest.mark.unit
    def test_valid_with_zero_std(self):
        """Test BenchmarkMetrics with zero standard deviation."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=0.0,  # Valid: zero std
            latency_p50_ms=100.0,
            latency_p95_ms=100.0,
            latency_p99_ms=100.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=8.0,
            vram_allocated_gb=6.0,
            cache_hit_rate=0.5,
            num_runs=1,
            warmup_runs=0,
        )
        assert metrics.latency_std_ms == 0.0

    @pytest.mark.unit
    def test_valid_with_equal_percentiles(self):
        """Test BenchmarkMetrics with equal percentiles."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=0.0,
            latency_p50_ms=100.0,
            latency_p95_ms=100.0,
            latency_p99_ms=100.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=8.0,
            vram_allocated_gb=6.0,
            cache_hit_rate=0.5,
            num_runs=1,
            warmup_runs=0,
        )
        assert metrics.latency_p50_ms == metrics.latency_p95_ms == metrics.latency_p99_ms


class TestBenchmarkMetricsValidation:
    """Test validation errors for invalid BenchmarkMetrics inputs."""

    @pytest.mark.unit
    def test_invalid_latency_mean_zero(self):
        """Test that latency_mean_ms <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="latency_mean_ms must be positive"):
            BenchmarkMetrics(
                latency_mean_ms=0.0,  # Invalid: zero
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_latency_mean_negative(self):
        """Test that negative latency_mean_ms raises ValueError."""
        with pytest.raises(ValueError, match="latency_mean_ms must be positive"):
            BenchmarkMetrics(
                latency_mean_ms=-100.0,  # Invalid: negative
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_latency_std_negative(self):
        """Test that negative latency_std_ms raises ValueError."""
        with pytest.raises(ValueError, match="latency_std_ms must be non-negative"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=-5.0,  # Invalid: negative
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_num_runs_zero(self):
        """Test that num_runs < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_runs must be >= 1"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=0,  # Invalid: zero
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_warmup_runs_negative(self):
        """Test that negative warmup_runs raises ValueError."""
        with pytest.raises(ValueError, match="warmup_runs must be non-negative"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=-1,  # Invalid: negative
            )

    @pytest.mark.unit
    def test_invalid_cache_hit_rate_negative(self):
        """Test that negative cache_hit_rate raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be between"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=-0.1,  # Invalid: negative
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_cache_hit_rate_above_one(self):
        """Test that cache_hit_rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="cache_hit_rate must be between"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=1.5,  # Invalid: above 1.0
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_vram_peak_negative(self):
        """Test that negative vram_peak_gb raises ValueError."""
        with pytest.raises(ValueError, match="vram_peak_gb must be non-negative"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=-1.0,  # Invalid: negative
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_vram_allocated_negative(self):
        """Test that negative vram_allocated_gb raises ValueError."""
        with pytest.raises(ValueError, match="vram_allocated_gb must be non-negative"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=150.0,
                latency_p99_ms=200.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=-1.0,  # Invalid: negative
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_percentile_ordering_p50_gt_p95(self):
        """Test that p50 > p95 raises ValueError."""
        with pytest.raises(ValueError, match="Percentiles must be ordered"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=200.0,  # Invalid: p50 > p95
                latency_p95_ms=150.0,
                latency_p99_ms=250.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )

    @pytest.mark.unit
    def test_invalid_percentile_ordering_p95_gt_p99(self):
        """Test that p95 > p99 raises ValueError."""
        with pytest.raises(ValueError, match="Percentiles must be ordered"):
            BenchmarkMetrics(
                latency_mean_ms=100.0,
                latency_std_ms=10.0,
                latency_p50_ms=100.0,
                latency_p95_ms=300.0,  # Invalid: p95 > p99
                latency_p99_ms=250.0,
                throughput_images_per_sec=10.0,
                vram_peak_gb=8.0,
                vram_allocated_gb=6.0,
                cache_hit_rate=0.5,
                num_runs=10,
                warmup_runs=3,
            )


class TestBenchmarkMetricsProperties:
    """Test computed properties of BenchmarkMetrics."""

    @pytest.mark.unit
    def test_latency_cv_calculation(self):
        """Test latency_cv (coefficient of variation) calculation."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=200.0,
            latency_std_ms=20.0,
            latency_p50_ms=195.0,
            latency_p95_ms=230.0,
            latency_p99_ms=250.0,
            throughput_images_per_sec=5.0,
            vram_peak_gb=10.0,
            vram_allocated_gb=8.0,
            cache_hit_rate=0.6,
            num_runs=10,
            warmup_runs=3,
        )
        # CV = std / mean = 20 / 200 = 0.1
        assert metrics.latency_cv == 0.1

    @pytest.mark.unit
    def test_latency_cv_zero_std(self):
        """Test latency_cv with zero standard deviation."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=0.0,
            latency_p50_ms=100.0,
            latency_p95_ms=100.0,
            latency_p99_ms=100.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=8.0,
            vram_allocated_gb=6.0,
            cache_hit_rate=0.5,
            num_runs=1,
            warmup_runs=0,
        )
        assert metrics.latency_cv == 0.0

    @pytest.mark.unit
    def test_expected_throughput_calculation(self):
        """Test expected_throughput calculation from mean latency."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=250.0,
            latency_std_ms=15.0,
            latency_p50_ms=245.0,
            latency_p95_ms=280.0,
            latency_p99_ms=300.0,
            throughput_images_per_sec=4.0,
            vram_peak_gb=12.0,
            vram_allocated_gb=10.0,
            cache_hit_rate=0.75,
            num_runs=10,
            warmup_runs=3,
        )
        # expected_throughput = 1000 / latency_mean_ms = 1000 / 250 = 4.0
        assert metrics.expected_throughput == 4.0

    @pytest.mark.unit
    def test_expected_throughput_fast_inference(self):
        """Test expected_throughput for fast inference."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=5.0,
            latency_p50_ms=98.0,
            latency_p95_ms=110.0,
            latency_p99_ms=120.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=8.0,
            vram_allocated_gb=6.0,
            cache_hit_rate=0.8,
            num_runs=20,
            warmup_runs=5,
        )
        # expected_throughput = 1000 / 100 = 10.0
        assert metrics.expected_throughput == 10.0

    @pytest.mark.unit
    def test_is_vram_compliant_true(self):
        """Test is_vram_compliant returns True when within limit."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=5.0,
            latency_p50_ms=98.0,
            latency_p95_ms=110.0,
            latency_p99_ms=120.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=12.0,  # Below T4 limit
            vram_allocated_gb=10.0,
            cache_hit_rate=0.5,
            num_runs=10,
            warmup_runs=3,
        )
        assert metrics.is_vram_compliant is True

    @pytest.mark.unit
    def test_is_vram_compliant_at_limit(self):
        """Test is_vram_compliant returns True at exact limit."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=5.0,
            latency_p50_ms=98.0,
            latency_p95_ms=110.0,
            latency_p99_ms=120.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=T4_VRAM_LIMIT_GB,  # At T4 limit
            vram_allocated_gb=10.0,
            cache_hit_rate=0.5,
            num_runs=10,
            warmup_runs=3,
        )
        assert metrics.is_vram_compliant is True

    @pytest.mark.unit
    def test_is_vram_compliant_false(self):
        """Test is_vram_compliant returns False when exceeding limit."""
        metrics = BenchmarkMetrics(
            latency_mean_ms=100.0,
            latency_std_ms=5.0,
            latency_p50_ms=98.0,
            latency_p95_ms=110.0,
            latency_p99_ms=120.0,
            throughput_images_per_sec=10.0,
            vram_peak_gb=16.0,  # Above T4 limit
            vram_allocated_gb=14.0,
            cache_hit_rate=0.5,
            num_runs=10,
            warmup_runs=3,
        )
        assert metrics.is_vram_compliant is False


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntryValidConstruction:
    """Test valid construction of CacheEntry."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_valid_cache_entry(self):
        """Test creating a valid CacheEntry with CUDA tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        entry = CacheEntry(
            timestep=5,
            block_idx=2,
            features=features,
            created_at=10,
        )
        assert entry.timestep == 5
        assert entry.block_idx == 2
        assert entry.access_count == 0
        assert entry.size_bytes > 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_valid_with_initial_access_count(self):
        """Test CacheEntry with non-zero initial access_count."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 640, 32, 32, device="cuda")
        entry = CacheEntry(
            timestep=10,
            block_idx=5,
            features=features,
            created_at=20,
            access_count=3,
        )
        assert entry.access_count == 3

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_size_bytes_calculation(self):
        """Test that size_bytes is correctly calculated."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a tensor with known size: 1 * 320 * 64 * 64 = 1,310,720 elements
        # float32 = 4 bytes per element
        features = torch.randn(1, 320, 64, 64, device="cuda", dtype=torch.float32)
        entry = CacheEntry(
            timestep=0,
            block_idx=0,
            features=features,
            created_at=0,
        )
        expected_size = 1 * 320 * 64 * 64 * 4  # 5,242,880 bytes
        assert entry.size_bytes == expected_size

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_size_bytes_fp16(self):
        """Test size_bytes calculation for FP16 tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # FP16 = 2 bytes per element
        features = torch.randn(1, 320, 64, 64, device="cuda", dtype=torch.float16)
        entry = CacheEntry(
            timestep=0,
            block_idx=0,
            features=features,
            created_at=0,
        )
        expected_size = 1 * 320 * 64 * 64 * 2  # 2,621,440 bytes
        assert entry.size_bytes == expected_size


class TestCacheEntryValidation:
    """Test validation errors for invalid CacheEntry inputs."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_invalid_timestep_negative(self):
        """Test that negative timestep raises ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="timestep must be non-negative"):
            CacheEntry(
                timestep=-1,  # Invalid: negative
                block_idx=0,
                features=features,
                created_at=0,
            )

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_invalid_block_idx_negative(self):
        """Test that negative block_idx raises ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="block_idx must be non-negative"):
            CacheEntry(
                timestep=0,
                block_idx=-1,  # Invalid: negative
                features=features,
                created_at=0,
            )

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_invalid_created_at_negative(self):
        """Test that negative created_at raises ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="created_at must be non-negative"):
            CacheEntry(
                timestep=0,
                block_idx=0,
                features=features,
                created_at=-1,  # Invalid: negative
            )

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_invalid_access_count_negative(self):
        """Test that negative access_count raises ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="access_count must be non-negative"):
            CacheEntry(
                timestep=0,
                block_idx=0,
                features=features,
                created_at=0,
                access_count=-1,  # Invalid: negative
            )

    @pytest.mark.unit
    def test_invalid_features_not_cuda(self):
        """Test that CPU tensor raises ValueError."""
        features = torch.randn(1, 320, 64, 64, device="cpu")  # CPU tensor
        with pytest.raises(ValueError, match="features tensor must be on CUDA device"):
            CacheEntry(
                timestep=0,
                block_idx=0,
                features=features,
                created_at=0,
            )


class TestCacheEntryMethods:
    """Test methods of CacheEntry."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_record_access_increments_count(self):
        """Test that record_access() increments access_count."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        entry = CacheEntry(
            timestep=5,
            block_idx=2,
            features=features,
            created_at=10,
        )
        assert entry.access_count == 0
        
        entry.record_access()
        assert entry.access_count == 1
        
        entry.record_access()
        assert entry.access_count == 2
        
        entry.record_access()
        assert entry.access_count == 3

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_record_access_multiple_times(self):
        """Test record_access() called many times."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(1, 320, 64, 64, device="cuda")
        entry = CacheEntry(
            timestep=0,
            block_idx=0,
            features=features,
            created_at=0,
        )
        
        for i in range(100):
            entry.record_access()
        
        assert entry.access_count == 100


class TestCacheEntryProperties:
    """Test computed properties of CacheEntry."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_size_mb_calculation(self):
        """Test size_mb property calculation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # 1 * 320 * 64 * 64 * 4 = 5,242,880 bytes = 5.0 MB
        features = torch.randn(1, 320, 64, 64, device="cuda", dtype=torch.float32)
        entry = CacheEntry(
            timestep=0,
            block_idx=0,
            features=features,
            created_at=0,
        )
        expected_mb = 5242880 / (1024 * 1024)  # 5.0 MB
        assert entry.size_mb == expected_mb

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_size_gb_calculation(self):
        """Test size_gb property calculation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # 1 * 320 * 64 * 64 * 4 = 5,242,880 bytes
        features = torch.randn(1, 320, 64, 64, device="cuda", dtype=torch.float32)
        entry = CacheEntry(
            timestep=0,
            block_idx=0,
            features=features,
            created_at=0,
        )
        expected_gb = 5242880 / (1024 * 1024 * 1024)
        assert abs(entry.size_gb - expected_gb) < 1e-9

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_size_properties_consistency(self):
        """Test that size_mb and size_gb are consistent with size_bytes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        features = torch.randn(2, 640, 32, 32, device="cuda", dtype=torch.float32)
        entry = CacheEntry(
            timestep=3,
            block_idx=4,
            features=features,
            created_at=5,
        )
        
        # Verify consistency
        assert entry.size_mb == entry.size_bytes / (1024 * 1024)
        assert entry.size_gb == entry.size_bytes / (1024 * 1024 * 1024)
        assert entry.size_gb == entry.size_mb / 1024
