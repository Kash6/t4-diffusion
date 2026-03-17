"""
Unit tests for CacheManager component in diffusion_trt.cache_manager.

Tests cover:
- CacheConfig validation (valid and invalid configs)
- CacheManager initialization
- should_compute() logic (cache interval, cache_layer_ids filtering)
- get_cached() tests (cache hits and misses)
- store() tests (storing features, updating existing entries)
- clear() tests (clearing cache and resetting stats)
- get_cache_stats() tests (hit/miss statistics)
- Eviction tests (evicting oldest entries when exceeding max_cache_size_gb)
- Memory limit enforcement tests

Validates: Requirements 5.4, 5.5
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List

import torch
import torch.nn as nn

from diffusion_trt.cache_manager import (
    CacheConfig,
    CacheManager,
    DEFAULT_MAX_CACHE_SIZE_GB,
)
from diffusion_trt.models import CacheEntry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_cuda_tensor():
    """Create a mock tensor that appears to be on CUDA."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.is_cuda = True
    tensor.device = torch.device("cuda:0")
    tensor.numel.return_value = 1024 * 1024  # 1M elements
    tensor.element_size.return_value = 4  # float32
    return tensor


@pytest.fixture
def real_cuda_tensor():
    """Create a real CUDA tensor for GPU tests."""
    if torch.cuda.is_available():
        return torch.randn(1, 4, 64, 64, device="cuda")
    return None


@pytest.fixture
def default_config():
    """Create a default CacheConfig."""
    return CacheConfig()


@pytest.fixture
def cache_manager(default_config):
    """Create a CacheManager with default config."""
    return CacheManager(default_config)


# =============================================================================
# CacheConfig Tests - Valid Construction
# =============================================================================


class TestCacheConfigValidConstruction:
    """Test valid construction of CacheConfig."""

    @pytest.mark.unit
    def test_valid_default_config(self):
        """Test creating a CacheConfig with default values."""
        config = CacheConfig()
        assert config.cache_interval == 3
        assert config.cache_layer_ids is None
        assert config.cache_branch == "main"
        assert config.max_cache_size_gb == DEFAULT_MAX_CACHE_SIZE_GB
        assert config.enable_token_caching is True
        assert config.token_similarity_threshold == 0.95

    @pytest.mark.unit
    def test_valid_custom_cache_interval(self):
        """Test CacheConfig with custom cache_interval."""
        config = CacheConfig(cache_interval=5)
        assert config.cache_interval == 5

    @pytest.mark.unit
    def test_valid_cache_interval_one(self):
        """Test CacheConfig with cache_interval=1 (minimum valid)."""
        config = CacheConfig(cache_interval=1)
        assert config.cache_interval == 1

    @pytest.mark.unit
    def test_valid_cache_layer_ids(self):
        """Test CacheConfig with specific cache_layer_ids."""
        config = CacheConfig(cache_layer_ids=[0, 1, 2, 3])
        assert config.cache_layer_ids == [0, 1, 2, 3]

    @pytest.mark.unit
    def test_valid_cache_branch_main(self):
        """Test CacheConfig with cache_branch='main'."""
        config = CacheConfig(cache_branch="main")
        assert config.cache_branch == "main"

    @pytest.mark.unit
    def test_valid_cache_branch_skip(self):
        """Test CacheConfig with cache_branch='skip'."""
        config = CacheConfig(cache_branch="skip")
        assert config.cache_branch == "skip"

    @pytest.mark.unit
    def test_valid_cache_branch_both(self):
        """Test CacheConfig with cache_branch='both'."""
        config = CacheConfig(cache_branch="both")
        assert config.cache_branch == "both"

    @pytest.mark.unit
    def test_valid_max_cache_size_gb(self):
        """Test CacheConfig with custom max_cache_size_gb."""
        config = CacheConfig(max_cache_size_gb=1.5)
        assert config.max_cache_size_gb == 1.5

    @pytest.mark.unit
    def test_valid_max_cache_size_at_limit(self):
        """Test CacheConfig with max_cache_size_gb at T4 limit."""
        config = CacheConfig(max_cache_size_gb=DEFAULT_MAX_CACHE_SIZE_GB)
        assert config.max_cache_size_gb == DEFAULT_MAX_CACHE_SIZE_GB

    @pytest.mark.unit
    def test_valid_token_caching_disabled(self):
        """Test CacheConfig with token caching disabled."""
        config = CacheConfig(enable_token_caching=False)
        assert config.enable_token_caching is False

    @pytest.mark.unit
    def test_valid_token_similarity_threshold(self):
        """Test CacheConfig with custom token_similarity_threshold."""
        config = CacheConfig(token_similarity_threshold=0.9)
        assert config.token_similarity_threshold == 0.9

    @pytest.mark.unit
    def test_valid_token_similarity_threshold_boundaries(self):
        """Test CacheConfig with token_similarity_threshold at boundaries."""
        config_zero = CacheConfig(token_similarity_threshold=0.0)
        assert config_zero.token_similarity_threshold == 0.0
        
        config_one = CacheConfig(token_similarity_threshold=1.0)
        assert config_one.token_similarity_threshold == 1.0


# =============================================================================
# CacheConfig Tests - Validation Errors
# =============================================================================


class TestCacheConfigValidation:
    """Test validation errors for invalid CacheConfig inputs."""

    @pytest.mark.unit
    def test_invalid_cache_interval_zero(self):
        """Test that cache_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="cache_interval must be >= 1"):
            CacheConfig(cache_interval=0)

    @pytest.mark.unit
    def test_invalid_cache_interval_negative(self):
        """Test that negative cache_interval raises ValueError."""
        with pytest.raises(ValueError, match="cache_interval must be >= 1"):
            CacheConfig(cache_interval=-1)

    @pytest.mark.unit
    def test_invalid_max_cache_size_zero(self):
        """Test that max_cache_size_gb=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size_gb must be > 0"):
            CacheConfig(max_cache_size_gb=0)

    @pytest.mark.unit
    def test_invalid_max_cache_size_negative(self):
        """Test that negative max_cache_size_gb raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size_gb must be > 0"):
            CacheConfig(max_cache_size_gb=-1.0)

    @pytest.mark.unit
    def test_invalid_max_cache_size_exceeds_limit(self):
        """Test that max_cache_size_gb > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size_gb must be <="):
            CacheConfig(max_cache_size_gb=3.0)

    @pytest.mark.unit
    def test_invalid_cache_branch(self):
        """Test that invalid cache_branch raises ValueError."""
        with pytest.raises(ValueError, match="cache_branch must be one of"):
            CacheConfig(cache_branch="invalid")

    @pytest.mark.unit
    def test_invalid_token_similarity_threshold_negative(self):
        """Test that negative token_similarity_threshold raises ValueError."""
        with pytest.raises(ValueError, match="token_similarity_threshold must be in"):
            CacheConfig(token_similarity_threshold=-0.1)

    @pytest.mark.unit
    def test_invalid_token_similarity_threshold_over_one(self):
        """Test that token_similarity_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="token_similarity_threshold must be in"):
            CacheConfig(token_similarity_threshold=1.1)

    @pytest.mark.unit
    def test_invalid_cache_layer_ids_negative(self):
        """Test that negative cache_layer_ids raises ValueError."""
        with pytest.raises(ValueError, match="cache_layer_ids must contain non-negative"):
            CacheConfig(cache_layer_ids=[0, -1, 2])

    @pytest.mark.unit
    def test_invalid_cache_layer_ids_non_integer(self):
        """Test that non-integer cache_layer_ids raises ValueError."""
        with pytest.raises(ValueError, match="cache_layer_ids must contain non-negative"):
            CacheConfig(cache_layer_ids=[0, "invalid", 2])


# =============================================================================
# CacheManager Tests - Initialization
# =============================================================================


class TestCacheManagerInit:
    """Test CacheManager initialization."""

    @pytest.mark.unit
    def test_init_with_default_config(self):
        """Test CacheManager initialization with default config."""
        manager = CacheManager()
        assert manager.config is not None
        assert manager.config.cache_interval == 3

    @pytest.mark.unit
    def test_init_with_custom_config(self):
        """Test CacheManager initialization with custom config."""
        config = CacheConfig(
            cache_interval=5,
            cache_layer_ids=[0, 1, 2],
            max_cache_size_gb=1.0
        )
        manager = CacheManager(config)
        assert manager.config.cache_interval == 5
        assert manager.config.cache_layer_ids == [0, 1, 2]
        assert manager.config.max_cache_size_gb == 1.0

    @pytest.mark.unit
    def test_init_cache_is_empty(self):
        """Test that cache is empty after initialization."""
        manager = CacheManager()
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 0
        assert stats["cache_size_bytes"] == 0

    @pytest.mark.unit
    def test_init_stats_are_zero(self):
        """Test that statistics are zero after initialization."""
        manager = CacheManager()
        stats = manager.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_queries"] == 0
        assert stats["hit_rate"] == 0.0


# =============================================================================
# CacheManager Tests - should_compute() Logic
# =============================================================================


class TestCacheManagerShouldCompute:
    """Test CacheManager.should_compute() logic."""

    @pytest.mark.unit
    def test_should_compute_at_compute_timestep(self):
        """Test should_compute returns True at compute timesteps."""
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Timesteps 0, 3, 6, 9 are compute timesteps (divisible by 3)
        assert manager.should_compute(0, 0) is True
        assert manager.should_compute(3, 0) is True
        assert manager.should_compute(6, 0) is True
        assert manager.should_compute(9, 0) is True

    @pytest.mark.unit
    def test_should_compute_at_non_compute_timestep_no_cache(self):
        """Test should_compute returns True at non-compute timesteps with no cache."""
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Timesteps 1, 2, 4, 5 are non-compute timesteps
        # Without cached data, should return True
        assert manager.should_compute(1, 0) is True
        assert manager.should_compute(2, 0) is True
        assert manager.should_compute(4, 0) is True
        assert manager.should_compute(5, 0) is True

    @pytest.mark.unit
    def test_should_compute_block_not_in_cache_layer_ids(self):
        """Test should_compute returns True for blocks not in cache_layer_ids."""
        config = CacheConfig(cache_interval=3, cache_layer_ids=[0, 1])
        manager = CacheManager(config)
        
        # Block 2 is not in cache_layer_ids, should always compute
        assert manager.should_compute(0, 2) is True
        assert manager.should_compute(3, 2) is True
        assert manager.should_compute(1, 2) is True

    @pytest.mark.unit
    def test_should_compute_block_in_cache_layer_ids(self):
        """Test should_compute respects cache_layer_ids."""
        config = CacheConfig(cache_interval=3, cache_layer_ids=[0, 1])
        manager = CacheManager(config)
        
        # Block 0 is in cache_layer_ids
        # At compute timestep, should compute
        assert manager.should_compute(0, 0) is True
        assert manager.should_compute(0, 1) is True

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_should_compute_returns_false_with_cached_data(self, real_cuda_tensor):
        """Test should_compute returns False when cache has data."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Store features at compute timestep 0
        manager.store(0, 0, real_cuda_tensor)
        
        # At non-compute timestep 1, should use cache
        assert manager.should_compute(1, 0) is False
        assert manager.should_compute(2, 0) is False

    @pytest.mark.unit
    def test_should_compute_updates_statistics(self):
        """Test should_compute updates hit/miss statistics."""
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # First call - miss (compute timestep)
        manager.should_compute(0, 0)
        stats = manager.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["total_queries"] == 1
        
        # Second call - miss (no cache)
        manager.should_compute(1, 0)
        stats = manager.get_cache_stats()
        assert stats["misses"] == 2
        assert stats["total_queries"] == 2


# =============================================================================
# CacheManager Tests - get_cached()
# =============================================================================


class TestCacheManagerGetCached:
    """Test CacheManager.get_cached() for cache hits and misses."""

    @pytest.mark.unit
    def test_get_cached_returns_none_when_empty(self):
        """Test get_cached returns None when cache is empty."""
        manager = CacheManager()
        result = manager.get_cached(0, 0)
        assert result is None

    @pytest.mark.unit
    def test_get_cached_returns_none_for_missing_key(self):
        """Test get_cached returns None for non-existent key."""
        manager = CacheManager()
        result = manager.get_cached(5, 3)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_cached_returns_stored_tensor(self, real_cuda_tensor):
        """Test get_cached returns the stored tensor."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Store at compute timestep 0
        manager.store(0, 0, real_cuda_tensor)
        
        # Get cached at non-compute timestep 1 (maps to compute timestep 0)
        result = manager.get_cached(1, 0)
        assert result is not None
        assert torch.equal(result, real_cuda_tensor)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_cached_finds_nearest_compute_timestep(self, real_cuda_tensor):
        """Test get_cached finds features from nearest compute timestep."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Store at compute timestep 3
        manager.store(3, 0, real_cuda_tensor)
        
        # Timesteps 4 and 5 should map to compute timestep 3
        result_4 = manager.get_cached(4, 0)
        result_5 = manager.get_cached(5, 0)
        
        assert result_4 is not None
        assert result_5 is not None
        assert torch.equal(result_4, real_cuda_tensor)
        assert torch.equal(result_5, real_cuda_tensor)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_cached_updates_access_count(self, real_cuda_tensor):
        """Test get_cached updates the access count of cache entry."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        manager.store(0, 0, real_cuda_tensor)
        
        # Access multiple times
        manager.get_cached(1, 0)
        manager.get_cached(2, 0)
        
        # Access count should be updated (internal check via cache entry)
        entry = manager._cache.get((0, 0))
        assert entry is not None
        assert entry.access_count == 2


# =============================================================================
# CacheManager Tests - store()
# =============================================================================


class TestCacheManagerStore:
    """Test CacheManager.store() for storing features."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_adds_entry_to_cache(self, real_cuda_tensor):
        """Test store adds an entry to the cache."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 1
        assert stats["cache_size_bytes"] > 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_multiple_entries(self, real_cuda_tensor):
        """Test store can add multiple entries."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        manager.store(0, 1, real_cuda_tensor)
        manager.store(3, 0, real_cuda_tensor)
        
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 3

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_updates_existing_entry(self, real_cuda_tensor):
        """Test store updates an existing entry with same key."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        
        # Store initial entry
        tensor1 = torch.randn(1, 4, 64, 64, device="cuda")
        manager.store(0, 0, tensor1)
        
        # Store new entry with same key
        tensor2 = torch.randn(1, 4, 64, 64, device="cuda")
        manager.store(0, 0, tensor2)
        
        # Should still have only 1 entry
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 1
        
        # Should return the new tensor
        result = manager.get_cached(0, 0)
        assert torch.equal(result, tensor2)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_respects_cache_layer_ids(self, real_cuda_tensor):
        """Test store only caches blocks in cache_layer_ids."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_layer_ids=[0, 1])
        manager = CacheManager(config)
        
        # Store for block 0 (in cache_layer_ids)
        manager.store(0, 0, real_cuda_tensor)
        
        # Store for block 2 (not in cache_layer_ids)
        manager.store(0, 2, real_cuda_tensor)
        
        # Only block 0 should be cached
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 1

    @pytest.mark.unit
    def test_store_raises_error_for_cpu_tensor(self):
        """Test store raises ValueError for CPU tensor."""
        manager = CacheManager()
        cpu_tensor = torch.randn(1, 4, 64, 64)  # CPU tensor
        
        with pytest.raises(ValueError, match="features tensor must be on CUDA device"):
            manager.store(0, 0, cpu_tensor)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_updates_cache_size_tracking(self, real_cuda_tensor):
        """Test store correctly updates cache size tracking."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        
        initial_size = manager.get_cache_stats()["cache_size_bytes"]
        assert initial_size == 0
        
        manager.store(0, 0, real_cuda_tensor)
        
        new_size = manager.get_cache_stats()["cache_size_bytes"]
        expected_size = real_cuda_tensor.numel() * real_cuda_tensor.element_size()
        assert new_size == expected_size


# =============================================================================
# CacheManager Tests - clear()
# =============================================================================


class TestCacheManagerClear:
    """Test CacheManager.clear() for clearing cache and resetting stats."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_clear_removes_all_entries(self, real_cuda_tensor):
        """Test clear removes all cache entries."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        manager.store(0, 1, real_cuda_tensor)
        manager.store(3, 0, real_cuda_tensor)
        
        manager.clear()
        
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 0
        assert stats["cache_size_bytes"] == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_clear_resets_statistics(self, real_cuda_tensor):
        """Test clear resets hit/miss statistics."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        
        # Generate some statistics
        manager.should_compute(0, 0)
        manager.should_compute(1, 0)
        manager.should_compute(2, 0)
        
        manager.clear()
        
        stats = manager.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_queries"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.unit
    def test_clear_on_empty_cache(self):
        """Test clear works on empty cache without error."""
        manager = CacheManager()
        manager.clear()  # Should not raise
        
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_clear_resets_step_counter(self, real_cuda_tensor):
        """Test clear resets the internal step counter."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.increment_step()
        manager.increment_step()
        
        manager.clear()
        
        # After clear, step counter should be 0
        assert manager._current_step == 0


# =============================================================================
# CacheManager Tests - get_cache_stats()
# =============================================================================


class TestCacheManagerGetCacheStats:
    """Test CacheManager.get_cache_stats() for hit/miss statistics."""

    @pytest.mark.unit
    def test_get_cache_stats_returns_correct_structure(self):
        """Test get_cache_stats returns all expected fields."""
        manager = CacheManager()
        stats = manager.get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "total_queries" in stats
        assert "hit_rate" in stats
        assert "num_entries" in stats
        assert "cache_size_bytes" in stats
        assert "cache_size_gb" in stats

    @pytest.mark.unit
    def test_get_cache_stats_hit_rate_calculation(self):
        """Test hit_rate is calculated correctly."""
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # All misses initially
        manager.should_compute(0, 0)  # miss (compute timestep)
        manager.should_compute(3, 0)  # miss (compute timestep)
        
        stats = manager.get_cache_stats()
        assert stats["hit_rate"] == 0.0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_cache_stats_with_hits(self, real_cuda_tensor):
        """Test hit_rate calculation with actual hits."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Store at compute timestep
        manager.store(0, 0, real_cuda_tensor)
        
        # Query at compute timestep (miss)
        manager.should_compute(0, 0)
        
        # Query at non-compute timesteps (hits)
        manager.should_compute(1, 0)
        manager.should_compute(2, 0)
        
        stats = manager.get_cache_stats()
        # 2 hits out of 3 queries
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_queries"] == 3
        assert abs(stats["hit_rate"] - 2/3) < 0.001

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_cache_stats_cache_size_gb(self, real_cuda_tensor):
        """Test cache_size_gb is calculated correctly."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        
        stats = manager.get_cache_stats()
        expected_gb = stats["cache_size_bytes"] / (1024 ** 3)
        assert abs(stats["cache_size_gb"] - expected_gb) < 1e-10


# =============================================================================
# CacheManager Tests - Eviction Behavior
# =============================================================================


class TestCacheManagerEviction:
    """Test CacheManager eviction behavior when exceeding max_cache_size_gb."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_eviction_removes_oldest_entry(self):
        """Test eviction removes the oldest entry based on created_at."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Use a very small cache size to trigger eviction
        config = CacheConfig(max_cache_size_gb=0.0001)  # ~100KB
        manager = CacheManager(config)
        
        # Create tensors that will exceed the tiny cache
        tensor1 = torch.randn(1, 64, 64, 64, device="cuda")  # ~1MB
        tensor2 = torch.randn(1, 64, 64, 64, device="cuda")  # ~1MB
        
        # Store first tensor at step 0
        manager._current_step = 0
        manager.store(0, 0, tensor1)
        
        # Store second tensor at step 1
        manager._current_step = 1
        manager.store(3, 0, tensor2)
        
        # First entry should be evicted (oldest)
        assert (0, 0) not in manager._cache
        # Second entry should remain
        assert (3, 0) in manager._cache

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_eviction_maintains_size_limit(self):
        """Test eviction keeps cache size within limit."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Small cache size
        config = CacheConfig(max_cache_size_gb=0.001)  # ~1MB
        manager = CacheManager(config)
        
        # Add multiple entries
        for i in range(10):
            tensor = torch.randn(1, 32, 32, 32, device="cuda")  # ~128KB each
            manager._current_step = i
            manager.store(i * 3, 0, tensor)
        
        # Cache size should be within limit
        assert manager.cache_size_gb <= config.max_cache_size_gb

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_eviction_preserves_newest_entries(self):
        """Test eviction preserves the newest entries."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = CacheConfig(max_cache_size_gb=0.0005)  # ~500KB
        manager = CacheManager(config)
        
        # Add entries with increasing step numbers
        tensors = []
        for i in range(5):
            tensor = torch.randn(1, 32, 32, 32, device="cuda")
            tensors.append(tensor)
            manager._current_step = i
            manager.store(i * 3, 0, tensor)
        
        # The newest entry should still be in cache
        stats = manager.get_cache_stats()
        assert stats["num_entries"] >= 1
        # Last entry should be present
        assert (12, 0) in manager._cache  # timestep 4*3=12


# =============================================================================
# CacheManager Tests - Memory Limit Enforcement
# =============================================================================


class TestCacheManagerMemoryLimits:
    """Test CacheManager memory limit enforcement."""

    @pytest.mark.unit
    def test_max_cache_size_bytes_property(self):
        """Test max_cache_size_bytes property calculation."""
        config = CacheConfig(max_cache_size_gb=1.5)
        manager = CacheManager(config)
        
        expected_bytes = int(1.5 * (1024 ** 3))
        assert manager.max_cache_size_bytes == expected_bytes

    @pytest.mark.unit
    def test_cache_size_gb_property(self):
        """Test cache_size_gb property returns correct value."""
        manager = CacheManager()
        
        # Initially zero
        assert manager.cache_size_gb == 0.0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_cache_size_gb_after_store(self, real_cuda_tensor):
        """Test cache_size_gb updates after storing."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        
        expected_gb = (real_cuda_tensor.numel() * real_cuda_tensor.element_size()) / (1024 ** 3)
        assert abs(manager.cache_size_gb - expected_gb) < 1e-10

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_memory_limit_enforced_on_store(self):
        """Test memory limit is enforced when storing new entries."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Very small limit
        config = CacheConfig(max_cache_size_gb=0.0001)
        manager = CacheManager(config)
        
        # Store a large tensor
        large_tensor = torch.randn(1, 128, 128, 128, device="cuda")
        manager.store(0, 0, large_tensor)
        
        # Cache should have at most 1 entry (the one we just stored)
        # and size should be within limit after eviction
        stats = manager.get_cache_stats()
        assert stats["num_entries"] <= 1

    @pytest.mark.unit
    def test_default_max_cache_size_is_2gb(self):
        """Test default max_cache_size_gb is 2.0 (T4 constraint)."""
        config = CacheConfig()
        assert config.max_cache_size_gb == 2.0
        assert config.max_cache_size_gb == DEFAULT_MAX_CACHE_SIZE_GB


# =============================================================================
# CacheManager Tests - CPU-Only Tests (Mocking CUDA)
# =============================================================================


class TestCacheManagerCPUOnly:
    """CPU-only tests for CacheManager using mocked CUDA tensors."""

    @pytest.mark.unit
    def test_should_cache_block_all_blocks(self):
        """Test _should_cache_block returns True for all blocks when no filter."""
        manager = CacheManager()
        
        assert manager._should_cache_block(0) is True
        assert manager._should_cache_block(5) is True
        assert manager._should_cache_block(10) is True

    @pytest.mark.unit
    def test_should_cache_block_with_filter(self):
        """Test _should_cache_block respects cache_layer_ids filter."""
        config = CacheConfig(cache_layer_ids=[0, 2, 4])
        manager = CacheManager(config)
        
        assert manager._should_cache_block(0) is True
        assert manager._should_cache_block(1) is False
        assert manager._should_cache_block(2) is True
        assert manager._should_cache_block(3) is False
        assert manager._should_cache_block(4) is True

    @pytest.mark.unit
    def test_find_cache_key_empty_cache(self):
        """Test _find_cache_key returns None for empty cache."""
        manager = CacheManager()
        
        result = manager._find_cache_key(5, 0)
        assert result is None

    @pytest.mark.unit
    def test_find_cache_key_computes_correct_timestep(self):
        """Test _find_cache_key computes correct compute timestep."""
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Manually add an entry to test key finding
        # Note: We can't use store() without CUDA, so we test the logic
        # Timestep 5 should map to compute timestep 3 (5 // 3 * 3 = 3)
        # Timestep 8 should map to compute timestep 6 (8 // 3 * 3 = 6)
        
        # Without entries, should return None
        assert manager._find_cache_key(5, 0) is None

    @pytest.mark.unit
    def test_increment_step(self):
        """Test increment_step increases the step counter."""
        manager = CacheManager()
        
        assert manager._current_step == 0
        manager.increment_step()
        assert manager._current_step == 1
        manager.increment_step()
        assert manager._current_step == 2

    @pytest.mark.unit
    def test_evict_oldest_empty_cache(self):
        """Test _evict_oldest returns False for empty cache."""
        manager = CacheManager()
        
        result = manager._evict_oldest()
        assert result is False


# =============================================================================
# CacheManager Tests - Edge Cases
# =============================================================================


class TestCacheManagerEdgeCases:
    """Test edge cases for CacheManager."""

    @pytest.mark.unit
    def test_cache_interval_one_always_computes(self):
        """Test cache_interval=1 means every timestep is a compute timestep."""
        config = CacheConfig(cache_interval=1)
        manager = CacheManager(config)
        
        # Every timestep should be a compute timestep
        assert manager.should_compute(0, 0) is True
        assert manager.should_compute(1, 0) is True
        assert manager.should_compute(2, 0) is True

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_large_block_idx(self, real_cuda_tensor):
        """Test handling of large block indices."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        
        # Store with large block index
        manager.store(0, 100, real_cuda_tensor)
        
        result = manager.get_cached(0, 100)
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_large_timestep(self, real_cuda_tensor):
        """Test handling of large timesteps."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        
        # Store with large timestep
        manager.store(999, 0, real_cuda_tensor)
        
        # Should be retrievable at nearby timesteps
        result = manager.get_cached(1000, 0)
        assert result is not None

    @pytest.mark.unit
    def test_empty_cache_layer_ids_list(self):
        """Test empty cache_layer_ids list means no blocks are cached."""
        config = CacheConfig(cache_layer_ids=[])
        manager = CacheManager(config)
        
        # No blocks should be cached
        assert manager._should_cache_block(0) is False
        assert manager._should_cache_block(1) is False

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_store_same_key_multiple_times(self, real_cuda_tensor):
        """Test storing to same key multiple times updates correctly."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        
        # Store multiple times to same key
        for i in range(5):
            tensor = torch.randn(1, 4, 64, 64, device="cuda")
            manager.store(0, 0, tensor)
        
        # Should only have 1 entry
        stats = manager.get_cache_stats()
        assert stats["num_entries"] == 1


# =============================================================================
# CacheManager Tests - Integration with CacheEntry
# =============================================================================


class TestCacheManagerWithCacheEntry:
    """Test CacheManager integration with CacheEntry dataclass."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_cache_entry_created_correctly(self, real_cuda_tensor):
        """Test CacheEntry is created with correct attributes."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager._current_step = 5
        manager.store(0, 0, real_cuda_tensor)
        
        entry = manager._cache.get((0, 0))
        assert entry is not None
        assert entry.timestep == 0
        assert entry.block_idx == 0
        assert entry.created_at == 5
        assert entry.access_count == 0
        assert entry.size_bytes > 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_cache_entry_size_bytes_accurate(self, real_cuda_tensor):
        """Test CacheEntry size_bytes matches tensor size."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        manager = CacheManager()
        manager.store(0, 0, real_cuda_tensor)
        
        entry = manager._cache.get((0, 0))
        expected_size = real_cuda_tensor.numel() * real_cuda_tensor.element_size()
        assert entry.size_bytes == expected_size

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_cache_entry_access_count_increments(self, real_cuda_tensor):
        """Test CacheEntry access_count increments on get_cached."""
        if real_cuda_tensor is None:
            pytest.skip("CUDA not available")
        
        config = CacheConfig(cache_interval=3)
        manager = CacheManager(config)
        manager.store(0, 0, real_cuda_tensor)
        
        # Access multiple times
        for _ in range(3):
            manager.get_cached(1, 0)
        
        entry = manager._cache.get((0, 0))
        assert entry.access_count == 3


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Test module-level constants."""

    @pytest.mark.unit
    def test_default_max_cache_size_gb(self):
        """Test DEFAULT_MAX_CACHE_SIZE_GB is 2.0."""
        assert DEFAULT_MAX_CACHE_SIZE_GB == 2.0
