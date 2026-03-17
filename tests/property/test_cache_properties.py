"""
Property-based tests for CacheManager component.

Property 4: Cache Consistency
- Verify cached features produce same output as computed features (rtol=1e-3)

Validates: Requirements 5.2, 5.3

Note: These tests require GPU for CUDA tensor operations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple

import torch

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
def valid_cache_config(draw):
    """Generate valid CacheConfig instances."""
    cache_interval = draw(st.integers(min_value=1, max_value=10))
    max_cache_size_gb = draw(st.floats(min_value=0.1, max_value=2.0))
    cache_branch = draw(st.sampled_from(["main", "skip", "both"]))
    token_similarity_threshold = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Optionally include cache_layer_ids
    include_layer_ids = draw(st.booleans())
    if include_layer_ids:
        cache_layer_ids = draw(
            st.lists(
                st.integers(min_value=0, max_value=11),
                min_size=1,
                max_size=6,
                unique=True
            )
        )
    else:
        cache_layer_ids = None
    
    return CacheConfig(
        cache_interval=cache_interval,
        cache_layer_ids=cache_layer_ids,
        cache_branch=cache_branch,
        max_cache_size_gb=max_cache_size_gb,
        enable_token_caching=draw(st.booleans()),
        token_similarity_threshold=token_similarity_threshold,
    )


@st.composite
def feature_tensor_params(draw):
    """Generate parameters for creating feature tensors."""
    batch_size = draw(st.integers(min_value=1, max_value=2))
    channels = draw(st.sampled_from([320, 640, 1280]))
    height = draw(st.sampled_from([8, 16, 32, 64]))
    width = draw(st.sampled_from([8, 16, 32, 64]))
    
    return (batch_size, channels, height, width)


@st.composite
def cache_key_params(draw):
    """Generate valid (timestep, block_idx) cache key parameters."""
    timestep = draw(st.integers(min_value=0, max_value=999))
    block_idx = draw(st.integers(min_value=0, max_value=11))
    
    return (timestep, block_idx)


# =============================================================================
# Property Tests - Cache Consistency (Property 4)
# =============================================================================


class TestCacheConsistencyProperties:
    """
    Property tests for cache consistency.
    
    Property 4: Cache Consistency
    - Verify cached features produce same output as computed features (rtol=1e-3)
    
    Validates: Requirements 5.2, 5.3
    """

    @pytest.mark.gpu
    @given(
        tensor_params=feature_tensor_params(),
        cache_key=cache_key_params(),
    )
    @settings(max_examples=50, deadline=None)
    def test_cached_features_match_stored_features(
        self, 
        tensor_params: Tuple[int, int, int, int],
        cache_key: Tuple[int, int],
    ):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: Cached features exactly match the features that were stored.
        
        For any feature tensor stored in the cache with a (timestep, block_idx) key,
        retrieving the cached features should return a tensor that matches the
        original within rtol=1e-3.
        """
        batch_size, channels, height, width = tensor_params
        timestep, block_idx = cache_key
        
        # Create cache manager with default config (caches all blocks)
        config = CacheConfig()
        cache_manager = CacheManager(config)
        
        # Generate random feature tensor on CUDA
        original_features = torch.randn(
            batch_size, channels, height, width,
            device="cuda",
            dtype=torch.float32
        )
        
        # Compute the aligned timestep for storage (must be on cache interval boundary)
        # This ensures the features will be stored and retrievable
        aligned_timestep = (timestep // config.cache_interval) * config.cache_interval
        
        # Store features in cache
        cache_manager.store(aligned_timestep, block_idx, original_features)
        
        # Retrieve cached features
        cached_features = cache_manager.get_cached(aligned_timestep, block_idx)
        
        # Verify cached features are not None
        assert cached_features is not None, (
            f"Cache miss for key ({aligned_timestep}, {block_idx}) - "
            f"expected cache hit after store"
        )
        
        # Verify cached features match original within tolerance
        assert torch.allclose(
            original_features, 
            cached_features, 
            rtol=1e-3, 
            atol=1e-5
        ), (
            f"Cached features do not match original features within rtol=1e-3. "
            f"Max diff: {(original_features - cached_features).abs().max().item()}"
        )
        
        # Cleanup
        cache_manager.clear()

    @pytest.mark.gpu
    @given(
        tensor_params=feature_tensor_params(),
        timestep=st.integers(min_value=0, max_value=999),
        block_idx=st.integers(min_value=0, max_value=11),
        cache_interval=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_cache_retrieval_from_nearby_timesteps(
        self,
        tensor_params: Tuple[int, int, int, int],
        timestep: int,
        block_idx: int,
        cache_interval: int,
    ):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: Features stored at compute timesteps can be retrieved
        from nearby non-compute timesteps.
        
        When features are stored at a compute timestep (timestep % cache_interval == 0),
        they should be retrievable from any timestep in the same interval window,
        and the retrieved features should match the original within rtol=1e-3.
        """
        batch_size, channels, height, width = tensor_params
        
        # Create cache manager with specified interval
        config = CacheConfig(cache_interval=cache_interval)
        cache_manager = CacheManager(config)
        
        # Calculate the compute timestep for this interval
        compute_timestep = (timestep // cache_interval) * cache_interval
        
        # Generate random feature tensor on CUDA
        original_features = torch.randn(
            batch_size, channels, height, width,
            device="cuda",
            dtype=torch.float32
        )
        
        # Store features at the compute timestep
        cache_manager.store(compute_timestep, block_idx, original_features)
        
        # Try to retrieve from the original timestep (may or may not be compute timestep)
        cached_features = cache_manager.get_cached(timestep, block_idx)
        
        # Verify cached features are retrievable
        assert cached_features is not None, (
            f"Cache miss for timestep {timestep} (compute timestep: {compute_timestep}, "
            f"interval: {cache_interval}) - expected cache hit"
        )
        
        # Verify cached features match original within tolerance
        assert torch.allclose(
            original_features,
            cached_features,
            rtol=1e-3,
            atol=1e-5
        ), (
            f"Cached features do not match original features within rtol=1e-3. "
            f"Max diff: {(original_features - cached_features).abs().max().item()}"
        )
        
        # Cleanup
        cache_manager.clear()

    @pytest.mark.gpu
    @given(
        tensor_params=feature_tensor_params(),
        num_entries=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_multiple_cache_entries_consistency(
        self,
        tensor_params: Tuple[int, int, int, int],
        num_entries: int,
    ):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: Multiple cache entries maintain consistency independently.
        
        When multiple feature tensors are stored with different (timestep, block_idx)
        keys, each entry should be retrievable and match its original within rtol=1e-3.
        """
        batch_size, channels, height, width = tensor_params
        
        # Create cache manager
        config = CacheConfig(cache_interval=1)  # Cache every timestep
        cache_manager = CacheManager(config)
        
        # Store multiple entries with different keys
        stored_features = {}
        for i in range(num_entries):
            timestep = i * 10  # Spread out timesteps
            block_idx = i % 12  # Cycle through block indices
            
            features = torch.randn(
                batch_size, channels, height, width,
                device="cuda",
                dtype=torch.float32
            )
            
            cache_manager.store(timestep, block_idx, features)
            stored_features[(timestep, block_idx)] = features
        
        # Verify all entries are retrievable and match
        for (timestep, block_idx), original_features in stored_features.items():
            cached_features = cache_manager.get_cached(timestep, block_idx)
            
            assert cached_features is not None, (
                f"Cache miss for key ({timestep}, {block_idx})"
            )
            
            assert torch.allclose(
                original_features,
                cached_features,
                rtol=1e-3,
                atol=1e-5
            ), (
                f"Cached features for key ({timestep}, {block_idx}) do not match. "
                f"Max diff: {(original_features - cached_features).abs().max().item()}"
            )
        
        # Cleanup
        cache_manager.clear()

    @pytest.mark.gpu
    @given(
        tensor_params=feature_tensor_params(),
        cache_key=cache_key_params(),
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_update_consistency(
        self,
        tensor_params: Tuple[int, int, int, int],
        cache_key: Tuple[int, int],
    ):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: Updating a cache entry replaces the old value consistently.
        
        When features are stored at a key that already exists, the new features
        should replace the old ones, and retrieval should return the new features
        within rtol=1e-3.
        """
        batch_size, channels, height, width = tensor_params
        timestep, block_idx = cache_key
        
        # Create cache manager
        config = CacheConfig(cache_interval=1)
        cache_manager = CacheManager(config)
        
        # Store initial features
        initial_features = torch.randn(
            batch_size, channels, height, width,
            device="cuda",
            dtype=torch.float32
        )
        cache_manager.store(timestep, block_idx, initial_features)
        
        # Store updated features at the same key
        updated_features = torch.randn(
            batch_size, channels, height, width,
            device="cuda",
            dtype=torch.float32
        )
        cache_manager.store(timestep, block_idx, updated_features)
        
        # Retrieve and verify it's the updated features
        cached_features = cache_manager.get_cached(timestep, block_idx)
        
        assert cached_features is not None, (
            f"Cache miss for key ({timestep}, {block_idx}) after update"
        )
        
        # Should match updated features, not initial
        assert torch.allclose(
            updated_features,
            cached_features,
            rtol=1e-3,
            atol=1e-5
        ), (
            f"Cached features do not match updated features. "
            f"Max diff: {(updated_features - cached_features).abs().max().item()}"
        )
        
        # Should NOT match initial features (unless they happen to be very similar)
        # We check that the cache was actually updated
        if not torch.allclose(initial_features, updated_features, rtol=1e-3):
            assert not torch.allclose(
                initial_features,
                cached_features,
                rtol=1e-3,
                atol=1e-5
            ), "Cache still contains initial features after update"
        
        # Cleanup
        cache_manager.clear()

    @pytest.mark.gpu
    @given(
        tensor_params=feature_tensor_params(),
        cache_layer_ids=st.lists(
            st.integers(min_value=0, max_value=11),
            min_size=1,
            max_size=6,
            unique=True
        ),
        timestep=st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_layer_filtering_consistency(
        self,
        tensor_params: Tuple[int, int, int, int],
        cache_layer_ids: list,
        timestep: int,
    ):
        """
        **Validates: Requirements 5.2, 5.3**
        
        Property: Cache layer filtering maintains consistency for allowed blocks.
        
        When cache_layer_ids is specified, only features from those blocks should
        be cached. Features from allowed blocks should be retrievable and match
        the original within rtol=1e-3.
        """
        batch_size, channels, height, width = tensor_params
        
        # Create cache manager with layer filtering
        config = CacheConfig(
            cache_interval=1,
            cache_layer_ids=cache_layer_ids
        )
        cache_manager = CacheManager(config)
        
        # Test with a block that IS in cache_layer_ids
        allowed_block_idx = cache_layer_ids[0]
        
        features = torch.randn(
            batch_size, channels, height, width,
            device="cuda",
            dtype=torch.float32
        )
        
        cache_manager.store(timestep, allowed_block_idx, features)
        cached_features = cache_manager.get_cached(timestep, allowed_block_idx)
        
        assert cached_features is not None, (
            f"Cache miss for allowed block {allowed_block_idx}"
        )
        
        assert torch.allclose(
            features,
            cached_features,
            rtol=1e-3,
            atol=1e-5
        ), (
            f"Cached features for allowed block do not match. "
            f"Max diff: {(features - cached_features).abs().max().item()}"
        )
        
        # Test with a block that is NOT in cache_layer_ids
        excluded_block_idx = next(
            (i for i in range(12) if i not in cache_layer_ids),
            None
        )
        
        if excluded_block_idx is not None:
            excluded_features = torch.randn(
                batch_size, channels, height, width,
                device="cuda",
                dtype=torch.float32
            )
            
            # Store should be a no-op for excluded blocks
            cache_manager.store(timestep, excluded_block_idx, excluded_features)
            
            # Retrieval should return None for excluded blocks
            cached_excluded = cache_manager.get_cached(timestep, excluded_block_idx)
            assert cached_excluded is None, (
                f"Excluded block {excluded_block_idx} should not be cached"
            )
        
        # Cleanup
        cache_manager.clear()

