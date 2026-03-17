"""
Cache Manager for DeepCache/X-Slim style feature caching.

This module implements feature caching for training-free acceleration of
diffusion model inference. It caches intermediate UNet features that can
be reused at similar timesteps to skip redundant computations.

Target VRAM constraint: 2GB for feature cache (within 15.6GB total T4 limit)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import gc

import torch

from diffusion_trt.models import CacheEntry


# Default maximum cache size in GB (per Requirement 8.3)
DEFAULT_MAX_CACHE_SIZE_GB = 2.0


@dataclass
class CacheConfig:
    """
    Configuration for the CacheManager.
    
    Attributes:
        cache_interval: Cache every N timesteps (default: 3)
        cache_layer_ids: Which UNet blocks to cache (None = all blocks)
        cache_branch: Which branch to cache ("main", "skip", "both")
        max_cache_size_gb: Maximum cache size in GB (default: 2.0)
        enable_token_caching: Enable token-level caching for cross-attention
        token_similarity_threshold: Threshold for token similarity caching
    
    Validation Rules:
        - cache_interval must be >= 1
        - max_cache_size_gb must be > 0 and <= 2.0 (T4 constraint)
        - token_similarity_threshold must be in [0.0, 1.0]
    """
    cache_interval: int = 3
    cache_layer_ids: Optional[list] = None
    cache_branch: str = "main"
    max_cache_size_gb: float = DEFAULT_MAX_CACHE_SIZE_GB
    enable_token_caching: bool = True
    token_similarity_threshold: float = 0.95
    
    def __post_init__(self) -> None:
        """Validate configuration fields after initialization."""
        # Validate cache_interval >= 1
        if self.cache_interval < 1:
            raise ValueError(
                f"cache_interval must be >= 1, got {self.cache_interval}"
            )
        
        # Validate max_cache_size_gb is positive and within T4 constraint
        if self.max_cache_size_gb <= 0:
            raise ValueError(
                f"max_cache_size_gb must be > 0, got {self.max_cache_size_gb}"
            )
        if self.max_cache_size_gb > DEFAULT_MAX_CACHE_SIZE_GB:
            raise ValueError(
                f"max_cache_size_gb must be <= {DEFAULT_MAX_CACHE_SIZE_GB} "
                f"(T4 constraint), got {self.max_cache_size_gb}"
            )
        
        # Validate cache_branch
        valid_branches = {"main", "skip", "both"}
        if self.cache_branch not in valid_branches:
            raise ValueError(
                f"cache_branch must be one of {valid_branches}, "
                f"got '{self.cache_branch}'"
            )
        
        # Validate token_similarity_threshold
        if not 0.0 <= self.token_similarity_threshold <= 1.0:
            raise ValueError(
                f"token_similarity_threshold must be in [0.0, 1.0], "
                f"got {self.token_similarity_threshold}"
            )
        
        # Validate cache_layer_ids if provided
        if self.cache_layer_ids is not None:
            for layer_id in self.cache_layer_ids:
                if not isinstance(layer_id, int) or layer_id < 0:
                    raise ValueError(
                        f"cache_layer_ids must contain non-negative integers, "
                        f"got {layer_id}"
                    )


class CacheManager:
    """
    Manages feature caching for DeepCache/X-Slim style inference acceleration.
    
    Implements timestep-based caching where features are computed only every
    N timesteps (cache_interval) and reused for intermediate steps. Supports
    block-level caching to only cache features from specified UNet blocks.
    
    Cache keys are (timestep, block_idx) tuples.
    
    Attributes:
        config: CacheConfig with caching parameters
        
    Requirements:
        - 5.1: Compute features only every N timesteps
        - 5.2: Return cached features on cache hit
        - 5.3: Track timestep and block index as cache keys
        - 5.5: Provide cache hit/miss statistics
        - 5.6: Release all cached tensors on clear()
        - 5.7: Only cache features from specified UNet blocks
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the CacheManager.
        
        Args:
            config: CacheConfig with caching parameters. Uses defaults if None.
        """
        self.config = config or CacheConfig()
        
        # Internal cache storage: (timestep, block_idx) -> CacheEntry
        self._cache: Dict[Tuple[int, int], CacheEntry] = {}
        
        # Statistics tracking
        self._hits: int = 0
        self._misses: int = 0
        self._total_queries: int = 0
        
        # Current inference step counter
        self._current_step: int = 0
        
        # Track total cache size in bytes
        self._cache_size_bytes: int = 0
    
    def should_compute(self, timestep: int, block_idx: int) -> bool:
        """
        Determine if block should be computed or use cache.
        
        Returns True if:
        - The block is not in cache_layer_ids (if specified)
        - The timestep is a compute timestep (timestep % cache_interval == 0)
        - No cached features exist for this (timestep, block_idx)
        
        Returns False if valid cached features exist and can be reused.
        
        Args:
            timestep: Current diffusion timestep
            block_idx: Index of the UNet block
            
        Returns:
            True if block should be computed, False if cache can be used
            
        Validates: Requirements 5.1, 5.7
        """
        self._total_queries += 1
        
        # Check if this block should be cached at all
        if not self._should_cache_block(block_idx):
            # Block not in cache_layer_ids, always compute
            self._misses += 1
            return True
        
        # Check if this is a compute timestep
        if timestep % self.config.cache_interval == 0:
            # Compute timestep - must compute and store
            self._misses += 1
            return True
        
        # Check if we have cached features for a nearby compute timestep
        cache_key = self._find_cache_key(timestep, block_idx)
        if cache_key is not None:
            self._hits += 1
            return False
        
        # No cache available, must compute
        self._misses += 1
        return True
    
    def get_cached(
        self, 
        timestep: int, 
        block_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached features if available.
        
        Looks up cached features for the given timestep and block index.
        If exact match not found, looks for features from the nearest
        compute timestep (based on cache_interval).
        
        Args:
            timestep: Current diffusion timestep
            block_idx: Index of the UNet block
            
        Returns:
            Cached feature tensor if available, None otherwise
            
        Validates: Requirements 5.2, 5.3
        """
        cache_key = self._find_cache_key(timestep, block_idx)
        if cache_key is None:
            return None
        
        entry = self._cache.get(cache_key)
        if entry is not None:
            entry.record_access()
            return entry.features
        
        return None
    
    def store(
        self,
        timestep: int,
        block_idx: int,
        features: torch.Tensor
    ) -> None:
        """
        Store features in cache with timestep and block index as keys.
        
        Creates a CacheEntry and stores it in the internal cache dictionary.
        Only stores if the block is in cache_layer_ids (if specified).
        Evicts oldest entries if cache exceeds max_cache_size_gb.
        
        Args:
            timestep: Diffusion timestep when features were computed
            block_idx: Index of the UNet block that produced these features
            features: Feature tensor to cache (must be on CUDA device)
            
        Validates: Requirements 5.3, 5.4, 5.7, 8.3
        """
        # Check if this block should be cached
        if not self._should_cache_block(block_idx):
            return
        
        # Validate features tensor is on CUDA
        if not features.is_cuda:
            raise ValueError(
                f"features tensor must be on CUDA device, "
                f"got device: {features.device}"
            )
        
        cache_key = (timestep, block_idx)
        
        # Remove existing entry if present (update case)
        if cache_key in self._cache:
            old_entry = self._cache[cache_key]
            self._cache_size_bytes -= old_entry.size_bytes
        
        # Create new cache entry
        entry = CacheEntry(
            timestep=timestep,
            block_idx=block_idx,
            features=features,
            created_at=self._current_step,
            access_count=0
        )
        
        # Store entry and update size tracking
        self._cache[cache_key] = entry
        self._cache_size_bytes += entry.size_bytes
        
        # Evict oldest entries if cache exceeds max size (Requirement 5.4, 8.3)
        while self._cache_size_bytes > self.max_cache_size_bytes and len(self._cache) > 1:
            self._evict_oldest()
    
    def clear(self) -> None:
        """
        Release all cached tensors and free GPU memory.
        
        Clears the internal cache dictionary, resets statistics,
        and triggers garbage collection to free GPU memory.
        
        Validates: Requirements 5.6
        """
        # Clear all cache entries
        self._cache.clear()
        
        # Reset size tracking
        self._cache_size_bytes = 0
        
        # Reset statistics
        self._hits = 0
        self._misses = 0
        self._total_queries = 0
        self._current_step = 0
        
        # Force garbage collection to free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_stats(self) -> dict:
        """
        Return cache hit/miss statistics.
        
        Returns:
            Dictionary containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - total_queries: Total number of should_compute calls
            - hit_rate: Fraction of hits (0.0 to 1.0)
            - num_entries: Number of entries in cache
            - cache_size_bytes: Total size of cached tensors in bytes
            - cache_size_gb: Total size of cached tensors in GB
            
        Validates: Requirements 5.5
        """
        hit_rate = 0.0
        if self._total_queries > 0:
            hit_rate = self._hits / self._total_queries
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_queries": self._total_queries,
            "hit_rate": hit_rate,
            "num_entries": len(self._cache),
            "cache_size_bytes": self._cache_size_bytes,
            "cache_size_gb": self._cache_size_bytes / (1024 ** 3),
        }
    
    def increment_step(self) -> None:
        """Increment the current inference step counter."""
        self._current_step += 1
    
    @property
    def cache_size_gb(self) -> float:
        """Return current cache size in GB."""
        return self._cache_size_bytes / (1024 ** 3)
    
    @property
    def max_cache_size_bytes(self) -> int:
        """Return maximum cache size in bytes."""
        return int(self.config.max_cache_size_gb * (1024 ** 3))
    
    def _should_cache_block(self, block_idx: int) -> bool:
        """
        Check if a block should be cached based on cache_layer_ids.
        
        Args:
            block_idx: Index of the UNet block
            
        Returns:
            True if block should be cached, False otherwise
        """
        if self.config.cache_layer_ids is None:
            # No restriction, cache all blocks
            return True
        return block_idx in self.config.cache_layer_ids
    
    def _find_cache_key(
        self, 
        timestep: int, 
        block_idx: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find the cache key for a given timestep and block index.
        
        For non-compute timesteps, finds the nearest compute timestep
        that would have cached features.
        
        Args:
            timestep: Current diffusion timestep
            block_idx: Index of the UNet block
            
        Returns:
            Cache key tuple (compute_timestep, block_idx) if found, None otherwise
        """
        # Calculate the nearest compute timestep
        interval = self.config.cache_interval
        compute_timestep = (timestep // interval) * interval
        
        cache_key = (compute_timestep, block_idx)
        if cache_key in self._cache:
            return cache_key
        
        return None
    
    def _evict_oldest(self) -> bool:
        """
        Evict the oldest cache entry based on created_at timestamp.
        
        Finds the entry with the smallest created_at value and removes it
        from the cache, updating the cache size tracking.
        
        Returns:
            True if an entry was evicted, False if cache is empty
            
        Validates: Requirements 5.4, 8.3
        """
        if not self._cache:
            return False
        
        # Find the oldest entry (smallest created_at)
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        
        # Remove the oldest entry
        oldest_entry = self._cache.pop(oldest_key)
        self._cache_size_bytes -= oldest_entry.size_bytes
        
        return True
