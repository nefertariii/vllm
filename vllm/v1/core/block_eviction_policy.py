# 檔案：vllm/v1/core/block_eviction_policy.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


@dataclass
class BlockEvictionStats:
    """Statistics for a cached block to compute eviction priority."""
    
    block_id: int
    last_access_time: float  # Unix timestamp in seconds
    reference_count: int  # Current ref_cnt
    token_length: int  # Number of tokens in this block (prefix length)
    access_count: int  # Total number of times accessed (frequency)
    creation_time: float  # When the block was created
    
    def get_lifetime_seconds(self) -> float:
        """Time since block was created."""
        return time.time() - self.creation_time
    
    def get_idle_time_seconds(self) -> float:
        """Time since last access."""
        return time.time() - self.last_access_time


class BlockEvictionPolicy(ABC):
    """Abstract base class for block eviction policies."""
    
    @abstractmethod
    def compute_priority_score(self, stats: BlockEvictionStats) -> float:
        """
        Compute eviction priority score for a block.
        
        Lower score = higher priority for eviction (will be evicted first)
        Higher score = lower priority for eviction (should be kept)
        
        Args:
            stats: Statistics of the block
            
        Returns:
            Priority score (float)
        """
        raise NotImplementedError
    
    @abstractmethod
    def should_protect_block(self, stats: BlockEvictionStats) -> bool:
        """
        Check if a block should be protected from eviction.
        
        Args:
            stats: Statistics of the block
            
        Returns:
            True if block should be protected, False otherwise
        """
        raise NotImplementedError


class LRUEvictionPolicy(BlockEvictionPolicy):
    """Simple LRU (Least Recently Used) eviction policy."""
    
    def compute_priority_score(self, stats: BlockEvictionStats) -> float:
        """
        LRU score: blocks accessed longer ago have higher eviction priority.
        Score = idle_time_seconds (higher idle time = lower score = evict first)
        """
        return stats.get_idle_time_seconds()
    
    def should_protect_block(self, stats: BlockEvictionStats) -> bool:
        """Protect blocks that are currently referenced."""
        return stats.reference_count > 0


class LFUEvictionPolicy(BlockEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def compute_priority_score(self, stats: BlockEvictionStats) -> float:
        """
        LFU score: blocks accessed less frequently have higher eviction priority.
        Score = -access_count (less frequently used = lower score = evict first)
        Add small idle time factor for tie-breaking.
        """
        # Avoid division by zero
        access_frequency = max(1, stats.access_count)
        idle_time = stats.get_idle_time_seconds()
        
        # Lower access_count → higher priority for eviction (lower score)
        # Ties broken by recency
        return -access_frequency + (idle_time / 1000.0)  # Small weight for tie-breaking
    
    def should_protect_block(self, stats: BlockEvictionStats) -> bool:
        """Protect blocks that are currently referenced."""
        return stats.reference_count > 0


class CostAwareEvictionPolicy(BlockEvictionPolicy):
    """
    Cost-Aware eviction policy that considers:
    1. Last access time (recency)
    2. Reference count (current usage)
    3. Token length / prefix length (recomputation cost)
    4. Access frequency (historical usage pattern)
    
    Priority Score = (1 - access_weight) * recency_score 
                     + access_weight * cost_score
                     + frequency_weight * frequency_score
    
    Lower score = evict first (higher priority for eviction)
    Higher score = keep (lower priority for eviction)
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        access_weight: float = 0.4,
        frequency_weight: float = 0.3,
        cost_sensitivity: float = 1.0,
        min_idle_time_for_eviction: float = 0.1,
    ):
        """
        Initialize Cost-Aware eviction policy.
        
        Args:
            recency_weight: Weight for recency score (0.0-1.0)
                           Lower = prioritize keeping recently accessed blocks
            access_weight: Weight for access count score (0.0-1.0)
                          Higher = prioritize keeping frequently accessed blocks
            frequency_weight: Weight for frequency/cost score (0.0-1.0)
                             Higher = prioritize keeping high-cost blocks
            cost_sensitivity: Sensitivity to token length cost (>0)
                            Higher = more sensitive to block size
            min_idle_time_for_eviction: Minimum idle time before a block can be evicted (seconds)
        """
        assert 0.0 <= recency_weight <= 1.0, "recency_weight must be in [0, 1]"
        assert 0.0 <= access_weight <= 1.0, "access_weight must be in [0, 1]"
        assert 0.0 <= frequency_weight <= 1.0, "frequency_weight must be in [0, 1]"
        
        # Normalize weights to sum to 1.0
        total_weight = recency_weight + access_weight + frequency_weight
        self.recency_weight = recency_weight / total_weight
        self.access_weight = access_weight / total_weight
        self.frequency_weight = frequency_weight / total_weight
        
        self.cost_sensitivity = cost_sensitivity
        self.min_idle_time_for_eviction = min_idle_time_for_eviction
        
        logger.info(
            f"CostAwareEvictionPolicy initialized: "
            f"recency_weight={self.recency_weight:.3f}, "
            f"access_weight={self.access_weight:.3f}, "
            f"frequency_weight={self.frequency_weight:.3f}, "
            f"cost_sensitivity={self.cost_sensitivity}"
        )
    
    def compute_priority_score(self, stats: BlockEvictionStats) -> float:
        """
        Compute combined priority score for a block.
        
        Components:
        1. Recency Score: Higher idle time = higher priority for eviction
                         Normalized to [0, 1]
        2. Access Count Score: Lower access count = higher priority for eviction
                              Normalized to [0, 1]
        3. Cost Score: Higher token count = LOWER priority for eviction (protect it!)
                      Inverse relationship: tokens_count → protection
        """
        idle_time = stats.get_idle_time_seconds()
        
        # Component 1: Recency Score (0-1, higher = more evictable)
        # Exponential decay: recently accessed blocks get high score (less evictable)
        # FORMULA: recency_score = 1 - exp(-idle_time / time_scale)
        # Where time_scale = 3600 seconds (1 hour default)
        time_scale = 3600.0  # seconds
        recency_score = 1.0 - (1.0 / (1.0 + idle_time / time_scale))
        # Alternative simple version:
        # recency_score = min(1.0, idle_time / time_scale)
        
        # Component 2: Access Count Score (0-1, higher = more evictable)
        # Blocks accessed more frequently get lower scores (less evictable)
        # FORMULA: access_score = 1.0 / (1.0 + access_count)
        access_score = 1.0 / (1.0 + stats.access_count)
        
        # Component 3: Cost Score (0-1, INVERTED - higher token_count = LOWER eviction priority)
        # Long-context blocks should be protected
        # FORMULA: cost_score = 1.0 / (1.0 + cost_sensitivity * token_length)
        # This means: more tokens → lower score → kept longer
        cost_score = 1.0 / (1.0 + self.cost_sensitivity * stats.token_length / 1000.0)
        
        # Combined score
        priority_score = (
            self.recency_weight * recency_score
            + self.access_weight * access_score
            + self.frequency_weight * cost_score
        )
        
        return priority_score
    
    def should_protect_block(self, stats: BlockEvictionStats) -> bool:
        """
        Determine if a block should be protected from eviction.
        
        Protection criteria:
        1. Block is currently referenced (ref_cnt > 0)
        2. Block is too new (hasn't been idle long enough)
        3. Block has very high cost (token_length > threshold)
        """
        # Always protect blocks that are in use
        if stats.reference_count > 0:
            return True
        
        # Protect blocks that are too new
        if stats.get_idle_time_seconds() < self.min_idle_time_for_eviction:
            return True
        
        # Protect very large blocks (high recomputation cost)
        # If a block has > 5000 tokens, protect it unless absolutely necessary
        if stats.token_length > 5000:
            return True
        
        return False


class HybridCostAwareEvictionPolicy(BlockEvictionPolicy):
    """
    Hybrid policy combining Cost-Aware with protection for common pool blocks.
    Used when operating with dual pool structure.
    """
    
    def __init__(
        self,
        recency_weight: float = 0.25,
        access_weight: float = 0.35,
        frequency_weight: float = 0.4,
        cost_sensitivity: float = 1.2,
        is_common_pool: bool = False,
        common_pool_protection_factor: float = 2.0,
    ):
        """
        Initialize hybrid cost-aware eviction policy.
        
        Args:
            recency_weight: Weight for recency component
            access_weight: Weight for access count component
            frequency_weight: Weight for frequency/cost component
            cost_sensitivity: Sensitivity to token length
            is_common_pool: Whether this is for common pool (gets more protection)
            common_pool_protection_factor: Multiplier for protecting common pool blocks
        """
        self.core_policy = CostAwareEvictionPolicy(
            recency_weight=recency_weight,
            access_weight=access_weight,
            frequency_weight=frequency_weight,
            cost_sensitivity=cost_sensitivity,
        )
        self.is_common_pool = is_common_pool
        self.common_pool_protection_factor = common_pool_protection_factor
    
    def compute_priority_score(self, stats: BlockEvictionStats) -> float:
        """
        Compute priority score with common pool protection.
        
        If this is a common pool:
        - Apply protection factor to make common pool blocks harder to evict
        - Score = core_policy.score / common_pool_protection_factor
        - Lower score = higher priority for eviction
        - So dividing by protection_factor makes score HIGHER = LESS evictable
        """
        core_score = self.core_policy.compute_priority_score(stats)
        
        if self.is_common_pool:
            # Higher score = less evictable = more protection
            return core_score / self.common_pool_protection_factor
        
        return core_score
    
    def should_protect_block(self, stats: BlockEvictionStats) -> bool:
        """
        Hybrid protection: common pool blocks get extra protection.
        """
        # Core protection rules
        if self.core_policy.should_protect_block(stats):
            return True
        
        # Additional protection for common pool
        if self.is_common_pool:
            # Protect blocks with very high reuse likelihood
            if stats.access_count >= 3:  # Been accessed 3+ times
                return True
        
        return False
