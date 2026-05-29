# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

from vllm.v1.core.block_eviction_policy import (
    BlockEvictionPolicy,
    CostAwareEvictionPolicy,
    HybridCostAwareEvictionPolicy,
    BlockEvictionStats,
)

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")

class DualPoolBlockCache:
    """
    Dual-pool cache structure inspired by Jenga paper.
    
    - common_pool: Stores blocks with high predicted reuse probability
                  Protected from eviction when possible
    - cached_pool: Regular cached blocks, evicted first under pressure
    """
    
    def __init__(self, max_common_pool_size: int = 1000):
        """
        Initialize dual-pool cache.
        
        Args:
            max_common_pool_size: Maximum number of pages in common pool
        """
        self.common_pool: dict[int, tuple[int, float]] = {}  # block_id -> (token_length, last_access)
        self.cached_pool: dict[int, tuple[int, float]] = {}  # block_id -> (token_length, last_access)
        self.max_common_pool_size = max_common_pool_size
        self.hash_to_blocks: BlockHashToBlockMap = BlockHashToBlockMap()
    
    def add_page_to_pool(
        self,
        block: "KVCacheBlock",
        token_length: int,
        is_common_reuse: bool = False,
    ) -> None:
        """
        Add a page to appropriate pool (common or cached).
        
        Args:
            block: The KVCacheBlock to add
            token_length: Number of tokens in this block
            is_common_reuse: Whether this block likely to be reused across requests
        """
        now = time.time()
        
        if is_common_reuse:
            # Try to add to common pool
            if len(self.common_pool) >= self.max_common_pool_size:
                # Common pool full - evict oldest from common pool to cached pool
                oldest_block_id = min(
                    self.common_pool.keys(),
                    key=lambda bid: self.common_pool[bid][1]  # min by last_access_time
                )
                token_len, _ = self.common_pool.pop(oldest_block_id)
                self.cached_pool[oldest_block_id] = (token_len, now)
            
            self.common_pool[block.block_id] = (token_length, now)
        else:
            # Add to cached pool directly
            self.cached_pool[block.block_id] = (token_length, now)
    
    def remove_from_pool(self, block_id: int) -> bool:
        """
        Remove block from either pool.
        
        Returns:
            True if block was in a pool, False otherwise
        """
        if block_id in self.common_pool:
            del self.common_pool[block_id]
            return True
        if block_id in self.cached_pool:
            del self.cached_pool[block_id]
            return True
        return False
    
    def get_eviction_candidates_from_cached_pool(
        self,
    ) -> list[int]:
        """Get block IDs that can be evicted from cached pool."""
        return list(self.cached_pool.keys())
    
    def get_all_cached_blocks(self) -> dict[int, tuple[int, float]]:
        """Get all blocks (both pools) for inspection."""
        return {**self.common_pool, **self.cached_pool}


class BlockPool:
    """BlockPool with dual-pool and cost-aware eviction support."""
    
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: "KVCacheMetricsCollector | None" = None,
        eviction_policy: "BlockEvictionPolicy | None" = None,
        enable_dual_pool: bool = True,
        max_common_pool_size: int = 1000,
    ):
        """
        Initialize BlockPool with optional dual-pool and custom eviction policy.
        
        Args:
            num_gpu_blocks: Number of GPU blocks
            enable_caching: Whether to enable prefix caching
            hash_block_size: Block size for hashing
            enable_kv_cache_events: Whether to enable KV cache events
            metrics_collector: Optional metrics collector
            eviction_policy: Custom eviction policy (defaults to CostAwareEvictionPolicy)
            enable_dual_pool: Whether to use dual-pool structure
            max_common_pool_size: Max size of common pool
        """
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.enable_dual_pool = enable_dual_pool
        
        # All kv-cache blocks
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        
        # Free block queue
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        
        # Dual-pool cache structure
        if self.enable_dual_pool:
            self.dual_pool = DualPoolBlockCache(max_common_pool_size)
        else:
            self.dual_pool = None
        
        # Eviction policy
        if eviction_policy is None:
            eviction_policy = CostAwareEvictionPolicy(
                recency_weight=0.3,
                access_weight=0.4,
                frequency_weight=0.3,
                cost_sensitivity=1.0,
            )
        self.eviction_policy = eviction_policy
        
        # Track block statistics for eviction decisions
        self.block_stats: dict[int, BlockEvictionStats] = {}
        
        # Placeholder null block
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list["KVCacheEvent"] = []
        self.metrics_collector = metrics_collector
        
        logger.info(
            f"BlockPool initialized with {num_gpu_blocks} blocks, "
            f"dual_pool={'enabled' if enable_dual_pool else 'disabled'}, "
            f"eviction_policy={eviction_policy.__class__.__name__}"
        )
    
    def _update_block_stats(self, block: KVCacheBlock, token_length: int = 0) -> None:
        """Update statistics for a block used in eviction decisions."""
        block_id = block.block_id
        now = time.time()
        
        if block_id not in self.block_stats:
            self.block_stats[block_id] = BlockEvictionStats(
                block_id=block_id,
                last_access_time=now,
                reference_count=block.ref_cnt,
                token_length=token_length,
                access_count=1,
                creation_time=now,
            )
        else:
            stats = self.block_stats[block_id]
            stats.last_access_time = now
            stats.reference_count = block.ref_cnt
            stats.access_count += 1
            if token_length > 0:
                stats.token_length = token_length
    
    def _get_eviction_candidates(self) -> list[tuple[KVCacheBlock, float]]:
        """
        Get blocks eligible for eviction sorted by priority score.
        
        Returns:
            List of (block, priority_score) tuples sorted by priority (lowest first)
        """
        candidates = []
        
        # Get blocks to consider for eviction
        if self.enable_dual_pool and self.dual_pool:
            # Only evict from cached pool when using dual pool
            eviction_block_ids = self.dual_pool.get_eviction_candidates_from_cached_pool()
        else:
            # Evict from all free blocks
            eviction_block_ids = [
                b.block_id for b in self.free_block_queue.get_all_free_blocks()
                if not b.is_null
            ]
        
        for block_id in eviction_block_ids:
            block = self.blocks[block_id]
            
            # Skip protected blocks
            if block.block_id in self.block_stats:
                stats = self.block_stats[block.block_id]
                if self.eviction_policy.should_protect_block(stats):
                    continue
                
                priority_score = self.eviction_policy.compute_priority_score(stats)
                candidates.append((block, priority_score))
        
        # Sort by priority score (lowest first = highest priority for eviction)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _perform_eviction_if_needed(self) -> bool:
        """
        Perform eviction if needed to free blocks.
        
        Returns:
            True if eviction was performed, False otherwise
        """
        if self.get_num_free_blocks() > 0:
            return False  # No need to evict
        
        candidates = self._get_eviction_candidates()
        if not candidates:
            logger.warning("No eviction candidates found")
            return False
        
        block, priority_score = candidates[0]
        logger.debug(
            f"Evicting block {block.block_id} with priority score {priority_score:.4f}"
        )
        
        self._maybe_evict_cached_block(block)
        return True
    
    def cache_full_blocks(
        self,
        request: "Request",
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        predict_common_reuse: bool = False,
    ) -> None:
        """
        Cache full blocks with optional common pool placement.
        
        Args:
            request: The request
            blocks: List of blocks to cache
            num_cached_blocks: Number already cached
            num_full_blocks: Number to cache now
            block_size: Block size in tokens
            kv_cache_group_id: KV cache group ID
            predict_common_reuse: Whether to predict this block for common pool
        """
        if num_cached_blocks >= num_full_blocks:
            return
        
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        
        if block_size == self.hash_block_size:
            block_hashes = request.block_hashes
        else:
            assert block_size % self.hash_block_size == 0
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )
        
        new_block_hashes = block_hashes[num_cached_blocks:]
        
        # Calculate total token length for the prefix up to this block
        token_length = num_full_blocks * block_size
        
        new_hashes = [] if self.enable_kv_cache_events else None
        
        for i, blk in enumerate(new_full_blocks):
            if blk.is_null:
                continue
            
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]
            
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            
            # Update block statistics
            self._update_block_stats(blk, token_length=token_length)
            
            # Add to appropriate pool
            if self.enable_dual_pool and self.dual_pool:
                self.dual_pool.add_page_to_pool(
                    blk,
                    token_length=token_length,
                    is_common_reuse=predict_common_reuse,
                )
            
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))
        
        if self.enable_kv_cache_events:
            # ... 現有的 event 處理代碼 ...
            pass
    
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """
        Get new blocks, with cost-aware eviction if needed.
        """
        if num_blocks > self.get_num_free_blocks():
            # Try eviction before failing
            logger.warning(
                f"Insufficient free blocks ({self.get_num_free_blocks()} < {num_blocks}), "
                f"attempting eviction"
            )
            while self.get_num_free_blocks() < num_blocks:
                if not self._perform_eviction_if_needed():
                    break
            
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        
        ret = self.free_block_queue.popleft_n(num_blocks)
        
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
                
                # Update stats for newly allocated block
                self._update_block_stats(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        
        return ret
    
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        Evict a cached block, with dual-pool awareness.
        """
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)
        
        block_hash = block.block_hash
        if block_hash is None:
            return False
        
        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False
        
        # Remove from dual pool if present
        if self.enable_dual_pool and self.dual_pool:
            self.dual_pool.remove_from_pool(block.block_id)
        
        # Clean up stats
        self.block_stats.pop(block.block_id, None)
        
        block.reset_hash()
        
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        
        return True
    
    def get_eviction_stats(self) -> dict:
        """
        Get current eviction statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_blocks": self.num_gpu_blocks,
            "free_blocks": self.get_num_free_blocks(),
            "used_blocks": self.num_gpu_blocks - self.get_num_free_blocks(),
            "cached_blocks": len(self.block_stats),
            "common_pool_size": len(self.dual_pool.common_pool) if self.dual_pool else 0,
            "cached_pool_size": len(self.dual_pool.cached_pool) if self.dual_pool else 0,
            "eviction_policy": self.eviction_policy.__class__.__name__,
        }
    
    # 保持現有方法...
    def touch(self, blocks: "Sequence[KVCacheBlock]") -> None:
        """Touch blocks and update statistics."""
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)
            
            # Update access statistics
            self._update_block_stats(block)
    
    def free_blocks(self, ordered_blocks: "Iterable[KVCacheBlock]") -> None:
        """Free blocks with statistics update."""
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )
        
    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

