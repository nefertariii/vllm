# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

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


@dataclass
class EvictionPolicyBreakdown:
    """Per-interval breakdown stats for the dual-pool cost-aware eviction policy.

    Call drain() to read and reset all counters atomically.
    """

    # ── Pool occupancy (snapshot at drain time) ─────────────────────────────
    common_pool_size: int = 0
    cached_pool_size: int = 0

    # ── Cache-hit breakdown ──────────────────────────────────────────────────
    hits_from_common_pool: int = 0   # hit found in protected common pool
    hits_from_cached_pool: int = 0   # hit found in normal cached pool
    cache_misses: int = 0            # no hit in either pool

    # ── Eviction breakdown ───────────────────────────────────────────────────
    evictions_from_cached_pool: int = 0       # normal evictions (cachePool → freed)
    demotions_common_to_cached: int = 0       # commonPool full → LRU → cachePool

    # ── Score-component analysis on evicted blocks ───────────────────────────
    # Sum accumulators (divide by evictions_from_cached_pool for averages)
    _sum_evicted_access_count: int = 0
    _sum_evicted_chain_blocks: int = 0
    _sum_evicted_idle_s: float = 0.0

    # ── Counterfactual: did LFU/cost change the choice vs pure LRU? ─────────
    lru_policy_would_differ: int = 0

    def record_eviction(self, block: "KVCacheBlock", now_ns: int) -> None:
        idle_s = (
            (now_ns - block.last_access_time_ns) / 1e9
            if block.last_access_time_ns else 0.0
        )
        self.evictions_from_cached_pool += 1
        self._sum_evicted_access_count += block.access_count
        self._sum_evicted_chain_blocks += block.chain_total_blocks
        self._sum_evicted_idle_s += idle_s

    def drain(self) -> dict:
        """Return current stats as a dict and reset all counters."""
        n = max(1, self.evictions_from_cached_pool)
        total_lookups = (
            self.hits_from_common_pool + self.hits_from_cached_pool + self.cache_misses
        )
        stats = {
            # Pool state
            "common_pool_size": self.common_pool_size,
            "cached_pool_size": self.cached_pool_size,
            # Hit breakdown
            "hits_from_common_pool": self.hits_from_common_pool,
            "hits_from_cached_pool": self.hits_from_cached_pool,
            "cache_misses": self.cache_misses,
            "overall_hit_rate": (total_lookups - self.cache_misses) / max(1, total_lookups),
            "common_pool_hit_fraction": self.hits_from_common_pool / max(1, total_lookups),
            # Eviction breakdown
            "evictions_from_cached_pool": self.evictions_from_cached_pool,
            "demotions_common_to_cached": self.demotions_common_to_cached,
            # Score component averages on evicted blocks
            "avg_evicted_access_count": self._sum_evicted_access_count / n,
            "avg_evicted_chain_blocks": self._sum_evicted_chain_blocks / n,
            "avg_evicted_idle_s": self._sum_evicted_idle_s / n,
            # Counterfactual
            "lru_policy_would_differ": self.lru_policy_would_differ,
        }
        # Reset
        self.common_pool_size = 0
        self.cached_pool_size = 0
        self.hits_from_common_pool = 0
        self.hits_from_cached_pool = 0
        self.cache_misses = 0
        self.evictions_from_cached_pool = 0
        self.demotions_common_to_cached = 0
        self._sum_evicted_access_count = 0
        self._sum_evicted_chain_blocks = 0
        self._sum_evicted_idle_s = 0.0
        self.lru_policy_would_differ = 0
        return stats


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        # ── Dual-pool + cost-aware eviction ─────────────────────────────────
        # enable_dual_pool: promote frequently-reused blocks to a protected
        # common pool (Jenga-style). Eviction falls back to common pool only
        # when the cached pool is exhausted.
        enable_dual_pool: bool = True,
        # enable_cost_scoring: use composite LRU+LFU+cost score for eviction
        # instead of pure LRU. Independent of enable_dual_pool.
        enable_cost_scoring: bool = True,
        # A block is promoted to the common (protected) pool when its
        # access_count reaches this threshold (i.e. reused across requests).
        common_pool_min_access: int = 2,
        # Upper bound on the number of blocks in the common pool.
        # -1 means 20 % of total blocks (mirrors Jenga's default).
        common_pool_max_blocks: int = -1,
        # Eviction-score weights for the cached pool.
        # score = -w_lru*recency + w_lfu*frequency + w_cost*cost  (lower → evict first)
        w_lru: float = 0.4,
        w_lfu: float = 0.3,
        w_cost: float = 0.3,
        # Set True to collect EvictionPolicyBreakdown stats.
        enable_breakdown: bool = False,
        # Number of blocks to sample when selecting the eviction candidate.
        # Replaces the O(n) full-scan with O(eviction_sample_k) sampling from
        # the head of the queue (oldest-first). Redis uses k=5; k=50 gives
        # ≥99% of full-scan quality while being ~560× faster for n=28k blocks.
        eviction_sample_k: int = 50,
        # ── Adaptive scoring ─────────────────────────────────────────────────
        # Dynamically adjust w_lru/w_lfu/w_cost based on four runtime signals:
        #   (1) cache fill ratio       → eviction pressure
        #   (2) cost CV in free pools  → how well cost differentiates blocks
        #   (3) total_blocks (GPU RAM) → baseline eviction pressure
        #   (4) model parameter count  → recomputation expense
        # Only active when enable_cost_scoring=True.
        enable_adaptive_scoring: bool = True,
        # Model size in billions of parameters (e.g. 9.0 for Gemma-2-9b).
        # Larger models make KV recomputation more expensive → upweight cost.
        num_params_B: float = 8.0,
        # Recompute adaptive weights every N calls to _evict_one_scored.
        adaptive_update_interval: int = 50,
        # Cap chain_total_blocks at (total_blocks * cost_cap_fraction) before
        # computing the cost component. Prevents extreme-length articles from
        # dominating the score and suppressing the LRU/LFU signals.
        cost_cap_fraction: float = 0.05,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size

        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # ── Dual-pool queues ─────────────────────────────────────────────────
        # cached_pool_queue: normal eviction candidates (replaces free_block_queue).
        # All blocks start here; eviction uses composite LRU+LFU+cost score.
        self.cached_pool_queue = FreeKVCacheBlockQueue(self.blocks)
        # common_pool_queue: protected blocks (high-reuse prefix pages).
        # Evicted only when the pool is full (LRU demotion to cached pool).
        self.common_pool_queue = FreeKVCacheBlockQueue([])

        # Dual-pool configuration
        self.enable_dual_pool = enable_dual_pool
        self.enable_cost_scoring = enable_cost_scoring
        self.common_pool_min_access = common_pool_min_access
        if common_pool_max_blocks == -1:
            self.common_pool_max_blocks = max(1, num_gpu_blocks // 5)
        else:
            self.common_pool_max_blocks = common_pool_max_blocks
        self.w_lru = w_lru
        self.w_lfu = w_lfu
        self.w_cost = w_cost

        # ── Adaptive scoring ─────────────────────────────────────────────────
        self.enable_adaptive_scoring = enable_adaptive_scoring and enable_cost_scoring

        # Static factors (computed once at init, never change):
        #   capacity_factor: smaller cache → higher pressure → upweight cost
        #     N_ref=4000 ≈ Gemma-2-9b on A6000 (small GPU reference).
        _N_ref = 4_000.0
        self._capacity_factor = min(max(_N_ref / num_gpu_blocks, 0.25), 2.0)
        #   model_factor: larger model → costlier recompute → upweight cost
        #     log2 scale avoids linear blow-up for 70B+ models.
        self._model_factor = math.log2(num_params_B + 1.0) / math.log2(9.0)
        #   cost_cap: clamps chain_total_blocks so a single extreme-length
        #     article (e.g. 90k tokens in Jenga) cannot drown out LRU/LFU.
        self._cost_cap_blocks = float(num_gpu_blocks) * cost_cap_fraction

        # Dynamic weights updated every adaptive_update_interval evictions:
        self._adaptive_update_interval = adaptive_update_interval
        self._eviction_count = 0
        # Initialised to the static base weights; overwritten on first update.
        self._adaptive_w_lru = w_lru
        self._adaptive_w_lfu = w_lfu
        self._adaptive_w_cost = w_cost

        # ── Welford online mean/variance for cost CV (O(1) per block freed) ──
        # Tracks the distribution of chain_total_blocks across freed blocks.
        # Updated in free_blocks(); read by _update_adaptive_weights().
        # Uses Welford's algorithm: numerically stable, single-pass, O(1).
        self._wf_n: int = 0           # sample count
        self._wf_mean: float = 0.0    # running mean
        self._wf_M2: float = 0.0      # running sum of squared deviations

        self._eviction_sample_k = eviction_sample_k

        # Scheduler-aware eviction: set of block IDs that should not be
        # evicted because a waiting request's prefix depends on them.
        # Refreshed every scheduling step by KVCacheManager.
        self.protected_block_ids: set[int] = set()

        # Breakdown stats (only populated when enable_breakdown=True)
        self.enable_breakdown = enable_breakdown
        self.breakdown: EvictionPolicyBreakdown | None = (
            EvictionPolicyBreakdown() if enable_breakdown else None
        )

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.cached_pool_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Blocks are found in the unified hash map (covers both common pool and
        cached pool). Breakdown tracking records which pool each hit came from.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        hit_in_common = False
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                if self.breakdown is not None:
                    self.breakdown.cache_misses += 1
                return None
            if block.in_common_pool:
                hit_in_common = True
            cached_blocks.append(block)

        if self.breakdown is not None:
            if hit_in_common:
                self.breakdown.hits_from_common_pool += 1
            else:
                self.breakdown.hits_from_cached_pool += 1
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        When dual-pool eviction is enabled (enable_dual_pool=True and
        enable_caching=True), blocks are selected from the cached pool using
        a composite LRU+LFU+cost score so that cheap-to-recompute and
        infrequently-reused blocks are evicted first.  The common pool is
        only touched as a last resort.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new blocks.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        use_scored = self.enable_cost_scoring and self.enable_caching
        use_dual_pool = self.enable_dual_pool and self.enable_caching

        has_protected = bool(self.protected_block_ids) and self.enable_caching
        if use_dual_pool or use_scored or has_protected:
            ret = [self._pop_one_free_block(use_scored) for _ in range(num_blocks)]
        else:
            ret = self.cached_pool_queue.popleft_n(num_blocks)

        now_ns = time.monotonic_ns()
        for block in ret:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            block.last_access_time_ns = now_ns
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    # ── Dual-pool helpers ────────────────────────────────────────────────────

    def _pop_one_free_block(self, use_scored: bool) -> KVCacheBlock:
        """Pop one block from the cached pool (scored or LRU), falling back to
        the common pool (always LRU) when the cached pool is empty.

        Args:
            use_scored: If True, use composite LRU+LFU+cost score; else LRU.
        """
        if self.cached_pool_queue.num_free_blocks > 0:
            if use_scored:
                # Adaptive fallback: skip scoring when cost weight is negligible.
                # Avoids O(k) scan overhead when adaptive scoring has reduced
                # w_cost to near-zero (low fill ratio → LRU is sufficient).
                if self.enable_adaptive_scoring and self._adaptive_w_cost < 0.01:
                    block = self._lru_skip_protected(self.cached_pool_queue)
                else:
                    block = self._evict_one_sampled(self.cached_pool_queue)
            else:
                block = self._lru_skip_protected(self.cached_pool_queue)
            block.in_common_pool = False
            return block

        # Cached pool exhausted: demote LRU from common pool.
        if self.common_pool_queue.num_free_blocks > 0:
            block = self._lru_skip_protected(self.common_pool_queue)
            block.in_common_pool = False
            if self.breakdown is not None:
                self.breakdown.demotions_common_to_cached += 1
            return block

        raise ValueError("Both pools are empty — this should not happen.")

    def _lru_skip_protected(self, queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        """Pop the oldest (LRU-head) block that is not in protected_block_ids.
        Falls back to the true LRU head if every block is protected."""
        if not self.protected_block_ids:
            return queue.popleft()

        curr = queue.fake_free_list_head.next_free_block
        while curr is not queue.fake_free_list_tail and curr is not None:
            if curr.block_id not in self.protected_block_ids:
                queue.remove(curr)
                return curr
            curr = curr.next_free_block

        # All blocks are protected; evict LRU head as a last resort.
        return queue.popleft()

    def _evict_one_sampled(self, queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        """Sample self._eviction_sample_k blocks from the head of *queue* and
        evict the one with the lowest score. O(k) instead of O(n).

        Sampling from the head (oldest-first by LRU ordering) maximises the
        chance of finding good eviction candidates: stale blocks tend to be at
        the front, and the scoring formula further distinguishes cheap-to-
        recompute blocks among them.

        Triggers periodic adaptive weight updates (same as _evict_one_scored).
        """
        if self.enable_adaptive_scoring:
            self._eviction_count += 1
            if self._eviction_count % self._adaptive_update_interval == 0:
                self._update_adaptive_weights()

        now_ns = time.monotonic_ns()
        best_block: KVCacheBlock | None = None
        best_score = float("inf")
        lru_head: KVCacheBlock | None = None

        curr = queue.fake_free_list_head.next_free_block
        count = 0
        first = True
        while (curr is not queue.fake_free_list_tail
               and curr is not None
               and count < self._eviction_sample_k):
            if first:
                lru_head = curr
                first = False
            # Skip blocks needed by waiting requests; still count toward sample
            # budget so we don't scan the entire queue when many are protected.
            if curr.block_id not in self.protected_block_ids:
                score = self._compute_eviction_score(curr, now_ns)
                if score < best_score:
                    best_score = score
                    best_block = curr
            curr = curr.next_free_block
            count += 1

        # Fallback: all sampled blocks were protected → evict LRU head anyway.
        if best_block is None:
            best_block = lru_head

        assert best_block is not None, "Tried to evict from an empty queue"

        if self.breakdown is not None:
            if lru_head is not None and best_block is not lru_head:
                self.breakdown.lru_policy_would_differ += 1
            self.breakdown.record_eviction(best_block, now_ns)

        queue.remove(best_block)
        return best_block

    def _evict_one_scored(self, queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        """Linear scan of *queue* to find and remove the block with the
        lowest eviction priority score.

        Lower score → evict first.
        Fresh (never-accessed) blocks always score −∞ so they are consumed
        before any cached block, preserving the original behaviour.

        NOTE: O(n) scan is acceptable for benchmarking. A production
        implementation would maintain a heap or bucket structure.
        """
        if self.enable_adaptive_scoring:
            self._eviction_count += 1
            if self._eviction_count % self._adaptive_update_interval == 0:
                self._update_adaptive_weights()

        now_ns = time.monotonic_ns()
        best_block: KVCacheBlock | None = None
        best_score = float("inf")
        lru_head: KVCacheBlock | None = None

        curr = queue.fake_free_list_head.next_free_block
        first = True
        while curr is not queue.fake_free_list_tail and curr is not None:
            if first:
                lru_head = curr
                first = False
            score = self._compute_eviction_score(curr, now_ns)
            if score < best_score:
                best_score = score
                best_block = curr
            curr = curr.next_free_block

        assert best_block is not None, "Tried to evict from an empty queue"

        if self.breakdown is not None:
            if lru_head is not None and best_block is not lru_head:
                self.breakdown.lru_policy_would_differ += 1
            self.breakdown.record_eviction(best_block, now_ns)

        queue.remove(best_block)
        return best_block

    def _compute_eviction_score(self, block: KVCacheBlock, now_ns: int) -> float:
        """Composite eviction score. Lower → evict first.

        Components (each normalized to roughly [0, 1] for typical workloads):
          recency  = idle_time_s / 60          (higher idle → more stale → evict sooner)
          lfu      = access_count / 10          (higher count → more valuable → keep longer)
          cost     = chain_total_blocks / 500   (longer chain → pricier recompute → keep longer)

        score = -w_lru * recency + w_lfu * lfu + w_cost * cost

        A higher score means "keep this block longer". The block with the
        lowest score is evicted first.

        chain_total_blocks is set at free() time, so even fresh blocks
        (access_count == 0) carry correct cost information. Fresh long-article
        blocks (chain_total_blocks ≈ 344) score higher than fresh short-article
        blocks (chain_total_blocks ≈ 25), correctly deferring their eviction.

        When enable_adaptive_scoring=True the weights (w_lru, w_lfu, w_cost)
        are replaced by self._adaptive_w_* which are updated periodically by
        _update_adaptive_weights(). The cost value is also capped at
        self._cost_cap_blocks to prevent extreme-length articles from drowning
        out the recency and frequency signals.
        """
        idle_s = (now_ns - block.last_access_time_ns) / 1e9
        recency = idle_s / 60.0
        lfu = block.access_count / 10.0

        if self.enable_adaptive_scoring:
            raw = min(float(block.chain_total_blocks), self._cost_cap_blocks)
            cost = raw / 500.0
            w_lru = self._adaptive_w_lru
            w_lfu = self._adaptive_w_lfu
            w_cost = self._adaptive_w_cost
        else:
            cost = block.chain_total_blocks / 500.0
            w_lru = self.w_lru
            w_lfu = self.w_lfu
            w_cost = self.w_cost

        return -w_lru * recency + w_lfu * lfu + w_cost * cost

    def _update_adaptive_weights(self) -> None:
        """Recompute adaptive LRU/LFU/cost weights from current runtime state.

        Four signals are combined (see __init__ docstring for full rationale):

        Dynamic (recomputed each call):
          pressure      – sigmoid of fill_ratio; near-full cache → cost matters more.
          heterogeneity – tanh of cost CV; diverse costs → cost signal is informative.

        Static (computed once at init):
          _capacity_factor – smaller GPU (fewer blocks) → baseline pressure higher.
          _model_factor    – larger model → costlier recompute → keep long chains.

        The final adaptive w_cost = w_cost_base × capacity × model × pressure
        × (0.5 + 0.5 × heterogeneity), then all three weights are renormalized
        so they sum to 1.
        """
        # ── Dynamic signal 1: fill ratio → eviction pressure ────────────────
        total_usable = max(1, self.num_gpu_blocks - 1)  # exclude null block
        fill_ratio = 1.0 - (self.get_num_free_blocks() / total_usable)
        # Smooth sigmoid centred at 75% fill:
        #   fill < 60% → pressure ≈ 0.5 (cost weight halved vs base)
        #   fill > 90% → pressure ≈ 1.0 (cost weight at full adaptive value)
        # Floor at 0.30 so cost scoring is always active (even at fill=0%).
        # Previous formula (floor≈0) made w_cost≈0 during early cache fill-up,
        # causing expensive blocks to be evicted before cost protection kicked in.
        # New range: [0.30 at fill=0%, 0.65 at fill=75%, 1.00 at fill=100%]
        pressure = 0.65 + 0.35 * math.tanh((fill_ratio - 0.75) * 8.0)

        # ── Dynamic signal 2: cost CV → heterogeneity (O(1) via Welford) ──────
        # _wf_n/_wf_mean/_wf_M2 are maintained incrementally in free_blocks().
        # CV = std / mean; tanh(CV) → 0 when all costs equal, → 1 when diverse.
        if self._wf_n >= 2:
            variance = self._wf_M2 / self._wf_n
            cv = math.sqrt(max(0.0, variance)) / (self._wf_mean + 1e-6)
        else:
            cv = 0.0
        heterogeneity = math.tanh(cv)

        # ── Combine into adaptive w_cost ─────────────────────────────────────
        w_cost_adapt = (
            self.w_cost
            * self._capacity_factor
            * self._model_factor
            * pressure
            * (0.5 + 0.5 * heterogeneity)
        )

        # ── Renormalize so all weights sum to 1 ──────────────────────────────
        total_w = self.w_lru + self.w_lfu + w_cost_adapt
        if total_w > 0:
            self._adaptive_w_lru = self.w_lru / total_w
            self._adaptive_w_lfu = self.w_lfu / total_w
            self._adaptive_w_cost = w_cost_adapt / total_w

    def _add_to_common_pool(self, block: KVCacheBlock) -> None:
        """Add *block* to the common (protected) pool.

        If the pool is already at capacity, the lowest-score block is demoted
        to the cached pool first (per spec: evict lowest score from common pool).
        Falls back to LRU when cost scoring is disabled.
        """
        if self.common_pool_queue.num_free_blocks >= self.common_pool_max_blocks:
            if self.enable_cost_scoring:
                evicted = self._evict_one_sampled(self.common_pool_queue)
            else:
                evicted = self.common_pool_queue.popleft()
            evicted.in_common_pool = False
            self.cached_pool_queue.append(evicted)
            if self.breakdown is not None:
                self.breakdown.demotions_common_to_cached += 1

        self.common_pool_queue.append(block)
        block.in_common_pool = True

    def get_breakdown(self) -> dict | None:
        """Return and reset the breakdown stats dict, or None if disabled."""
        if self.breakdown is None:
            return None
        self.breakdown.common_pool_size = self.common_pool_queue.num_free_blocks
        self.breakdown.cached_pool_size = self.cached_pool_queue.num_free_blocks
        return self.breakdown.drain()

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Also updates LFU access_count and the LRU timestamp used by the
        composite eviction score.

        Args:
            blocks: A list of blocks to touch.
        """
        now_ns = time.monotonic_ns()
        for block in blocks:
            # ref_cnt=0 means this block is in one of the free pools
            # (eviction candidate), so remove it from the correct queue.
            if block.ref_cnt == 0 and not block.is_null:
                if block.in_common_pool:
                    self.common_pool_queue.remove(block)
                    block.in_common_pool = False
                else:
                    self.cached_pool_queue.remove(block)
            block.ref_cnt += 1
            block.access_count += 1
            block.last_access_time_ns = now_ns
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        When dual-pool eviction is enabled, blocks with access_count >=
        common_pool_min_access are promoted to the protected common pool;
        all others go to the cached pool.

        Callers (SingleTypeKVCacheManager.free) must set
        block.chain_total_blocks before calling this method so that the
        cost component of the eviction score is accurate.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1

        ready = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]

        # Update Welford running statistics for cost CV (O(1) per block).
        if self.enable_adaptive_scoring:
            for b in ready:
                x = float(b.chain_total_blocks)
                if x > 0:
                    self._wf_n += 1
                    delta = x - self._wf_mean
                    self._wf_mean += delta / self._wf_n
                    self._wf_M2 += delta * (x - self._wf_mean)

        if self.enable_dual_pool and self.enable_caching:
            for block in ready:
                if block.access_count >= self.common_pool_min_access:
                    self._add_to_common_pool(block)
                else:
                    self.cached_pool_queue.append(block)
        else:
            self.cached_pool_queue.append_n(ready)

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

        # Remove all hashes from all blocks and reset dual-pool state.
        for block in self.blocks:
            block.reset_hash()
            block.access_count = 0
            block.chain_total_blocks = 0
            block.last_access_time_ns = 0
            block.in_common_pool = False

        # Reset Welford statistics.
        self._wf_n = 0
        self._wf_mean = 0.0
        self._wf_M2 = 0.0
        self._eviction_count = 0
        self._adaptive_w_lru = self.w_lru
        self._adaptive_w_lfu = self.w_lfu
        self._adaptive_w_cost = self.w_cost

        # Move all common-pool blocks back to cached pool.
        while self.common_pool_queue.num_free_blocks > 0:
            block = self.common_pool_queue.popleft()
            self.cached_pool_queue.append(block)

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool (both pools combined).

        Returns:
            The number of free blocks.
        """
        return (
            self.cached_pool_queue.num_free_blocks
            + self.common_pool_queue.num_free_blocks
        )

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

