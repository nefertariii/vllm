#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: LRU vs Dual-Pool Cost-Aware KV Cache Eviction Policy.

Simulates prefix-caching behaviour on real datasets (ShareGPT, arXiv QA)
using the same scoring formula as the modified vllm BlockPool, without
requiring a GPU or running actual LLM inference.

Breakdown metrics collected every --interval requests:
  - Cache hit rate (overall, common pool, cached pool)
  - Total recompute blocks when cache missed
  - Eviction breakdown: short/long chain, LFU/cost influence
  - Policy counterfactual: how often composite score differs from pure LRU

Usage (ShareGPT):
    python benchmark_dual_pool_eviction.py \\
        --model google/gemma-2-9b-it \\
        --datasets sharegpt \\
        --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-requests 1000 --num-blocks 2000

Usage (arXiv QA, auto-downloads from HuggingFace):
    python benchmark_dual_pool_eviction.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --datasets arxiv_qa --num-requests 500

Usage (both datasets, both models):
    python benchmark_dual_pool_eviction.py \\
        --model google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct \\
        --datasets sharegpt arxiv_qa \\
        --sharegpt-path /path/to/ShareGPT_V3.json \\
        --num-requests 800 --output results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

PolicyName = Literal["lru", "dual_pool"]


@dataclass
class WorkloadRequest:
    """One simulated inference request."""

    prefix_id: str          # identifies the shared context group
    prefix_block_hashes: list[int]   # hashes for each prefix block
    query_block_hashes: list[int]    # hashes for each query block
    num_context_tokens: int          # total context tokens
    num_query_tokens: int


@dataclass
class IntervalBreakdown:
    """Per-interval (rolling-window) stats, mirroring EvictionPolicyBreakdown."""

    interval_start: int
    interval_end: int
    policy: str

    # ── Pool occupancy
    common_pool_size: int = 0
    cached_pool_size: int = 0

    # ── Hit breakdown
    hits_from_common_pool: int = 0
    hits_from_cached_pool: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    common_pool_hit_fraction: float = 0.0

    # ── Eviction breakdown
    evictions_total: int = 0
    evictions_from_cached_pool: int = 0
    demotions_common_to_cached: int = 0

    # ── Evicted block characteristics
    avg_evicted_access_count: float = 0.0
    avg_evicted_chain_blocks: float = 0.0
    avg_evicted_idle_steps: float = 0.0

    # ── Cost-analysis: blocks that had to be recomputed
    total_recompute_blocks: int = 0

    # ── Counterfactual
    lru_policy_would_differ: int = 0


@dataclass
class BenchmarkResult:
    """Full result for one (policy, dataset, model) run."""

    timestamp: str
    model: str
    dataset: str
    policy: PolicyName
    num_requests: int
    num_blocks: int
    block_size: int

    # ── Policy config (dual_pool only)
    common_pool_min_access: int = 2
    common_pool_fraction: float = 0.2
    w_lru: float = 0.4
    w_lfu: float = 0.3
    w_cost: float = 0.3

    # ── Aggregate totals
    total_hits: int = 0
    total_misses: int = 0
    overall_hit_rate: float = 0.0
    total_recompute_blocks: int = 0
    total_evictions: int = 0

    # ── Interval breakdown series
    intervals: list[dict] = field(default_factory=list)

    # ── Unique prefixes seen
    unique_prefixes: int = 0


# ---------------------------------------------------------------------------
# Cache simulator
# ---------------------------------------------------------------------------

class CacheBlock:
    __slots__ = (
        "block_hash",
        "access_count",
        "chain_total_blocks",
        "last_access_step",
        "in_common_pool",
    )

    def __init__(self, block_hash: int, chain_total_blocks: int, step: int):
        self.block_hash = block_hash
        # access_count starts at 1: the creating request counts as one use.
        # Blocks with access_count >= common_pool_min_access go to common pool.
        self.access_count: int = 1
        self.chain_total_blocks: int = chain_total_blocks
        self.last_access_step: int = step
        self.in_common_pool: bool = False


class CacheSimulator:
    """
    Standalone prefix-cache simulator that mirrors BlockPool's dual-pool logic.

    Two policies:
      "lru"       — pure LRU, evicts the least recently used cached block.
      "dual_pool" — Jenga-style dual pool with composite LRU+LFU+cost score.

    Block lifecycle (mirrors vllm ref_cnt semantics):
      1. Prefix block allocated → access_count=1, enters eviction pool.
      2. Future prefix hit     → access_count+1, pool re-evaluated.
      3. Query block allocated → NOT added to eviction pool (ephemeral).

    All time references use integer "steps" (request index) for determinism.
    """

    def __init__(
        self,
        num_blocks: int,
        policy: PolicyName = "lru",
        common_pool_min_access: int = 2,
        common_pool_fraction: float = 0.2,
        w_lru: float = 0.4,
        w_lfu: float = 0.3,
        w_cost: float = 0.3,
    ):
        self.num_blocks = num_blocks
        self.policy = policy
        self.common_pool_min_access = common_pool_min_access
        self.common_pool_max = max(1, int(num_blocks * common_pool_fraction))
        self.w_lru = w_lru
        self.w_lfu = w_lfu
        self.w_cost = w_cost

        # All cached prefix blocks: hash → CacheBlock
        self._cache: dict[int, CacheBlock] = {}

        # Eviction pool sets (dual_pool only).
        # Only prefix blocks are in these sets; query blocks are excluded.
        self._common_pool: set[int] = set()   # protected from eviction
        self._cached_pool: set[int] = set()   # normal eviction candidates

        self._reset_interval()

    # ── Public API ──────────────────────────────────────────────────────────

    def process_request(self, req: WorkloadRequest, step: int) -> tuple[int, int]:
        """Simulate one inference request.

        Returns (num_hit_prefix_blocks, num_miss_prefix_blocks).
        """
        # 1. Prefix lookup
        hit = self._lookup_prefix(req.prefix_block_hashes, step)
        miss = len(req.prefix_block_hashes) - hit
        self._iv_misses += miss

        # 2. Allocate missed prefix blocks → enter eviction pool
        missed_hashes = req.prefix_block_hashes[hit:]
        chain_len = len(req.prefix_block_hashes)
        self._allocate_prefix_blocks(missed_hashes, chain_len, step)

        # 3. Query blocks are ephemeral: they use GPU KV cache during
        #    generation but are freed immediately after and never prefix-cached.
        #    We don't model them in the eviction pool at all — in vllm the
        #    scheduler reserves headroom for active query blocks separately.

        self._iv_recompute_blocks += miss
        return hit, miss

    def get_interval_breakdown(
        self, interval_start: int, interval_end: int, policy: str
    ) -> IntervalBreakdown:
        n_evict = max(1, self._iv_evictions)
        bd = IntervalBreakdown(
            interval_start=interval_start,
            interval_end=interval_end,
            policy=policy,
            common_pool_size=len(self._common_pool),
            cached_pool_size=len(self._cached_pool),
            hits_from_common_pool=self._iv_common_hits,
            hits_from_cached_pool=self._iv_cached_hits,
            cache_misses=self._iv_misses,
            evictions_total=self._iv_evictions,
            evictions_from_cached_pool=self._iv_evict_from_cached,
            demotions_common_to_cached=self._iv_demotions,
            total_recompute_blocks=self._iv_recompute_blocks,
            lru_policy_would_differ=self._iv_lru_differ,
        )
        total = self._iv_common_hits + self._iv_cached_hits + self._iv_misses
        bd.hit_rate = (
            (self._iv_common_hits + self._iv_cached_hits) / total if total else 0.0
        )
        bd.common_pool_hit_fraction = (
            self._iv_common_hits / total if total else 0.0
        )
        bd.avg_evicted_access_count = self._iv_sum_evict_access / n_evict
        bd.avg_evicted_chain_blocks = self._iv_sum_evict_chain / n_evict
        bd.avg_evicted_idle_steps = self._iv_sum_evict_idle / n_evict
        self._reset_interval()
        return bd

    # ── Private helpers ─────────────────────────────────────────────────────

    def _reset_interval(self) -> None:
        self._iv_common_hits = 0
        self._iv_cached_hits = 0
        self._iv_misses = 0
        self._iv_evictions = 0
        self._iv_evict_from_cached = 0
        self._iv_demotions = 0
        self._iv_recompute_blocks = 0
        self._iv_lru_differ = 0
        self._iv_sum_evict_access = 0
        self._iv_sum_evict_chain = 0
        self._iv_sum_evict_idle = 0.0

    def _lookup_prefix(self, hashes: list[int], step: int) -> int:
        """Return the number of contiguous prefix blocks found in cache."""
        hit = 0
        for h in hashes:
            if h not in self._cache:
                break
            blk = self._cache[h]
            blk.access_count += 1
            blk.last_access_step = step
            # Re-evaluate pool membership after access_count change
            if self.policy == "dual_pool":
                self._reclassify(h, blk)
            if blk.in_common_pool:
                self._iv_common_hits += 1
            else:
                self._iv_cached_hits += 1
            hit += 1
        return hit

    def _allocate_prefix_blocks(
        self, hashes: list[int], chain_len: int, step: int
    ) -> None:
        """Allocate and cache new prefix blocks, entering the eviction pool."""
        for h in hashes:
            if h in self._cache:
                continue
            if len(self._cache) >= self.num_blocks:
                self._evict_one(step)
            blk = CacheBlock(h, chain_len, step)  # access_count starts at 1
            self._cache[h] = blk
            if self.policy == "dual_pool":
                # access_count=1 < common_pool_min_access → cached pool
                self._cached_pool.add(h)

    def _reclassify(self, h: int, blk: CacheBlock) -> None:
        """Move block between pools if access_count threshold crossed."""
        should_be_common = blk.access_count >= self.common_pool_min_access
        if should_be_common and not blk.in_common_pool:
            self._cached_pool.discard(h)
            self._promote_to_common(h, blk)
        # Demotion from common → cached doesn't happen on access (only on overflow)

    def _promote_to_common(self, h: int, blk: CacheBlock) -> None:
        """Add to common pool, demoting LRU if at capacity."""
        if len(self._common_pool) >= self.common_pool_max:
            lru_h = min(
                self._common_pool,
                key=lambda x: self._cache[x].last_access_step,
            )
            self._cache[lru_h].in_common_pool = False
            self._common_pool.discard(lru_h)
            self._cached_pool.add(lru_h)
            self._iv_demotions += 1
        blk.in_common_pool = True
        self._common_pool.add(h)

    def _evict_one(self, step: int) -> None:
        """Evict one block using the active policy."""
        if not self._cache:
            return

        if self.policy == "lru":
            victim_h = min(
                self._cache,
                key=lambda x: self._cache[x].last_access_step,
            )
            self._record_eviction(victim_h, step, from_cached=False)
            del self._cache[victim_h]

        else:  # dual_pool
            pool = self._cached_pool if self._cached_pool else self._common_pool
            from_cached = bool(self._cached_pool)

            lru_h = min(pool, key=lambda x: self._cache[x].last_access_step)
            scored_h = min(pool, key=lambda x: self._score(self._cache[x], step))
            if scored_h != lru_h:
                self._iv_lru_differ += 1

            self._record_eviction(scored_h, step, from_cached=from_cached)
            pool.discard(scored_h)
            # Also remove from the other pool if it ended up in both (shouldn't happen)
            self._common_pool.discard(scored_h)
            self._cached_pool.discard(scored_h)
            del self._cache[scored_h]

    def _score(self, blk: CacheBlock, current_step: int) -> float:
        """Composite eviction score. Lower → evict first.

        Mirrors BlockPool._compute_eviction_score():
          score = -w_lru * recency + w_lfu * lfu + w_cost * cost
        """
        idle = current_step - blk.last_access_step
        recency = idle / 100.0          # reference: 100 steps ≈ 1 scheduling round
        lfu = blk.access_count / 10.0
        cost = blk.chain_total_blocks / 500.0
        return -self.w_lru * recency + self.w_lfu * lfu + self.w_cost * cost

    def _record_eviction(self, h: int, step: int, from_cached: bool) -> None:
        blk = self._cache[h]
        self._iv_evictions += 1
        if from_cached:
            self._iv_evict_from_cached += 1
        self._iv_sum_evict_access += blk.access_count
        self._iv_sum_evict_chain += blk.chain_total_blocks
        self._iv_sum_evict_idle += step - blk.last_access_step


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _hash_token_block(tokens: list[int]) -> int:
    """Stable hash for a block of token IDs."""
    return int(hashlib.md5(
        b"".join(t.to_bytes(3, "little") for t in tokens)
    ).hexdigest()[:16], 16)


def _tokens_to_block_hashes(
    tokens: list[int], block_size: int, prefix_salt: bytes = b""
) -> list[int]:
    """Split token list into blocks and return one hash per block."""
    hashes = []
    for i in range(0, len(tokens), block_size):
        block = tokens[i: i + block_size]
        if len(block) < block_size:
            break  # skip partial tail block (matches vllm behaviour)
        raw = prefix_salt + b"".join(t.to_bytes(3, "little") for t in block)
        hashes.append(int(hashlib.md5(raw).hexdigest()[:16], 16))
    return hashes


def _synthetic_tokens(length: int, vocab_size: int = 32000) -> list[int]:
    return [random.randint(1, vocab_size - 1) for _ in range(length)]


def load_mixed_hot_cold(
    block_size: int,
    num_blocks: int,
    num_requests: int,
    seed: int = 42,
) -> list[WorkloadRequest]:
    """Adversarial hot/cold workload — the canonical case where dual_pool wins.

    Pattern: alternating "warm" phases (N_hot hot-prefix requests) and
    "scan" phases (N_scan one-shot cold requests that fill the cache).

    During each scan phase, the cold documents flush the hot prefix out of
    LRU's cache.  When the next warm phase starts, LRU must recompute the
    hot prefix from scratch.  dual_pool keeps the hot prefix in the common
    pool and survives the scan.

    Design parameters:
      hot_prefix:   64 blocks (~1k tokens) — reused every warm phase
      cold docs:    each fills ~80% of cache (large one-shot documents)
      warm_size:    30 requests to the hot prefix
      scan_size:    20 cold one-shot documents (each ~80% of cache blocks)
    """
    rng = random.Random(seed)

    # Hot prefix: 64 blocks, reused periodically
    hot_blocks = 64
    hot_tokens = _synthetic_tokens(block_size * hot_blocks)
    hot_hashes = _tokens_to_block_hashes(
        hot_tokens, block_size, prefix_salt=b"hot-prefix"
    )

    # Cold one-shot documents: large enough to displace the hot prefix
    # Each cold doc fills ~80% of cache → a single scan of N docs
    # evicts all hot blocks from LRU cache.
    cold_doc_blocks = max(hot_blocks + 1, int(num_blocks * 0.8))
    # Enough unique cold docs to guarantee full cache eviction per scan
    num_scans_per_cycle = math.ceil(num_blocks / cold_doc_blocks) + 1
    # Generate many unique cold docs (each used only once per cycle)
    num_cold_docs = num_scans_per_cycle * (num_requests // 50 + 10)
    cold_docs: list[list[int]] = [
        _tokens_to_block_hashes(
            _synthetic_tokens(block_size * cold_doc_blocks),
            block_size,
            prefix_salt=f"cold-{c}".encode(),
        )
        for c in range(num_cold_docs)
    ]

    warm_phase = 30   # requests to hot prefix per cycle
    scan_phase = num_scans_per_cycle  # cold docs per cycle
    cold_idx = 0

    requests: list[WorkloadRequest] = []
    while len(requests) < num_requests:
        # Warm phase: queries against the shared hot prefix
        for _ in range(warm_phase):
            if len(requests) >= num_requests:
                break
            qry_ids = _synthetic_tokens(rng.randint(16, 32))
            requests.append(WorkloadRequest(
                prefix_id="hot",
                prefix_block_hashes=hot_hashes,
                query_block_hashes=_tokens_to_block_hashes(qry_ids, block_size),
                num_context_tokens=len(hot_tokens),
                num_query_tokens=len(qry_ids),
            ))
        # Scan phase: one-shot cold documents that flush the cache
        for _ in range(scan_phase):
            if len(requests) >= num_requests:
                break
            if cold_idx >= num_cold_docs:
                cold_idx = 0
            qry_ids = _synthetic_tokens(rng.randint(16, 32))
            requests.append(WorkloadRequest(
                prefix_id=f"cold-{cold_idx}",
                prefix_block_hashes=cold_docs[cold_idx],
                query_block_hashes=_tokens_to_block_hashes(qry_ids, block_size),
                num_context_tokens=block_size * cold_doc_blocks,
                num_query_tokens=len(qry_ids),
            ))
            cold_idx += 1

    return requests[:num_requests]


def load_sharegpt(
    path: str,
    tokenizer,
    num_requests: int,
    block_size: int,
    seed: int = 42,
) -> list[WorkloadRequest]:
    """Load ShareGPT conversations.

    Each conversation turn is treated as a query that shares the earlier
    conversation history (system prompt + prior turns) as its prefix.
    """
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    rng.shuffle(data)

    requests: list[WorkloadRequest] = []
    for conv in data:
        if len(requests) >= num_requests:
            break
        convs = conv["conversations"]
        # Build prefix = all turns before the last user turn
        context_text = " ".join(
            c["value"] for c in convs[:-1] if c.get("from") == "human"
        )
        query_text = convs[-1]["value"] if convs[-1].get("from") == "human" else convs[-1]["value"]

        if tokenizer is not None:
            ctx_ids = tokenizer.encode(context_text)
            qry_ids = tokenizer.encode(query_text)
        else:
            # Fallback: approximate token count by character / 4
            ctx_ids = _synthetic_tokens(max(16, len(context_text) // 4))
            qry_ids = _synthetic_tokens(max(8, len(query_text) // 4))

        prefix_id = f"sharegpt-{conv.get('id', len(requests))}"
        ctx_hashes = _tokens_to_block_hashes(ctx_ids, block_size,
                                              prefix_salt=prefix_id.encode())
        qry_hashes = _tokens_to_block_hashes(qry_ids, block_size)
        if not ctx_hashes:
            continue

        requests.append(WorkloadRequest(
            prefix_id=prefix_id,
            prefix_block_hashes=ctx_hashes,
            query_block_hashes=qry_hashes,
            num_context_tokens=len(ctx_ids),
            num_query_tokens=len(qry_ids),
        ))

    return requests[:num_requests]


def load_arxiv_qa(
    tokenizer,
    num_requests: int,
    block_size: int,
    num_blocks: int = 4000,
    seed: int = 42,
    papers_per_group: int = 5,
) -> list[WorkloadRequest]:
    """Load arXiv QA from HuggingFace (taesiri/arxiv_qa).

    Each paper's abstract/text is the shared prefix; each Q&A pair is a
    separate query against that prefix. papers_per_group controls how many
    questions per paper (simulates repeated access to the same document).
    Falls back to synthetic data if the dataset cannot be downloaded.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore
        print("Downloading arXiv QA dataset from HuggingFace...")
        ds = load_dataset("taesiri/arxiv_qa", split="train")
        rows = list(ds)
        rng.shuffle(rows)
    except Exception as e:
        print(f"[arXiv QA] Could not load dataset ({e}). Using synthetic fallback.")
        rows = None

    requests: list[WorkloadRequest] = []

    if rows:
        # Group rows by paper_id so multiple questions share the same prefix
        from collections import defaultdict
        by_paper: dict[str, list] = defaultdict(list)
        for row in rows:
            pid = row.get("paper_id") or row.get("id", str(len(by_paper)))
            by_paper[pid].append(row)

        paper_ids = list(by_paper.keys())
        rng.shuffle(paper_ids)

        for pid in paper_ids:
            if len(requests) >= num_requests:
                break
            paper_rows = by_paper[pid]
            # taesiri/arxiv_qa has no paper text — simulate a shared paper prefix
            # using a deterministic synthetic token sequence seeded by paper_id.
            # Multiple questions per paper still exercise prefix reuse correctly.
            paper_text = (
                f"You are a research assistant analyzing arXiv paper {pid}. "
                f"This paper covers topics in machine learning, computer science, "
                f"and related fields. Below is a detailed summary of the paper content "
                f"followed by questions. Answer each question based on the paper.\n"
                + "Paper content: " + ("x " * 200)  # ~200 word synthetic body
            )
            if tokenizer is not None:
                ctx_ids = tokenizer.encode(paper_text)
            else:
                ctx_ids = _synthetic_tokens(max(64, len(paper_text) // 4))

            ctx_hashes = _tokens_to_block_hashes(ctx_ids, block_size,
                                                  prefix_salt=pid.encode())
            if not ctx_hashes:
                continue

            for row in paper_rows[:papers_per_group]:
                if len(requests) >= num_requests:
                    break
                q_text = row.get("question", "") + " " + row.get("answer", "")
                if tokenizer is not None:
                    qry_ids = tokenizer.encode(q_text)
                else:
                    qry_ids = _synthetic_tokens(max(8, len(q_text) // 4))
                qry_hashes = _tokens_to_block_hashes(qry_ids, block_size)

                requests.append(WorkloadRequest(
                    prefix_id=f"arxiv-{pid}",
                    prefix_block_hashes=ctx_hashes,
                    query_block_hashes=qry_hashes,
                    num_context_tokens=len(ctx_ids),
                    num_query_tokens=len(qry_ids),
                ))
    else:
        # Synthetic arXiv QA.
        # Paper length is capped at block_size * (num_blocks // 3) tokens so
        # that a single paper does not overflow the entire simulated cache,
        # which would make every eviction policy equally bad.
        max_paper_tokens = block_size * max(64, num_blocks // 3)
        min_paper_tokens = block_size * 32  # at least 32 blocks
        num_papers = max(10, num_requests // papers_per_group)
        for p in range(num_papers):
            if len(requests) >= num_requests:
                break
            ctx_len = rng.randint(min_paper_tokens, max_paper_tokens)
            ctx_ids = _synthetic_tokens(ctx_len)
            ctx_hashes = _tokens_to_block_hashes(
                ctx_ids, block_size, prefix_salt=f"paper-{p}".encode()
            )
            if not ctx_hashes:
                continue
            for _ in range(papers_per_group):
                if len(requests) >= num_requests:
                    break
                qry_ids = _synthetic_tokens(rng.randint(32, 128))
                qry_hashes = _tokens_to_block_hashes(qry_ids, block_size)
                requests.append(WorkloadRequest(
                    prefix_id=f"arxiv-synthetic-{p}",
                    prefix_block_hashes=ctx_hashes,
                    query_block_hashes=qry_hashes,
                    num_context_tokens=len(ctx_ids),
                    num_query_tokens=len(qry_ids),
                ))

    return requests[:num_requests]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    policy: PolicyName,
    requests: list[WorkloadRequest],
    model: str,
    dataset: str,
    num_blocks: int,
    block_size: int,
    interval: int,
    common_pool_min_access: int,
    common_pool_fraction: float,
    w_lru: float,
    w_lfu: float,
    w_cost: float,
) -> BenchmarkResult:
    sim = CacheSimulator(
        num_blocks=num_blocks,
        policy=policy,
        common_pool_min_access=common_pool_min_access,
        common_pool_fraction=common_pool_fraction,
        w_lru=w_lru,
        w_lfu=w_lfu,
        w_cost=w_cost,
    )

    result = BenchmarkResult(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        dataset=dataset,
        policy=policy,
        num_requests=len(requests),
        num_blocks=num_blocks,
        block_size=block_size,
        common_pool_min_access=common_pool_min_access,
        common_pool_fraction=common_pool_fraction,
        w_lru=w_lru,
        w_lfu=w_lfu,
        w_cost=w_cost,
        unique_prefixes=len({r.prefix_id for r in requests}),
    )

    interval_start = 0
    for step, req in enumerate(requests):
        hit, miss = sim.process_request(req, step)
        result.total_hits += hit
        result.total_misses += miss
        result.total_recompute_blocks += miss

        if (step + 1) % interval == 0 or step == len(requests) - 1:
            bd = sim.get_interval_breakdown(interval_start, step + 1, policy)
            result.intervals.append(asdict(bd))
            result.total_evictions += bd.evictions_total
            interval_start = step + 1

    total_lookups = result.total_hits + result.total_misses
    result.overall_hit_rate = result.total_hits / total_lookups if total_lookups else 0.0
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a compact side-by-side comparison table."""
    header = (
        f"{'Policy':<12} {'Dataset':<12} {'Model':<30} "
        f"{'Hit%':>7} {'Recompute':>10} {'Evictions':>10} "
        f"{'LRU≠score':>10} {'CommonHit%':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        # Aggregate across intervals
        lru_differ = sum(
            iv.get("lru_policy_would_differ", 0) for iv in r.intervals
        )
        common_hits = sum(iv.get("hits_from_common_pool", 0) for iv in r.intervals)
        total_lookups = r.total_hits + r.total_misses
        common_frac = common_hits / total_lookups if total_lookups else 0.0

        short_model = r.model.split("/")[-1][:28]
        print(
            f"{r.policy:<12} {r.dataset:<12} {short_model:<30} "
            f"{_fmt_pct(r.overall_hit_rate):>7} "
            f"{r.total_recompute_blocks:>10,} "
            f"{r.total_evictions:>10,} "
            f"{lru_differ:>10,} "
            f"{_fmt_pct(common_frac):>12}"
        )
    print("=" * len(header))


def print_breakdown_sample(result: BenchmarkResult, n: int = 5) -> None:
    """Print the last n intervals for a single result."""
    print(f"\n── Interval breakdown (last {n}) — {result.policy} / {result.dataset}")
    print(
        f"  {'Interval':>12} {'HitRate':>8} {'CmnHit%':>8} "
        f"{'Evict':>7} {'Demote':>7} {'RecompBlk':>10} "
        f"{'AvgAccCnt':>10} {'AvgChain':>9} {'LRU≠':>6}"
    )
    for iv in result.intervals[-n:]:
        print(
            f"  {iv['interval_start']:>6}-{iv['interval_end']:<5} "
            f"{_fmt_pct(iv['hit_rate']):>8} "
            f"{_fmt_pct(iv['common_pool_hit_fraction']):>8} "
            f"{iv['evictions_total']:>7} "
            f"{iv['demotions_common_to_cached']:>7} "
            f"{iv['total_recompute_blocks']:>10} "
            f"{iv['avg_evicted_access_count']:>10.2f} "
            f"{iv['avg_evicted_chain_blocks']:>9.1f} "
            f"{iv['lru_policy_would_differ']:>6}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.FlexibleArgumentParser(
        description="Dual-pool eviction policy benchmark"
    ) if False else argparse.ArgumentParser(
        description="Dual-pool eviction policy benchmark"
    )
    p.add_argument(
        "--model", nargs="+",
        default=["google/gemma-2-9b-it"],
        help="HuggingFace model IDs (used for tokenisation)",
    )
    p.add_argument(
        "--datasets", nargs="+",
        choices=["sharegpt", "arxiv_qa", "mixed_hot_cold"],
        default=["sharegpt", "mixed_hot_cold"],
    )
    p.add_argument("--sharegpt-path", default=None,
                   help="Local path to ShareGPT JSON file")
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--num-blocks", type=int, default=4000,
                   help="Total GPU cache blocks to simulate")
    p.add_argument("--block-size", type=int, default=16,
                   help="Tokens per block (should match vllm config)")
    p.add_argument("--interval", type=int, default=100,
                   help="Collect breakdown stats every N requests")
    p.add_argument("--seed", type=int, default=42)
    # Dual-pool knobs
    p.add_argument("--common-pool-min-access", type=int, default=2)
    p.add_argument("--common-pool-fraction", type=float, default=0.2,
                   help="Fraction of blocks reserved for common pool")
    p.add_argument("--w-lru", type=float, default=0.4)
    p.add_argument("--w-lfu", type=float, default=0.3)
    p.add_argument("--w-cost", type=float, default=0.3)
    p.add_argument("--output", default=None,
                   help="Write JSON results to this file")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="Skip tokenisation (use synthetic token lengths)")
    p.add_argument("--policies", nargs="+",
                   choices=["lru", "dual_pool"],
                   default=["lru", "dual_pool"],
                   help="Policies to benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    all_results: list[BenchmarkResult] = []

    for model_id in args.model:
        # Load tokenizer
        tokenizer = None
        if not args.no_tokenizer:
            try:
                from transformers import AutoTokenizer  # type: ignore
                print(f"Loading tokenizer for {model_id} ...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                print("Tokenizer ready.")
            except Exception as e:
                print(f"[warn] Could not load tokenizer ({e}). Using synthetic lengths.")

        for dataset_name in args.datasets:
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name}  |  Model: {model_id}")
            print("=" * 60)

            # ── Load dataset ─────────────────────────────────────────────
            if dataset_name == "sharegpt":
                if args.sharegpt_path is None:
                    print("[warn] --sharegpt-path not provided; using synthetic data.")
                    # Generate FIXED tokens per prefix_id so hashes are
                    # identical across requests with the same prefix.
                    # Use Zipf access: ~20% of prefixes get ~80% of traffic
                    # (power-law with exponent 1.2), mirroring real workloads.
                    num_prefixes = 100
                    rng_sg = random.Random(args.seed)
                    prefix_tokens_map = {
                        pid: _synthetic_tokens(rng_sg.randint(64, 512))
                        for pid in range(num_prefixes)
                    }
                    # Zipf weights: prefix 0 is hottest
                    zipf_weights = [1.0 / (i + 1) ** 1.2
                                    for i in range(num_prefixes)]
                    total_w = sum(zipf_weights)
                    zipf_probs = [w / total_w for w in zipf_weights]
                    requests = []
                    for _ in range(args.num_requests):
                        pid = rng_sg.choices(
                            range(num_prefixes), weights=zipf_probs, k=1
                        )[0]
                        pid_str = f"sg-{pid}"
                        ctx_ids = prefix_tokens_map[pid]
                        qry_ids = _synthetic_tokens(rng_sg.randint(16, 64))
                        ctx_hashes = _tokens_to_block_hashes(
                            ctx_ids, args.block_size,
                            prefix_salt=pid_str.encode(),
                        )
                        if not ctx_hashes:
                            continue
                        requests.append(WorkloadRequest(
                            prefix_id=pid_str,
                            prefix_block_hashes=ctx_hashes,
                            query_block_hashes=_tokens_to_block_hashes(
                                qry_ids, args.block_size,
                            ),
                            num_context_tokens=len(ctx_ids),
                            num_query_tokens=len(qry_ids),
                        ))
                else:
                    requests = load_sharegpt(
                        args.sharegpt_path, tokenizer,
                        args.num_requests, args.block_size, args.seed,
                    )
            elif dataset_name == "arxiv_qa":
                requests = load_arxiv_qa(
                    tokenizer, args.num_requests, args.block_size,
                    num_blocks=args.num_blocks, seed=args.seed,
                )
            else:  # mixed_hot_cold
                requests = load_mixed_hot_cold(
                    block_size=args.block_size,
                    num_blocks=args.num_blocks,
                    num_requests=args.num_requests,
                    seed=args.seed,
                )

            if not requests:
                print("[warn] No requests generated, skipping.")
                continue

            print(
                f"Loaded {len(requests)} requests, "
                f"{len({r.prefix_id for r in requests})} unique prefixes."
            )
            avg_ctx = sum(len(r.prefix_block_hashes) for r in requests) / len(requests)
            print(f"Avg prefix blocks: {avg_ctx:.1f}  (block_size={args.block_size})")

            # ── Run each policy ──────────────────────────────────────────
            for policy in args.policies:
                print(f"\nRunning policy: {policy} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                result = run_benchmark(
                    policy=policy,
                    requests=requests,
                    model=model_id,
                    dataset=dataset_name,
                    num_blocks=args.num_blocks,
                    block_size=args.block_size,
                    interval=args.interval,
                    common_pool_min_access=args.common_pool_min_access,
                    common_pool_fraction=args.common_pool_fraction,
                    w_lru=args.w_lru,
                    w_lfu=args.w_lfu,
                    w_cost=args.w_cost,
                )
                elapsed = time.perf_counter() - t0
                print(f"done in {elapsed:.1f}s")
                print(
                    f"  hit_rate={_fmt_pct(result.overall_hit_rate)}  "
                    f"recompute_blocks={result.total_recompute_blocks:,}  "
                    f"evictions={result.total_evictions:,}"
                )
                all_results.append(result)

            # Print breakdown for this (dataset, model) pair
            for policy in args.policies:
                matching = [
                    r for r in all_results
                    if r.policy == policy
                    and r.dataset == dataset_name
                    and r.model == model_id
                ]
                if matching:
                    print_breakdown_sample(matching[-1], n=5)

    # ── Final comparison table ───────────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison(all_results)

    # ── Improvement summary ──────────────────────────────────────────────────
    print("\n── Dual-pool improvement over LRU ─────────────────────────────")
    for model_id in args.model:
        for dataset_name in args.datasets:
            lru_r = next(
                (r for r in all_results
                 if r.policy == "lru" and r.dataset == dataset_name and r.model == model_id),
                None,
            )
            dp_r = next(
                (r for r in all_results
                 if r.policy == "dual_pool" and r.dataset == dataset_name and r.model == model_id),
                None,
            )
            if lru_r and dp_r:
                hit_delta = (dp_r.overall_hit_rate - lru_r.overall_hit_rate) * 100
                recompute_delta = lru_r.total_recompute_blocks - dp_r.total_recompute_blocks
                recompute_pct = (
                    recompute_delta / lru_r.total_recompute_blocks * 100
                    if lru_r.total_recompute_blocks else 0.0
                )
                print(
                    f"  {dataset_name:<12} {model_id.split('/')[-1]:<28} "
                    f"hit_rate: {'+' if hit_delta >= 0 else ''}{hit_delta:.2f}pp  "
                    f"recompute: {'+' if recompute_delta >= 0 else ''}{recompute_delta:,} "
                    f"blocks ({recompute_pct:.1f}% {'saved' if recompute_delta >= 0 else 'worse'})"
                )

    # ── Save JSON ────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps([asdict(r) for r in all_results], indent=2)
        )
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
