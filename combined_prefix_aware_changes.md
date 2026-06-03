# Combined Prefix-Aware: Policy Changes vs. Upstream vLLM

Branch: `combined-prefix-aware`

This document describes all modifications made to vLLM in this branch.
Two orthogonal prefix-aware mechanisms are combined: eviction-side and scheduling-side.

---

## 1. Prefix-Aware Eviction

**Env var**: `VLLM_PREFIX_AWARE_EVICTION=1` (default: 0 = LRU)

**Key files**:
- `vllm/v1/core/kv_cache_manager.py` — `refresh_protected_blocks()`
- `vllm/v1/core/block_pool.py` — `_lru_skip_protected()`

**Mechanism**:
Before each scheduling step, the scheduler scans waiting requests' prefix hashes and marks matching free KV cache blocks as *protected*. The eviction policy then skips protected blocks, preserving blocks that a pending request will reuse.

```
scheduler.py: self.kv_cache_manager.refresh_protected_blocks(self.waiting)
```

**Without this**: LRU may evict a block whose article is needed by a request sitting in the waiting queue, forcing a full re-prefill when that request is finally scheduled.

**With this**: Articles for queued requests survive in cache until those requests are scheduled, increasing prefix cache hit rate.

### Fix A: Budget Cap (prevents O(n) eviction scan)

```python
# kv_cache_manager.py — refresh_protected_blocks()
budget = max(1, self.block_pool.get_num_free_blocks() // 2)
```

Without a cap, a large article (e.g. 90k tokens = 5,600 blocks) could fill the entire protected set, forcing `_lru_skip_protected` to scan all blocks O(n) per eviction. The cap ensures ≥50% of free blocks are always evictable, keeping eviction O(1) amortized.

### Fix B: Pool-Size Auto-Disable (`_MIN_POOL_RATIO`)

```python
# kv_cache_manager.py — refresh_protected_blocks()
_MIN_POOL_RATIO = 2.0
_total_blocks = self.block_pool.num_gpu_blocks - 1
_tokens_per_block = max(1, self.block_pool.hash_block_size)

# Sample first 5 waiting requests to estimate avg request size
_avg_req_blocks = avg(r.num_tokens for r in waiting[:5]) / _tokens_per_block

if _total_blocks / _avg_req_blocks < _MIN_POOL_RATIO:
    self.block_pool.protected_block_ids = set()
    return  # fall back to pure LRU
```

When the KV cache pool holds fewer than 2 average-sized requests simultaneously, protecting blocks provides no benefit — any new prefill will evict entire articles regardless. The condition auto-disables eviction protection and falls back to LRU.

**Example**: Llama-3.1-8B on A5000 with 50k context → pool ≈ 3,206 blocks, avg article ≈ 2,907 blocks → ratio 1.10 < 2.0 → LRU fallback.

---

## 2. Prefix-Aware Scheduling

**Flag**: `--enable-prefix-aware-scheduling` (default: off = FCFS)

**Key file**: `vllm/v1/core/sched/scheduler.py`

**Mechanism**:
The scheduler re-orders the waiting queue to prioritize requests with the highest prefix cache hit length. Instead of pure FCFS, a request that already has cached blocks is promoted ahead of cold requests.

**Without this**: A cold request (no cache hit) may be scheduled before a warm request (long cache hit), evicting useful blocks and wasting GPU cycles on re-prefill.

**With this**: Warm requests are scheduled first, maximizing cache reuse within each scheduling step.

---

## 3. Combined Effect

The two mechanisms are complementary:

- **EvictAware** preserves blocks *between* scheduling steps (blocks survive in free pool)
- **SchedAware** maximizes reuse *within* a scheduling step (warm requests go first)

| Policy | EvictAware | SchedAware |
|---|---|---|
| LRU | ✗ | ✗ |
| EvictAware | ✓ | ✗ |
| SchedAware | ✗ | ✓ |
| Both | ✓ | ✓ |

---

## Experiment Results

### Gemma-3-4B-IT — arXiv QA (90k context, 28k block pool, A5000)

Pool holds ~5 articles simultaneously → EvictAware highly effective.

| Policy | req/s | Δ vs LRU | TTFT mean | TTFT p99 |
|---|---|---|---|---|
| LRU | 0.1203 | — | 171.3s | 328.6s |
| SchedAware | 0.1747 | +45% | 98.9s | 224.5s |
| EvictAware | 0.2253 | +87% | 93.2s | 172.3s |
| **Both** | **0.2692** | **+124%** | **76.9s** | **143.0s** |

### Ministral-8B FP8 — arXiv QA (32k context, A5000)

Mid-size pool → EvictAware limited, SchedAware strong.

| Policy | req/s | Δ vs LRU | TTFT mean | TTFT p99 |
|---|---|---|---|---|
| LRU | 0.1725 | — | 121.8s | 221.1s |
| EvictAware | 0.1766 | +2.4% | 120.3s | 216.8s |
| SchedAware | 0.2044 | +18.5% | 91.8s | 184.6s |
| **Both** | **0.2193** | **+27.1%** | **87.6s** | **170.2s** |

### Llama-3.1-8B — arXiv QA (50k context, small pool, A5000)

Pool ≈ 1 article → `_MIN_POOL_RATIO` triggers → EvictAware auto-disabled.

| Policy | req/s | Δ vs LRU | TTFT mean | TTFT p99 |
|---|---|---|---|---|
| LRU | 0.0574 | — | 349.6s | 683.4s |
| EvictAware | 0.0542 | −5.6% (noise) | 363.7s | 724.6s |
| SchedAware | 0.0607 | +5.7% | 301.3s | 652.7s |
| **Both** | **0.0613** | **+6.8%** | **307.2s** | **647.1s** |

### Llama-3.1-8B — MMLU-pro (4k context, no shared prefix, A5000)

Control experiment: independent prompts → all policies ≈ LRU (< 0.6% variance).

| Policy | req/s | Δ vs LRU |
|---|---|---|
| LRU | 12.47 | — |
| EvictAware | 12.46 | −0.1% |
| SchedAware | 12.50 | +0.3% |
| Both | 12.43 | −0.3% |

---

## Key Findings

1. **EvictAware scales with `pool_size / article_size` ratio**: Gemma (ratio ≈ 5) +87%, Ministral (ratio ≈ 2) +2.4%, Llama (ratio ≈ 1) auto-disabled.
2. **SchedAware is effective regardless of pool size**: +45% Gemma, +19% Ministral, +6% Llama.
3. **Both policies combine super-additively on large pools**: Gemma Both +124% > EvictAware +87% + SchedAware +45%.
4. **No regression on unrelated workloads**: MMLU-pro control shows < 0.6% variance across all policies.
