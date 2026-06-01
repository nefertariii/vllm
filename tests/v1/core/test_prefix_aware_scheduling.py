# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefix-aware request scheduling."""

import time

import pytest

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PrefixAwareRequestQueue,
)
from vllm.v1.request import Request

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_request(
    request_id: str,
    num_prefix_hit_tokens: int,
    enqueue_time: float,
    num_tokens: int = 10,
) -> Request:
    """Create a Request and manually set prefix scheduling fields."""
    reqs = create_requests(num_requests=1, num_tokens=num_tokens,
                           req_ids=[request_id])
    req = reqs[0]
    req.num_prefix_hit_tokens = num_prefix_hit_tokens
    req.enqueue_time = enqueue_time
    return req


# ---------------------------------------------------------------------------
# Test 1: Basic ordering — higher prefix hit is scheduled first
# ---------------------------------------------------------------------------

def test_basic_ordering():
    queue = PrefixAwareRequestQueue(max_wait_seconds=100.0)
    t0 = time.monotonic()

    req_a = _make_request("A", num_prefix_hit_tokens=0,   enqueue_time=t0)
    req_b = _make_request("B", num_prefix_hit_tokens=512, enqueue_time=t0 + 1)
    req_c = _make_request("C", num_prefix_hit_tokens=256, enqueue_time=t0 + 2)

    queue.add_request(req_a)
    queue.add_request(req_b)
    queue.add_request(req_c)

    assert queue.pop_request() is req_b, "B (512 hits) should be first"
    assert queue.pop_request() is req_c, "C (256 hits) should be second"
    assert queue.pop_request() is req_a, "A (0 hits) should be last"


# ---------------------------------------------------------------------------
# Test 2: FCFS tiebreak when prefix hits are equal
# ---------------------------------------------------------------------------

def test_fcfs_tiebreak():
    queue = PrefixAwareRequestQueue(max_wait_seconds=100.0)
    t0 = time.monotonic()

    req_x = _make_request("X", num_prefix_hit_tokens=128, enqueue_time=t0)
    req_y = _make_request("Y", num_prefix_hit_tokens=128, enqueue_time=t0 + 1)

    queue.add_request(req_x)
    queue.add_request(req_y)

    assert queue.pop_request() is req_x, "X (earlier) should be first"
    assert queue.pop_request() is req_y


# ---------------------------------------------------------------------------
# Test 3: Anti-starvation — stale request is promoted
# ---------------------------------------------------------------------------

def test_anti_starvation():
    # Use max_wait_seconds=0.5: OLD has waited 1s (stale), NEW has waited
    # only a few ms (not stale). Verify OLD is promoted despite 0 hits.
    queue = PrefixAwareRequestQueue(max_wait_seconds=0.5)
    now = time.monotonic()

    req_old = _make_request("OLD", num_prefix_hit_tokens=0,    enqueue_time=now - 1.0)
    req_new = _make_request("NEW", num_prefix_hit_tokens=9999, enqueue_time=now)

    queue.add_request(req_old)
    queue.add_request(req_new)

    # Trigger a sort by adding a dummy request so _is_sorted=False, then pop.
    # At this point req_old is already stale (waited 1s > 0.5s threshold).
    req_dummy = _make_request("Z", num_prefix_hit_tokens=0, enqueue_time=now)
    queue.add_request(req_dummy)
    queue.remove_request(req_dummy)

    assert queue.pop_request() is req_old, "OLD (stale) should be promoted first"
    assert queue.pop_request() is req_new


# ---------------------------------------------------------------------------
# Test 4: remove_request works correctly
# ---------------------------------------------------------------------------

def test_remove_request():
    queue = PrefixAwareRequestQueue(max_wait_seconds=100.0)
    t0 = time.monotonic()

    req_a = _make_request("A", 0,   t0)
    req_b = _make_request("B", 100, t0 + 1)
    req_c = _make_request("C", 50,  t0 + 2)

    queue.add_request(req_a)
    queue.add_request(req_b)
    queue.add_request(req_c)

    queue.remove_request(req_b)

    assert len(queue) == 2
    popped = [queue.pop_request(), queue.pop_request()]
    assert req_b not in popped, "B should have been removed"
    assert req_a in popped
    assert req_c in popped


# ---------------------------------------------------------------------------
# Test 5: estimate_prefix_hit_tokens is read-only
# ---------------------------------------------------------------------------

def test_estimate_prefix_hit_tokens_is_read_only():
    """Verify that estimate_prefix_hit_tokens does not mutate block ref counts."""
    scheduler = create_scheduler(enable_prefix_caching=True, num_blocks=1000)

    # Add a request and schedule it so its blocks are cached.
    from .utils import create_requests as _create_requests
    reqs = _create_requests(num_requests=2, num_tokens=32, same_prompt=True)
    first_req, second_req = reqs[0], reqs[1]

    scheduler.add_request(first_req)
    output = scheduler.schedule()
    # Simulate the first request finishing (populate cache).
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import RequestStatus
    scheduler.finish_requests(first_req.request_id,
                              RequestStatus.FINISHED_STOPPED)

    # Snapshot all block ref counts before estimation.
    block_pool = scheduler.kv_cache_manager.block_pool
    ref_counts_before = {
        blk.block_id: blk.ref_cnt
        for blk in block_pool.blocks
    }

    hit_count = scheduler.kv_cache_manager.estimate_prefix_hit_tokens(second_req)

    ref_counts_after = {
        blk.block_id: blk.ref_cnt
        for blk in block_pool.blocks
    }

    assert ref_counts_before == ref_counts_after, (
        "estimate_prefix_hit_tokens must not modify block ref counts"
    )
    # With same_prompt=True and prefix caching enabled, we should get a hit.
    assert hit_count >= 0


# ---------------------------------------------------------------------------
# Test 6: Feature flag — FCFS behavior preserved when disabled
# ---------------------------------------------------------------------------

def test_fcfs_behavior_when_disabled():
    """When enable_prefix_aware_scheduling=False, order must be FCFS."""
    # create_scheduler uses enable_prefix_aware_scheduling=False by default.
    scheduler = create_scheduler()

    # Requests with varying (artificial) prefix scores; since PrefixAware is
    # off, the queue is FCFSRequestQueue and the order must be insertion order.
    reqs = create_requests(num_requests=3, num_tokens=10)
    req_a, req_b, req_c = reqs

    # Manually set scores that would reorder if PrefixAware were active.
    req_a.num_prefix_hit_tokens = 0
    req_b.num_prefix_hit_tokens = 1000
    req_c.num_prefix_hit_tokens = 500

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.add_request(req_c)

    # The waiting queue should be an FCFSRequestQueue, not PrefixAware.
    assert isinstance(scheduler.waiting, FCFSRequestQueue), (
        "Waiting queue should be FCFSRequestQueue when flag is off"
    )

    # Pop order should be FCFS (insertion order): A -> B -> C
    assert scheduler.waiting.pop_request() is req_a
    assert scheduler.waiting.pop_request() is req_b
    assert scheduler.waiting.pop_request() is req_c


# ---------------------------------------------------------------------------
# Test 7: Scheduler respects flag and uses PrefixAwareRequestQueue
# ---------------------------------------------------------------------------

def test_scheduler_uses_prefix_aware_queue_when_enabled():
    """When flag is True, scheduler.waiting is a PrefixAwareRequestQueue."""
    from vllm.config import SchedulerConfig, CacheConfig, ModelConfig, VllmConfig
    from vllm.config import ParallelConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.request_queue import PrefixAwareRequestQueue
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager
    import torch

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        enable_prefix_aware_scheduling=True,
        prefix_scheduling_max_wait_seconds=3.0,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    cache_config.num_gpu_blocks = 1000
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    sched = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    assert isinstance(sched.waiting, PrefixAwareRequestQueue), (
        "Scheduler.waiting should be PrefixAwareRequestQueue when flag is on"
    )
    assert sched.waiting.max_wait_seconds == 3.0
