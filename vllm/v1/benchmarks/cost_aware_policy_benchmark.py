# SPDX-License-Identifier: Apache-2.0
"""
Ablation microbenchmark for cost-aware KV cache eviction.

Convention used in this benchmark:
    score = keep/protection score
    lower score = evict first
    higher score = keep longer

This matches the desired design:
    low score evicted first.

The benchmark is intentionally engine-free. It tests eviction-policy logic,
not full vLLM serving throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import numpy as np


PolicyName = Literal[
    "lru",
    "mru",
    "cost_only",
    "recency_only",
    "frequency_only",
    "recency_cost",
    "frequency_cost",
    "recency_frequency",
    "full_cost_aware",
    "dual_pool_lru",
    "dual_pool_cost_only",
    "dual_pool_full",
]


@dataclass
class CacheEntry:
    prefix_id: int
    token_count: int
    reuse_probability: float
    created_at_step: int
    last_access_step: int
    access_count: int = 1
    pool: Literal["single", "common", "cached"] = "single"


@dataclass
class BenchResult:
    workload: str
    policy_name: str
    num_requests: int
    num_blocks: int
    unique_prefixes: int

    cache_hits: int
    cache_misses: int
    cache_hit_rate: float

    total_evictions: int
    total_recompute_tokens: int
    avg_evicted_tokens: float

    short_evictions: int
    medium_evictions: int
    long_evictions: int
    common_pool_evictions: int
    cached_pool_evictions: int

    final_common_pool_size: int
    final_cached_pool_size: int

    total_time_seconds: float


def get_git_info() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    return {
        "branch": run(["git", "branch", "--show-current"]) or "detached",
        "commit": run(["git", "rev-parse", "HEAD"]),
        "describe": run(["git", "describe", "--tags", "--always", "--dirty"]),
        "status_short": run(["git", "status", "--short"]),
    }


def generate_requests(workload: str, num_requests: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    requests: list[dict] = []

    if workload == "uniform":
        for i in range(num_requests):
            requests.append({
                "prefix_id": i % 500,
                "token_count": 512,
                "reuse_probability": 0.30,
            })

    elif workload == "long_tail":
        for i in range(num_requests):
            if i % 20 == 0:
                requests.append({
                    "prefix_id": 10_000 + (i % 50),
                    "token_count": int(rng.integers(5000, 15000)),
                    "reuse_probability": 0.80,
                })
            else:
                requests.append({
                    "prefix_id": i % 700,
                    "token_count": int(rng.integers(100, 1000)),
                    "reuse_probability": 0.20,
                })

    elif workload == "mixed":
        for i in range(num_requests):
            r = i % 100
            if r < 10:
                requests.append({
                    "prefix_id": 20_000 + (i % 40),
                    "token_count": int(rng.integers(8000, 20000)),
                    "reuse_probability": 0.85,
                })
            elif r < 25:
                requests.append({
                    "prefix_id": 30_000 + (i % 30),
                    "token_count": 128,
                    "reuse_probability": 0.95,
                })
            else:
                requests.append({
                    "prefix_id": i % 800,
                    "token_count": int(rng.integers(500, 2500)),
                    "reuse_probability": 0.35,
                })

    elif workload == "adversarial":
        hot = [
            {
                "prefix_id": 40_000 + j,
                "token_count": int(rng.integers(10000, 24000)),
                "reuse_probability": 0.95,
            }
            for j in range(60)
        ]

        cold_scan = [
            {
                "prefix_id": 50_000 + j,
                "token_count": int(rng.integers(256, 2048)),
                "reuse_probability": 0.05,
            }
            for j in range(num_requests)
        ]

        while len(requests) < num_requests:
            requests.extend(hot)
            requests.extend(cold_scan[:300])
            requests.extend(hot)

        requests = requests[:num_requests]

    else:
        raise ValueError(f"Unknown workload: {workload}")

    return requests


def normalize_positive(x: float, scale: float) -> float:
    return min(1.0, max(0.0, x / scale))


def keep_score(
    entry: CacheEntry,
    step: int,
    policy: PolicyName,
    max_token_seen: int,
) -> float:
    """
    Return keep/protection score.
    Lower score = evict first.
    Higher score = keep longer.
    """
    age_since_access = step - entry.last_access_step

    # Recently used blocks should be kept.
    # If age is small, recency_keep is high.
    recency_keep = 1.0 / (1.0 + age_since_access)

    # Frequently accessed blocks should be kept.
    frequency_keep = normalize_positive(entry.access_count, 10.0)

    # Expensive long-prefix blocks should be kept.
    denom = max(1, max_token_seen)
    cost_keep = normalize_positive(entry.token_count, denom)

    if policy in ("lru", "dual_pool_lru"):
        # Older last_access_step is smaller, so low score evicts oldest first.
        return float(entry.last_access_step)

    if policy == "mru":
        # Newer last_access_step should be evicted first.
        # Low score evicts first, so negate.
        return -float(entry.last_access_step)

    if policy in ("cost_only", "dual_pool_cost_only"):
        return cost_keep

    if policy == "recency_only":
        return recency_keep

    if policy == "frequency_only":
        return frequency_keep

    if policy == "recency_cost":
        return 0.5 * recency_keep + 0.5 * cost_keep

    if policy == "frequency_cost":
        return 0.5 * frequency_keep + 0.5 * cost_keep

    if policy == "recency_frequency":
        return 0.5 * recency_keep + 0.5 * frequency_keep

    if policy in ("full_cost_aware", "dual_pool_full"):
        return (
            0.30 * recency_keep
            + 0.40 * frequency_keep
            + 0.30 * cost_keep
        )

    raise ValueError(f"Unknown policy: {policy}")


def uses_dual_pool(policy: PolicyName) -> bool:
    return policy.startswith("dual_pool_")


def is_common_candidate(
    req: dict,
    common_reuse_threshold: float,
    common_token_threshold: int,
) -> bool:
    return (
        req["reuse_probability"] >= common_reuse_threshold
        and req["token_count"] >= common_token_threshold
    )


def classify_token_size(token_count: int) -> str:
    if token_count < 1000:
        return "short"
    if token_count < 5000:
        return "medium"
    return "long"


def run_policy(
    workload: str,
    requests: list[dict],
    policy: PolicyName,
    num_blocks: int,
    common_pool_ratio: float,
    common_reuse_threshold: float,
    common_token_threshold: int,
) -> BenchResult:
    start = time.time()
    max_token_seen = max(r["token_count"] for r in requests)

    single_cache: dict[int, CacheEntry] = {}
    common_pool: dict[int, CacheEntry] = {}
    cached_pool: dict[int, CacheEntry] = {}

    max_common_pool_size = max(1, int(num_blocks * common_pool_ratio))

    hits = 0
    evictions = 0
    recompute_tokens = 0
    evicted_tokens: list[int] = []

    short_evictions = 0
    medium_evictions = 0
    long_evictions = 0
    common_pool_evictions = 0
    cached_pool_evictions = 0

    def total_cached() -> int:
        if uses_dual_pool(policy):
            return len(common_pool) + len(cached_pool)
        return len(single_cache)

    def evict_from(pool: dict[int, CacheEntry], step: int) -> None:
        nonlocal evictions
        nonlocal recompute_tokens
        nonlocal short_evictions
        nonlocal medium_evictions
        nonlocal long_evictions
        nonlocal common_pool_evictions
        nonlocal cached_pool_evictions

        victim_id = min(
            pool,
            key=lambda k: keep_score(pool[k], step, policy, max_token_seen),
        )
        victim = pool.pop(victim_id)

        evictions += 1
        recompute_tokens += victim.token_count
        evicted_tokens.append(victim.token_count)

        size_class = classify_token_size(victim.token_count)
        if size_class == "short":
            short_evictions += 1
        elif size_class == "medium":
            medium_evictions += 1
        else:
            long_evictions += 1

        if victim.pool == "common":
            common_pool_evictions += 1
        elif victim.pool == "cached":
            cached_pool_evictions += 1

    for step, req in enumerate(requests):
        prefix_id = req["prefix_id"]

        if uses_dual_pool(policy):
            if prefix_id in common_pool:
                hits += 1
                ent = common_pool[prefix_id]
                ent.last_access_step = step
                ent.access_count += 1
                continue

            if prefix_id in cached_pool:
                hits += 1
                ent = cached_pool[prefix_id]
                ent.last_access_step = step
                ent.access_count += 1
                continue

            while total_cached() >= num_blocks:
                # Dual-pool rule: evict cached pool first; common pool only as fallback.
                if cached_pool:
                    evict_from(cached_pool, step)
                else:
                    evict_from(common_pool, step)

            if is_common_candidate(
                req,
                common_reuse_threshold=common_reuse_threshold,
                common_token_threshold=common_token_threshold,
            ):
                if len(common_pool) >= max_common_pool_size:
                    # Demote lowest-keep common entry to cached pool.
                    demote_id = min(
                        common_pool,
                        key=lambda k: keep_score(
                            common_pool[k],
                            step,
                            policy,
                            max_token_seen,
                        ),
                    )
                    demoted = common_pool.pop(demote_id)
                    demoted.pool = "cached"
                    cached_pool[demote_id] = demoted

                common_pool[prefix_id] = CacheEntry(
                    prefix_id=prefix_id,
                    token_count=req["token_count"],
                    reuse_probability=req["reuse_probability"],
                    created_at_step=step,
                    last_access_step=step,
                    pool="common",
                )
            else:
                cached_pool[prefix_id] = CacheEntry(
                    prefix_id=prefix_id,
                    token_count=req["token_count"],
                    reuse_probability=req["reuse_probability"],
                    created_at_step=step,
                    last_access_step=step,
                    pool="cached",
                )

        else:
            if prefix_id in single_cache:
                hits += 1
                ent = single_cache[prefix_id]
                ent.last_access_step = step
                ent.access_count += 1
                continue

            while total_cached() >= num_blocks:
                evict_from(single_cache, step)

            single_cache[prefix_id] = CacheEntry(
                prefix_id=prefix_id,
                token_count=req["token_count"],
                reuse_probability=req["reuse_probability"],
                created_at_step=step,
                last_access_step=step,
                pool="single",
            )

    misses = len(requests) - hits

    return BenchResult(
        workload=workload,
        policy_name=policy,
        num_requests=len(requests),
        num_blocks=num_blocks,
        unique_prefixes=len({r["prefix_id"] for r in requests}),
        cache_hits=hits,
        cache_misses=misses,
        cache_hit_rate=hits / len(requests) if requests else 0.0,
        total_evictions=evictions,
        total_recompute_tokens=recompute_tokens,
        avg_evicted_tokens=float(np.mean(evicted_tokens)) if evicted_tokens else 0.0,
        short_evictions=short_evictions,
        medium_evictions=medium_evictions,
        long_evictions=long_evictions,
        common_pool_evictions=common_pool_evictions,
        cached_pool_evictions=cached_pool_evictions,
        final_common_pool_size=len(common_pool),
        final_cached_pool_size=len(cached_pool),
        total_time_seconds=time.time() - start,
    )


def summarize_vs_lru(results: list[BenchResult]) -> list[dict]:
    grouped: dict[str, list[BenchResult]] = {}
    for r in results:
        grouped.setdefault(r.workload, []).append(r)

    summaries = []
    for workload, rs in grouped.items():
        base = next(r for r in rs if r.policy_name == "lru")
        for r in rs:
            if r.policy_name == "lru":
                continue

            summaries.append({
                "workload": workload,
                "policy_name": r.policy_name,
                "hit_rate_delta_abs": r.cache_hit_rate - base.cache_hit_rate,
                "hit_rate_delta_percentage_points": (
                    r.cache_hit_rate - base.cache_hit_rate
                ) * 100.0,
                "recompute_token_reduction_pct": (
                    (base.total_recompute_tokens - r.total_recompute_tokens)
                    / base.total_recompute_tokens
                    * 100.0
                    if base.total_recompute_tokens else 0.0
                ),
                "eviction_reduction_pct": (
                    (base.total_evictions - r.total_evictions)
                    / base.total_evictions
                    * 100.0
                    if base.total_evictions else 0.0
                ),
                "long_eviction_reduction_pct": (
                    (base.long_evictions - r.long_evictions)
                    / base.long_evictions
                    * 100.0
                    if base.long_evictions else 0.0
                ),
                "time_overhead_ratio": (
                    r.total_time_seconds / base.total_time_seconds
                    if base.total_time_seconds else None
                ),
            })

    return summaries


def print_table(results: list[BenchResult]) -> None:
    by_workload: dict[str, list[BenchResult]] = {}
    for r in results:
        by_workload.setdefault(r.workload, []).append(r)

    for workload, rs in by_workload.items():
        print("\n" + "=" * 120)
        print(f"WORKLOAD: {workload}")
        print("=" * 120)
        print(
            f"{'policy':<26} {'hit%':>8} {'evict':>7} "
            f"{'recompute':>12} {'shortE':>7} {'medE':>7} {'longE':>7} "
            f"{'commonE':>8} {'time':>9}"
        )
        print("-" * 120)
        for r in rs:
            print(
                f"{r.policy_name:<26} "
                f"{100*r.cache_hit_rate:>7.2f}% "
                f"{r.total_evictions:>7} "
                f"{r.total_recompute_tokens:>12} "
                f"{r.short_evictions:>7} "
                f"{r.medium_evictions:>7} "
                f"{r.long_evictions:>7} "
                f"{r.common_pool_evictions:>8} "
                f"{r.total_time_seconds:>8.4f}s"
            )


def write_csv(path: str, results: list[BenchResult]) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=5000)
    parser.add_argument("--num-blocks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-pool-ratio", type=float, default=0.20)
    parser.add_argument("--common-reuse-threshold", type=float, default=0.60)
    parser.add_argument("--common-token-threshold", type=int, default=512)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["uniform", "long_tail", "mixed", "adversarial"],
        choices=["uniform", "long_tail", "mixed", "adversarial"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "lru",
            "mru",
            "cost_only",
            "recency_only",
            "frequency_only",
            "recency_cost",
            "frequency_cost",
            "recency_frequency",
            "full_cost_aware",
            "dual_pool_lru",
            "dual_pool_cost_only",
            "dual_pool_full",
        ],
    )
    parser.add_argument("--output-json", default="cost_aware_ablation_results.json")
    parser.add_argument("--output-csv", default="cost_aware_ablation_results.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    results: list[BenchResult] = []

    for workload in args.workloads:
        requests = generate_requests(workload, args.num_requests, args.seed)

        for policy in args.policies:
            result = run_policy(
                workload=workload,
                requests=requests,
                policy=policy,
                num_blocks=args.num_blocks,
                common_pool_ratio=args.common_pool_ratio,
                common_reuse_threshold=args.common_reuse_threshold,
                common_token_threshold=args.common_token_threshold,
            )
            results.append(result)

    output = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "config": {
            "num_requests": args.num_requests,
            "num_blocks": args.num_blocks,
            "seed": args.seed,
            "common_pool_ratio": args.common_pool_ratio,
            "common_reuse_threshold": args.common_reuse_threshold,
            "common_token_threshold": args.common_token_threshold,
            "workloads": args.workloads,
            "policies": args.policies,
            "score_convention": "lower score evicted first; higher score kept longer",
        },
        "results": [asdict(r) for r in results],
        "summaries_vs_lru": summarize_vs_lru(results),
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    write_csv(args.output_csv, results)
    print_table(results)
    print(f"\nWrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
