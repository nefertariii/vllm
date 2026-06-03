#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LRU vs Dual-Pool Cost-Aware KV Cache Eviction Benchmark
=========================================================

Dataset
-------
    ccdv/arxiv-summarization
    HuggingFace : https://huggingface.co/datasets/ccdv/arxiv-summarization
    Paper       : Cohan et al. (2018), "A Discourse-Aware Attention Model for
                  Abstractive Summarization of Long Documents", NAACL 2018.
                  https://arxiv.org/abs/1804.05685
    License     : Apache 2.0

Benchmark design
----------------
    Phase 1 — Warmup  (not timed)
        Submit num_warmup_papers × 1 question each.
        Submission order: LONG papers first, then SHORT papers.
        This ensures the KV cache holds SHORT papers as the "most recent"
        entries at the end of warmup — forcing LRU and cost-aware policies
        to make opposite decisions when eviction pressure arrives.

    Phase 2 — Timed
        One mixed batch of:
          • Re-requests  — warmup papers answered with a DIFFERENT question.
                           Expected to be in cache unless evicted.
          • New papers   — never-seen documents; guaranteed eviction triggers.
        The eviction policy decides which warmup paper blocks survive into the
        timed phase; re-request hit rates expose the quality of that decision.

    Key insight
        At end of warmup, SHORT papers are most-recently-used.
          LRU  keeps SHORT papers  → evicts LONG papers (expensive to recompute)
          DualPool keeps LONG papers → evicts SHORT papers (cheap to recompute)
        Re-request hit rates on LONG papers quantify this trade-off.

Usage
-----
    python benchmark_eviction_policy.py \\
        --model google/gemma-2-9b-it \\
        --num-warmup-papers 40 --num-new-papers 20 \\
        --max-prompt-words 4000 --max-tokens 128

    python benchmark_eviction_policy.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --num-warmup-papers 40 --num-new-papers 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_MAX_WORDS = 1500   # papers with ≤ this many words → "short / cheap"
LONG_MIN_WORDS  = 4000   # papers with ≥ this many words → "long / expensive"

# Five distinct question templates; warmup uses Q0, timed re-requests use Q1.
QUESTION_TEMPLATES = [
    "What is the main contribution of this paper?",
    "What methodology or technical approach does this paper propose?",
    "What datasets or benchmarks were used to evaluate the method?",
    "What are the main quantitative results reported in the paper?",
    "What limitations or future work directions does the paper identify?",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_arxiv_summarization(
    num_warmup_papers: int,
    num_new_papers: int,
    max_prompt_words: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Load ccdv/arxiv-summarization and build four prompt lists.

    Returns
    -------
    warmup_prompts   : Q0 for each warmup paper (long papers first, then short)
    re_prompts       : Q1 for each warmup paper (same order; should be cache hits)
    new_prompts      : Q0 for each new paper (never seen; triggers eviction)
    re_labels        : "short" or "long" label for each re_prompt entry
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    print("  Loading ccdv/arxiv-summarization …")
    ds = load_dataset("ccdv/arxiv-summarization", split="test")
    rng = random.Random(seed)

    articles: list[tuple[int, str]] = [(i, r["article"]) for i, r in enumerate(ds)]
    rng.shuffle(articles)

    short_pool = [(i, a) for i, a in articles if len(a.split()) <= SHORT_MAX_WORDS]
    long_pool  = [(i, a) for i, a in articles if len(a.split()) >= LONG_MIN_WORDS]

    n_long_warmup  = num_warmup_papers * 2 // 3
    n_short_warmup = num_warmup_papers - n_long_warmup

    if len(long_pool) < n_long_warmup + num_new_papers:
        raise RuntimeError(
            f"Not enough long papers in dataset "
            f"(need {n_long_warmup + num_new_papers}, got {len(long_pool)})"
        )
    if len(short_pool) < n_short_warmup:
        raise RuntimeError(
            f"Not enough short papers in dataset "
            f"(need {n_short_warmup}, got {len(short_pool)})"
        )

    # Warmup pool: long papers first (will be oldest / LRU-evictable),
    # then short papers (will be most-recent at eviction time).
    long_warmup  = long_pool[:n_long_warmup]
    short_warmup = short_pool[:n_short_warmup]
    warmup_pool  = long_warmup + short_warmup   # order matters!

    # New papers come from the next slice of long_pool (unseen during warmup)
    new_pool = long_pool[n_long_warmup : n_long_warmup + num_new_papers]

    def build_prompt(article: str, question: str) -> str:
        words = article.split()[:max_prompt_words]
        text  = " ".join(words)
        return f"Paper:\n{text}\n\nQuestion: {question}\nAnswer:"

    warmup_prompts = [build_prompt(a, QUESTION_TEMPLATES[0]) for _, a in warmup_pool]
    re_prompts     = [build_prompt(a, QUESTION_TEMPLATES[1]) for _, a in warmup_pool]
    new_prompts    = [build_prompt(a, QUESTION_TEMPLATES[0]) for _, a in new_pool]

    re_labels = (
        ["long"]  * n_long_warmup +
        ["short"] * n_short_warmup
    )

    total_warmup_words = sum(min(len(a.split()), max_prompt_words)
                             for _, a in warmup_pool)
    total_warmup_tokens = int(total_warmup_words * 1.3)
    print(f"  Warmup : {n_long_warmup} long + {n_short_warmup} short papers")
    print(f"  New    : {num_new_papers} long papers")
    print(f"  Est. warmup token volume : ~{total_warmup_tokens:,} "
          f"(A6000 KV cache ~73k)")

    return warmup_prompts, re_prompts, new_prompts, re_labels


# ---------------------------------------------------------------------------
# Cache hit rate helpers
# ---------------------------------------------------------------------------

def _compute_hit_rate(outputs) -> float:
    """hit_rate = Σ num_cached_tokens / Σ prompt_tokens, per request."""
    total_prompt  = sum(len(o.prompt_token_ids) for o in outputs
                        if o.prompt_token_ids)
    total_cached  = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outputs)
    return total_cached / total_prompt if total_prompt > 0 else -1.0


def _hit_rate_by_label(outputs, labels: list[str], target: str) -> float:
    subset = [o for o, l in zip(outputs, labels) if l == target]
    return _compute_hit_rate(subset)


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    policy: str
    model: str
    num_warmup_papers: int
    num_new_papers: int
    total_time_s: float
    throughput_req_per_s: float
    avg_ttft_ms: float
    # Overall hit rates
    re_cache_hit_rate: float          # re-requests on warmup papers
    new_cache_hit_rate: float         # new papers (should be ~0)
    # Breakdown by paper length
    re_long_hit_rate: float           # long paper re-requests (expensive)
    re_short_hit_rate: float          # short paper re-requests (cheap)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_policy(
    model: str,
    warmup_prompts: list[str],
    re_prompts: list[str],
    new_prompts: list[str],
    re_labels: list[str],
    max_tokens: int,
    enable_dual_pool: bool,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    seed: int,
) -> PolicyResult:
    """
    Run one policy (LRU or DualPool) through warmup + timed phases.

    Warmup fills the KV cache with long-then-short papers.
    Timed submits re-requests (warmup papers, different question) mixed with
    new papers (never seen).  Hit rates on re-requests measure eviction quality.
    """
    from vllm import LLM, SamplingParams

    policy_name = "DUAL_POOL" if enable_dual_pool else "LRU"
    print(f"\n{'='*60}")
    print(f"Policy: {policy_name}  |  Model: {model}")
    print(f"{'='*60}")

    llm = LLM(
        model=model,
        enable_prefix_caching=True,
        enable_dual_pool=enable_dual_pool,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        disable_log_stats=True,
    )

    warmup_sp = SamplingParams(max_tokens=1,          temperature=0.0)
    timed_sp  = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # ── Phase 1: warmup ──────────────────────────────────────────────────────
    n_w = len(warmup_prompts)
    print(f"  [warmup] {n_w} papers (long-first order) …")
    llm.generate(warmup_prompts, sampling_params=warmup_sp)

    # ── Phase 2: timed ───────────────────────────────────────────────────────
    timed_prompts = re_prompts + new_prompts
    print(f"  [timed ] {len(re_prompts)} re-requests + "
          f"{len(new_prompts)} new papers …")

    ttft_list: list[float] = []
    t0 = time.perf_counter()
    timed_outputs = llm.generate(timed_prompts, sampling_params=timed_sp)
    total_time = time.perf_counter() - t0

    for out in timed_outputs:
        m = out.metrics
        if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
            ttft_list.append((m.first_token_time - m.arrival_time) * 1000)

    re_outputs  = timed_outputs[:len(re_prompts)]
    new_outputs = timed_outputs[len(re_prompts):]

    re_hit   = _compute_hit_rate(re_outputs)
    new_hit  = _compute_hit_rate(new_outputs)
    long_hit = _hit_rate_by_label(re_outputs, re_labels, "long")
    short_hit= _hit_rate_by_label(re_outputs, re_labels, "short")

    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else -1.0
    n_timed  = len(timed_outputs)

    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    result = PolicyResult(
        policy=policy_name,
        model=model,
        num_warmup_papers=len(warmup_prompts),
        num_new_papers=len(new_prompts),
        total_time_s=round(total_time, 2),
        throughput_req_per_s=round(n_timed / total_time, 3),
        avg_ttft_ms=round(avg_ttft, 2),
        re_cache_hit_rate=round(re_hit, 4) if re_hit >= 0 else -1.0,
        new_cache_hit_rate=round(new_hit, 4) if new_hit >= 0 else -1.0,
        re_long_hit_rate=round(long_hit, 4) if long_hit >= 0 else -1.0,
        re_short_hit_rate=round(short_hit, 4) if short_hit >= 0 else -1.0,
    )

    def fmt(v: float) -> str:
        return f"{v:.1%}" if v >= 0 else "N/A"

    print(f"\n  Results ({policy_name}):")
    print(f"    Total time          : {result.total_time_s:.1f}s")
    print(f"    Throughput          : {result.throughput_req_per_s:.2f} req/s")
    print(f"    Avg TTFT            : {result.avg_ttft_ms:.1f}ms"
          if avg_ttft >= 0 else "    Avg TTFT            : N/A")
    print(f"    Re-request hit rate : {fmt(re_hit)}  "
          f"(long={fmt(long_hit)}, short={fmt(short_hit)})")
    print(f"    New-paper hit rate  : {fmt(new_hit)}  (expected ~0%)")

    return result


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LRU vs Dual-Pool KV cache eviction benchmark "
                    "(dataset: ccdv/arxiv-summarization)"
    )
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--num-warmup-papers", type=int, default=40,
                   help="Papers used in warmup phase (⅔ long, ⅓ short)")
    p.add_argument("--num-new-papers",    type=int, default=20,
                   help="Unseen papers added in timed phase (triggers eviction)")
    p.add_argument("--max-prompt-words",  type=int, default=4000,
                   help="Truncate paper text to this many words "
                        "(safe for gemma 8k context window)")
    p.add_argument("--max-tokens",        type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size",   type=int,   default=1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--output",  default=None,
                   help="Path for JSON results (auto-named if omitted)")
    p.add_argument("--policies", nargs="+",
                   choices=["lru", "dual_pool"],
                   default=["lru", "dual_pool"])
    return p.parse_args()


def _pct_delta(new: float, old: float) -> str:
    if old and old > 0:
        return f"{(new - old) / old * 100:+.1f}%"
    return "N/A"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    warmup_prompts, re_prompts, new_prompts, re_labels = load_arxiv_summarization(
        num_warmup_papers=args.num_warmup_papers,
        num_new_papers=args.num_new_papers,
        max_prompt_words=args.max_prompt_words,
        seed=args.seed,
    )

    results: list[PolicyResult] = []

    for policy in args.policies:
        r = run_policy(
            model=args.model,
            warmup_prompts=warmup_prompts,
            re_prompts=re_prompts,
            new_prompts=new_prompts,
            re_labels=re_labels,
            max_tokens=args.max_tokens,
            enable_dual_pool=(policy == "dual_pool"),
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )
        results.append(r)

    # ── Comparison table ─────────────────────────────────────────────────────
    if len(results) == 2:
        lru = next(r for r in results if r.policy == "LRU")
        dp  = next(r for r in results if r.policy == "DUAL_POOL")
        W = 65
        print(f"\n{'='*W}")
        print("COMPARISON: LRU  vs  Dual-Pool Cost-Aware")
        print(f"{'='*W}")
        print(f"{'Metric':<36} {'LRU':>10} {'DualPool':>10} {'Δ':>8}")
        print("-"*W)

        def row(label, lv, dv, fmt_fn=lambda x: f"{x:.2f}"):
            print(f"{label:<36} {fmt_fn(lv):>10} {fmt_fn(dv):>10} "
                  f"{_pct_delta(dv, lv):>8}")

        row("Total time (s)",           lru.total_time_s,         dp.total_time_s)
        row("Throughput (req/s)",        lru.throughput_req_per_s, dp.throughput_req_per_s)
        if lru.avg_ttft_ms >= 0:
            row("Avg TTFT (ms)",         lru.avg_ttft_ms,          dp.avg_ttft_ms)
        row("Re-request hit rate (all)", lru.re_cache_hit_rate,   dp.re_cache_hit_rate,
            lambda x: f"{x:.1%}" if x >= 0 else "N/A")
        row("Re-request hit rate (LONG)",lru.re_long_hit_rate,    dp.re_long_hit_rate,
            lambda x: f"{x:.1%}" if x >= 0 else "N/A")
        row("Re-request hit rate (SHORT)",lru.re_short_hit_rate,  dp.re_short_hit_rate,
            lambda x: f"{x:.1%}" if x >= 0 else "N/A")
        row("New-paper hit rate",        lru.new_cache_hit_rate,  dp.new_cache_hit_rate,
            lambda x: f"{x:.1%}" if x >= 0 else "N/A")
        print("-"*W)
        print("DualPool advantage: higher LONG hit rate at cost of lower SHORT hit rate")
        print(f"{'='*W}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    model_tag = args.model.replace("/", "_")
    out_path = args.output or (
        f"results_eviction/result_{model_tag}_arxiv.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": "ccdv/arxiv-summarization",
            "citation": (
                "Cohan et al. (2018), 'A Discourse-Aware Attention Model for "
                "Abstractive Summarization of Long Documents', NAACL 2018. "
                "https://arxiv.org/abs/1804.05685"
            ),
            "hf_url": "https://huggingface.co/datasets/ccdv/arxiv-summarization",
            "split": "test",
        },
        "config": {
            "model": args.model,
            "num_warmup_papers": args.num_warmup_papers,
            "num_new_papers": args.num_new_papers,
            "max_prompt_words": args.max_prompt_words,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "warmup_order": "long_first_then_short",
        },
        "results": [asdict(r) for r in results],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
