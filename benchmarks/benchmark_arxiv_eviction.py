#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LRU vs Cost-Aware KV Cache Eviction — arXiv Benchmark
Adapted from: Jenga SOSP'25 AE  (heheda12345/Jenga-SOSP25-AE)

Dataset : liyucheng/arxiv-march-2023  (same as Jenga paper)
Format  : article_text + question     (same as Jenga, no separator)
Questions: Jenga's 10 canonical questions

Design
------
We load N_long long articles (truncated to long_tokens each) and N_short
short articles (truncated to short_tokens each).  For each article we
generate Q questions, yielding N_long*Q + N_short*Q total requests.
Requests are shuffled (fixed seed) and submitted in a single batch.

Because N_long * long_tokens > KV-cache capacity, the cache is under
constant eviction pressure throughout the benchmark.  The eviction policy
decides which articles to keep:

  LRU        → evicts the LEAST-RECENTLY-USED blocks, regardless of cost
  Cost-Aware → evicts blocks belonging to SHORT articles first
               (chain_total_blocks is smaller → lower score → evict first)

Key metrics
-----------
  re_long_hit_rate  : cache hit rate on Q2+ questions for long articles
  re_short_hit_rate : cache hit rate on Q2+ questions for short articles
  throughput        : requests / second (timed phase only)

Expected result
---------------
  Cost-Aware: higher re_long_hit_rate, lower re_short_hit_rate
  LRU       : lower re_long_hit_rate, higher re_short_hit_rate

Usage
-----
  # Gemma defaults (58k-token KV cache)
  python benchmark_arxiv_eviction.py --model google/gemma-2-9b-it

  # Llama defaults (174k-token KV cache)
  python benchmark_arxiv_eviction.py \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --num-long 6 --long-tokens 25000 \\
      --num-short 6 --short-tokens 2000

  # Custom
  python benchmark_arxiv_eviction.py \\
      --model google/gemma-2-9b-it \\
      --num-long 10 --long-tokens 5000 \\
      --num-short 6 --short-tokens 400 \\
      --num-questions 4 \\
      --output results_arxiv_eviction.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Jenga's 10 canonical questions (from heheda12345/Jenga-SOSP25-AE)
QUESTIONS = [
    "What is the author of the paper?",
    "What is the title of the paper?",
    "What is the abstract of the paper?",
    "What is the introduction of the paper?",
    "Please summarize the paper.",
    "What is the conclusion of the paper?",
    "What is the main contribution of the paper?",
    "When was the paper published?",
    "What is the categories of the paper?",
    "What is the URL of the paper?",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_articles(
    n_long: int,
    n_short: int,
    long_tokens: int,
    short_tokens: int,
    long_min_tokens: int,
    tokenizer,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Load articles from liyucheng/arxiv-march-2023.

    Long articles: only papers with >= long_min_tokens tokens qualify;
    they are truncated to exactly long_tokens.  This ensures the "long"
    group truly occupies long_tokens each in the KV cache.

    Short articles: any paper truncated to short_tokens.  Drawn from a
    separate slice of the shuffled index so there is no prefix overlap
    with the long group.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    print("  Loading liyucheng/arxiv-march-2023 …")
    ds = load_dataset("liyucheng/arxiv-march-2023", split="train")
    print(f"  Dataset: {len(ds):,} articles")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    def truncate(text: str, max_tok: int) -> str:
        ids = tokenizer(text, truncation=False).input_ids
        if len(ids) <= max_tok:
            return text
        return tokenizer.decode(ids[:max_tok], skip_special_tokens=True)

    long_arts: list[str] = []
    short_arts: list[str] = []
    used_indices: set[int] = set()

    # Pass 1: fill long pool (enforce minimum length)
    for idx in indices:
        if len(long_arts) >= n_long:
            break
        text: str = ds[idx]["text"]
        tok_count = len(tokenizer(text, truncation=False).input_ids)
        if tok_count >= long_min_tokens:
            long_arts.append(truncate(text, long_tokens))
            used_indices.add(idx)

    # Pass 2: fill short pool from remaining articles
    for idx in indices:
        if len(short_arts) >= n_short:
            break
        if idx in used_indices:
            continue
        text = ds[idx]["text"]
        tok_count = len(tokenizer(text, truncation=False).input_ids)
        if tok_count >= short_tokens:  # must have at least short_tokens of content
            short_arts.append(truncate(text, short_tokens))
            used_indices.add(idx)

    if len(long_arts) < n_long:
        raise RuntimeError(
            f"Not enough long articles (need {n_long}, got {len(long_arts)}). "
            f"Try lowering --long-min-tokens or --num-long."
        )
    if len(short_arts) < n_short:
        raise RuntimeError(
            f"Not enough short articles (need {n_short}, got {len(short_arts)}). "
            "Try lowering --num-short."
        )

    long_tok_lens  = [len(tokenizer(a, truncation=False).input_ids) for a in long_arts]
    short_tok_lens = [len(tokenizer(a, truncation=False).input_ids) for a in short_arts]
    long_total  = sum(long_tok_lens)
    short_total = sum(short_tok_lens)

    print(f"  Long  articles : {n_long} × {long_tokens:,} tokens "
          f"(actual avg {long_total//n_long:,}, total {long_total:,})")
    print(f"  Short articles : {n_short} × {short_tokens:,} tokens "
          f"(actual avg {short_total//n_short:,}, total {short_total:,})")
    print(f"  Total unique content : {long_total + short_total:,} tokens")

    return long_arts, short_arts


def build_requests(
    long_arts: list[str],
    short_arts: list[str],
    num_questions: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """
    Build (prompt, group, article_idx, question_idx) tuples, then shuffle.

    Returns:
        prompts      : list of prompt strings
        groups       : list of "long" or "short" for each prompt
        article_idxs : which article index within its group
        question_idxs: which question index (0 = first visit, 1+ = re-visit)
    """
    qs = QUESTIONS[:num_questions]
    entries = []
    for a_idx, art in enumerate(long_arts):
        for q_idx, q in enumerate(qs):
            entries.append((art + q, "long", a_idx, q_idx))
    for a_idx, art in enumerate(short_arts):
        for q_idx, q in enumerate(qs):
            entries.append((art + q, "short", a_idx, q_idx))

    rng = random.Random(seed)
    rng.shuffle(entries)

    prompts       = [e[0] for e in entries]
    groups        = [e[1] for e in entries]
    article_idxs  = [e[2] for e in entries]
    question_idxs = [e[3] for e in entries]
    return prompts, groups, article_idxs, question_idxs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _hit_rate(outputs, indices: list[int]) -> float:
    subset = [outputs[i] for i in indices]
    total_prompt  = sum(len(o.prompt_token_ids) for o in subset
                        if o.prompt_token_ids)
    total_cached  = sum(getattr(o, "num_cached_tokens", 0) or 0
                        for o in subset)
    return total_cached / total_prompt if total_prompt > 0 else -1.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    policy: str
    model: str
    n_long: int
    n_short: int
    long_tokens: int
    short_tokens: int
    num_questions: int
    total_requests: int
    total_time_s: float
    throughput_req_per_s: float
    avg_ttft_ms: float
    # Overall hit rate
    overall_hit_rate: float
    # Hit rate by group × question position
    long_q1_hit_rate: float    # first question (cold, should be ~0)
    long_reqs_hit_rate: float  # Q2+ questions for long articles (key metric)
    short_q1_hit_rate: float
    short_reqs_hit_rate: float # Q2+ questions for short articles


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_policy(
    model: str,
    prompts: list[str],
    groups: list[str],
    question_idxs: list[int],
    max_tokens: int,
    enable_dual_pool: bool,
    enable_cost_scoring: bool,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    seed: int,
    n_long: int,
    n_short: int,
    long_tokens: int,
    short_tokens: int,
    num_questions: int,
) -> PolicyResult:
    from vllm import LLM, SamplingParams

    if not enable_cost_scoring and not enable_dual_pool:
        policy_name = "LRU"
    elif enable_cost_scoring and not enable_dual_pool:
        policy_name = "Cost-Aware"
    elif not enable_cost_scoring and enable_dual_pool:
        policy_name = "LRU+DualPool"
    else:
        policy_name = "Full(Cost+DualPool)"

    print(f"\n{'='*62}")
    print(f"Policy : {policy_name}")
    print(f"Model  : {model}")
    print(f"{'='*62}")

    llm = LLM(
        model=model,
        enable_prefix_caching=True,
        enable_dual_pool=enable_dual_pool,
        enable_cost_scoring=enable_cost_scoring,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    print(f"  Submitting {len(prompts)} requests (shuffled) …")
    ttft_list: list[float] = []
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sp)
    total_time = time.perf_counter() - t0

    for out in outputs:
        m = out.metrics
        if (m and getattr(m, "first_token_time", None)
                and getattr(m, "arrival_time", None)):
            ttft_list.append((m.first_token_time - m.arrival_time) * 1000)

    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else -1.0
    n = len(outputs)

    # ── Compute hit rates by group × question position ───────────────────
    long_q1_idx   = [i for i, (g, q) in enumerate(zip(groups, question_idxs))
                     if g == "long"  and q == 0]
    long_re_idx   = [i for i, (g, q) in enumerate(zip(groups, question_idxs))
                     if g == "long"  and q > 0]
    short_q1_idx  = [i for i, (g, q) in enumerate(zip(groups, question_idxs))
                     if g == "short" and q == 0]
    short_re_idx  = [i for i, (g, q) in enumerate(zip(groups, question_idxs))
                     if g == "short" and q > 0]

    all_idx         = list(range(n))
    overall_hit     = _hit_rate(outputs, all_idx)
    long_q1_hit     = _hit_rate(outputs, long_q1_idx)
    long_re_hit     = _hit_rate(outputs, long_re_idx)
    short_q1_hit    = _hit_rate(outputs, short_q1_idx)
    short_re_hit    = _hit_rate(outputs, short_re_idx)

    def fmt(v: float) -> str:
        return f"{v:.1%}" if v >= 0 else "N/A"

    print(f"\n  [{policy_name}] Results:")
    print(f"    Throughput          : {n/total_time:.2f} req/s")
    if avg_ttft >= 0:
        print(f"    Avg TTFT            : {avg_ttft:.1f} ms")
    print(f"    Overall hit rate    : {fmt(overall_hit)}")
    print(f"    Long  Q1 (cold)     : {fmt(long_q1_hit)}")
    print(f"    Long  Q2+ (re-req)  : {fmt(long_re_hit)}  ← key metric")
    print(f"    Short Q1 (cold)     : {fmt(short_q1_hit)}")
    print(f"    Short Q2+ (re-req)  : {fmt(short_re_hit)}")

    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    return PolicyResult(
        policy=policy_name,
        model=model,
        n_long=n_long,
        n_short=n_short,
        long_tokens=long_tokens,
        short_tokens=short_tokens,
        num_questions=num_questions,
        total_requests=n,
        total_time_s=round(total_time, 2),
        throughput_req_per_s=round(n / total_time, 3),
        avg_ttft_ms=round(avg_ttft, 2),
        overall_hit_rate=round(overall_hit, 4)   if overall_hit   >= 0 else -1.0,
        long_q1_hit_rate=round(long_q1_hit, 4)   if long_q1_hit   >= 0 else -1.0,
        long_reqs_hit_rate=round(long_re_hit, 4) if long_re_hit   >= 0 else -1.0,
        short_q1_hit_rate=round(short_q1_hit, 4) if short_q1_hit  >= 0 else -1.0,
        short_reqs_hit_rate=round(short_re_hit, 4) if short_re_hit >= 0 else -1.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "LRU vs Cost-Aware KV cache eviction benchmark "
            "(liyucheng/arxiv-march-2023, Jenga-style)"
        )
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument(
        "--num-long", type=int, default=10,
        help="Number of long articles. n_long * long_tokens should exceed "
             "KV-cache capacity to ensure eviction pressure.",
    )
    ap.add_argument(
        "--long-tokens", type=int, default=5500,
        help="Tokens per long article after truncation. "
             "Default 5500 fits Gemma's 8192-token context.",
    )
    ap.add_argument(
        "--long-min-tokens", type=int, default=None,
        help="Minimum token count for a paper to qualify as 'long'. "
             "Defaults to --long-tokens (i.e. only papers that reach the full "
             "truncation limit are used). Increase to guarantee actual overflow.",
    )
    ap.add_argument(
        "--num-short", type=int, default=8,
        help="Number of short articles.",
    )
    ap.add_argument(
        "--short-tokens", type=int, default=400,
        help="Tokens per short article after truncation.",
    )
    ap.add_argument(
        "--num-questions", type=int, default=4,
        help="Questions per article (uses Jenga's canonical 10 questions). "
             "Q1 is the cold hit; Q2+ are re-requests that test cache.",
    )
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Max output tokens per request.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None,
                    help="JSON output file path (auto-named if omitted).")
    args = ap.parse_args()

    if args.num_questions > len(QUESTIONS):
        ap.error(f"--num-questions max is {len(QUESTIONS)}")

    long_min = args.long_min_tokens if args.long_min_tokens else args.long_tokens

    print("=" * 62)
    print("arXiv Eviction Benchmark  (Jenga-style)")
    print(f"Model  : {args.model}")
    print(f"Dataset: liyucheng/arxiv-march-2023")
    print(f"Long   : {args.num_long} × {args.long_tokens:,} tok "
          f"(min qualify: {long_min:,}) "
          f"= ~{args.num_long * args.long_tokens:,} total")
    print(f"Short  : {args.num_short} × {args.short_tokens:,} tokens "
          f"= ~{args.num_short * args.short_tokens:,} total")
    print(f"Q/art  : {args.num_questions}")
    print("=" * 62)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Articles ──────────────────────────────────────────────────────────
    print("\nLoading articles …")
    long_arts, short_arts = load_articles(
        n_long=args.num_long,
        n_short=args.num_short,
        long_tokens=args.long_tokens,
        short_tokens=args.short_tokens,
        long_min_tokens=long_min,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    # ── Requests ──────────────────────────────────────────────────────────
    prompts, groups, article_idxs, question_idxs = build_requests(
        long_arts=long_arts,
        short_arts=short_arts,
        num_questions=args.num_questions,
        seed=args.seed,
    )
    print(f"\nTotal requests: {len(prompts)} "
          f"({args.num_long * args.num_questions} long + "
          f"{args.num_short * args.num_questions} short)")

    # ── Run 4 policies (2×2 matrix) ───────────────────────────────────────
    # (enable_cost_scoring, enable_dual_pool)
    policy_configs = [
        (False, False),   # LRU
        (True,  False),   # Cost-Aware
        (False, True),    # LRU + DualPool
        (True,  True),    # Full (Cost + DualPool)
    ]
    results: list[PolicyResult] = []
    for enable_cost_scoring, enable_dual_pool in policy_configs:
        r = run_policy(
            model=args.model,
            prompts=prompts,
            groups=groups,
            question_idxs=question_idxs,
            max_tokens=args.max_tokens,
            enable_dual_pool=enable_dual_pool,
            enable_cost_scoring=enable_cost_scoring,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            n_long=args.num_long,
            n_short=args.num_short,
            long_tokens=args.long_tokens,
            short_tokens=args.short_tokens,
            num_questions=args.num_questions,
        )
        results.append(r)

    # ── Comparison table ──────────────────────────────────────────────────
    def _pct(v: float) -> str:
        return f"{v:.1%}" if v >= 0 else "N/A"

    def _delta(base: float, cmp: float) -> str:
        if base < 0 or cmp < 0:
            return "N/A"
        return f"{(cmp - base) * 100:+.1f}"

    # Column headers (policy names)
    names = [r.policy for r in results]
    lru = results[0]

    col_w = 18
    hdr_w = 28

    print(f"\n{'='*(hdr_w + col_w * len(results) + 4)}")
    print("RESULTS TABLE  (arXiv Eviction Benchmark)")
    print(f"{'='*(hdr_w + col_w * len(results) + 4)}")
    header = f"{'Metric':<{hdr_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * (hdr_w + col_w * len(results) + 4))

    def _row(label: str, vals: list[str]) -> None:
        print(f"  {label:<{hdr_w - 2}}" + "".join(f"{v:>{col_w}}" for v in vals))

    _row("Throughput (req/s)",
         [f"{r.throughput_req_per_s:.2f}" for r in results])
    _row("Overall hit rate",
         [_pct(r.overall_hit_rate) for r in results])
    _row("Long  Q1 (cold)",
         [_pct(r.long_q1_hit_rate) for r in results])
    _row("Long  Q2+ hit  ★",
         [_pct(r.long_reqs_hit_rate) for r in results])
    _row("Short Q1 (cold)",
         [_pct(r.short_q1_hit_rate) for r in results])
    _row("Short Q2+ hit",
         [_pct(r.short_reqs_hit_rate) for r in results])

    print("-" * (hdr_w + col_w * len(results) + 4))

    # Δ vs LRU baseline
    print(f"\n  Δ vs LRU baseline:")
    _row("  Long  Q2+ Δ (pp)",
         ["—"] + [_delta(lru.long_reqs_hit_rate, r.long_reqs_hit_rate)
                  for r in results[1:]])
    _row("  Short Q2+ Δ (pp)",
         ["—"] + [_delta(lru.short_reqs_hit_rate, r.short_reqs_hit_rate)
                  for r in results[1:]])
    _row("  Throughput Δ (%)",
         ["—"] + [
             f"{(r.throughput_req_per_s - lru.throughput_req_per_s) / lru.throughput_req_per_s * 100:+.1f}"
             for r in results[1:]
         ])

    print(f"\n  ★ Long Q2+ hit: higher = better "
          f"(cost-aware preserves expensive long-article prefixes)")

    # ── Save ──────────────────────────────────────────────────────────────
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": [asdict(r) for r in results],
    }
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"results_arxiv_eviction_{ts}.json")

    out_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
