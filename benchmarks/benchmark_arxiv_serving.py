#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Online serving benchmark for arXiv eviction policy evaluation.

2-Phase design:
  Phase 1 (warmup)  : Submit Q1 for ALL articles concurrently.
                      Cache fills up; eviction policy determines which articles survive.
  Phase 2 (measure) : Submit Q2+ for ALL articles concurrently.
                      Only articles whose blocks survived Phase 1 eviction get cache hits.

The server must be started externally before running this script:

  python -m vllm.entrypoints.openai.api_server \\
      --model google/gemma-2-9b-it \\
      --enable-prefix-caching \\
      --enable-prompt-tokens-details \\
      [--enable-dual-pool | --no-enable-dual-pool] \\
      [--enable-cost-scoring | --no-enable-cost-scoring] \\
      --port 8000

Usage:
  python benchmarks/benchmark_arxiv_serving.py \\
      --host localhost --port 8000 \\
      --model google/gemma-2-9b-it \\
      --policy-name LRU \\
      --output results_arxiv/result_lru_serving.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Reuse article loading + request building from the offline benchmark
from benchmark_arxiv_eviction import load_articles, QUESTIONS

# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class ReqResult:
    group: str          # "long" or "short"
    article_idx: int
    question_idx: int   # 0 = Q1 (warmup), 1+ = Q2+ (measured)
    prompt_tokens: int
    cached_tokens: int  # from usage.prompt_tokens_details.cached_tokens
    output_tokens: int
    ttft_ms: float
    e2e_ms: float

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0

    @property
    def recomputed_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def send_one(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    group: str,
    article_idx: int,
    question_idx: int,
) -> ReqResult:
    from openai import AsyncOpenAI  # imported lazily to avoid import-time overhead

    t0 = time.perf_counter()
    first_token_t: float | None = None
    prompt_tokens = 0
    cached_tokens = 0

    stream = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if first_token_t is None and chunk.choices:
            first_token_t = time.perf_counter()
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0

    e2e_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = ((first_token_t - t0) * 1000) if first_token_t else -1.0
    output_tokens = max_tokens  # approximate; server may stop early

    return ReqResult(
        group=group,
        article_idx=article_idx,
        question_idx=question_idx,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
    )


async def run_phase(
    client,
    model: str,
    entries: list[tuple[str, str, int, int]],
    max_tokens: int,
    phase_name: str,
) -> tuple[list[ReqResult], float]:
    """Send all entries concurrently, return (results, wall_time_s)."""
    print(f"  [{phase_name}] sending {len(entries)} requests concurrently …")
    t0 = time.perf_counter()
    tasks = [
        send_one(client, model, prompt, max_tokens, group, a_idx, q_idx)
        for prompt, group, a_idx, q_idx in entries
    ]
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    print(f"  [{phase_name}] done in {wall:.1f}s  ({len(entries)/wall:.2f} req/s)")
    return list(results), wall


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _hit_rate_for(results: list[ReqResult], group: str, q_idx_min: int) -> float:
    subset = [r for r in results if r.group == group and r.question_idx >= q_idx_min]
    if not subset:
        return -1.0
    tot_prompt = sum(r.prompt_tokens for r in subset)
    tot_cached = sum(r.cached_tokens for r in subset)
    return tot_cached / tot_prompt if tot_prompt > 0 else 0.0


def _recomputed_tokens(results: list[ReqResult], group: str, q_idx_min: int) -> int:
    subset = [r for r in results if r.group == group and r.question_idx >= q_idx_min]
    return sum(r.recomputed_tokens for r in subset)


def _avg_ttft(results: list[ReqResult], group: str, q_idx_min: int) -> float:
    vals = [r.ttft_ms for r in results
            if r.group == group and r.question_idx >= q_idx_min and r.ttft_ms >= 0]
    return sum(vals) / len(vals) if vals else -1.0


def get_server_cache_hit_rate(host: str, port: int) -> float:
    """Query vLLM's /metrics endpoint for overall prefix-cache hit rate."""
    try:
        import urllib.request
        url = f"http://{host}:{port}/metrics"
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
        queries = hits = 0.0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if "vllm:prefix_cache_queries" in line:
                queries = float(line.split()[-1])
            elif "vllm:prefix_cache_hits" in line and "external" not in line:
                hits = float(line.split()[-1])
        return hits / queries if queries > 0 else -1.0
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Online serving benchmark (2-phase arXiv eviction test)"
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--policy-name", default="LRU",
                    help="Human-readable policy name for the JSON output.")
    ap.add_argument("--num-long",  type=int, default=20)
    ap.add_argument("--long-tokens",  type=int, default=5500)
    ap.add_argument("--long-min-tokens", type=int, default=None)
    ap.add_argument("--num-short", type=int, default=8)
    ap.add_argument("--short-tokens", type=int, default=400)
    ap.add_argument("--num-questions", type=int, default=4,
                    help="Questions per article. Q1 is warmup; Q2+ are measured.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    # Compare mode: print table from multiple JSON files
    ap.add_argument("--compare", nargs="+", metavar="JSON",
                    help="Print a comparison table from multiple result JSON files.")
    args = ap.parse_args()

    if args.compare:
        _compare_mode(args.compare)
        return

    long_min = args.long_min_tokens or args.long_tokens

    print("=" * 62)
    print("arXiv Serving Benchmark  (2-phase, online)")
    print(f"Policy : {args.policy_name}")
    print(f"Model  : {args.model}")
    print(f"Server : {args.host}:{args.port}")
    print(f"Long   : {args.num_long} × {args.long_tokens:,} tok")
    print(f"Short  : {args.num_short} × {args.short_tokens:,} tok")
    print(f"Q/art  : {args.num_questions}  (Q1=warmup, Q2+=measured)")
    print("=" * 62)

    from transformers import AutoTokenizer
    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading articles …")
    long_arts, short_arts = load_articles(
        n_long=args.num_long, n_short=args.num_short,
        long_tokens=args.long_tokens, short_tokens=args.short_tokens,
        long_min_tokens=long_min, tokenizer=tokenizer, seed=args.seed,
    )

    qs = QUESTIONS[:args.num_questions]

    # Build per-phase entry lists
    import random
    rng = random.Random(args.seed)

    warmup_entries: list[tuple[str, str, int, int]] = []
    measure_entries: list[tuple[str, str, int, int]] = []

    for a_idx, art in enumerate(long_arts):
        warmup_entries.append((art + qs[0], "long", a_idx, 0))
        for q_idx in range(1, len(qs)):
            measure_entries.append((art + qs[q_idx], "long", a_idx, q_idx))
    for a_idx, art in enumerate(short_arts):
        warmup_entries.append((art + qs[0], "short", a_idx, 0))
        for q_idx in range(1, len(qs)):
            measure_entries.append((art + qs[q_idx], "short", a_idx, q_idx))

    rng.shuffle(warmup_entries)
    rng.shuffle(measure_entries)

    print(f"\nPhase 1 (warmup) : {len(warmup_entries)} requests  "
          f"(Q1 for all {args.num_long + args.num_short} articles)")
    print(f"Phase 2 (measure): {len(measure_entries)} requests  "
          f"(Q2-Q{args.num_questions} for all articles)")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="EMPTY",
    )

    # Read baseline cache metrics before warmup
    hits_before = get_server_cache_hit_rate(args.host, args.port)

    async def _run():
        warmup_results, warmup_time = await run_phase(
            client, args.model, warmup_entries, args.max_tokens, "warmup"
        )
        measure_results, measure_time = await run_phase(
            client, args.model, measure_entries, args.max_tokens, "measure"
        )
        return warmup_results, warmup_time, measure_results, measure_time

    warmup_res, warmup_time, measure_res, measure_time = asyncio.run(_run())

    hits_after = get_server_cache_hit_rate(args.host, args.port)

    # ── Metrics ──────────────────────────────────────────────────────────────
    total_reqs = len(warmup_res) + len(measure_res)
    total_time = warmup_time + measure_time

    long_q2_hit   = _hit_rate_for(measure_res, "long",  1)
    short_q2_hit  = _hit_rate_for(measure_res, "short", 1)
    long_ttft     = _avg_ttft(measure_res, "long",  1)
    short_ttft    = _avg_ttft(measure_res, "short", 1)
    long_recomp   = _recomputed_tokens(measure_res, "long",  1)
    short_recomp  = _recomputed_tokens(measure_res, "short", 1)
    total_recomp  = long_recomp + short_recomp
    # token throughput: output tokens / wall time (prompt reuse saves prefill time)
    total_out_tok = sum(r.output_tokens for r in measure_res)
    tok_throughput = (total_out_tok + sum(r.cached_tokens for r in measure_res)) / measure_time

    def _pct(v: float) -> str:
        return f"{v:.1%}" if v >= 0 else "N/A"

    print(f"\n  [{args.policy_name}] Phase 2 results:")
    print(f"    Throughput (req/s)         : {len(measure_res)/measure_time:.2f}")
    print(f"    Token throughput (tok/s)   : {tok_throughput:.0f}")
    print(f"    Long  Q2+ hit rate    ★    : {_pct(long_q2_hit)}")
    print(f"    Short Q2+ hit rate         : {_pct(short_q2_hit)}")
    print(f"    Long  recomputed tokens    : {long_recomp:,}")
    print(f"    Short recomputed tokens    : {short_recomp:,}")
    print(f"    Total recomputed tokens    : {total_recomp:,}")
    print(f"    Long  avg TTFT (ms)        : {long_ttft:.0f}")
    print(f"    Short avg TTFT (ms)        : {short_ttft:.0f}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "policy": args.policy_name,
        "model": args.model,
        "config": {
            "num_long": args.num_long, "long_tokens": args.long_tokens,
            "num_short": args.num_short, "short_tokens": args.short_tokens,
            "num_questions": args.num_questions,
        },
        "warmup": {
            "num_requests": len(warmup_res),
            "wall_time_s": round(warmup_time, 2),
            "throughput_req_s": round(len(warmup_res) / warmup_time, 3),
        },
        "measure": {
            "num_requests": len(measure_res),
            "wall_time_s": round(measure_time, 2),
            "throughput_req_s": round(len(measure_res) / measure_time, 3),
            "token_throughput": round(tok_throughput, 1),
            "long_q2_hit_rate": round(long_q2_hit, 4),
            "short_q2_hit_rate": round(short_q2_hit, 4),
            "long_recomputed_tokens": long_recomp,
            "short_recomputed_tokens": short_recomp,
            "total_recomputed_tokens": total_recomp,
            "long_avg_ttft_ms": round(long_ttft, 1),
            "short_avg_ttft_ms": round(short_ttft, 1),
            "server_hit_rate": round(hits_after, 4) if hits_after >= 0 else -1,
        },
        "per_request": [asdict(r) for r in measure_res],
    }

    out_path = Path(args.output) if args.output else \
        Path(f"results_arxiv/serving_{args.policy_name.replace('+','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Compare mode: print table from multiple JSON files
# ---------------------------------------------------------------------------

def _compare_mode(json_files: list[str]) -> None:
    """Print comparison table. Groups files by policy name and averages multiple runs."""
    import statistics

    raw: dict[str, list[dict]] = {}
    for f in json_files:
        data = json.loads(Path(f).read_text())
        p = data["policy"]
        raw.setdefault(p, []).append(data)

    # Ordered policy list (preserve insertion order)
    policies = list(raw.keys())

    def mean_std(vals: list[float]) -> str:
        if len(vals) == 1:
            v = vals[0]
            return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
        m = statistics.mean(vals)
        s = statistics.stdev(vals)
        if abs(m) < 100:
            return f"{m:.2f}±{s:.2f}"
        return f"{m:.0f}±{s:.0f}"

    def mean_pct(vals: list[float]) -> str:
        valid = [v for v in vals if v >= 0]
        if not valid:
            return "N/A"
        m = statistics.mean(valid)
        if len(valid) == 1:
            return f"{m:.1%}"
        s = statistics.stdev(valid)
        return f"{m:.1%}±{s*100:.1f}pp"

    def mean_int(vals: list[int]) -> str:
        m = statistics.mean(vals)
        if len(vals) == 1:
            return f"{int(m):,}"
        s = statistics.stdev(vals)
        return f"{int(m):,}±{int(s):,}"

    def get_vals(policy: str, path: str) -> list[float]:
        out = []
        for d in raw[policy]:
            node = d
            for key in path.split("."):
                node = node[key]
            out.append(float(node))
        return out

    def get_int_vals(policy: str, path: str) -> list[int]:
        return [int(v) for v in get_vals(policy, path)]

    n_runs = {p: len(raw[p]) for p in policies}
    col_w = 22
    hdr_w = 32

    runs_info = ", ".join(f"{p}:{n_runs[p]}runs" for p in policies)
    print(f"\n{'=' * (hdr_w + col_w * len(policies) + 4)}")
    print(f"SERVING BENCHMARK  ({runs_info})")
    print(f"{'=' * (hdr_w + col_w * len(policies) + 4)}")
    hdr = f"{'Metric':<{hdr_w}}" + "".join(f"{p:>{col_w}}" for p in policies)
    print(hdr)
    print("-" * (hdr_w + col_w * len(policies) + 4))

    def _row(label: str, vals: list[str]) -> None:
        print(f"  {label:<{hdr_w - 2}}" + "".join(f"{v:>{col_w}}" for v in vals))

    _row("Throughput-warmup (req/s)",
         [mean_std(get_vals(p, "warmup.throughput_req_s")) for p in policies])
    _row("Throughput-measure (req/s)",
         [mean_std(get_vals(p, "measure.throughput_req_s")) for p in policies])
    _row("Token throughput (tok/s)",
         [mean_std(get_vals(p, "measure.token_throughput")) for p in policies])
    _row("Long  Q2+ hit rate  ★",
         [mean_pct(get_vals(p, "measure.long_q2_hit_rate")) for p in policies])
    _row("Short Q2+ hit rate",
         [mean_pct(get_vals(p, "measure.short_q2_hit_rate")) for p in policies])
    _row("Long  recomputed tok",
         [mean_int(get_int_vals(p, "measure.long_recomputed_tokens")) for p in policies])
    _row("Short recomputed tok",
         [mean_int(get_int_vals(p, "measure.short_recomputed_tokens")) for p in policies])
    _row("Total recomputed tok",
         [mean_int(get_int_vals(p, "measure.total_recomputed_tokens")) for p in policies])
    _row("Long  avg TTFT (ms)",
         [mean_std(get_vals(p, "measure.long_avg_ttft_ms")) for p in policies])
    _row("Short avg TTFT (ms)",
         [mean_std(get_vals(p, "measure.short_avg_ttft_ms")) for p in policies])

    print("-" * (hdr_w + col_w * len(policies) + 4))

    # Δ vs LRU (first policy assumed baseline)
    base_policy = policies[0]
    print(f"\n  Δ vs {base_policy} (baseline)  [mean values]:")

    def _delta_pct(base_vals: list[float], cmp_vals: list[float]) -> str:
        bm = statistics.mean([v for v in base_vals if v >= 0] or [0])
        cm = statistics.mean([v for v in cmp_vals  if v >= 0] or [0])
        if bm <= 0:
            return "N/A"
        return f"{(cm - bm) * 100:+.1f}pp"

    def _delta_rel(base_vals: list[float], cmp_vals: list[float]) -> str:
        bm = statistics.mean(base_vals)
        cm = statistics.mean(cmp_vals)
        if bm == 0:
            return "N/A"
        return f"{(cm - bm) / bm * 100:+.1f}%"

    base_long  = get_vals(base_policy, "measure.long_q2_hit_rate")
    base_short = get_vals(base_policy, "measure.short_q2_hit_rate")
    base_tput  = get_vals(base_policy, "measure.throughput_req_s")
    base_recomp = get_vals(base_policy, "measure.total_recomputed_tokens")

    _row("  Long Q2+ Δ (pp)",
         ["—"] + [_delta_pct(base_long,  get_vals(p, "measure.long_q2_hit_rate"))
                  for p in policies[1:]])
    _row("  Short Q2+ Δ (pp)",
         ["—"] + [_delta_pct(base_short, get_vals(p, "measure.short_q2_hit_rate"))
                  for p in policies[1:]])
    _row("  Throughput Δ (%)",
         ["—"] + [_delta_rel(base_tput,  get_vals(p, "measure.throughput_req_s"))
                  for p in policies[1:]])
    _row("  Recomputed tok Δ (%)",
         ["—"] + [_delta_rel(base_recomp, get_vals(p, "measure.total_recomputed_tokens"))
                  for p in policies[1:]])

    print(f"\n  ★ Long Q2+ hit / lower recomputed tokens = better eviction policy")


if __name__ == "__main__":
    main()
