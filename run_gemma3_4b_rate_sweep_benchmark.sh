#!/bin/bash
# Rate sweep benchmark — Gemma-3-4B-IT on A5000 (combined-prefix-aware branch)
# Figure15-style: vary request rate, measure E2E latency for LRU vs Both.
#
# Policies:
#   LRU   eviction=LRU,  sched=FCFS
#   Both  eviction=PA,   sched=PA    (VLLM_PREFIX_AWARE_EVICTION=1 + --enable-prefix-aware-scheduling)
#
# Rates tested (req/s): 0.2:0.2:4.0  (20 points, same as Jenga figure15)
#   LRU peak throughput ≈ 0.12, Both peak ≈ 0.26
#   → all rates above LRU saturation; 0.2-0.26 shows Both advantage clearly
#
# Dataset: liyucheng/arxiv-march-2023, subset=ministral
#   10 articles × 4 questions = 40 prompts, shuffled seed=55555
#   hf_max_len=90000, hf_output_len=150, burstiness=1.0 (Poisson)
#   percentile-metrics: ttft,tpot,itl,e2el
#
# Usage:
#   tmux new-session -d -s bench_gemma_rate_sweep \
#     "bash run_gemma3_4b_rate_sweep_benchmark.sh 2>&1 | tee bench_gemma_rate_sweep.log"
#
# Estimated time: ~5-7 hours (20 rates × 2 policies, cold-start per run)

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/r14922173/vllm_work/cache/huggingface

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/gemma3_4b_rate_sweep
PORT=8768
MODEL=google/gemma-3-4b-it
NUM_PROMPTS=40
RATES="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0"

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== Rate sweep benchmark — Gemma-3-4B-IT (LRU vs Both) ===" | tee "$LOG"
echo "Start  : $(date)"                                              | tee -a "$LOG"
echo "GPU    : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "Branch : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "Rates  : $RATES"                                               | tee -a "$LOG"

# ── Helpers ──────────────────────────────────────────────────────────────────

start_server() {
    local policy=$1
    local extra_flags=$2
    echo "" | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy" | tee -a "$LOG"
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --no-enable-dual-pool \
        --no-enable-cost-scoring \
        --no-enable-adaptive-scoring \
        --port $PORT \
        --no-enable-log-requests \
        $extra_flags \
        >> "$LOG" 2>&1 &
    SERVER_PID=$!

    echo "    Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
    for _w in $(seq 1 300); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && \
            echo "    Ready after $((_w*2))s" | tee -a "$LOG" && return 0
        sleep 2
    done
    echo "ERROR: server did not start in 600s" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null; exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    echo "    Waiting for GPU 0 memory to clear..." | tee -a "$LOG"
    for i in $(seq 1 30); do
        FREE=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$FREE" -gt 20000 ] && break
        sleep 2
    done
    echo "    GPU 0 cleared." | tee -a "$LOG"
}

run_client() {
    local policy=$1
    local rate=$2
    # replace dot with underscore for filename safety
    local rate_str
    rate_str=$(echo "$rate" | tr '.' '_')
    local out=$OUTDIR/${policy}_rate${rate_str}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy  rate=$rate req/s" | tee -a "$LOG"
    cd benchmarks
    $PYTHON jenga_benchmark_serving.py \
        --port "$PORT" \
        --model "$MODEL" \
        --dataset-path liyucheng/arxiv-march-2023 \
        --dataset-name hf \
        --hf-subset ministral \
        --hf-split train \
        --num-prompts "$NUM_PROMPTS" \
        --seed 55555 \
        --hf-output-len 150 \
        --hf-max-len 90000 \
        --ignore-eos \
        --request-rate "$rate" \
        --burstiness 1.0 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-filename "../$out" \
        2>&1 | tee -a "../$LOG"
    cd ..
    echo "    Results saved → $out" | tee -a "$LOG"
}

# ── Policy 1: LRU ─────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "====== POLICY: LRU ======" | tee -a "$LOG"
export VLLM_PREFIX_AWARE_EVICTION=0
for rate in $RATES; do
    start_server "LRU" ""
    run_client "LRU" "$rate"
    stop_server
done

# ── Policy 2: Both (PA eviction + PA scheduling) ──────────────────────────────
echo "" | tee -a "$LOG"
echo "====== POLICY: Both ======" | tee -a "$LOG"
export VLLM_PREFIX_AWARE_EVICTION=1
for rate in $RATES; do
    start_server "Both" "--enable-prefix-aware-scheduling"
    run_client "Both" "$rate"
    stop_server
done

echo "" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
echo "Results in $OUTDIR/"       | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
