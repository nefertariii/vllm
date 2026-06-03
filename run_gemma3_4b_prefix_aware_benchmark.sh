#!/bin/bash
# LRU vs Scheduler-Aware (prefix-aware eviction) — Gemma-3-4B-IT on A5000
#
# Policies compared:
#   LRU              VLLM_PREFIX_AWARE_EVICTION=0  (pure LRU, baseline)
#   Prefix-Aware     VLLM_PREFIX_AWARE_EVICTION=1  (scheduler-aware eviction)
#
# Request sending follows Jenga SOSP'25 AE benchmark_serving.py:
#   dataset: liyucheng/arxiv-march-2023, subset=ministral
#   4 questions per article, shuffled, rate=inf
#
# Usage:
#   tmux new-session -d -s bench_gemma_pa \
#     "bash run_gemma3_4b_prefix_aware_benchmark.sh 2>&1 | tee bench_gemma_pa.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/gemma3_4b_prefix_aware
PORT=8768
MODEL=google/gemma-3-4b-it
NUM_PROMPTS=40    # 10 articles × 4 questions, shuffled (follows Jenga AE)

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs Prefix-Aware — Gemma-3-4B-IT on A5000 ===" | tee "$LOG"
echo "Start : $(date)"                                         | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "Model : $MODEL"                                          | tee -a "$LOG"
echo "Prompts per run: $NUM_PROMPTS (10 articles × 4 questions, shuffled)" | tee -a "$LOG"

start_server() {
    echo ""                                                            | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$1"         | tee -a "$LOG"
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
        >> "$LOG" 2>&1 &
    SERVER_PID=$!

    echo "    Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
    for _w in $(seq 1 300); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "    Ready after $((_w*2))s" | tee -a "$LOG"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: server did not start in 600s" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null; exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a "$LOG"
    PGID=$(ps -o pgid= -p $SERVER_PID 2>/dev/null | tr -d ' ')
    [ -n "$PGID" ] && kill -TERM -$PGID 2>/dev/null || true
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
    local out=$OUTDIR/${policy}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy" | tee -a "$LOG"
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
        --save-result \
        --result-filename "../$out" \
        2>&1 | tee -a "../$LOG"
    cd ..
    echo "    Results saved → $out" | tee -a "$LOG"
}

# ── Policy 1: LRU (baseline) ────────────────────────────────────────────────
export VLLM_PREFIX_AWARE_EVICTION=0
start_server "LRU"
run_client "LRU"
stop_server

# ── Policy 2: Scheduler-Aware (prefix-protected eviction) ───────────────────
export VLLM_PREFIX_AWARE_EVICTION=1
start_server "PrefixAware"
run_client "PrefixAware"
stop_server

echo ""                              | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="    | tee -a "$LOG"
echo "Results in $OUTDIR/"          | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
