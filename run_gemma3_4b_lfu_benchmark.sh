#!/bin/bash
# LRU vs LRU+LFU benchmark — Gemma-3-4B-it on A5000 (GPU 0)
#
# 2 policies:
#   LRU      --no-enable-cost-scoring          (O(1) popleft, pure recency)
#   LRU+LFU  --enable-cost-scoring --eviction-w-cost 0.0  (O(n) scan, recency+LFU only)
#
# Usage:
#   tmux new-session -d -s benchlfu \
#     "bash run_gemma3_4b_lfu_benchmark.sh 2>&1 | tee bench_lfu.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/jenga_gemma3_4b_lfu
PORT=8766
MODEL=google/gemma-3-4b-it
GPU_MEM=0.85

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs LRU+LFU — Gemma-3-4B-it ===" | tee "$LOG"
echo "Start : $(date)"                           | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

start_server() {
    local extra_flags=$1 policy=$2
    echo ""                                                            | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy"    | tee -a "$LOG"
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --no-enable-dual-pool \
        --no-enable-adaptive-scoring \
        --gpu-memory-utilization $GPU_MEM \
        --port $PORT \
        --no-enable-log-requests \
        $extra_flags \
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
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
}

run_client() {
    local policy=$1
    local safe=${policy//+/_}
    local out=$OUTDIR/${safe}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy" | tee -a "$LOG"
    cd benchmarks
    $PYTHON jenga_benchmark_serving.py \
        --port "$PORT" \
        --model "$MODEL" \
        --dataset-path liyucheng/arxiv-march-2023 \
        --dataset-name hf \
        --hf-subset ministral \
        --hf-split train \
        --num-prompts 40 \
        --seed 55555 \
        --hf-output-len 150 \
        --ignore-eos \
        --save-result \
        --result-filename "../$out" \
        2>&1 | tee -a "../$LOG"
    cd ..
    echo "    Results saved → $out" | tee -a "$LOG"
}

# Policy 1: LRU — O(1) popleft, no scoring
start_server "--no-enable-cost-scoring" "LRU"
run_client "LRU"
stop_server

# Policy 2: LRU+LFU — O(n) scan, w_cost=0 (cost component disabled)
start_server "--enable-cost-scoring --eviction-w-cost 0.0" "LRU+LFU"
run_client "LRU+LFU"
stop_server

echo ""                              | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="    | tee -a "$LOG"
echo "Results in $OUTDIR/"          | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
