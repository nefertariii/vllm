#!/bin/bash
# LRU vs Adaptive-v2 (Welford CV) — Gemma-3-4B-it on A5000 GPU 3
#
# 2 policies:
#   LRU       --no-enable-cost-scoring
#   Adaptive  --enable-cost-scoring --enable-adaptive-scoring (Welford O(1) CV)
#
# Usage:
#   tmux new-session -d -s benchadaptive \
#     "bash run_gemma3_4b_adaptive_v2_benchmark.sh 2>&1 | tee bench_adaptive_v2.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
# GPU 3 = RTX A5000 24GB, Bus 65:00.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/jenga_gemma3_4b_adaptive_v2
PORT=8768
MODEL=google/gemma-3-4b-it
GPU_MEM=0.85
NUM_PARAMS_B=4.0

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs Adaptive-v2 (Welford) — Gemma-3-4B-it ===" | tee "$LOG"
echo "Start : $(date)"                                         | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 3 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

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
        --num-params-b "$NUM_PARAMS_B" \
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

# Policy 1: LRU
start_server "--no-enable-cost-scoring --no-enable-adaptive-scoring" "LRU"
run_client "LRU"
stop_server

# Policy 2: Adaptive-v2 (Welford CV, O(1) per freed block)
start_server "--enable-cost-scoring --enable-adaptive-scoring" "Adaptive"
run_client "Adaptive"
stop_server

echo ""                              | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="    | tee -a "$LOG"
echo "Results in $OUTDIR/"          | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
