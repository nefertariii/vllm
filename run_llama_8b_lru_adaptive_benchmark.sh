#!/bin/bash
# LRU vs Adaptive (random sampling) — Llama-3.1-8B-Instruct on A5000 GPU 3
#
# 2 policies:
#   LRU      --no-enable-cost-scoring
#   Adaptive --enable-cost-scoring --enable-adaptive-scoring (O(50) sampling)
#
# max-model-len=27000: Llama native ctx=131072 but A5000 only has ~3.4GB KV cache
#
# Usage:
#   tmux new-session -d -s benchllama2 \
#     "bash run_llama_8b_lru_adaptive_benchmark.sh 2>&1 | tee bench_llama_adaptive.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/jenga_llama_8b_adaptive
PORT=8767
MODEL=meta-llama/Llama-3.1-8B-Instruct
GPU_MEM=0.85
NUM_PARAMS_B=8.0
MAX_MODEL_LEN=27000
HF_MAX_LEN=26850   # 27000 - 150 output tokens

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs Adaptive — Llama-3.1-8B-Instruct ===" | tee "$LOG"
echo "Start : $(date)"                                    | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 3 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "max_model_len : $MAX_MODEL_LEN"                    | tee -a "$LOG"

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
        --max-model-len "$MAX_MODEL_LEN" \
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
        --hf-max-len "$HF_MAX_LEN" \
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

# Policy 2: Adaptive (random sampling O(50), pressure floor=0.30)
start_server "--enable-cost-scoring --enable-adaptive-scoring" "Adaptive"
run_client "Adaptive"
stop_server

echo ""                              | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="    | tee -a "$LOG"
echo "Results in $OUTDIR/"          | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
