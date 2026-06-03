#!/bin/bash
# Llama-3.1-8B max_model_len=65000 — GPU 3, gpu_util=0.99
set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/llama_8b_65k_gpu3
PORT=8767
MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=54000
HF_MAX_LEN=53850
NUM_PROMPTS=40

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs PrefixAware — Llama-3.1-8B max_model_len=65000 GPU 3 ===" | tee "$LOG"
echo "Start : $(date)" | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 3 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "max_model_len : $MAX_MODEL_LEN  gpu_util=0.99  (A5000 max ~54800)" | tee -a "$LOG"

start_server() {
    echo "" | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$1" | tee -a "$LOG"
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --no-enable-dual-pool \
        --no-enable-cost-scoring \
        --no-enable-adaptive-scoring \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.99 \
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
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    # Wait for CUDA memory to fully release before starting next server
    sleep 15
}

run_client() {
    local policy=$1
    local out=$OUTDIR/${policy}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy" | tee -a "$LOG"
    cd benchmarks
    $PYTHON jenga_benchmark_serving.py \
        --port "$PORT" --model "$MODEL" \
        --dataset-path liyucheng/arxiv-march-2023 \
        --dataset-name hf --hf-subset ministral --hf-split train \
        --num-prompts "$NUM_PROMPTS" --seed 55555 \
        --hf-output-len 150 --hf-max-len "$HF_MAX_LEN" \
        --ignore-eos --save-result \
        --result-filename "../$out" \
        2>&1 | tee -a "../$LOG"
    cd ..
    echo "    Results saved → $out" | tee -a "$LOG"
}

export VLLM_PREFIX_AWARE_EVICTION=0
start_server "LRU"
run_client "LRU"
stop_server

export VLLM_PREFIX_AWARE_EVICTION=1
start_server "PrefixAware"
run_client "PrefixAware"
stop_server

echo "" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
