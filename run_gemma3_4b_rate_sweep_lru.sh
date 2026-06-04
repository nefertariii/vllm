#!/bin/bash
# Rate sweep — Gemma-3-4B-IT, LRU only
# GPU 0 (A5000), port 8768
# Rates: 0.2:0.2:4.0 (20 points, same as Jenga figure15)

set -u
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:-hf_placeholder}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/r14922173/vllm_work/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_PREFIX_AWARE_EVICTION=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/gemma3_4b_rate_sweep
PORT=8768
MODEL=google/gemma-3-4b-it
NUM_PROMPTS=40
RATES="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0"

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/lru_${TS}.log

echo "=== Rate sweep — Gemma-3-4B-IT LRU (GPU 0) ===" | tee "$LOG"
echo "Start  : $(date)"                                 | tee -a "$LOG"
echo "GPU    : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "Branch : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "Rates  : $RATES"                                  | tee -a "$LOG"

start_server() {
    echo "" | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting LRU server (rate=$1)" | tee -a "$LOG"
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
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && \
            echo "    Ready after $((_w*2))s" | tee -a "$LOG" && return 0
        sleep 2
    done
    echo "ERROR: server did not start in 600s — skipping this rate" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null
    return 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    for i in $(seq 1 30); do
        FREE=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$FREE" -gt 20000 ] && break
        sleep 2
    done
    echo "    GPU 0 cleared." | tee -a "$LOG"
}

run_client() {
    local rate=$1
    local rate_str
    rate_str=$(echo "$rate" | tr '.' '_')
    local out=$OUTDIR/LRU_rate${rate_str}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: LRU  rate=$rate req/s" | tee -a "$LOG"
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
    echo "    Saved → $out" | tee -a "$LOG"
}

for rate in $RATES; do
    if start_server "$rate"; then
        run_client "$rate" || echo "    WARN: client failed for rate=$rate, continuing" | tee -a "$LOG"
        stop_server
    else
        echo "    SKIP rate=$rate (server failed to start)" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== LRU DONE: $(date) ===" | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
