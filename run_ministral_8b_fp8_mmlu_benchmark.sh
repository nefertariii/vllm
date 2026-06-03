#!/bin/bash
# LRU vs PrefixAware — Ministral-8B-Instruct FP8 quantization + MMLU-pro
# GPU 3 (A5000 24GB), port 8769
#
# Usage:
#   tmux new-session -d -s bench_ministral_fp8_mmlu \
#     "bash run_ministral_8b_fp8_mmlu_benchmark.sh 2>&1 | tee bench_ministral_fp8_mmlu.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/r14922173/vllm_work/cache/huggingface

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/ministral_8b_fp8_mmlu
PORT=8769
MODEL=mistralai/Ministral-8B-Instruct-2410
MAX_MODEL_LEN=4096
HF_MAX_LEN=3946    # 4096 - 150
HF_OUTPUT_LEN=20   # MMLU multiple-choice answers are short
NUM_PROMPTS=40

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== LRU vs PrefixAware — Ministral-8B FP8 + MMLU-pro on A5000 ===" | tee "$LOG"
echo "Start         : $(date)"                                              | tee -a "$LOG"
echo "GPU           : $(nvidia-smi -i 3 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "Model         : $MODEL (FP8 quantization)"                           | tee -a "$LOG"
echo "Dataset       : MMLU-pro (meta-llama/Llama-3.1-405B-evals)"         | tee -a "$LOG"
echo "max_model_len : $MAX_MODEL_LEN"                                      | tee -a "$LOG"
echo "hf_output_len : $HF_OUTPUT_LEN"                                     | tee -a "$LOG"
echo "Prompts/run   : $NUM_PROMPTS"                                        | tee -a "$LOG"

start_server() {
    local policy=$1
    echo ""                                                                | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy"        | tee -a "$LOG"
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --quantization fp8 \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --no-enable-dual-pool \
        --no-enable-cost-scoring \
        --no-enable-adaptive-scoring \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.95 \
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
    kill $SERVER_PID 2>/dev/null || true
    exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a "$LOG"
    PGID=$(ps -o pgid= -p $SERVER_PID 2>/dev/null | tr -d ' ')
    [ -n "$PGID" ] && kill -TERM -$PGID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    echo "    Waiting for GPU 3 memory to clear..." | tee -a "$LOG"
    for i in $(seq 1 30); do
        FREE=$(nvidia-smi -i 3 --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$FREE" -gt 20000 ] && break
        sleep 2
    done
    echo "    GPU 3 cleared." | tee -a "$LOG"
}

run_client() {
    local policy=$1
    local out=$OUTDIR/${policy}_${TS}.json
    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy" | tee -a "$LOG"
    cd benchmarks
    $PYTHON jenga_benchmark_serving.py \
        --port "$PORT" \
        --model "$MODEL" \
        --dataset-path "meta-llama/Llama-3.1-405B-evals" \
        --dataset-name hf \
        --hf-subset "llama_31_405b_evals__mmlu_pro__details" \
        --hf-split "latest" \
        --num-prompts "$NUM_PROMPTS" \
        --seed 55555 \
        --hf-output-len "$HF_OUTPUT_LEN" \
        --hf-max-len "$HF_MAX_LEN" \
        --ignore-eos \
        --save-result \
        --result-filename "../$out" \
        2>&1 | tee -a "../$LOG"
    cd ..
    echo "    Results saved → $out" | tee -a "$LOG"
}

# ── Policy 1: LRU ─────────────────────────────────────────────────────────────
export VLLM_PREFIX_AWARE_EVICTION=0
start_server "LRU"
run_client "LRU"
stop_server

# ── Policy 2: PrefixAware ─────────────────────────────────────────────────────
export VLLM_PREFIX_AWARE_EVICTION=1
start_server "PrefixAware"
run_client "PrefixAware"
stop_server

echo ""                           | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
echo "Results in $OUTDIR/"       | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
