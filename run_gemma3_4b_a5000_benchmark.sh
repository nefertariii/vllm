#!/bin/bash
# Gemma-3-4B-it benchmark: Jenga AE 相同設定 × 3 eviction policies
#
# GPU    : A5000 (24GB, Bus 17:00.0, CUDA_VISIBLE_DEVICES=0)
# Model  : google/gemma-3-4b-it
# Dataset: liyucheng/arxiv-march-2023, subset=ministral (同 Jenga figure14)
# Client : jenga_benchmark_serving.py — 完全遵循 Jenga AE 參數
#          (無 --max-model-len, 無 --hf-max-len)
#
# 3 policies:
#   LRU        --no-enable-cost-scoring  --no-enable-adaptive-scoring
#   Cost-Aware --enable-cost-scoring     --no-enable-adaptive-scoring  (靜態權重)
#   Adaptive   --enable-cost-scoring     --enable-adaptive-scoring     (動態權重)
#
# Usage:
#   tmux new-session -d -s bench4b \
#     "bash run_gemma3_4b_a5000_benchmark.sh 2>&1 | tee bench_gemma3_4b.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
# GPU 0 = RTX A5000 24GB, Bus 17:00.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/jenga_gemma3_4b
PORT=8766
MODEL=google/gemma-3-4b-it
GPU_MEM=0.85
NUM_PARAMS_B=4.0   # used by Adaptive policy (model_factor in adaptive scoring)

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== Jenga Gemma-3-4B-it Benchmark (3 policies) ===" | tee "$LOG"
echo "Start : $(date)"                                      | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "Model : $MODEL"                                       | tee -a "$LOG"
echo "gpu_memory_utilization : $GPU_MEM"                   | tee -a "$LOG"

# ── Helper: start server ──────────────────────────────────────────────────
start_server() {
    local cost_flag=$1 adaptive_flag=$2 policy=$3

    echo ""                                                             | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy"     | tee -a "$LOG"

    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --no-enable-dual-pool \
        $cost_flag \
        $adaptive_flag \
        --num-params-b "$NUM_PARAMS_B" \
        --gpu-memory-utilization $GPU_MEM \
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

# ── 3 policies ────────────────────────────────────────────────────────────
POLICIES=("LRU"                  "Cost-Aware"              "Adaptive")
COST=(    "--no-enable-cost-scoring" "--enable-cost-scoring"   "--enable-cost-scoring")
ADAPTIVE=("--no-enable-adaptive-scoring" "--no-enable-adaptive-scoring" "--enable-adaptive-scoring")

for i in 0 1 2; do
    start_server "${COST[$i]}" "${ADAPTIVE[$i]}" "${POLICIES[$i]}"
    run_client "${POLICIES[$i]}"
    stop_server
done

echo ""                               | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="     | tee -a "$LOG"
echo "Results in $OUTDIR/"           | tee -a "$LOG"
echo "Log: $LOG"

fuser -k ${PORT}/tcp 2>/dev/null || true
echo "Server port $PORT cleared."
