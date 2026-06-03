#!/bin/bash
# Gemma-3-12b-it benchmark: Jenga AE 相同設定 × 4 eviction policies
#
# GPU: A6000 with gpu_memory_utilization=0.533 → 模擬 L4 24GB
# Dataset: liyucheng/arxiv-march-2023, subset=ministral (同 Jenga figure14)
# Client: jenga_benchmark_serving.py (Jenga AE 原版)
#
# Usage:
#   tmux new-session -d -s jenga "bash run_jenga_gemma3_benchmark.sh 2>&1 | tee bench_jenga_gemma3.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/jenga_gemma3
PORT=8765
MODEL=google/gemma-3-12b-it
GPU_MEM=0.85    # A6000 46GB

mkdir -p $OUTDIR
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/serving_${TS}.log

echo "=== Jenga Gemma-3 Benchmark (4 policies) ===" | tee $LOG
echo "Start : $(date)" | tee -a $LOG
echo "GPU   : $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader 2>/dev/null)" | tee -a $LOG
echo "Model : $MODEL" | tee -a $LOG
echo "gpu_memory_utilization: $GPU_MEM" | tee -a $LOG

start_server() {
    local cost_flag=$1 pool_flag=$2 policy=$3

    echo "" | tee -a $LOG
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy" | tee -a $LOG

    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        $cost_flag \
        $pool_flag \
        --gpu-memory-utilization $GPU_MEM \
        --port $PORT \
        --no-enable-log-requests \
        >> $LOG 2>&1 &
    SERVER_PID=$!

    echo "    Waiting for server PID=$SERVER_PID ..." | tee -a $LOG
    for _w in $(seq 1 300); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "    Ready after $((_w*2))s" | tee -a $LOG
            return 0
        fi
        sleep 2
    done
    echo "ERROR: server did not start in 600s" | tee -a $LOG
    kill $SERVER_PID 2>/dev/null || true
    exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a $LOG
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
}

run_client() {
    local policy=$1
    local safe=${policy//+/_}
    local out=$OUTDIR/${safe}_${TS}.json

    echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy" | tee -a $LOG

    # Jenga AE 完全相同的參數 (figure14 Gemma-3 設定)
    cd benchmarks
    $PYTHON jenga_benchmark_serving.py \
        --port $PORT \
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
    echo "    Results saved → $out" | tee -a $LOG
}

POLICIES=("LRU" "Cost-Aware" "LRU+DualPool" "Full")
COST=("--no-enable-cost-scoring" "--enable-cost-scoring"
      "--no-enable-cost-scoring" "--enable-cost-scoring")
POOL=("--no-enable-dual-pool"   "--no-enable-dual-pool"
      "--enable-dual-pool"      "--enable-dual-pool")

for i in 0 1 2 3; do
    start_server "${COST[$i]}" "${POOL[$i]}" "${POLICIES[$i]}"
    run_client "${POLICIES[$i]}"
    stop_server
done

echo "" | tee -a $LOG
echo "=== ALL DONE: $(date) ===" | tee -a $LOG
echo "Results in $OUTDIR/" | tee -a $LOG

# 確認 server 完全停止
fuser -k ${PORT}/tcp 2>/dev/null || true
echo "Server port $PORT cleared." | tee -a $LOG
