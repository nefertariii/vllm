#!/bin/bash
# Online Serving Benchmark — 4-Policy Comparison (2-phase arXiv)
#
# Usage:
#   bash run_serving_benchmark.sh gemma   # Gemma-2-9b-it only
#   bash run_serving_benchmark.sh llama   # Llama-3.1-8B only
#   bash run_serving_benchmark.sh all     # Both models (default)
#
# Run fully automated in tmux (no interactive prompts):
#   tmux new-session -d -s bench "bash run_serving_benchmark.sh gemma 2>&1 | tee bench.log"
#
# 4 policies tested (2x2 matrix):
#   LRU          --no-enable-cost-scoring  --no-enable-dual-pool
#   Cost-Aware   --enable-cost-scoring     --no-enable-dual-pool
#   LRU+DualPool --no-enable-cost-scoring  --enable-dual-pool
#   Full         --enable-cost-scoring     --enable-dual-pool

set -e
cd /home/r14922173/vllm

TARGET=${1:-all}

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
# Pin GPU1 (A6000 46068MiB, Bus 31:00.0)
# CUDA_DEVICE_ORDER=PCI_BUS_ID ensures index 1 = second GPU by PCI bus = GPU1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
CLIENT=benchmarks/benchmark_arxiv_serving.py
OUTDIR=results_arxiv/serving
GPU_MEM=0.85
MAX_TOKENS=64
PORT=8765
RUNS=3   # number of repeated runs per policy (for mean ± std)

mkdir -p $OUTDIR
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/serving_${TARGET}_${TS}.log

echo "=== Online Serving Benchmark (target=$TARGET) ===" | tee $LOG
echo "Start : $(date)" | tee -a $LOG
echo "GPU   : $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader 2>/dev/null)" | tee -a $LOG

# ── Helper: start server, wait until healthy ──────────────────────────────
start_server() {
    local model=$1 cost_flag=$2 pool_flag=$3 policy=$4

    echo "" | tee -a $LOG
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy" | tee -a $LOG

    # Kill any previous server on this port
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        $cost_flag \
        $pool_flag \
        --gpu-memory-utilization $GPU_MEM \
        --port $PORT \
        >> $LOG 2>&1 &
    SERVER_PID=$!

    echo "    Waiting for server PID=$SERVER_PID …" | tee -a $LOG
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "    Ready after $((i*2))s" | tee -a $LOG
            return 0
        fi
        sleep 2
    done
    echo "ERROR: server did not start in 240s" | tee -a $LOG
    kill $SERVER_PID 2>/dev/null || true
    exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a $LOG
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
}

# ── Run one model across all 4 policies ──────────────────────────────────
run_model() {
    local model=$1 tag=$2
    local num_long=$3 long_tok=$4 num_short=$5 short_tok=$6

    echo "" | tee -a $LOG
    echo "======================================" | tee -a $LOG
    echo "MODEL: $model" | tee -a $LOG
    echo "  ${num_long}×${long_tok}tok long + ${num_short}×${short_tok}tok short" | tee -a $LOG
    echo "======================================" | tee -a $LOG

    local ALL_RESULTS=()

    # policy name / cost flag / pool flag
    local POLICIES=("LRU" "Cost-Aware" "LRU+DualPool" "Full")
    local COST=("--no-enable-cost-scoring" "--enable-cost-scoring"
                "--no-enable-cost-scoring" "--enable-cost-scoring")
    local POOL=("--no-enable-dual-pool"    "--no-enable-dual-pool"
                "--enable-dual-pool"       "--enable-dual-pool")

    for i in 0 1 2 3; do
        local policy=${POLICIES[$i]}
        local safe=${policy//+/_}

        start_server "$model" "${COST[$i]}" "${POOL[$i]}" "$policy"

        for run in $(seq 1 $RUNS); do
            local out=$OUTDIR/${tag}_${safe}_run${run}_${TS}.json
            ALL_RESULTS+=("$out")

            echo ">>> [$(date +%H:%M:%S)] Client: policy=$policy run=$run/$RUNS" | tee -a $LOG
            $PYTHON $CLIENT \
                --host localhost --port $PORT \
                --model "$model" \
                --policy-name "$policy" \
                --num-long  $num_long  --long-tokens  $long_tok \
                --num-short $num_short --short-tokens $short_tok \
                --num-questions 4 \
                --max-tokens $MAX_TOKENS \
                --seed 42 \
                --output "$out" \
                2>&1 | tee -a $LOG
        done

        stop_server
    done

    echo "" | tee -a $LOG
    echo "====== $tag COMPARISON TABLE (${RUNS} runs/policy, mean±std) ======" | tee -a $LOG
    $PYTHON $CLIENT --compare "${ALL_RESULTS[@]}" 2>&1 | tee -a $LOG
}

# ── Dispatch ──────────────────────────────────────────────────────────────
# Gemma: 20 long×5500 + 8 short×400 = 113.2k >> 58.6k cache
# Llama: 8 long×25000 + 6 short×2000 = 212k   >> 174k  cache

if [[ "$TARGET" == "gemma" || "$TARGET" == "all" ]]; then
    run_model google/gemma-2-9b-it gemma 20 5500 8 400
fi

if [[ "$TARGET" == "llama" || "$TARGET" == "all" ]]; then
    run_model meta-llama/Llama-3.1-8B-Instruct llama 8 25000 6 2000
fi

echo "" | tee -a $LOG
echo "=== ALL DONE: $(date) ===" | tee -a $LOG
echo "Results in $OUTDIR/"
echo "Log: $LOG"
