#!/bin/bash
# 4-policy benchmark — Llama-3.1-8B-Instruct + MMLU-pro on A5000 (combined-prefix-aware branch)
#
# Policies:
#   LRU          eviction=LRU,  sched=FCFS
#   EvictAware   eviction=PA,   sched=FCFS   (VLLM_PREFIX_AWARE_EVICTION=1)
#   SchedAware   eviction=LRU,  sched=PA     (--enable-prefix-aware-scheduling)
#   Both         eviction=PA,   sched=PA
#
# Dataset: MMLU-pro (meta-llama/Llama-3.1-405B-evals)
#   40 prompts, max_model_len=4096, hf_output_len=20, seed=55555, ignore-eos
#
# NOTE: MMLU prompts are independent (no shared prefix) → expect all policies ≈ LRU.
#       This is a control/baseline experiment for the combined-prefix-aware branch.
#
# Usage:
#   tmux new-session -d -s bench_llama_mmlu \
#     "bash run_llama_8b_mmlu_combined_benchmark.sh 2>&1 | tee bench_llama_mmlu.log"

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/r14922173/vllm_work/cache/huggingface

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/llama_8b_mmlu_combined
PORT=8767
MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=4096
HF_MAX_LEN=3946          # 4096 - 150 output tokens
HF_OUTPUT_LEN=20
NUM_PROMPTS=40

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== 4-policy benchmark — Llama-3.1-8B-Instruct + MMLU-pro (combined branch) ===" | tee "$LOG"
echo "Start         : $(date)"                                                            | tee -a "$LOG"
echo "GPU           : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "max_model_len : $MAX_MODEL_LEN"                                                     | tee -a "$LOG"
echo "Dataset       : MMLU-pro (40 prompts, no shared prefix — control experiment)"      | tee -a "$LOG"
echo "Branch        : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee -a "$LOG"

start_server() {
    local policy=$1
    local extra_flags=$2
    echo ""                                                                   | tee -a "$LOG"
    echo ">>> [$(date +%H:%M:%S)] Starting server: policy=$policy"           | tee -a "$LOG"
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --no-enable-dual-pool \
        --no-enable-cost-scoring \
        --no-enable-adaptive-scoring \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.95 \
        --port $PORT \
        --no-enable-log-requests \
        $extra_flags \
        >> "$LOG" 2>&1 &
    SERVER_PID=$!

    echo "    Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
    for _w in $(seq 1 300); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && \
            echo "    Ready after $((_w*2))s" | tee -a "$LOG" && return 0
        sleep 2
    done
    echo "ERROR: server did not start in 600s" | tee -a "$LOG"
    kill $SERVER_PID 2>/dev/null; exit 1
}

stop_server() {
    echo "    Stopping server PID=$SERVER_PID" | tee -a "$LOG"
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
start_server "LRU" ""
run_client "LRU"
stop_server

# ── Policy 2: EvictAware (scheduler-aware eviction only) ─────────────────────
export VLLM_PREFIX_AWARE_EVICTION=1
start_server "EvictAware" ""
run_client "EvictAware"
stop_server

# ── Policy 3: SchedAware (prefix-aware scheduling only) ──────────────────────
export VLLM_PREFIX_AWARE_EVICTION=0
start_server "SchedAware" "--enable-prefix-aware-scheduling"
run_client "SchedAware"
stop_server

# ── Policy 4: Both ───────────────────────────────────────────────────────────
export VLLM_PREFIX_AWARE_EVICTION=1
start_server "Both" "--enable-prefix-aware-scheduling"
run_client "Both"
stop_server

echo ""                               | tee -a "$LOG"
echo "=== ALL DONE: $(date) ==="     | tee -a "$LOG"
echo "Results in $OUTDIR/"           | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true
