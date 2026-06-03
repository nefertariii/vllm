#!/bin/bash
# 4-policy benchmark — Ministral-8B-Instruct FP8 + arXiv QA on A5000 (combined-prefix-aware branch)
#
# Policies:
#   LRU          eviction=LRU,  sched=FCFS
#   EvictAware   eviction=PA,   sched=FCFS   (VLLM_PREFIX_AWARE_EVICTION=1)
#   SchedAware   eviction=LRU,  sched=PA     (--enable-prefix-aware-scheduling)
#   Both         eviction=PA,   sched=PA
#
# Dataset: liyucheng/arxiv-march-2023, subset=ministral
#   10 articles × 4 questions = 40 prompts, shuffled seed=55555
#   hf_max_len=32618, hf_output_len=150, rate=inf, ignore-eos
#
# Usage:
#   tmux new-session -d -s bench_ministral_combined \
#     "bash run_ministral_8b_fp8_combined_benchmark.sh 2>&1 | tee bench_ministral_combined.log"

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
OUTDIR=results_arxiv/ministral_8b_fp8_combined
PORT=8769
MODEL=mistralai/Ministral-8B-Instruct-2410
MAX_MODEL_LEN=32768
HF_MAX_LEN=32618
NUM_PROMPTS=40

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== 4-policy benchmark — Ministral-8B FP8 + arXiv QA (combined branch) ===" | tee "$LOG"
echo "Start         : $(date)"                                                       | tee -a "$LOG"
echo "GPU           : $(nvidia-smi -i 3 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"
echo "max_model_len : $MAX_MODEL_LEN"                                                | tee -a "$LOG"
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
        --dataset-path liyucheng/arxiv-march-2023 \
        --dataset-name hf \
        --hf-subset ministral \
        --hf-split train \
        --num-prompts "$NUM_PROMPTS" \
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

# ── Policy 1: LRU ─────────────────────────────────────────────────────────────
export VLLM_PREFIX_AWARE_EVICTION=0
start_server "LRU" ""
run_client "LRU"
stop_server

# ── Policy 2: EvictAware (prefix-aware eviction only) ────────────────────────
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
