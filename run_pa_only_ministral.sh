#!/bin/bash
set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/r14922173/vllm_work/cache/huggingface
export VLLM_PREFIX_AWARE_EVICTION=1

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/ministral_8b_prefix_aware
PORT=8769
MODEL=mistralai/Ministral-8B-Instruct-2410
MAX_MODEL_LEN=32768
HF_MAX_LEN=32618
NUM_PROMPTS=40
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_pa_only_${TS}.log

mkdir -p "$OUTDIR"
echo "=== PrefixAware ONLY — Ministral-8B on A5000 ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --no-enable-dual-pool --no-enable-cost-scoring --no-enable-adaptive-scoring \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.95 \
    --port $PORT \
    --no-enable-log-requests \
    >> "$LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
for _w in $(seq 1 300); do
    curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && echo "Ready after $((_w*2))s" | tee -a "$LOG" && break
    sleep 2
done

cd benchmarks
$PYTHON jenga_benchmark_serving.py \
    --port "$PORT" --model "$MODEL" \
    --dataset-path liyucheng/arxiv-march-2023 \
    --dataset-name hf --hf-subset ministral --hf-split train \
    --num-prompts "$NUM_PROMPTS" --seed 55555 \
    --hf-output-len 150 --hf-max-len "$HF_MAX_LEN" \
    --ignore-eos --save-result \
    --result-filename "../$OUTDIR/PrefixAware_${TS}.json" \
    2>&1 | tee -a "../$LOG"
cd ..

kill $SERVER_PID 2>/dev/null || true
fuser -k ${PORT}/tcp 2>/dev/null || true
echo "=== DONE: $(date) ===" | tee -a "$LOG"
