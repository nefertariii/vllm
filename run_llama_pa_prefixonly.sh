#!/bin/bash
# Llama-3.1-8B pre-fix PrefixAware only (Direction A reverted, buggy O(n) path)
set -e
cd /home/r14922173/vllm
export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PREFIX_AWARE_EVICTION=1

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/llama_8b_prefix_aware
PORT=8767
MODEL=meta-llama/Llama-3.1-8B-Instruct
mkdir -p "$OUTDIR"
LOG=$OUTDIR/bench_pa_prefixonly.log

echo "=== Llama pre-fix PrefixAware only (buggy O(n)) ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --no-enable-dual-pool \
    --no-enable-cost-scoring \
    --no-enable-adaptive-scoring \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.97 \
    --port $PORT \
    --no-enable-log-requests >> "$LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
for _w in $(seq 1 300); do
    curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && echo "Ready after $((_w*2))s" | tee -a "$LOG" && break
    sleep 2
done

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
    --hf-max-len 49850 \
    --ignore-eos \
    --save-result \
    --result-filename "../$OUTDIR/PrefixAware_prefix_buggy_50k.json" \
    2>&1 | tee -a "../$LOG"
cd ..

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
fuser -k ${PORT}/tcp 2>/dev/null || true
echo "=== DONE ===" | tee -a "$LOG"
