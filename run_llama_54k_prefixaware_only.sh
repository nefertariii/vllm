#!/bin/bash
# PrefixAware only — Llama-3.1-8B 54k GPU 3 (LRU already done)
set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_PREFIX_AWARE_EVICTION=1

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/llama_8b_65k_gpu3
PORT=8767
MODEL=meta-llama/Llama-3.1-8B-Instruct
TS=20260602_034415   # same TS as LRU run

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 10  # wait for CUDA memory to fully release

LOG=$OUTDIR/bench_${TS}.log

echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Restarting PrefixAware server" | tee -a "$LOG"

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --enable-prefix-caching \
    --no-enable-dual-pool \
    --no-enable-cost-scoring \
    --no-enable-adaptive-scoring \
    --max-model-len 54000 \
    --gpu-memory-utilization 0.99 \
    --port $PORT \
    --no-enable-log-requests \
    >> "$LOG" 2>&1 &
SERVER_PID=$!

echo "    Waiting for server PID=$SERVER_PID ..." | tee -a "$LOG"
for _w in $(seq 1 300); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "    Ready after $((_w*2))s" | tee -a "$LOG"
        break
    fi
    sleep 2
done

cd benchmarks
$PYTHON jenga_benchmark_serving.py \
    --port "$PORT" --model "$MODEL" \
    --dataset-path liyucheng/arxiv-march-2023 \
    --dataset-name hf --hf-subset ministral --hf-split train \
    --num-prompts 40 --seed 55555 \
    --hf-output-len 150 --hf-max-len 53850 \
    --ignore-eos --save-result \
    --result-filename "../$OUTDIR/PrefixAware_${TS}.json" \
    2>&1 | tee -a "../$LOG"
cd ..

kill $SERVER_PID 2>/dev/null || true
echo "=== PrefixAware DONE ===" | tee -a "$LOG"
