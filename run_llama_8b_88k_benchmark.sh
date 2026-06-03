#!/bin/bash
# Llama-3.1-8B max_model_len=88000 test — GPU 0
set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
OUTDIR=results_arxiv/llama_8b_88k
PORT=8769
MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=88000
HF_MAX_LEN=87850
NUM_PROMPTS=40

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUTDIR/bench_${TS}.log

echo "=== Llama-3.1-8B max_model_len=88000 on GPU 0 ===" | tee "$LOG"
echo "Start : $(date)" | tee -a "$LOG"
echo "GPU   : $(nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

export VLLM_PREFIX_AWARE_EVICTION=0
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --enable-prefix-caching \
    --no-enable-dual-pool \
    --no-enable-cost-scoring \
    --no-enable-adaptive-scoring \
    --max-model-len "$MAX_MODEL_LEN" \
    --port $PORT \
    --no-enable-log-requests \
    >> "$LOG" 2>&1 &
SERVER_PID=$!

echo "Server PID=$SERVER_PID, waiting..." | tee -a "$LOG"
for _w in $(seq 1 300); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Ready after $((_w*2))s" | tee -a "$LOG"
        break
    fi
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
    --result-filename "../$OUTDIR/LRU_88k_${TS}.json" \
    2>&1 | tee -a "../$LOG"
cd ..

kill $SERVER_PID 2>/dev/null || true
echo "=== DONE ===" | tee -a "$LOG"
