#!/bin/bash
# LRU vs Dual-Pool KV Cache Eviction Benchmark
# Dataset : ccdv/arxiv-summarization (Cohan et al., NAACL 2018)
# Models  : google/gemma-2-9b-it  |  meta-llama/Llama-3.1-8B-Instruct
# GPU     : RTX A6000 (CUDA_VISIBLE_DEVICES=1)

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0   # skip flashinfer JIT (no nvcc needed)

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
SCRIPT=benchmarks/benchmark_eviction_policy.py
OUTDIR=results_eviction
GPU_MEM=0.85

# Benchmark parameters
# A6000 KV cache ≈ 73k tokens.
# 40 warmup papers (≈ 27 long × 4000w + 13 short × 1500w ≈ 162k warmup tokens)
# → 2× cache overflow ensures real eviction pressure.
# 20 new papers in timed phase add another eviction wave.
NUM_WARMUP=40
NUM_NEW=20
MAX_WORDS=4000   # safe for gemma 8192 ctx window
MAX_TOKENS=128

mkdir -p $OUTDIR
LOG=$OUTDIR/benchmark_$(date +%Y%m%d_%H%M%S).log

echo "=== Eviction Policy Benchmark ===" | tee $LOG
echo "Start  : $(date)"                  | tee -a $LOG
echo "GPU    : $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader)" | tee -a $LOG
echo "Dataset: ccdv/arxiv-summarization (Cohan et al., NAACL 2018)" | tee -a $LOG
echo ""  | tee -a $LOG

run_one() {
    local model=$1
    local tag=$(echo "$model" | tr '/' '_')
    local outfile="$OUTDIR/result_${tag}_arxiv.json"

    echo ">>> [$tag] START $(date)" | tee -a $LOG
    $PYTHON $SCRIPT \
        --model "$model" \
        --num-warmup-papers $NUM_WARMUP \
        --num-new-papers    $NUM_NEW    \
        --max-prompt-words  $MAX_WORDS  \
        --max-tokens        $MAX_TOKENS \
        --gpu-memory-utilization $GPU_MEM \
        --output "$outfile" \
        2>&1 | tee -a $LOG
    echo ">>> [$tag] DONE $(date)" | tee -a $LOG
    echo "" | tee -a $LOG
}

run_one "google/gemma-2-9b-it"
run_one "meta-llama/Llama-3.1-8B-Instruct"

echo "=== ALL DONE: $(date) ===" | tee -a $LOG
echo "Results in $OUTDIR/"
