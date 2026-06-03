#!/bin/bash
# arXiv KV Cache Eviction Benchmark (Jenga-style, 4-policy 2×2 matrix)
# Dataset : liyucheng/arxiv-march-2023
# Models  : google/gemma-2-9b-it  |  meta-llama/Llama-3.1-8B-Instruct
# GPU     : RTX A6000 (CUDA_VISIBLE_DEVICES=1)
#
# 4 policies compared:
#   LRU               (cost_scoring=F, dual_pool=F)  ← baseline
#   Cost-Aware        (cost_scoring=T, dual_pool=F)
#   LRU+DualPool      (cost_scoring=F, dual_pool=T)
#   Full(Cost+DP)     (cost_scoring=T, dual_pool=T)
#
# KV cache sizes observed on this machine:
#   gemma-2-9b-it            :  ~58,646 tokens  (block_size=16)
#   Llama-3.1-8B-Instruct    : ~174,336 tokens  (block_size=16)
#
# Gemma params: 20 long (5500 tok) + 8 short (400 tok) = 110k + 3.2k = 113.2k
#   → 113.2k >> 58.6k KV cache → guaranteed eviction pressure
#
# Llama params: 8 long (25000 tok) + 6 short (2000 tok) = 200k + 12k = 212k
#   → 212k >> 174k KV cache → guaranteed eviction pressure

set -e
cd /home/r14922173/vllm

export HF_TOKEN=${HF_TOKEN:?Please export HF_TOKEN before running this script}
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0   # skip flashinfer JIT (no nvcc on PATH)

PYTHON=/home/r14922173/vllm_work/envs/vllm-src-py312/bin/python
SCRIPT=benchmarks/benchmark_arxiv_eviction.py
OUTDIR=results_arxiv
GPU_MEM=0.85
MAX_TOKENS=64

mkdir -p $OUTDIR
LOG=$OUTDIR/benchmark_$(date +%Y%m%d_%H%M%S).log

echo "=== arXiv Eviction Benchmark ===" | tee $LOG
echo "Start : $(date)" | tee -a $LOG
echo "GPU   : $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader)" | tee -a $LOG
echo "" | tee -a $LOG

# ── Gemma-2-9b-it ─────────────────────────────────────────────────────────
# 20 long × 5500 + 8 short × 400 = 113.2k >> 58.6k cache → guaranteed eviction
echo ">>> [gemma-2-9b-it] START $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --model google/gemma-2-9b-it \
    --num-long 20 --long-tokens 5500 \
    --num-short 8  --short-tokens 400 \
    --num-questions 4 \
    --max-tokens $MAX_TOKENS \
    --gpu-memory-utilization $GPU_MEM \
    --seed 42 \
    --output $OUTDIR/result_gemma_arxiv.json \
    2>&1 | tee -a $LOG
echo ">>> [gemma-2-9b-it] DONE $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# ── Llama-3.1-8B-Instruct ─────────────────────────────────────────────────
# 8 long × 25000 + 6 short × 2000 = 212k >> 174k cache → guaranteed eviction
echo ">>> [Llama-3.1-8B] START $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-long 8  --long-tokens 25000 \
    --num-short 6 --short-tokens 2000 \
    --num-questions 4 \
    --max-tokens $MAX_TOKENS \
    --gpu-memory-utilization $GPU_MEM \
    --seed 42 \
    --output $OUTDIR/result_llama_arxiv.json \
    2>&1 | tee -a $LOG
echo ">>> [Llama-3.1-8B] DONE $(date)" | tee -a $LOG
echo "" | tee -a $LOG

echo "=== ALL DONE: $(date) ===" | tee -a $LOG
echo "Results in $OUTDIR/"
