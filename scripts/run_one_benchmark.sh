#!/bin/bash
# Run one benchmark pass: launch server → run client → save results → kill server.
#
# Usage:
#   bash scripts/run_one_benchmark.sh MODEL_TAG RESULT_SUBDIR PORT RATE MODEL [extra vllm args...]
#
# MODEL_TAG     : "llama" or "gemma"  (controls dataset subset selection)
# RESULT_SUBDIR : path under results/  e.g. "llama/baseline/run1"
# PORT          : server port
# RATE          : request rate in req/s  (use "inf" for max throughput)
# MODEL         : full HuggingFace model ID
# extra args    : forwarded to vllm api_server  (e.g. --enable-prefix-aware-scheduling
#                 --max-model-len 32768)
#
# Dataset used: liyucheng/arxiv-march-2023 via Jenga's benchmark_serving.py
# Client:       /home/r14922144/Jenga-SOSP25-AE/benchmark_serving.py

set -euo pipefail

MODEL_TAG=$1
RESULT_SUBDIR=$2
PORT=$3
RATE=$4
MODEL=$5
shift 5

RESULT_DIR="results/${RESULT_SUBDIR}"
mkdir -p "$RESULT_DIR"

JENGA_DIR=/home/r14922144/Jenga-SOSP25-AE
PYTHON=/home/r14922144/miniconda3/envs/vllm/bin/python
PID_FILE="/tmp/vllm_server_${PORT}.pid"

# ── Dataset parameters ────────────────────────────────────────────────────────
# Both models use the gemma2 subset (individual articles, ≤8k tokens each).
# ministral groups are 60-80k tokens, which overflows our 32k max-model-len.
HF_SUBSET="gemma2"
NUM_PROMPTS=80        # 20 articles × 4 questions
HF_MAX_LEN_ARG=""     # single articles fit comfortably in 8192 tokens

echo ""
echo "========================================================"
echo " Model  : ${MODEL_TAG} (${MODEL})"
echo " Config : ${RESULT_SUBDIR}"
echo " Rate   : ${RATE} req/s"
echo " Subset : ${HF_SUBSET}  N=${NUM_PROMPTS}"
echo "========================================================"

# ── Launch server ─────────────────────────────────────────────────────────────
STATUS=$(bash scripts/launch_server.sh "$RESULT_SUBDIR" "$PORT" "$MODEL" "$@")
if [[ "$STATUS" != *"READY"* ]]; then
    echo "ERROR: Server failed to start for ${RESULT_SUBDIR}"
    exit 1
fi
echo "Server is ready."

# ── Run benchmark client ──────────────────────────────────────────────────────
LOGFILE="$RESULT_DIR/benchmark.log"

$PYTHON "$JENGA_DIR/benchmark_serving.py" \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-path liyucheng/arxiv-march-2023 \
    --dataset-name hf \
    --hf-subset "$HF_SUBSET" \
    --hf-split train \
    --num-prompts "$NUM_PROMPTS" \
    --seed 55555 \
    --hf-output-len 150 \
    ${HF_MAX_LEN_ARG} \
    --request-rate "$RATE" \
    --ignore-eos \
    2>&1 | tee "$LOGFILE"

# ── Collect cache metrics from Prometheus ────────────────────────────────────
curl -s "http://localhost:${PORT}/metrics" \
    | grep -E "vllm:(gpu_cache_usage|prefix_cache_hits_total|num_preemptions)" \
    > "$RESULT_DIR/cache_metrics.txt" 2>/dev/null || true

# ── Kill server ────────────────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 5
fi

echo "Done: ${RESULT_SUBDIR}"
