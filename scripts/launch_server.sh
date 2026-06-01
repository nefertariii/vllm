#!/bin/bash
# Launch a vLLM server and wait until it is healthy.
#
# Usage:
#   bash scripts/launch_server.sh RESULT_SUBDIR PORT MODEL [extra vllm args...]
#
# RESULT_SUBDIR : path under results/  (used for the log file)
# PORT          : port to listen on
# MODEL         : HuggingFace model ID
# extra args    : forwarded verbatim to vllm api_server  (e.g. --max-model-len 32768)
#
# Exit codes: 0 = server is READY, 1 = FAILED
# Writes server PID to /tmp/vllm_server_${PORT}.pid

set -euo pipefail

RESULT_SUBDIR=$1
PORT=$2
MODEL=$3
shift 3

# ── Environment ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_HOME=/home/r14922144/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/r14922144/miniconda3/envs/vllm/bin/python
PID_FILE="/tmp/vllm_server_${PORT}.pid"
LOG_FILE="results/${RESULT_SUBDIR}/server.log"
mkdir -p "results/${RESULT_SUBDIR}"

# ── Kill any leftover server on this port ────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
fi

echo "Starting: model=${MODEL}  port=${PORT}"
echo "Extra args: $*"
echo "Log: ${LOG_FILE}"

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --no-enable-log-requests \
    "$@" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

# ── Poll until healthy ────────────────────────────────────────────────────────
TIMEOUT=300
ELAPSED=0
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "READY"
        exit 0
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "FAILED (server process died)"
        tail -40 "$LOG_FILE"
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

echo "FAILED (timeout after ${TIMEOUT}s)"
tail -40 "$LOG_FILE"
exit 1
