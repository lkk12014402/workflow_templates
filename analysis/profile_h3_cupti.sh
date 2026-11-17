#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
COMFY_DIR="$HERE/ComfyUI"
GPU_ID="${GPU_ID:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8189}"
OUT_DIR="${OUT_DIR:-$HERE/logs/cupti_h3_t2v}"
TRACE_NAME="${TRACE_NAME:-h3_t2v_cold}"
NSYS="${NSYS:-nsys}"

export TMPDIR="$HERE/.tmp"
export XDG_CACHE_HOME="$HERE/.cache"
export HF_HOME="$HERE/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [ -n "$(ss -H -ltn "( sport = :$PORT )")" ]; then
    echo "ERROR: port $PORT is already listening" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
cd "$COMFY_DIR"
exec "$NSYS" profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    --export=sqlite \
    --output="$OUT_DIR/$TRACE_NAME" \
    "$VENV/bin/python" main.py \
    --listen "$HOST" \
    --port "$PORT" \
    --disable-partner-nodes \
    --output-directory "$OUT_DIR/output" \
    --temp-directory "$OUT_DIR/temp"
