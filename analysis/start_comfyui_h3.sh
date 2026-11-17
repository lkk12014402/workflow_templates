#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
COMFY_DIR="$HERE/ComfyUI"
GPU_ID="${GPU_ID:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8188}"

export TMPDIR="$HERE/.tmp"
export XDG_CACHE_HOME="$HERE/.cache"
export HF_HOME="$HERE/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

[ -x "$VENV/bin/python" ] || {
    echo "ERROR: missing $VENV/bin/python; run setup_comfyui_h3.sh first."
    exit 1
}
[ -f "$COMFY_DIR/main.py" ] || {
    echo "ERROR: missing $COMFY_DIR/main.py; run setup_comfyui_h3.sh first."
    exit 1
}
if [ -n "$(ss -H -ltn "( sport = :$PORT )")" ]; then
    echo "ERROR: port $PORT is already listening:"
    ss -ltnp "( sport = :$PORT )"
    exit 1
fi

mkdir -p "$HERE/logs" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
nvidia-smi -i "$GPU_ID"
echo "Starting ComfyUI on http://$HOST:$PORT (physical GPU $GPU_ID)"
echo "Server IP addresses: $(hostname -I)"
echo "Workflow: $HERE/video_minimax_h3_t2v.json"

cd "$COMFY_DIR"
exec "$VENV/bin/python" main.py \
    --listen "$HOST" \
    --port "$PORT" \
    --disable-partner-nodes
