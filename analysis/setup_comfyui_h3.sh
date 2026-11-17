#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
COMFY_DIR="$HERE/ComfyUI"
WORKFLOW="$HERE/video_minimax_h3_t2v.json"
LOG_DIR="$HERE/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/setup_comfyui_h3.log}"
PYBIN="${PYBIN:-/usr/bin/python3}"
UV="${UV:-$(command -v uv)}"
COMFY_VERSION="v0.45.0"
WORKFLOW_COMMIT="ea7faae6999fb31338f43408ff4a575bd3f7d9a0"
HF_REPO="Comfy-Org/MiniMax-H3"
HF_REVISION="4b53a35191437de05370da548e59aeabe2a5a09a"
GPU_ID="${GPU_ID:-2}"

mkdir -p "$LOG_DIR" "$HERE/.tmp" "$HERE/.uv-cache" "$HERE/.cache/huggingface"
export TMPDIR="$HERE/.tmp"
export UV_CACHE_DIR="$HERE/.uv-cache"
export XDG_CACHE_HOME="$HERE/.cache"
export HF_HOME="$HERE/.cache/huggingface"
export VIRTUAL_ENV="$VENV"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec > >(tee -a "$LOG_FILE") 2>&1
PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -x

date -Is
uname -a
lsb_release -ds
"$PYBIN" --version
"$UV" --version
df -h / /tmp "$HERE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

[ -x "$PYBIN" ]
[ -x "$UV" ]

if [ ! -d "$COMFY_DIR/.git" ]; then
    if [ -e "$COMFY_DIR" ]; then
        echo "ERROR: $COMFY_DIR exists but is not a ComfyUI git checkout; refusing to overwrite it."
        exit 1
    fi
    git clone --depth 1 --branch "$COMFY_VERSION" \
        https://github.com/Comfy-Org/ComfyUI.git \
        "$COMFY_DIR"
fi

ACTUAL_COMFY_VERSION="$(git -C "$COMFY_DIR" describe --tags --exact-match)"
if [ "$ACTUAL_COMFY_VERSION" != "$COMFY_VERSION" ]; then
    echo "ERROR: expected $COMFY_VERSION, found $ACTUAL_COMFY_VERSION; refusing to change an existing checkout."
    exit 1
fi
git -C "$COMFY_DIR" rev-parse HEAD
git -C "$COMFY_DIR" status --short

if [ ! -x "$VENV/bin/python" ]; then
    "$PYBIN" -m venv "$VENV"
fi
"$VENV/bin/python" --version

"$UV" pip install --python "$VENV/bin/python" \
    --torch-backend=cu130 \
    "torch==2.13.0" \
    "torchvision==0.28.0" \
    "torchaudio==2.11.0"

"$UV" pip install --python "$VENV/bin/python" \
    -r "$COMFY_DIR/requirements.txt" \
    "huggingface_hub==1.32.0"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import torchaudio
import av

print("torch:", torch.__version__)
print("torchvision:", md.version("torchvision"))
print("torchaudio:", torchaudio.__version__)
print("PyAV:", av.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.version.cuda == "13.0", torch.version.cuda
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
assert torch.cuda.get_device_capability(0) == (10, 0)
assert "sm_100" in torch.cuda.get_arch_list()
PY

curl --fail --location --retry 5 --retry-delay 5 \
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/$WORKFLOW_COMMIT/templates/video_minimax_h3_t2v.json" \
    --output "$WORKFLOW"

"$VENV/bin/python" - "$WORKFLOW" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    workflow = json.load(f)
assert workflow["id"] == "e3f2b845-8f2c-4b5a-9caf-eac1029d3e7e"
assert workflow["extra"]["frontendVersion"] == "1.49.6"
core_nodes = [n for n in workflow["nodes"] if n.get("properties", {}).get("cnr_id") == "comfy-core"]
assert core_nodes
assert all(n["properties"]["ver"] == "0.33.0" for n in core_nodes)
print("workflow:", path)
print("top-level nodes:", len(workflow["nodes"]))
print("core node version:", sorted({n["properties"]["ver"] for n in core_nodes}))
print("frontend:", workflow["extra"]["frontendVersion"])
PY

USER_WORKFLOW_DIR="$COMFY_DIR/user/default/workflows"
USER_WORKFLOW="$USER_WORKFLOW_DIR/MiniMax_H3_T2V_Deployed.json"
mkdir -p "$USER_WORKFLOW_DIR"
cp --update=none "$WORKFLOW" "$USER_WORKFLOW"
cmp "$WORKFLOW" "$USER_WORKFLOW"

"$VENV/bin/hf" download "$HF_REPO" \
    --revision "$HF_REVISION" \
    --local-dir "$COMFY_DIR/models" \
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "vae/minimax_h3_video_vae_int8_convrot.safetensors" \
    "vae/minimax_h3_audio_vae_fp32.safetensors" \
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

cd "$COMFY_DIR/models"
sha256sum --check <<'EOF'
e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e  vae/minimax_h3_video_vae_int8_convrot.safetensors
8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48  vae/minimax_h3_audio_vae_fp32.safetensors
2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e  loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
EOF

"$UV" pip freeze --python "$VENV/bin/python" > "$LOG_DIR/environment.freeze.txt"
du -sh "$COMFY_DIR/models" "$VENV"
df -h "$HERE"
date -Is

cat <<EOF
Setup completed successfully.
ComfyUI:   $COMFY_DIR ($COMFY_VERSION)
Python:    $VENV
Workflow:  $WORKFLOW
Log:       $LOG_FILE
Start:     GPU_ID=$GPU_ID bash $HERE/start_comfyui_h3.sh
EOF
