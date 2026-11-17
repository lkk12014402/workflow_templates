# MiniMax H3 T2V ComfyUI 本机部署调研

## 1. 结论

这个 workflow 可以在本机部署，建议使用：

- ComfyUI **v0.45.0**（本次查询到的最新 stable release）
- Python 3.12、独立 `uv` 虚拟环境
- PyTorch **2.13.0+cu130**、torchvision 0.28.0+cu130、torchaudio 2.11.0+cu130
- 单张 NVIDIA B200；不需要多卡，不需要系统 CUDA Toolkit
- workflow 指定的 4 个必需模型文件，加上可选 Turbo LoRA 共 5 个文件
- 所有节点均为 `comfy-core`，**不需要 ComfyUI-Manager、第三方 custom nodes 或在线生成 API**

本机显存和内存容量充足。真正的安装风险是：`/` 只剩约 2.1 GiB，而 `/home` 还有约 6.1 TiB。因此代码、虚拟环境、模型、HF/uv 缓存和临时目录都应放在 `/home/kaokaolv/comfyui_h3` 下。

实际部署已经完成。可重复执行的脚本为 `setup_comfyui_h3.sh` 和 `start_comfyui_h3.sh`；服务监听 `0.0.0.0:8188`，使用物理 GPU 2。首次真实生成和音视频解码验证均已通过，实测结果见第 9 节，完整过程见 `run_book.md`。

## 2. Workflow 解析

来源：

- 用户指定模板：<https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json>
- 本次固定模板 commit：`ea7faae6999fb31338f43408ff4a575bd3f7d9a0`
- 官方教程：<https://docs.comfy.org/tutorials/video/minimax/minimax-h3>

### 2.1 节点与数据流

模板顶层有 6 个节点，核心计算封装在一个子图中。子图名虽为 `Image to Video (MiniMax H3)`，但 `first_frame` 和 `last_frame` 均未连接，因此 `MiniMaxH3ImageToVideo` 以纯文本条件运行，即 T2V。

数据流为：

1. `UNETLoader` 加载 H3 FL2VA diffusion model。
2. `CLIPLoader` 以 `type=minimax` 加载 Qwen3-VL-32B 文本编码器。
3. `MiniMaxH3ImageToVideo` 同时产生正条件和视频/音频联合 latent。
4. `SamplerCustomAdvanced` 使用 `res_multistep` sampler、`simple` scheduler 采样。
5. 两个 VAE 分别解码视频与音频。
6. `CreateVideo` 以 24 fps 合成带音轨的视频。
7. `SaveVideo` 保存到 ComfyUI 输出目录。

所有相关节点的 `properties.cnr_id` 都是 `comfy-core`，节点版本标为 `0.33.0`。官方教程对普通 T2V 的最低版本写的是 0.30.0，但为了保证与**这个 JSON 的当前版本**一致，不应使用低于 0.33.0 的 ComfyUI；这里直接固定更新的 stable v0.45.0。

模板声明的前端版本为 1.49.6；ComfyUI v0.45.0 的 `requirements.txt` 固定 `comfyui-frontend-package==1.58.5`，满足该要求，无需单独安装前端。

### 2.2 实际默认参数

子图内部控件值会被顶层连线覆盖，不能只读子图中的 1344×768、73 frames 等值。沿连线解析后的实际默认值是：

| 参数 | 实际默认值 | 依据 |
|---|---:|---|
| 分辨率 | 864×480 | 顶层 `ResolutionSelector`: 16:9、0.4 MP、32 对齐 |
| 时长输入 | 5 秒 | 顶层子图 `duration=5` |
| 实际帧数 | 124 帧，约 5.17 秒 | `round(5*24)=120`，向上对齐到 `17k+5` 网格 |
| 帧率 | 24 fps | `CreateVideo.fps=24` |
| 随机种子 | 757358688076805 | 顶层 `noise_seed` |
| Turbo | 关闭 | `turbo_mode=false` |
| 默认采样步数 | 20 | `ComfySwitchNode.on_false=20` |
| Turbo 分支 | 8 步、LoRA strength 1 | 仅 `turbo_mode=true` 时使用 |
| 官方 768p 设置 | 1344×768 | 将 Resolution Selector 调为 16:9、0.98 MP |

模板中的 10 个 style embeddings 均为可选项，默认 prompt 没有调用它们，首次部署无需下载。

### 2.3 本地能力边界

- 此模板本地运行的是 H3-Base FL2VA，可生成视频及原生双声道音频。
- 官方 ComfyUI 教程说明：开放权重面向约 1 MP、768 px 短边、24 fps、约 15 秒以内的生成。
- **2K 不是这个 JSON 的直接本地输出能力。** 官方 2K 流程需要单独的 H3-Regenerate-2K/hosted pass；MiniMax 模型卡也说明该模块尚未开源。
- MiniMax 的 H3-Context-IR hosted 预处理系统也不包含在此本地模板中；复杂 prompt 的官方完整托管效果不能直接等同于本地模板效果。

## 3. 模型文件与磁盘占用

模型来源：<https://huggingface.co/Comfy-Org/MiniMax-H3>

本次查询的仓库 revision：`4b53a35191437de05370da548e59aeabe2a5a09a`。以下大小和 SHA256 来自 Hugging Face API 的 LFS 元数据，而不是根据文件名推测。

| 目标目录 | 文件 | 字节数 | 是否必需 |
|---|---|---:|---|
| `models/diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | 是 |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | 是 |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` | 2,811,065,184 | 是 |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | 是 |
| `models/loras` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1,956,193,000 | 仅 Turbo 分支需要 |

- 4 个基础文件合计约 **37.32 GiB**。
- 加上 Turbo LoRA 合计 **42,030,035,159 bytes，约 39.14 GiB**。
- 建议一次下载 5 个文件，以便直接测试模板的 Turbo 开关；不要下载整个模型仓库，因为其中有大量不需要的精度变体。
- `Comfy-Org` 仓库中的 8-step LoRA 与模板原始 `lightx2v/Minimax-h3-Turbo` URL 指向的文件具有相同 blob ID 和 SHA256，因此从 `Comfy-Org` 统一下载是等价的。
- 模型卡明确建议：能使用 PyTorch cu130 时，diffusion model 优先 `int8_convrot`，这与模板选择一致；`nvfp4_awq` 文本编码器不要求必须是 Blackwell GPU，本机 B200 也可使用。

权重文件大小不等于运行峰值显存；文本编码、激活、视频帧数和分辨率都会增加占用。单张 B200 的容量看起来很充裕，但峰值仍需实测。

## 4. 本机环境检查结果

| 项目 | 实测结果 | 影响 |
|---|---|---|
| OS | Ubuntu 24.04.3 LTS，x86_64 | 使用 Linux manual install |
| GPU | 8×NVIDIA B200，每张 183,359 MiB | 单卡即可，无需多卡并行 |
| 驱动 | 580.178.04，CUDA Version 13.0 | 与 cu130 PyTorch 路线匹配 |
| 当前空闲卡 | GPU 1 仅 4 MiB，利用率 0% | 启动示例使用 GPU 1；运行前应重新检查 |
| RAM | 总计 3.0 TiB，available 约 1.6 TiB | 充足 |
| `/` | 仅剩约 2.1–2.2 GiB，100% | 不应把临时文件、venv 或模型放这里 |
| `/home` | 约 6.1 TiB 可用 | 可作为完整部署根目录 |
| Python | `/usr/bin/python3`，3.12.3 | 可用；无需 conda |
| uv | 0.12.15 | 可创建 venv 并安装包 |
| PyTorch | 系统 Python 未安装 | 必须在独立 venv 安装 cu130 版本 |
| pip | 系统无 `pip3` | 使用 `uv pip`，无需 apt |
| git | 2.43.0 | 可 clone ComfyUI |
| git-lfs | 未安装 | 使用 `hf download`，因此不需要 |
| ffmpeg CLI | 未安装 | 此模板不依赖外部 CLI；ComfyUI 使用 PyAV |
| ComfyUI | 尚未安装 | 需要 clone |
| 8188 端口 | 检查时未监听 | 可使用默认端口 |

ComfyUI v0.45.0 的依赖包含 `av>=17.0.0`；核心 `nodes_video.py` 也直接 `import av` 并调用视频对象的 `save_to`。因此不需要为了这个模板另外安装 `ffmpeg` 命令行。

## 5. 可复现部署步骤

以下命令只是部署方案，本次调研没有执行。全程不需要 `sudo`。

### 5.1 准备目录并固定 ComfyUI 版本

```bash
BASE="$HOME/comfyui_h3"
cd "$BASE"

mkdir -p "$BASE/.tmp" "$BASE/.uv-cache" "$BASE/.cache"
export TMPDIR="$BASE/.tmp"
export UV_CACHE_DIR="$BASE/.uv-cache"
export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$BASE/.cache/huggingface"

git clone --depth 1 --branch v0.45.0 \
  https://github.com/Comfy-Org/ComfyUI.git \
  "$BASE/ComfyUI"

uv venv --python /usr/bin/python3 "$BASE/.venv"
```

这些环境变量避免 uv、编译缓存和临时文件写入空间不足的 `/tmp` 或其他根分区目录。新开终端后需要重新 export。

### 5.2 安装 PyTorch cu130 与 ComfyUI 依赖

```bash
uv pip install --python "$BASE/.venv/bin/python" \
  --torch-backend=cu130 \
  "torch==2.13.0" \
  "torchvision==0.28.0" \
  "torchaudio==2.11.0"

uv pip install --python "$BASE/.venv/bin/python" \
  -r "$BASE/ComfyUI/requirements.txt" \
  "huggingface_hub==1.32.0"
```

先安装 CUDA wheel，再安装 `requirements.txt`，可避免 ComfyUI 的无版本 `torch` 依赖将 CUDA build 换成其他 build。`torchaudio` 也必须安装：H3 核心节点直接 `import torchaudio`，但 v0.45.0 的 requirements 没有单独列出它。

官方 ComfyUI README 的 NVIDIA 安装命令使用 cu130。这里固定 torch 2.13.0 / torchvision 0.28.0 / torchaudio 2.11.0，是因为本机已有环境的同一组合实测为 `2.13.0+cu130`，支持 B200 的 `sm_100`；新项目 venv 安装后也再次通过了 CUDA 自检。

下载大模型前先验证实际 GPU 和 CUDA build：

```bash
CUDA_VISIBLE_DEVICES=2 "$BASE/.venv/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
assert torch.version.cuda and torch.version.cuda.startswith("13.")
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
PY
```

预期看到 B200、CUDA 13.0.x，且 assertion 不报错。若显示 `torch CUDA: None` 或 `CUDA available: False`，应先修正 PyTorch 安装，不要继续下载模型。

### 5.3 下载指定 workflow

```bash
curl -fL \
  https://raw.githubusercontent.com/Comfy-Org/workflow_templates/ea7faae6999fb31338f43408ff4a575bd3f7d9a0/templates/video_minimax_h3_t2v.json \
  -o "$BASE/video_minimax_h3_t2v.json"
```

也可以在 ComfyUI 的 Template Library → Video 中选择 MiniMax H3 Text to Video；下载这个固定 JSON 则更便于复现用户给出的版本。

### 5.4 只下载所需的 5 个模型文件

```bash
"$BASE/.venv/bin/hf" download Comfy-Org/MiniMax-H3 \
  --revision 4b53a35191437de05370da548e59aeabe2a5a09a \
  --local-dir "$BASE/ComfyUI/models" \
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "vae/minimax_h3_video_vae_int8_convrot.safetensors" \
  "vae/minimax_h3_audio_vae_fp32.safetensors" \
  "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
```

`hf download --local-dir` 会保留仓库内目录结构，所以文件会正好进入 ComfyUI 期望的目录，并支持缓存元数据和断点恢复。无需 `git lfs pull`。

### 5.5 校验模型完整性

```bash
cd "$BASE/ComfyUI/models"
sha256sum -c <<'EOF'
e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e  vae/minimax_h3_video_vae_int8_convrot.safetensors
8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48  vae/minimax_h3_audio_vae_fp32.safetensors
2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e  loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
EOF
```

5 行都必须显示 `OK`。

### 5.6 启动 ComfyUI

先重新运行 `nvidia-smi`，确认 GPU 2 仍空闲；若已被其他任务使用，修改下面的 `CUDA_VISIBLE_DEVICES`。

```bash
BASE="$HOME/comfyui_h3"
mkdir -p "$BASE/.tmp" "$BASE/.cache"
export TMPDIR="$BASE/.tmp"
export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$BASE/.cache/huggingface"

cd "$BASE/ComfyUI"
CUDA_VISIBLE_DEVICES=2 "$BASE/.venv/bin/python" main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-partner-nodes
```

说明：

- `CUDA_VISIBLE_DEVICES=2` 让此进程只看到物理 GPU 2，避免占用其他任务的 GPU。
- 首次先使用默认显存管理，不添加 `--highvram`、`--gpu-only` 或 `--fast`。
- `--disable-partner-nodes` 禁用付费 API partner nodes，不影响这个全本地 workflow。
- 按用户要求使用 `--listen 0.0.0.0`，供内网外部浏览器访问；不要将无认证的 ComfyUI 直接暴露到公网。

内网浏览器打开 <http://172.26.46.130:8188>。若在其他网络环境部署，则改回 127.0.0.1 并使用 SSH tunnel。

### 5.7 验证并运行

服务启动后可先检查：

```bash
curl -fsS http://127.0.0.1:8188/system_stats
curl -fsS http://127.0.0.1:8188/object_info/MiniMaxH3ImageToVideo
```

第二条应返回节点 schema，而不是 404 或空错误。

在 Web UI 中：

1. 在 ComfyUI 的用户 workflow 列表中打开 `MiniMax_H3_T2V_Deployed.json`；原始文件也保留在 `$BASE/video_minimax_h3_t2v.json`。
2. 确认 4 个基础模型下拉框能选中第 3 节列出的文件。
3. 首次保持默认 864×480、5 秒、`turbo_mode=false`，点击 Run。
4. 成功后再测试 0.98 MP（1344×768）或打开 Turbo；不要一开始就直接尝试最高分辨率和最长时长。
5. 输出应位于 `ComfyUI/output/video/`，文件名前缀为 `MiniMax_H3`，并包含音轨。

## 6. 常见问题定位

### `Torch not compiled with CUDA enabled` / `torch.cuda.is_available() == False`

说明装到了 CPU wheel 或使用了错误的 Python。确认启动命令使用的是 `$BASE/.venv/bin/python`，并在同一个 venv 重装官方 cu130 wheel；不要改用系统 Python。

### 节点不存在或 JSON 无法加载

检查 ComfyUI 版本。这个 JSON 的节点导出版本为 0.33.0，本文使用 v0.45.0；不要拿旧版 ComfyUI 强行加载，也不需要通过 Manager 安装名称相似的第三方节点。

### 模型下拉框为空或提示文件不存在

核对完整路径和文件名，尤其是：

- diffusion model 在 `models/diffusion_models`，不是 `models/checkpoints`
- Qwen 文本编码器在 `models/text_encoders`
- 两个 VAE 都在 `models/vae`

修正后刷新模型列表或重启 ComfyUI。

### `No space left on device`

```bash
df -h / /tmp "$BASE"
```

确认 `TMPDIR`、`UV_CACHE_DIR`、`XDG_CACHE_HOME` 和 `HF_HOME` 都指向 `$BASE` 下。当前根分区空间极少，不能依赖默认 `/tmp`。

### CUDA OOM 或生成异常缓慢

- 先用 `nvidia-smi` 检查所选物理卡是否被其他进程占用。
- 回到默认 0.4 MP、5 秒、非 Turbo 路径验证。
- 首跑不要加 `--highvram` 或 `--gpu-only`。
- 权重总大小不是峰值显存；更高分辨率和更长帧数需要更多的激活与解码空间。
- 本机 `/home` 是挂载的网络文件系统，模型首次读取速度可能受存储吞吐影响；目前没有实测数据，不能给出加载时间承诺。

### 是否需要安装 FFmpeg

此模板不需要外部 `ffmpeg` 命令。ComfyUI 核心通过 `av`（PyAV）保存视频，requirements 已包含 `av>=17.0.0`。只有额外的外部处理脚本才可能需要 ffmpeg CLI。

## 7. 许可证注意事项

MiniMax 原始模型卡标明使用 **MiniMax H3 Community License Agreement**。ComfyUI 官方教程还明确说明：本地生成结果若用于商业用途，需要 MiniMax commercial license，Comfy 是官方 reseller。商业使用前应核对许可证，不应把“权重可下载”理解为“所有商业用途自动免费”。

来源：

- <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>

## 8. 主要证据链接

- Workflow 固定版本：<https://raw.githubusercontent.com/Comfy-Org/workflow_templates/ea7faae6999fb31338f43408ff4a575bd3f7d9a0/templates/video_minimax_h3_t2v.json>
- ComfyUI v0.45.0 README：<https://github.com/Comfy-Org/ComfyUI/blob/v0.45.0/README.md>
- ComfyUI v0.45.0 requirements：<https://github.com/Comfy-Org/ComfyUI/blob/v0.45.0/requirements.txt>
- H3 核心节点源码：<https://github.com/Comfy-Org/ComfyUI/blob/v0.33.0/comfy_extras/nodes_minimax_h3.py>
- 视频核心节点源码：<https://github.com/Comfy-Org/ComfyUI/blob/v0.33.0/comfy_extras/nodes_video.py>
- ComfyUI H3 官方教程：<https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- ComfyUI 重打包模型：<https://huggingface.co/Comfy-Org/MiniMax-H3>
- MiniMax 原始模型卡：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Hugging Face 下载文档：<https://huggingface.co/docs/huggingface_hub/guides/download>

## 9. 实际部署与验收结果

实际环境：

- ComfyUI v0.45.0，commit `989d93d0d968e704c589ce0146bb18c77f6b50cc`
- Python 3.12.3，torch 2.13.0+cu130、torchvision 0.28.0+cu130、torchaudio 2.11.0+cu130
- 物理 GPU 2（NVIDIA B200），监听 `0.0.0.0:8188`
- 内网地址：<http://172.26.46.130:8188>
- 用户 workflow：`MiniMax_H3_T2V_Deployed.json`
- 5 个模型文件均通过官方 LFS SHA256 校验

首次通过 `/prompt` 提交官方 prompt 的 14 节点执行图，参数为 864×480、124 帧、24 fps、20 步、非 Turbo。prompt ID：`3a119304-3b35-4156-8b80-0e8a3ba829f7`。服务器 history 为 `success/completed=true`，ComfyUI 报告执行时间 62.54 秒，客户端含轮询总耗时约 70.09 秒。

输出：

`ComfyUI/output/video/MiniMax_H3_API_Smoke_00001_.mp4`

PyAV 全量解码验证：

- 文件 1,526,558 bytes，时长 5.166688 秒
- H.264，864×480，24/1 fps，124 帧全部成功解码
- AAC，32 kHz，双声道 stereo
- 162 个音频帧成功解码，165,334 samples/channel
- 音频 max abs 0.64827，不是静音轨
- 视频各帧标准差均大于 34，不是空白画面
- 已通过视频读取工具重新读取 MP4

脚本与记录：

- `setup_comfyui_h3.sh`：完整环境、workflow、模型安装和校验
- `start_comfyui_h3.sh`：按 0.0.0.0、GPU 2 启动
- `logs/setup_comfyui_h3.log`：setup 完整 xtrace
- `logs/comfyui_server.log`：服务与生成日志
- `logs/environment.freeze.txt`：实际 Python 依赖版本
- `logs/h3_t2v_prompt.json`：首次真实 API 执行图
- `logs/first_generation_result.json`：成功 history 摘要
- `logs/media_validation.json`：音视频全量解码结果
- `run_book.md`：逐步命令、分析、问题与解决记录
