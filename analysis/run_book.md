# Run Book

## 任务与边界

用户要求调研 `video_minimax_h3_t2v.json`，判断如何搭建环境并在本机部署。

本次执行范围是调研：读取官方 workflow、官方教程、ComfyUI 源码与 release、Hugging Face 模型元数据，并检查本机环境。没有 clone/install ComfyUI，没有安装 Python 包，没有下载模型，没有启动服务，也没有生成视频。

最终部署调研与可复现命令见 [minimax_h3_t2v_research.md](./minimax_h3_t2v_research.md)。本文记录调查过程、命令、关键输出、错误及修正。

## 1. Workflow 官方来源

### FetchURL

读取：

- `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v.json`
- `https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json`

结果：raw JSON 为 67 KiB 左右、2183 行；完整工具输出保存于：

```text
/home/kaokaolv/.kimi-code/sessions/wd_comfyui_h3_ae9cbe5743a7/session_ecbcb023-6078-44eb-aae0-e2ad5773896f/agents/main/tool-results/FetchURL-tool_HZO8FVfuBGpt9Vl9tkPtQ02d-0b086aad-5753-44b7-ae1c-852d58131e18.txt
```

直接看到：核心节点版本 `0.33.0`、前端 `1.49.6`、模型仓库 `Comfy-Org/MiniMax-H3`、H3 PR 链接及模型下载说明。

### JSON 解析时出现的错误与修正

第一次在 Bash 中用 Python `json.loads` 解析工具输出：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/home/kaokaolv/.kimi-code/sessions/wd_comfyui_h3_ae9cbe5743a7/session_ecbcb023-6078-44eb-aae0-e2ad5773896f/agents/main/tool-results/FetchURL-tool_HZO8FVfuBGpt9Vl9tkPtQ02d-0b086aad-5753-44b7-ae1c-852d58131e18.txt')
s = p.read_text().strip()
if s.startswith('```'):
    s = s.split('\n', 1)[1].rsplit('```', 1)[0]
w = json.loads(s)
PY
```

失败：`JSONDecodeError: Expecting value: line 1 column 1`。原因是 FetchURL 文件在 Markdown fence 前还有两行说明文字。

用 Read 检查开头，确认第 1 行是工具说明、第 3 行是 ```、第 4 行才是 `{`。修正为：

```python
s = p.read_text()
w = json.loads(s[s.index('{'):s.rindex('}') + 1])
```

随后成功输出所有顶层节点及子图节点。关键结论：

- 顶层节点：`SaveVideo`、`ResolutionSelector`、3 个 `MarkdownNote`、1 个 H3 子图实例。
- 子图内部包括：`UNETLoader`、`CLIPLoader(type=minimax)`、2 个 `VAELoader`、`MiniMaxH3ImageToVideo`、`SamplerCustomAdvanced`、`VAEDecode`、`VAEDecodeAudio`、`CreateVideo`、LoRA/步数 Switch 等。
- 所有计算节点都是 `cnr_id=comfy-core`，无第三方 custom node。
- 模型：FL2VA pruned int8_convrot、Qwen3-VL-32B nvfp4_awq、video int8_convrot VAE、audio fp32 VAE、可选 8-step Turbo LoRA。
- `MiniMaxH3ImageToVideo` 的 first/last frame 没有输入，故为 T2V。

### 采样连线解析的错误与修正

最初把 `sg['links']` 当数组：

```python
lid,src,src_slot,dst,dst_slot,typ=link[:6]
```

失败：`KeyError: slice(None, 6, None)`。打印首条后确认 link 是 dict：

```json
{"id": 23, "origin_id": 120, "origin_slot": 0, "target_id": 121, "target_slot": 1, "type": "VAE"}
```

改为读取 dict 字段后又遇到 `KeyError: -10`，原因是 `-10` 表示子图输入，不是普通节点 ID。将 `origin_id == -10` 映射到 `sg['inputs'][origin_slot]` 后解析成功。

关键连线：

```text
UNETLoader -> LoraLoaderModelOnly -> Switch.on_true
UNETLoader -> Switch.on_false
PrimitiveInt(20) -> StepsSwitch.on_false
PrimitiveInt(6/顶层覆盖为8) -> StepsSwitch.on_true
PrimitiveBoolean(顶层 turbo_mode) -> 两个 Switch
Switch(model) -> BasicGuider + BasicScheduler
MathExpression -> MiniMaxH3ImageToVideo.length
ResolutionSelector -> 子图 width/height
```

由此修正了只看子图控件默认值造成的误判：

- 子图控件虽写 1344×768，但顶层 ResolutionSelector 实际覆盖为 16:9、0.4 MP，即 864×480。
- `duration=5` 经 `max(5, round(a*24)) + (5 - (... % 17)) % 17` 得 124 帧，约 5.17 秒。
- `turbo_mode=false`，默认实际为 20 步且不经过 LoRA；开启后才使用顶层指定的 8 步 Turbo 分支。

## 2. 本机硬件与现有环境

以下命令均在 `/home/kaokaolv/comfyui_h3` 执行。

### OS、内存与磁盘

```bash
uname -a && uname -m && (command -v lsb_release >/dev/null && lsb_release -a || true) && free -h && df -h / /home/kaokaolv/comfyui_h3
```

关键输出：

```text
Linux sc09super22-b200 6.8.0-146-generic x86_64
Ubuntu 24.04.3 LTS (noble)
RAM: total 3.0Ti, available 1.6Ti
/: 1.7T total, 2.2G available, 100%
/home: 73T total, 6.1T available
```

结论：不能使用根分区承接模型、venv、包缓存或临时文件；当前工作目录在 `/home`，适合作为部署根目录。

### GPU

```bash
if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo 'nvidia-smi: not found'; fi
if command -v rocminfo >/dev/null; then rocminfo; else echo 'rocminfo: not found'; fi
```

关键输出：

```text
NVIDIA-SMI 580.178.04
Driver Version: 580.178.04
CUDA Version: 13.0
8 × NVIDIA B200, each 183359 MiB
GPU 0: 1238 MiB, util 96%
GPU 1: 4 MiB, util 0%
GPU 2: 12986 MiB
GPU 3: 178886 MiB, util 100%
GPU 4: 178758 MiB, util 100%
GPU 5: 158024 MiB
GPU 6: 14132 MiB
GPU 7: 18550 MiB
rocminfo: not found
```

结论：使用单张 B200 即可；检查时 GPU 1 最空闲，因此部署示例选择物理 GPU 1。该状态是瞬时数据，实际启动前必须重新检查。

### Python 与工具

```bash
for cmd in python3 python pip3 git git-lfs ffmpeg conda uv; do
  if command -v "$cmd" >/dev/null; then
    echo "[$cmd] $(command -v "$cmd")"
    case "$cmd" in
      git-lfs) git lfs version ;;
      ffmpeg) ffmpeg -version | sed -n '1p' ;;
      *) "$cmd" --version 2>&1 | sed -n '1p' ;;
    esac
  else
    echo "[$cmd] not found"
  fi
done
if [ -d /home/kaokaolv/ComfyUI ]; then
  echo '[ComfyUI] /home/kaokaolv/ComfyUI exists'
else
  echo '[ComfyUI] /home/kaokaolv/ComfyUI not found'
fi
```

输出：

```text
python3: /usr/bin/python3, Python 3.12.3
python: not found
pip3: not found
git: 2.43.0
git-lfs: not found
ffmpeg: not found
conda: not found
uv: /home/kaokaolv/.local/bin/uv, 0.12.15
/home/kaokaolv/ComfyUI not found
```

另执行：

```bash
python3 - <<'PY'
import importlib.util
for name in ('torch','venv','ensurepip'):
    spec=importlib.util.find_spec(name)
    print(f'{name}: {spec.origin if spec else "not installed"}')
PY
ss -ltn '( sport = :8188 )' || true
ps -eo pid,args --sort=pid | python3 -c "import sys; [print(line,end='') for line in sys.stdin if 'ComfyUI' in line or 'main.py --listen' in line]"
```

结果：系统 Python 无 torch；有 venv/ensurepip；8188 未监听；没有既有 ComfyUI 进程。进程过滤只匹配到了本次检查命令本身，不是已有服务。

再检查：

```bash
for cmd in curl wget sha256sum; do
  if command -v "$cmd" >/dev/null; then echo "$cmd: $(command -v "$cmd")"; else echo "$cmd: not found"; fi
done
df -h /tmp /home/kaokaolv/.cache /home/kaokaolv/comfyui_h3
git status --short 2>/dev/null || echo 'not a git repository'
```

结果：curl、wget、sha256sum 均存在；`/tmp` 位于仅剩约 2.1 GiB 的根分区；`~/.cache` 和工作目录位于有 6.1 TiB 的 `/home`；当前目录不是 git repository。

决策：用 uv 独立 venv，不需要 apt/pip/conda；用 `hf download`，不需要 git-lfs；用 PyAV，不需要 ffmpeg CLI；显式设置 TMPDIR/UV_CACHE_DIR/XDG_CACHE_HOME/HF_HOME 到 `/home`。

## 3. 官方教程、源码和版本证据

### FetchURL 列表及结论

读取：

- `https://huggingface.co/Comfy-Org/MiniMax-H3`
  - 官方重打包模型；列出所有目录和精度变体。
  - 能使用 cu130 时 diffusion 优先 int8_convrot；fp8_scaled 是次选。
  - nvfp4_awq text encoder 不要求 Blackwell。
- `https://docs.comfy.org/tutorials/video/minimax/minimax-h3`
  - T2V 标注 ComfyUI 0.30.0+。
  - 开放权重目标约 1 MP、768 短边、24 fps、约 15 秒。
  - 2K 是 hosted-service output，ComfyUI 需单独 upscale pass。
  - 本地生成用于商业用途的许可证提醒。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/requirements.txt`
  - master 当前前端 1.59.1、templates 0.11.109。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/README.md`
  - Linux manual install；NVIDIA stable 使用 cu130；需要 torch/torchvision/torchaudio。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.45.0/requirements.txt`
  - v0.45.0 固定前端 1.58.5、templates 0.11.105，含 `av>=17.0.0`，不单独列 torchaudio。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.45.0/README.md`
  - 与 master 一样给出 NVIDIA cu130 安装路线。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.45.0/comfy/cli_args.py`
  - 默认 listen 127.0.0.1、port 8188。
  - 有 `--cuda-device`、`--disable-partner-nodes`、`--offline` 等参数。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.33.0/comfy_extras/nodes_minimax_h3.py`
  - H3 节点直接 `import torchaudio`。
  - T2V/I2V 共用 `MiniMaxH3ImageToVideo`，无图像输入即 T2V。
  - 124 帧约 5 秒；17k+5 帧网格；视频 24 fps、音频 latent 40 Hz。
  - 默认 1344×768，训练长度约 124–362 帧。
- `https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.33.0/comfy_extras/nodes_video.py`
  - `import av`；`SaveVideo` 调用 `video.save_to`；`CreateVideo` 可合并音频。
- `https://huggingface.co/docs/huggingface_hub/guides/download`
  - `hf download` 支持多文件、revision 和 local-dir；local-dir 保留目录结构并写下载元数据。
- `https://huggingface.co/MiniMaxAI/MiniMax-H3`
  - H3-Base FL2VA 支持无图像 T2V 和首/尾帧 I2V。
  - 33B dense Omni Transformer，Qwen3-VL-32B encoder。
  - H3-Context-IR 是 hosted preprocessing，不在开源 release。
  - H3-Regenerate-2K 尚未开源。
  - 使用 MiniMax H3 Community License Agreement。
- `https://www.minimax.io/blog/minimax-h3`
  - 官方模型能力背景；不以其宣传中的 2K 能力替代本地模板能力边界。

未采用为部署依据的页面：

- `https://github.com/Comfy-Org/ComfyUI/pull/15224`：FetchURL 抽取到的是一段扩展适配讨论，不足以作为核心安装依据，因此改用 release 源码和官方教程。
- `https://docs.comfy.org/tutorials/video/minimax/minimax-h3-commercial-license`：404；许可证提醒仍可由主教程和原始模型卡直接证实。
- `https://huggingface.co/api/models/Comfy-Org/MiniMax-H3`：FetchURL 无法提取 JSON，随后改用 Bash Python urllib 获取原始 API 数据成功。

## 4. Release、workflow 和模型元数据 API

执行 Python urllib 请求：

```text
https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest
https://api.github.com/repos/Comfy-Org/workflow_templates/commits?path=templates/video_minimax_h3_t2v.json&per_page=1
https://huggingface.co/api/models/Comfy-Org/MiniMax-H3?blobs=true
https://huggingface.co/api/models/lightx2v/Minimax-h3-Turbo?blobs=true
```

脚本使用 `urllib.request.Request(..., headers={'User-Agent':'comfyui-h3-research'})`，仅读取公开 API，没有认证信息。

结果：

```text
ComfyUI latest stable: v0.45.0, published 2026-11-10T21:42:42Z
workflow latest path commit: ea7faae6999fb31338f43408ff4a575bd3f7d9a0
Comfy-Org/MiniMax-H3 revision: 4b53a35191437de05370da548e59aeabe2a5a09a
license card value: other
```

模型 LFS 元数据：

```text
diffusion: 20970379616 bytes
sha256 e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a

text encoder: 15687142551 bytes
sha256 35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6

video VAE: 2811065184 bytes
sha256 52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e

audio VAE: 605254808 bytes
sha256 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48

8-step LoRA: 1956193000 bytes
sha256 2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e
```

Comfy-Org 与 lightx2v 的 8-step LoRA 具有相同 blob ID `9bc880321fcbed8375b4951e8caf696cac4390c7` 和相同 SHA256，故统一从 Comfy-Org 下载不会改变模板权重。

用 Python 求和：

```text
5 files: 42030035159 bytes = 39.14352055545896 GiB
without LoRA: 37.321673854254186 GiB
```

决策：固定 ComfyUI v0.45.0、workflow commit 和 HF revision；只下载 5 个目标文件，不下载整个仓库。

## 5. PyTorch cu130 wheel 查询

读取：

```text
https://download.pytorch.org/whl/cu130/torch/
https://download.pytorch.org/whl/cu130/torchvision/
https://download.pytorch.org/whl/cu130/torchaudio/
```

第一次解析筛选 `linux_x86_64`，返回空列表。检查实际 wheel 名称后发现平台字符串是 `manylinux_2_28_x86_64`，不是筛选表达式能匹配的连续 `linux_x86_64`。

诊断命令输出：

```text
wheel links: 291
cp312 Linux matches: 18
示例: torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
```

将匹配改为 `manylinux.*x86_64` 后，确认：

```text
torch stable versions tail: 2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1
torchvision stable versions tail: 0.27.0, 0.27.1, 0.28.0, 0.29.0, 0.29.1
torchaudio stable versions tail: 2.9.0, 2.9.1, 2.10.0, 2.11.0
```

决策：遵循 ComfyUI 官方安装命令，让 resolver 选择 torch、torchvision、torchaudio 的兼容组合，不仅凭各索引中的最大版本号分别固定。安装后必须检查 `torch.version.cuda`、`torch.cuda.is_available()`、GPU 名称、compute capability 和 arch list，再开始下载大模型。

## 6. 最终结论与交付

1. 本机可以部署，单张 B200 足够开始验证；没有证据需要多卡。
2. 使用 v0.45.0、Python 3.12、uv venv、PyTorch cu130。
3. 模板全部为核心节点，不安装 Manager/custom nodes。
4. 基础模型 4 个，Turbo LoRA 第 5 个可选；完整 5 文件约 39.14 GiB。
5. 首跑实际默认 864×480、124 帧、24 fps、20 步、非 Turbo。
6. 根分区空间是主要风险，所有临时目录和缓存必须显式放到 `/home`。
7. 此模板不是完整官方 2K/H3-Context-IR hosted pipeline。
8. 没有实测，不能宣称服务已启动、模型已跑通或给出未经验证的性能数据。

已写入：

- `minimax_h3_t2v_research.md`：完整调研、部署命令、SHA256、运行与排错步骤。
- `run_book.md`：本调查记录。

## 7. 交付文档静态校验

没有执行文档中的安装/下载命令，只提取全部 `bash` 代码块并交给 `bash -n` 做语法检查；SSH 示例中的占位符仅在检查进程中替换为 `user@server`。另用正则检查 SHA256 条目数量和文件名校验。

```bash
python3 - <<'PY'
from pathlib import Path
import re, subprocess
p=Path('/home/kaokaolv/comfyui_h3/minimax_h3_t2v_research.md')
text=p.read_text()
blocks=re.findall(r'```bash\n(.*?)```',text,re.S)
failed=[]
for i,block in enumerate(blocks,1):
    block=block.replace('<用户名>@<服务器地址>','user@server')
    result=subprocess.run(['bash','-n'],input=block,text=True,capture_output=True)
    print(f'bash block {i}:', 'syntax OK' if result.returncode == 0 else result.stderr.strip())
    if result.returncode: failed.append(i)
checks=re.findall(r'^([0-9a-f]{64})  (\S+\.safetensors)$',text,re.M)
print('SHA256 entries:',len(checks))
assert len(checks)==5
assert len({name for _,name in checks})==5
assert not failed, failed
print('Static validation: PASS')
PY
```

结果：10 个 Bash 代码块全部 `syntax OK`，5 个唯一模型的 SHA256 条目格式正确，`Static validation: PASS`。

最终 `ls -lah /home/kaokaolv/comfyui_h3` 只显示原有 `AGENTS.md` 和本次要求记录/调研产生的两个 Markdown 文件，没有测试输出、日志、模型或环境目录。

---

# 实际部署阶段

## 8. 新任务、授权与实施边界

用户要求实际把 MiniMax H3 ComfyUI workflow 部署起来，并明确：

1. 可参考 `/home/kaokaolv/.venv*` 与 `/home/kaokaolv/test-swe` 的建环境脚本。
2. 内网可用，服务监听地址必须是 `0.0.0.0`，供外部浏览器访问。
3. setup 脚本、每一步分析/结论、问题与解决、运行命令都要详细完整记录。

实施决定：只读取旧环境和 `test-swe`，不修改其中任何文件；在 `/home/kaokaolv/comfyui_h3` 创建独立 `.venv`、ComfyUI、模型、缓存、日志和脚本。服务将按用户授权绑定 `0.0.0.0:8188`。最终必须实际生成一个视频并检查音轨，不能只以进程启动作为完成标准。

## 9. 现有环境、示例脚本与实时资源检查

### 文件与目录检查

执行：

```bash
ls -la /home/kaokaolv
ls -la /home/kaokaolv/test-swe
```

结果：找到 `.venv-aqa-gpu`、`.venv-vllm-main`、`.venv-vllm-nightly`、`.venv-omini-latest` 等环境，以及 `test-swe/setup_vllm_env_*.sh`、`serve_*.sh` 示例。目录内容很多，后续不再做无差别全目录列举，改为只读取目标脚本和版本元数据。未读取 `.netrc`、`.env`、SSH 私钥等敏感文件。

尝试用 Glob 搜索：

```text
path=/home/kaokaolv/test-swe
pattern=**/*.{sh,py,md,toml,yaml,yml}
```

20 秒超时且没有收集到结果；不再递归扫整个目录，改为根据 `ls` 已确认的准确路径直接 Read。

读取：

- `/home/kaokaolv/AGENTS.md`
- `/home/kaokaolv/test-swe/AGENTS.md`
- `/home/kaokaolv/test-swe/setup_vllm_env_dsv4flash0731_onpar.sh`
- `/home/kaokaolv/test-swe/setup_vllm_env_qwen38fn.sh`
- `/home/kaokaolv/test-swe/serve_DeepSeek-V4-Flash-0731-MXFP4.sh`

采用的已有惯例：系统 Python 3.12 + venv + uv、`--torch-backend=cu130`、显式 `VIRTUAL_ENV`、启动前检查 GPU/端口、日志落盘。没有采用示例中的 vLLM 补丁、TP、多卡、NVLS 参数或停止其他进程的 `pkill` 行为，因为这些与 ComfyUI 单卡任务无关且可能影响其他任务。

### 实时 GPU、磁盘、端口和 IP

执行：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h / /tmp /home/kaokaolv/comfyui_h3
ss -ltn '( sport = :8188 )'
hostname -I
```

关键结果：

```text
GPU 0: 1588 MiB / 183359 MiB, 0%
GPU 1: 1588 MiB / 183359 MiB, 0%
GPU 2: 4 MiB / 183359 MiB, 0%
GPU 3/4: 约 179 GiB 已用，100%
GPU 5: 158024 MiB 已用
GPU 6/7: 有其他任务和较高利用率
/: 2.1 GiB available
/home: 6.1 TiB available
8188: 无监听
IP: 172.26.46.130（另有其他网卡地址）
```

决定：物理 GPU 从调研时的 1 改为当前完全空闲的 **2**；启动前脚本仍会再次打印 GPU 状态，避免静态假设。

### 旧 venv 版本元数据

对 `/home/kaokaolv/.venv*/bin/python` 逐个执行只读元数据查询，输出 Python、torch、torchvision、torchaudio、huggingface_hub、comfyui-frontend-package 的版本；没有 import vLLM 或启动旧服务，没有修改旧环境。

结果：多个环境使用 Python 3.12.3 和以下组合：

```text
torch 2.13.0
torchvision 0.28.0
torchaudio 2.11.0
```

部分 Omni 环境是 torch/torchvision `+cu132`、torchaudio `+cpu`，不适合作为本任务的直接参考；没有旧环境安装 ComfyUI frontend。决定新建隔离环境，不复用正在服务其他任务的 venv。

### 实测 CUDA build 与 B200 支持

执行：

```bash
CUDA_VISIBLE_DEVICES=2 /home/kaokaolv/.venv-vllm-main/bin/python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('arch list:', torch.cuda.get_arch_list())
PY
uv pip install --help | python3 -c "import sys; [print(line,end='') for line in sys.stdin if 'torch-backend' in line or 'extra-index-url' in line]"
```

输出：

```text
torch: 2.13.0+cu130
CUDA build: 13.0
CUDA available: True
GPU: NVIDIA B200
capability: (10, 0)
arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
uv 支持 --torch-backend 和 --extra-index-url
```

结论：固定 `torch==2.13.0`、`torchvision==0.28.0`、`torchaudio==2.11.0`，通过 uv `--torch-backend=cu130` 安装。这个组合有本机实测证据，不再根据 wheel 索引中的最大版本号盲目升级。

另检查 `/home/kaokaolv/models`，未发现现成 MiniMax H3/ComfyUI 模型目录，因此按既定 5 文件下载。

## 10. Setup/start 脚本设计与静态检查

创建：

- `setup_comfyui_h3.sh`
- `start_comfyui_h3.sh`

### setup 脚本行为

`setup_comfyui_h3.sh` 固定以下版本：

```text
ComfyUI v0.45.0
workflow commit ea7faae6999fb31338f43408ff4a575bd3f7d9a0
HF revision 4b53a35191437de05370da548e59aeabe2a5a09a
torch 2.13.0 + cu130
torchvision 0.28.0
torchaudio 2.11.0
huggingface_hub 1.32.0
```

脚本特性：

- `set -euo pipefail`，任何错误即停止。
- 所有实质命令通过 `set -x` 和 `tee` 记录到 `logs/setup_comfyui_h3.log`，包含行号和时间。
- TMPDIR、uv/XDG/HF cache 全部在项目目录，避免根分区写满。
- clone 前拒绝覆盖非 git 目录；已有 checkout 版本不符时退出，不擅自 reset。
- venv 可幂等复用；包安装可重复执行。
- 在下载 42 GB 模型前检查 CUDA 13.0、B200、`sm_100`、torchaudio 和 PyAV。
- 校验 workflow JSON ID、前端和核心节点版本。
- 只下载所需 5 个文件，然后逐个 SHA256 校验。
- 安装结束写 `logs/environment.freeze.txt`，记录实际依赖闭包。

### start 脚本行为

`start_comfyui_h3.sh` 默认：

```text
GPU_ID=2
HOST=0.0.0.0
PORT=8188
```

检查 venv、main.py 和端口占用，打印 GPU 与服务器 IP，最后 `exec` ComfyUI。没有 `--highvram`、`--gpu-only` 或未经验证的 `--fast`；保留默认显存管理。使用 `--disable-partner-nodes`，不影响本地 H3 节点。

### 静态检查命令

```bash
chmod +x /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh
bash -n /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh
bash -n /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh
if command -v shellcheck >/dev/null; then
  shellcheck /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh
else
  echo 'shellcheck: not installed; bash -n passed'
fi
```

结果：两个脚本 `bash -n` 均通过；机器没有 shellcheck，因此没有声称做过 shellcheck。下一步执行 setup，完整原始输出以脚本日志为准。

## 11. 执行 setup

执行命令：

```bash
bash /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh
```

运行方式：Bash background task，task ID `bash-kvn8hzvy`，PID `930809`，cwd `/home/kaokaolv/comfyui_h3`，任务超时设为 7200 秒。脚本自身完整记录到：

```text
/home/kaokaolv/comfyui_h3/logs/setup_comfyui_h3.log
```

脚本顺序为：版本/资源记录 → clone v0.45.0 → 创建独立 venv → 安装固定 cu130 兼容组合与 ComfyUI requirements → CUDA/B200/PyAV/torchaudio 实测 → 下载并解析 workflow → 下载 5 模型 → SHA256 → freeze 依赖。若任一步失败，`set -e` 会阻止后续步骤，不会跳过错误继续报成功。

## 12. 安装中间状态与 API 执行图源码核对

### setup 中间状态

第一次 WaitFor 等待 90 秒后任务仍在运行；随后用 TaskOutput 读取一次非阻塞快照。此时约运行 5 分钟，关键原始输出为：

```text
ComfyUI tag: v0.45.0
ComfyUI commit: 989d93d0d968e704c589ce0146bb18c77f6b50cc
Python: 3.12.3
torch: 2.13.0+cu130
torchvision: 0.28.0+cu130
torchaudio: 2.11.0+cu130
triton: 3.7.1
33 个 PyTorch 相关包安装成功
ComfyUI requirements 仍在下载/安装，尚未进入模型下载阶段
```

没有错误需要修复。完整逐命令输出持续写入 `logs/setup_comfyui_h3.log`，这里只记录阶段摘要，不重复粘贴数百行包下载列表。

### API 路由文档

读取官方 `https://docs.comfy.org/development/comfyui-server/comms_routes`，确认：

- POST `/prompt` 验证执行图并入队，返回 prompt_id/number 或 error/node_errors。
- GET `/object_info/{node_class}` 返回节点 schema。
- GET `/history/{prompt_id}` 返回执行历史。
- `/queue`、`/interrupt` 可查询或停止执行。

另一个尝试地址 `https://docs.comfy.org/development/core-concepts/api` 返回 404；没有用它作为依据，也没有继续猜测相似 URL，改用上述官方路由文档和本地 v0.45.0 源码。

### 新克隆仓库的 AGENTS.md

读取本地源码后发现并完整读取 `ComfyUI/AGENTS.md`。本任务不修改 ComfyUI 核心代码；只按用户明确授权下载模型到标准 models 目录。后续若遇到 bug，优先做最小、可证据化的修复，不引入额外联网路径或依赖。

### 执行图输入格式

直接检查本地源码：

- `comfy_extras/nodes_custom_sampler.py`
  - `RandomNoise` API 只需要 `noise_seed`，UI 的 control_after_generate 不是后端输入。
  - `BasicScheduler`: model/scheduler/steps/denoise。
  - `BasicGuider`: model/conditioning。
  - `SamplerCustomAdvanced`: noise/guider/sampler/sigmas/latent_image。
- `comfy_extras/nodes_audio.py`
  - `VAEDecodeAudio`: samples/vae。
- `comfy_extras/nodes_video.py`
  - v0.45.0 的 `SaveVideo` 已把 format 改为 DynamicCombo，并有可选隐藏 codec 向后兼容。
  - `CreateVideo` 新增可选 color_space/chroma_subsampling/codec；API 首跑会显式传 sRGB/auto/none，bit_depth=8。
- `comfy_api/latest/_io.py` 与 `execution.py`
  - DynamicCombo 通过 `format`、`format.codec` 等展开字段和 dynamic_paths 重组，不能简单把 UI 的字符串 widgets 原样 POST。

因此首个 API 执行图采用：

```text
UNETLoader(127) 直接连 BasicScheduler/BasicGuider（默认非 Turbo，20 步）
CLIPLoader(128, type=minimax)
MiniMaxH3ImageToVideo(131, 864x480, length=124)
RandomNoise(129, 模板原 seed)
res_multistep + simple scheduler
SamplerCustomAdvanced(125)
VAEDecode + VAEDecodeAudio
CreateVideo(24fps, 8bit, sRGB)
SaveVideo(format=mp4, format.codec=h264)
```

prompt 将从下载的官方 workflow 顶层子图控件原样提取，输出文件名前缀单独设为 `video/MiniMax_H3_API_Smoke`，便于识别验证产物。服务启动后先查询 object_info，再 POST；若 schema 校验失败，保存完整错误并修正，不会把 HTTP 200  alone 当作生成成功。

## 13. 问题：requirements 安装日志暂停与 NFS 诊断

现象：uv 在打印 `Downloaded comfyui-workflow-templates-media-video` 后数分钟没有新日志。没有立即 kill/retry，先检查进程与 I/O。

执行：

```bash
ps -eo pid,ppid,stat,etime,wchan:32,args | python3 -c "import sys; [print(line,end='') for line in sys.stdin if any(x in line for x in ('setup_comfyui_h3.sh','uv pip install','tee -a','hf download'))]"
du -sh /home/kaokaolv/comfyui_h3/.venv /home/kaokaolv/comfyui_h3/.uv-cache
df -h /home/kaokaolv/comfyui_h3 /tmp
```

进程显示：

```text
setup bash PID 930809: do_wait
uv PID 940236: futex_wait_queue
tee: pipe_read
```

但 `du -sh .venv .uv-cache` 在 NFS 上递归扫描超过 60 秒，诊断命令被工具转为后台 task `bash-kl1rb30j`。该 du 不是部署必需步骤且增加 NFS 压力，因此用 TaskStop 停止；没有停止 setup 主任务。此问题说明部署期间应避免对大型 uv cache 做递归扫描。

进一步执行：

```bash
ps -L -p 940236 -o pid,tid,stat,etime,wchan:32,comm
ss -tpn | python3 -c "import sys; [print(line,end='') for line in sys.stdin if '940236' in line]"
ls -l /proc/940236/fd
```

并读取 `/proc/940236/io`：

```text
rchar: 5216381
wchar: 1426086735
read_bytes: 4431872
write_bytes: 1484050432
一个 worker: rpc_wait_bit_killable
打开文件: .uv-cache/wheels-v6/pypi/comfyui-embedded-docs/.nfs...
仍有内网 ESTAB 连接
```

结论：不是依赖解析错误，也不是 GPU 问题；uv 已写入约 1.48 GB，正在等待 NFS 对 `comfyui-embedded-docs` 临时文件的 I/O 返回。处理决定是保留原进程等待，不清理缓存、不重复下载、不切换到只剩 2.1 GiB 的根分区。额外 du 诊断已停止以减少竞争。后续只有确认持续无 I/O 进展时才需要改变缓存策略。

## 14. NFS 等待恢复、模型下载开始与执行图生成

约 90 秒后继续检查时，uv 进程 PID 940236 已退出，说明 NFS 写入最终完成，没有人工重跑。日志已前进到：

```text
torch: 2.13.0+cu130
torchaudio: 2.11.0+cu130
torch CUDA: 13.0
CUDA available: True
GPU: NVIDIA B200
compute capability: (10, 0)
arch list 包含 sm_100
workflow 下载成功（68573 bytes）
workflow JSON 校验通过
hf download 已开始
```

workflow 自检输出 `top-level nodes: 6`，因此把调研文档此前误写的“7 个节点”更正为“6 个节点”（SaveVideo、ResolutionSelector、3 个 MarkdownNote、1 个子图实例）。这不影响数据流或部署。

模型下载命令已启动，初期出现多条 `.gitignore.lock` 等待信息，这是 5 文件并发下载时 huggingface_hub 对 local-dir 元数据锁的协调；首个文件随后完成，进度到 20%，暂不视为错误。HF 的新版本/hf-cli skill 提示与本任务无关，不升级、不安装 skill。

利用已下载的官方 JSON 生成真实 API 请求：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
# 从 node 140 提取原 prompt/seed，构造 14 节点执行图，
# 写入 logs/h3_t2v_prompt.json；完整脚本逻辑如下：
# 127 UNETLoader; 128 CLIPLoader; 119/120 VAELoader;
# 131 MiniMaxH3ImageToVideo; 129 RandomNoise; 123 KSamplerSelect;
# 124 BasicScheduler; 126 BasicGuider; 125 SamplerCustomAdvanced;
# 122 VAEDecode; 121 VAEDecodeAudio; 130 CreateVideo; 92 SaveVideo。
PY
```

实际生成文件：`/home/kaokaolv/comfyui_h3/logs/h3_t2v_prompt.json`。结果：

```text
14 个 API 节点
seed 757358688076805
864x480，124 帧，20 步，非 Turbo
prompt 1647 字符（官方模板原文）
SaveVideo: format=auto, format.codec=auto
输出前缀: video/MiniMax_H3_API_Smoke
```

执行图 JSON 已持久化，后续 POST 和故障复现都可使用同一份输入，不需要再次手动拼接。

### 第 14 节 API 请求生成命令的完整 here-doc

为满足完整留痕，补充上面实际执行的完整脚本（不是伪代码）：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from pathlib import Path
base=Path('/home/kaokaolv/comfyui_h3')
workflow=json.loads((base/'video_minimax_h3_t2v.json').read_text())
subgraph=next(n for n in workflow['nodes'] if n['id']==140)
prompt=subgraph['widgets_values_named']['prompt']
seed=subgraph['widgets_values_named']['noise_seed']
graph={
  '127': {'class_type':'UNETLoader','inputs':{'unet_name':'minimax_h3_fl2va_pruned_int8_convrot.safetensors','weight_dtype':'default'}},
  '128': {'class_type':'CLIPLoader','inputs':{'clip_name':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','type':'minimax','device':'default'}},
  '119': {'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_int8_convrot.safetensors'}},
  '120': {'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_audio_vae_fp32.safetensors'}},
  '131': {'class_type':'MiniMaxH3ImageToVideo','inputs':{'clip':['128',0],'vae':['119',0],'prompt':prompt,'width':864,'height':480,'length':124}},
  '129': {'class_type':'RandomNoise','inputs':{'noise_seed':seed}},
  '123': {'class_type':'KSamplerSelect','inputs':{'sampler_name':'res_multistep'}},
  '124': {'class_type':'BasicScheduler','inputs':{'model':['127',0],'scheduler':'simple','steps':20,'denoise':1.0}},
  '126': {'class_type':'BasicGuider','inputs':{'model':['127',0],'conditioning':['131',0]}},
  '125': {'class_type':'SamplerCustomAdvanced','inputs':{'noise':['129',0],'guider':['126',0],'sampler':['123',0],'sigmas':['124',0],'latent_image':['131',1]}},
  '122': {'class_type':'VAEDecode','inputs':{'samples':['125',0],'vae':['119',0]}},
  '121': {'class_type':'VAEDecodeAudio','inputs':{'samples':['125',0],'vae':['120',0]}},
  '130': {'class_type':'CreateVideo','inputs':{'images':['122',0],'audio':['121',0],'fps':24,'bit_depth':8,'color_space':'sRGB','chroma_subsampling':'auto','codec':'none'}},
  '92': {'class_type':'SaveVideo','inputs':{'video':['130',0],'filename_prefix':'video/MiniMax_H3_API_Smoke','format':'auto','format.codec':'auto'}},
}
out=base/'logs/h3_t2v_prompt.json'
out.write_text(json.dumps({'prompt':graph},indent=2,ensure_ascii=False)+'\n')
print('wrote:',out)
print('nodes:',len(graph))
print('seed:',seed)
print('resolution: 864x480; frames: 124; steps: 20; turbo: false')
print('prompt characters:',len(prompt))
PY
```

## 15. 模型下载完成与活动目录 NFS 诊断处理

按准确目录列出模型文件，确认文本编码器、两个 VAE 和 LoRA 已落盘；只剩 diffusion model。随后尝试：

```bash
ls -lah /home/kaokaolv/comfyui_h3/ComfyUI/models/.cache/huggingface/download/diffusion_models
ps -eo pid,ppid,stat,etime,wchan:28,args | python3 -c "import sys; [print(line,end='') for line in sys.stdin if 'hf download Comfy-Org/MiniMax-H3' in line]"
```

活动下载目录的 NFS `ls` 超过两分钟，被转为后台 task `bash-q9v1nory`。它与之前递归 du 一样不是必要验证且增加 I/O 竞争，因此已停止。后续不再对活动下载目录做额外元数据扫描，以 setup 日志和最终 SHA256 为准。

最终 HF 日志：

```text
Fetching 5 files: 100%
✓ Downloaded
path: /home/kaokaolv/comfyui_h3/ComfyUI/models
5 文件总耗时约 8 分 52 秒
```

当前执行到：

```bash
cd /home/kaokaolv/comfyui_h3/ComfyUI/models
sha256sum --check <<'EOF'
# 第 3 节记录的 5 组官方 LFS SHA256 与相对路径
EOF
```

42 GB 在 NFS 上读取校验需要时间；只有 5 个文件都输出 OK 才进入服务启动。

## 16. Setup 完成、服务验证与首次提交

### setup 最终状态

后台 task `bash-kvn8hzvy` 已以 exit code 0 完成，总时长约 1971 秒。日志最后确认：

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors: OK
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors: OK
vae/minimax_h3_video_vae_int8_convrot.safetensors: OK
vae/minimax_h3_audio_vae_fp32.safetensors: OK
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors: OK
models: 40G
.venv: 6.0G
Setup completed successfully.
```

完整依赖闭包在 `logs/environment.freeze.txt`，包括 `av==19.0.2`、frontend 1.58.5、templates 0.11.105、comfy-kitchen 0.2.41。

### 服务启动

执行：

```bash
bash /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh > /home/kaokaolv/comfyui_h3/logs/comfyui_server.log 2>&1
```

后台 task `bash-r5bhummb`，shell PID 983575，ComfyUI Python PID 983576；服务是长期进程，不设自动超时。实际 argv：

```text
main.py --listen 0.0.0.0 --port 8188 --disable-partner-nodes
```

`ss -ltnp` 实测：

```text
LISTEN 0 128 0.0.0.0:8188 0.0.0.0:* users:(("python",pid=983576,...))
```

### readiness 与模型/节点验证

用项目 venv Python `http.client` 每 2 秒请求 `/system_stats`，等待上限 120 秒；前 12 次为 connection refused，随后服务就绪。验证脚本还请求：

```text
/models/diffusion_models
/models/text_encoders
/models/vae
/models/loras
/object_info/MiniMaxH3ImageToVideo
/object_info/SaveVideo
/object_info/CreateVideo
/object_info/VAEDecodeAudio
```

结果：

- ComfyUI 0.45.0、frontend 1.58.5、templates 0.11.105、Python 3.12.3、torch 2.13.0+cu130。
- 设备为 NVIDIA B200，vram_total 191502876672 bytes。
- 所有 5 个模型都在对应 API 列表中。
- H3/SaveVideo/CreateVideo/VAEDecodeAudio schema 均返回 200。
- 完整结果保存为 `logs/server_validation.json`。
- `curl --noproxy '*' -fsS http://172.26.46.130:8188/system_stats > logs/system_stats_lan.json` 成功，证明绑定地址可通过内网 IP 访问，不只是 127.0.0.1。

外部浏览器地址：`http://172.26.46.130:8188`。原始 UI workflow：`/home/kaokaolv/comfyui_h3/video_minimax_h3_t2v.json`。

服务日志还确认：

```text
Using async weight offloading with 2 streams
DynamicVRAM support detected and enabled
Using pytorch attention
Found comfy_kitchen backend cuda: available=True
```

triton backend `disabled=True` 是 ComfyUI 默认策略；CUDA backend available=True，因此不需要为首跑额外 `--enable-triton-backend`。无 OpenGL_accelerate 是可选加速模块提示，不影响此 workflow。

### 首次真实 prompt 提交

后台 task `bash-7idia8t0`，执行保存的 `logs/h3_t2v_prompt.json`，完整 stdout/stderr 写入 `logs/first_generation.log`。

POST `/prompt` 返回：

```json
{"prompt_id":"3a119304-3b35-4156-8b80-0e8a3ba829f7","number":0,"node_errors":{}}
```

说明 API 图包括 DynamicCombo 在内已通过服务器校验。queue 显示 running=1、pending=0。服务日志已开始加载：

- audio fp32 VAE
- video int8_convrot VAE
- MiniMaxH3TEModel_，14956MB staged
- MiniMaxH3，19995MB staged
- 检测到 nvfp4/mixed precision 与 FLOW_AV

当前仍是执行中，不能据此提前宣布生成成功；后续以 history success、输出文件、PyAV 解码和实际媒体读取共同验收。

## 17. 首次生成、媒体验收与用户 workflow 发布

### 首次生成完整命令

下面是实际后台执行的完整命令；不是只 import/compile，而是 POST 真实执行图并等待 history：

```bash
set -o pipefail; /home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY' 2>&1 | tee /home/kaokaolv/comfyui_h3/logs/first_generation.log
import http.client, json, time
from pathlib import Path
base=Path('/home/kaokaolv/comfyui_h3')
request_body=(base/'logs/h3_t2v_prompt.json').read_bytes()

def request(method,path,body=None):
    conn=http.client.HTTPConnection('127.0.0.1',8188,timeout=60)
    headers={'Content-Type':'application/json'} if body is not None else {}
    conn.request(method,path,body=body,headers=headers)
    response=conn.getresponse()
    data=response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f'{method} {path}: HTTP {response.status}: {data.decode(errors="replace")}')
    return json.loads(data)

started=time.time()
response=request('POST','/prompt',request_body)
(base/'logs/first_prompt_response.json').write_text(json.dumps(response,indent=2)+'\n')
print('prompt response:',json.dumps(response),flush=True)
prompt_id=response['prompt_id']
print('prompt_id:',prompt_id,flush=True)

while True:
    history=request('GET','/history/'+prompt_id)
    if prompt_id in history:
        item=history[prompt_id]
        (base/'logs/first_history.json').write_text(json.dumps(item,indent=2,ensure_ascii=False)+'\n')
        status=item.get('status',{})
        print('history status:',json.dumps(status),flush=True)
        if status.get('completed') or status.get('status_str') in ('success','error'):
            break
    else:
        queue=request('GET','/queue')
        print(f'waiting {time.time()-started:.0f}s; running={len(queue.get("queue_running",[]))}; pending={len(queue.get("queue_pending",[]))}',flush=True)
    time.sleep(10)

files=[]
def collect(obj):
    if isinstance(obj,dict):
        if 'filename' in obj:
            files.append({k:obj.get(k) for k in ('filename','subfolder','type')})
        for value in obj.values(): collect(value)
    elif isinstance(obj,list):
        for value in obj: collect(value)
collect(item.get('outputs',{}))
result={'prompt_id':prompt_id,'elapsed_seconds':time.time()-started,'status':status,'files':files}
(base/'logs/first_generation_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('result:',json.dumps(result,ensure_ascii=False),flush=True)
assert status.get('status_str') == 'success', status
assert status.get('completed') is True, status
assert files, 'No output files in history'
PY
```

后台 task `bash-7idia8t0` exit code 0，总耗时 70.167 秒。服务器报告 `Prompt executed in 62.54 seconds`，采样为 20/20 步。结果文件：

```text
ComfyUI/output/video/MiniMax_H3_API_Smoke_00001_.mp4
```

### 媒体全量解码完整命令

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import av, json
import numpy as np
from pathlib import Path
path=Path('/home/kaokaolv/comfyui_h3/ComfyUI/output/video/MiniMax_H3_API_Smoke_00001_.mp4')
assert path.is_file(), path
report={'path':str(path),'size_bytes':path.stat().st_size}
with av.open(str(path)) as container:
    report['container_duration_seconds']=container.duration/1e6 if container.duration else None
    report['streams']=[]
    for stream in container.streams:
        item={'type':stream.type,'codec':stream.codec_context.name}
        if stream.type=='video':
            item.update(width=stream.codec_context.width,height=stream.codec_context.height,average_rate=str(stream.average_rate),frames=stream.frames)
        if stream.type=='audio':
            item.update(sample_rate=stream.codec_context.sample_rate,channels=stream.codec_context.channels,layout=stream.codec_context.layout.name)
        report['streams'].append(item)
    video=container.streams.video[0]
    audio=container.streams.audio[0]
    assert (video.codec_context.width,video.codec_context.height)==(864,480)
    assert video.average_rate==24
    assert audio.codec_context.channels==2
    assert audio.codec_context.sample_rate==32000
    frame_means=[]
    frame_stds=[]
    for frame in container.decode(video=0):
        array=frame.to_ndarray(format='rgb24')
        frame_means.append(float(array.mean()))
        frame_stds.append(float(array.std()))
report['decoded_video_frames']=len(frame_means)
report['video_mean_range']=[min(frame_means),max(frame_means)]
report['video_std_range']=[min(frame_stds),max(frame_stds)]
assert len(frame_means)==124
assert max(frame_stds)>5.0, 'Video appears blank or nearly blank'
with av.open(str(path)) as container:
    audio_frames=0
    audio_samples=0
    audio_max=0.0
    audio_abs_sum=0.0
    audio_value_count=0
    for frame in container.decode(audio=0):
        array=frame.to_ndarray().astype(np.float64)
        audio_frames+=1
        audio_samples+=frame.samples
        audio_max=max(audio_max,float(np.abs(array).max()))
        audio_abs_sum+=float(np.abs(array).sum())
        audio_value_count+=array.size
report['decoded_audio_frames']=audio_frames
report['decoded_audio_samples_per_channel']=audio_samples
report['audio_max_abs']=audio_max
report['audio_mean_abs']=audio_abs_sum/audio_value_count
assert audio_frames>0 and audio_samples>0
assert audio_max>1e-4, 'Audio track is silent'
out=Path('/home/kaokaolv/comfyui_h3/logs/media_validation.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print('Media validation: PASS')
PY
```

实测结果：

```text
size: 1526558 bytes
container duration: 5.166688 s
video: h264, 864x480, 24/1 fps, 124 frames
audio: AAC, 32000 Hz, 2 channels, stereo
decoded video frames: 124
video mean range: 48.2245–80.9200
video std range: 34.0704–55.4711
decoded audio frames: 162
samples/channel: 165334
audio max abs: 0.6482708454
audio mean abs: 0.1590286648
Media validation: PASS
```

另外调用 `ReadMediaFile` 读取该 MP4 成功，工具识别 `video/mp4`、大小 1526558 bytes。结合全帧解码与像素标准差，确认不是空文件或空白视频；音频最大值也确认不是静音轨。没有凭主观想象描述视频内容。

### 把 workflow 发布到外部浏览器可打开的位置

最初 `ComfyUI/user` 只有 `used_headlessly`，请求：

```bash
curl --noproxy '*' -fsS 'http://127.0.0.1:8188/userdata?dir=workflows'
```

返回 404。读取 `app/user_manager.py` 和 `folder_paths.py` 后确认：单用户模式的公开目录是 `ComfyUI/user/default`，workflows 子目录尚不存在，所以 404 是目录未创建，不是服务异常。

执行：

```bash
mkdir -p /home/kaokaolv/comfyui_h3/ComfyUI/user/default/workflows
cp -n /home/kaokaolv/comfyui_h3/video_minimax_h3_t2v.json \
  /home/kaokaolv/comfyui_h3/ComfyUI/user/default/workflows/MiniMax_H3_T2V_Deployed.json
sha256sum \
  /home/kaokaolv/comfyui_h3/video_minimax_h3_t2v.json \
  /home/kaokaolv/comfyui_h3/ComfyUI/user/default/workflows/MiniMax_H3_T2V_Deployed.json
curl --noproxy '*' -fsS 'http://127.0.0.1:8188/userdata?dir=workflows'
```

两个文件 SHA256 都是 `8149b64db94375cd01a7d46da0b4a4c7349d83dedf6b5e1026b460e5e202626f`，API 返回：

```json
["MiniMax_H3_T2V_Deployed.json"]
```

`cp -n` 给出 GNU coreutils 非可移植警告，功能成功；为以后可重复执行，setup 脚本改用 `cp --update=none`，并在复制后用 `cmp` 验证一致。新增脚本片段：

```bash
USER_WORKFLOW_DIR="$COMFY_DIR/user/default/workflows"
USER_WORKFLOW="$USER_WORKFLOW_DIR/MiniMax_H3_T2V_Deployed.json"
mkdir -p "$USER_WORKFLOW_DIR"
cp --update=none "$WORKFLOW" "$USER_WORKFLOW"
cmp "$WORKFLOW" "$USER_WORKFLOW"
```

这样外部浏览器无需访问服务器本地文件选择器，直接在用户 workflow 列表打开即可。

## 18. 最终健康检查与交付审计

执行：

```bash
bash -n /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh
bash -n /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh
curl --noproxy '*' -fsS http://127.0.0.1:8188/queue
curl --noproxy '*' -fsS 'http://172.26.46.130:8188/userdata?dir=workflows'
ss -ltnp '( sport = :8188 )'
nvidia-smi -i 2 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
git -C /home/kaokaolv/comfyui_h3/ComfyUI status --short
ls -lh /home/kaokaolv/comfyui_h3/logs
```

结果：

- 两个脚本语法检查通过。
- queue_running 和 queue_pending 均为空，首个任务已完成，没有遗留排队任务。
- 内网 IP 可读取用户 workflow 列表。
- 服务仍监听 `0.0.0.0:8188`，Python PID 983576。
- GPU 2 当前占用约 37722 MiB，是生成后保留的模型缓存；利用率 0%，不是仍在跑任务。
- ComfyUI git status 为空，没有修改核心源码。
- 日志目录包含 setup、freeze、server、API 请求/响应、history、生成结果、媒体验证和内网访问验证，全部保留供复核。

交付核对：

1. 参考旧 venv/test-swe：完成，且未修改旧环境。
2. 独立 setup/start 脚本：完成。
3. 每步命令、分析、问题与解决记录：记录在本文和 logs；NFS 两次诊断、404、节点计数修正均已写明。
4. 模型下载和完整性：5/5 SHA256 OK。
5. host=0.0.0.0：ss 与内网 HTTP 请求均验证。
6. workflow 可在用户列表打开：`MiniMax_H3_T2V_Deployed.json`。
7. 真实生成：history success，20/20 步。
8. 视频和音轨：全量解码及 ReadMediaFile 通过。
9. 服务保持运行，不停止，不占用其他 GPU。

## 19. 最终前端检查

补充执行：

```bash
bash -n /home/kaokaolv/comfyui_h3/setup_comfyui_h3.sh
bash -n /home/kaokaolv/comfyui_h3/start_comfyui_h3.sh
curl --noproxy '*' -fsS -o /dev/null -w 'frontend HTTP %{http_code}, %{content_type}\n' http://172.26.46.130:8188/
curl --noproxy '*' -fsS http://127.0.0.1:8188/queue
```

结果：`frontend HTTP 200, text/html`，queue 仍为空。调研文档已同步实际 GPU 2、0.0.0.0、固定 cu130 版本、用户 workflow 和实测结果；保留的 127.0.0.1 仅用于服务器本地 API 检查，不与服务监听要求冲突。

---

# 用户网页操作问题排查

## 20. 只看到文生图且 VAEDecode 报错

用户反馈：打开 `http://sc09super22-b200.sc.intel.com:8188/` 只看到生成图，最新 `comfyui_server.log` 报错，未看到视频。

### 已确认的生成方式

`MiniMax_H3_API_Smoke_00001_.mp4` 不是点击网页默认“生成图”产生的，而是将 `logs/h3_t2v_prompt.json` 中的 14 节点 H3 API 执行图 POST 到 `/prompt`，得到 prompt ID `3a119304-3b35-4156-8b80-0e8a3ba829f7`，随后由 SaveVideo 写出。它使用 MiniMaxH3ImageToVideo、VAEDecodeAudio、CreateVideo、SaveVideo，和默认文生图不是同一张图。

### 最新日志与 history 证据

读取 `logs/comfyui_server.log` 第 98–170 行，并通过 `/history` 枚举所有任务。history 共 3 条：

1. API H3 T2V：success，输出 MP4。
2. 网页任务 `a1532671-1567-4beb-86d9-b398b0854898`：error。
3. 网页任务 `5bc21114-ccef-47d5-963d-4a776baf0ee8`：error。

用以下完整命令打印两个错误任务的真实执行图，而不是依赖猜测：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import http.client,json
for pid in ('a1532671-1567-4beb-86d9-b398b0854898','5bc21114-ccef-47d5-963d-4a776baf0ee8'):
    c=http.client.HTTPConnection('127.0.0.1',8188,timeout=30)
    c.request('GET','/history/'+pid)
    item=json.loads(c.getresponse().read())[pid]; c.close()
    graph=item['prompt'][2]
    print('\nPROMPT',pid)
    for node_id,node in graph.items():
        print(node_id,node['class_type'],json.dumps(node['inputs'],ensure_ascii=False))
PY
```

两个错误图完全相同，关键节点为：

```text
SaveImage
CLIPLoader(type=lumina2)
VAELoader(minimax_h3_audio_vae_fp32.safetensors)
EmptySD3LatentImage(1024x1024)
ModelSamplingAuraFlow
KSampler
VAEDecode(samples=KSampler, vae=audio_vae)
```

因此用户当前浏览器运行的是 Lumina2 默认文生图 workflow，不是 H3 T2V。错误发生在 `nodes.py` 普通 `VAEDecode` 调用 `comfy/ldm/minimax/audio_vae.py:422`：

```text
RuntimeError: The size of tensor a (24) must match the size of tensor b (32) at non-singleton dimension 1
```

直接原因：该文生图把音频 VAE 当成图像 VAE 解码；更根本的问题是打开了错误的 workflow。服务首页不会自动切换到已部署的用户 workflow。

### 服务器内正确 workflow 复核

执行：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/home/kaokaolv/comfyui_h3/ComfyUI/user/default/workflows/MiniMax_H3_T2V_Deployed.json')
w=json.loads(p.read_text())
print('workflow id:',w['id'])
print('top-level types:',[n['type'] for n in w['nodes']])
sg=w['definitions']['subgraphs'][0]
print('subgraph name:',sg['name'])
print('has H3:',any(n['type']=='MiniMaxH3ImageToVideo' for n in sg['nodes']))
print('has CreateVideo:',any(n['type']=='CreateVideo' for n in sg['nodes']))
print('has SaveVideo:',any(n['type']=='SaveVideo' for n in w['nodes']))
PY
```

结果确认：服务器内的用户 workflow 是正确的 H3 T2V，包含 H3、CreateVideo、SaveVideo。

### 可直接访问的视频和 JSON

执行：

```bash
curl --noproxy '*' -fsS -o /dev/null -w 'video HTTP %{http_code}, %{content_type}, %{size_download} bytes\n' \
  'http://127.0.0.1:8188/view?filename=MiniMax_H3_API_Smoke_00001_.mp4&subfolder=video&type=output'
curl --noproxy '*' -fsS -o /dev/null -w 'workflow HTTP %{http_code}, %{content_type}, %{size_download} bytes\n' \
  'http://127.0.0.1:8188/userdata/workflows%2FMiniMax_H3_T2V_Deployed.json'
```

结果：

```text
video HTTP 200, video/mp4, 1526558 bytes
workflow HTTP 200, application/json, 68573 bytes
```

因此给用户两个浏览器入口：

- 已生成视频：`http://sc09super22-b200.sc.intel.com:8188/view?filename=MiniMax_H3_API_Smoke_00001_.mp4&subfolder=video&type=output`
- 正确 JSON：`http://sc09super22-b200.sc.intel.com:8188/userdata/workflows%2FMiniMax_H3_T2V_Deployed.json`

正确操作是打开用户 workflow 或下载 JSON 后拖入画布；画布应出现 Note: MiniMax H3、Resolution Selector、Image to Video (MiniMax H3) 和 SaveVideo。若仍看到 SaveImage、EmptySD3LatentImage、ModelSamplingAuraFlow，则仍是错误的 Lumina2 文生图，不能继续点运行。

本次只诊断并给出入口，没有修改核心源码、模型或正在运行的服务。

---

# Model card 全 workflow 可行性调研

## 21. 是否可以全部跑一遍

用户询问 `https://huggingface.co/Comfy-Org/MiniMax-H3` model card 中的 workflows 是否都可以跑一遍。本步骤只做依赖与版本调研，没有批量下载模型、素材或提交新生成任务。

### 官方 workflow 范围

读取 model card 页面及原始 README：

- `https://huggingface.co/Comfy-Org/MiniMax-H3`
- `https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/README.md`
- `https://docs.comfy.org/tutorials/video/minimax/minimax-h3`

README 的 Workflows 表格实际有 **12 个本地模板**：

1. `video_minimax_h3_i2v.json`
2. `video_minimax_h3_r2v.json`
3. `video_minimax_h3_t2v.json`
4. `video_pika_h3_flex_ref2va.json`
5. `image_p_h3_image_edit.json`
6. `video_pika_h3_flex_t2v.json`
7. `image_minimax_h3_t2i.json`
8. `image_p_h3_t2i.json`
9. `video_pika_h3_flex_fl2va.json`
10. `video_minimax_h3_i2v_continuation.json`
11. `video_minimax_h3_multiframe_reference.json`
12. `video_minimax_h3_fun_controlnet_union.json`

这里的结论不包括文档中另外列出的 FastH3 或 API-node 模板。

### JSON 依赖解析

用 Python 并发读取上述 12 个官方 raw JSON，递归展开顶层与 definitions.subgraphs 节点，提取：

- `properties.models` 的 name/directory/url
- LoadImage/LoadVideo 输入素材名
- frontend/core 版本
- 非 comfy-core custom nodes
- 与当前 `ComfyUI/models` 的文件逐一比对

完整结构化结果保存为：

`logs/model_card_workflows_analysis.json`

所有模板都没有第三方 custom node。当前 ComfyUI v0.45.0 高于模板中最高的显式 core 0.43.0，也高于官方教程对 Multiframe 0.34、Fun ControlNet 0.35/Union2 0.38、FastH3 0.36 的要求。

### 逐项结论

| Workflow | 当前模型状态 | 还需要 |
|---|---|---|
| MiniMax H3 T2V | 已具备，且已实测成功 | 无 |
| MiniMax H3 I2V | 模型已具备 | 1 张输入图 |
| MiniMax H3 I2V continuation | 模型已具备 | 输入图/上一段结果 |
| MiniMax H3 R2V | 缺模型 | Ref2VA diffusion、Ref2V Turbo LoRA、2 张参考图 |
| MiniMax H3 Multiframe Reference | 缺模型 | Ref2VA diffusion、Ref2V Turbo LoRA、4 张参考图 |
| Pika H3 Flex T2V | 缺 LoRA | Pika FL2VA LoRA |
| Pika H3 Flex FL2VA | 缺 LoRA | Pika FL2VA LoRA、2 张图片 |
| Pika H3 Flex R2V | 缺模型/LoRA | Ref2VA diffusion、Pika Ref2VA LoRA、2 张参考图 |
| MiniMax H3 T2I | 缺 VAE | `minimax_h3_video_vae_fp16.safetensors` |
| P-H3-Image T2I | 缺专用组件 | P-H3 text encoder、prompt upsampler、VAE、LoRA |
| P-H3-Image Edit | 缺专用组件 | 同上的 4 个 P-H3 文件，另加 1 张输入图 |
| MiniMax H3 Fun ControlNet Union | 缺模型 | Ref2VA、Ref2V Turbo、Union patch、SDPose、RT-DETR、1 个控制视频 |

所有需要下载的模型都保留模板指定的 `int8_convrot`/BF16 配置，没有用其他精度文件擅自替代。MiniMax T2I 模板指定 fp16 video VAE，因此不能把已安装的 int8 VAE 直接算作完全相同配置。

### 额外下载量

查询 Hugging Face API：

```text
https://huggingface.co/api/models/Comfy-Org/MiniMax-H3?blobs=true
https://huggingface.co/api/models/Comfy-Org/SDPose?blobs=true
https://huggingface.co/api/models/PrunaAI/p-h3-image-comfy?blobs=true
```

缺失文件包括：

- Ref2VA pruned int8：20,970,379,616 bytes
- Ref2V Turbo LoRA：1,956,193,000 bytes
- Pika FL2VA/Ref2VA LoRA：各 1,956,172,304 bytes
- MiniMax fp16 video VAE：5,207,808,496 bytes
- Fun ControlNet Union int8 patch：2,296,635,360 bytes
- SDPose checkpoint：1,916,645,792 bytes
- RT-DETR：123,968,978 bytes
- P-H3 LoRA：1,956,195,360 bytes
- P-H3 prompt upsampler：6,724,096,116 bytes
- P-H3 text encoder：5,120,870,504 bytes
- P-H3 VAE：3,146,996,440 bytes

去重后合计：

```text
53,332,134,270 bytes = 49.6694 GiB
```

当前 `/home` 约有数 TiB 可用，容量不是阻碍；但下载、SHA256 校验及首次加载仍需时间。

### 节点和素材实测

对当前运行服务器 GET `/object_info`，与 12 个模板的所有节点类型比对：

```text
unique node types: 48
missing on current server: []
```

对 13 个官方输入素材 URL 逐个发 HEAD 请求，全部 HTTP 200：

- 12 张图片（各模板合计）及 `dancer_field_pose.mp4`
- 素材链接位于 `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/`
- 这些素材不在当前已安装 template package 的 RECORD 中，因此批量运行前需要显式下载到 `ComfyUI/input` 或手动上传

### 最终判断

从服务器版本、节点注册、GPU、模型可得性和输入素材可得性看，**12 个都可以在本机跑一遍**。但当前只有 T2V 已真实生成验证；其余模板需要补齐去重后约 49.67 GiB 的模型和 13 个官方输入素材，再逐个提交并检查 history/输出，不能把“依赖可满足”写成“已经全部跑通”。

---

# 全 workflow 批量运行与显存分析

## 22. 任务、分组与 profiling 方法

用户要求：分析已跑通 T2V 的各模型/模块显存，并按 I2V → Pika → Ref2VA/Multiframe → ControlNet → P-H3 分组下载、运行、检查输出和显存。

实施分组：

- baseline：T2V 冷启动重跑，作为已验证 workflow 的显存基线。
- i2v：I2V、I2V continuation、MiniMax H3 T2I。
- pika：Pika T2V、Pika FL2VA、Pika R2V。
- ref2va：MiniMax R2V、Multiframe Reference。
- controlnet：Fun ControlNet Union Pose 示例。
- p-h3：P-H3 T2I、P-H3 Image Edit。

这样覆盖 model card 的 12 个模板，同时保持用户指定的大组顺序。Pika R2V 在 Pika 组先下载 Ref2VA 基础模型，后续 Ref2VA 组复用。

### 显存口径

新增 `ComfyUI/custom_nodes/h3_memory_profiler.py`，只注册两个本地 HTTP 接口，不修改执行逻辑：

- GET `/h3/memory`：读取 PyTorch allocated/reserved/max_allocated/max_reserved，以及 ComfyUI `current_loaded_models` 的 model_class、patcher_class、model_size、loaded_size、offloaded_size、device、dynamic、patch_count。
- POST `/h3/memory/reset`：只重置 PyTorch peak counters，不调用 empty_cache，不卸载模型。

批量执行器还在每次 workflow 前调用官方 `/free` 清理上一模板模型，再 reset peaks，实现冷启动对比。显存报告将区分：

1. 模型文件磁盘大小；
2. ComfyUI model_size；
3. 当前 loaded_size/offloaded_size；
4. PyTorch 整进程 allocated/reserved 及峰值；
5. NVML 整卡进程占用（包含 CUDA context、allocator 之外的库等）；
6. 节点阶段采样峰值。

这些口径不可直接相加，也不把模型文件大小当成显存。

### 重启与日志保留

添加 profiling 前队列已确认为空，原服务 task `bash-r5bhummb` 停止。原日志先复制为：

`logs/comfyui_server_initial.log`

新服务 task `bash-qooj8jlr`，日志为：

`logs/comfyui_server_profiled.log`

重启后 GET `/h3/memory` 返回 200，冷状态下 models=[]、allocated/reserved 均为 0，证明 profiling 可用且未携带上一进程模型缓存。

### 官方 UI workflow 转换器

新建 `h3_batch_runner.py`，功能：

- 固定模板 commit，下载官方 JSON 和所需 input assets。
- 按组调用 `hf_hub_download`，固定模型 repo/revision，并逐文件 SHA256。
- 将 UI JSON 展开为 API graph：支持嵌套子图、子图输入/输出、literal 与 link 映射、mode=4 bypass、DynamicCombo 展开字段、输出祖先裁剪。
- 通过 `/prompt` 提交，通过 `/history` 验收。
- 每个 workflow 保存 API graph、history、result、WebSocket events、0.5 秒 memory samples、媒体全解码报告。
- 成功后发布到 ComfyUI 用户 workflows 目录。

首次转换时误把全量 `/object_info` 当成单节点接口的嵌套格式，出现 `KeyError: SaveVideo`。检查实际响应确认全量接口本身就是 `{class_type: schema}`，修正为直接索引；这是转换器 bug，不是 ComfyUI 接口变化。

修正后 T2V 展开为 23 个 API 节点。相比首次手写 14 节点图，它保留官方 Switch 的两个候选分支；`turbo_mode=false` 仍选择 20 步非 LoRA 路径。子图 ID 形如 `140:131`，链接正确指向展开后的节点。

### baseline 执行

启动：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run baseline \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/baseline_driver.log 2>&1
```

后台 task `bash-x4v9z4rn`。该任务用于验证转换器和 profiling 的端到端正确性，完成后才能继续后续分组。

## 23. 基线加载耗时诊断与其余模板预转换

T2V profiling 启动后，日志长时间停在文本编码器/diffusion model 动态加载，耗时明显高于首次手工 API 运行。没有立即认定为 profiler 开销或模型计算变慢，进行了以下检查：

- GET `/h3/memory` 自身仅约 3 ms，可以正常返回。
- 文本编码器 model_size=15,686,911,147 bytes，采样时 loaded_size 从约 7.52 GB 增至 14.50 GB。
- diffusion model model_size=20,967,535,680 bytes，随后 loaded_size 约 7.65 GB。
- GPU 利用率为 0%、NVML 约 23.9 GB，说明当时主要在加载权重而不是采样计算。
- `/proc/1272402/io` 显示 rchar/read_bytes 已分别约 21.23/20.65 GB，磁盘读取确实发生并增长。

当前证据指向冷进程从 NFS 动态加载权重造成的 I/O 等待，而不是 CUDA 计算变慢；采样接口响应很快。首次运行时权重有既有 page cache/缓存状态，不能直接拿来与这次主动冷启动比较。保留这次运行以获取真实冷启动数据，不kill重跑。

并行完成的非生成工作：预先转换其余 11 个模板，全部成功：

```text
I2V: 24 API nodes
I2V continuation: 24
MiniMax T2I: 17
Pika T2V: 22
Pika FL2VA: 24
Pika R2V: 21
MiniMax R2V: 26
Multiframe: 37
ControlNet: 36
P-H3 T2I: 19
P-H3 Image Edit: 21
```

这些转换只下载 JSON、查询 object_info，没有提交新 prompt，也没有启动下一组模型下载，因此不会与 baseline 争抢 GPU。后续组必须等 baseline 验收后再开始。

## 24. T2V 显存基线结果与 I2V 启动

后台 task `bash-x4v9z4rn` 已完成，T2V 官方展开图执行成功：

```text
prompt_id: d16467d5-c6a1-47c1-97d7-a95a616aa1dc
output: ComfyUI/output/video/MiniMax_H3_00002_.mp4
视频: 864x480, 24fps, 124帧
音频: AAC, 32000Hz, 双声道
媒体验证: PASS
客户端总耗时: 953.3s（包含冷加载）
NVML 峰值: 38570 MiB
```

已用 ReadMediaFile 重新读取该 MP4。

### 各模型模块（峰值采样）

```text
MiniMaxH3TEModel_:
  model_size 14.61 GiB
  loaded_size 最大 13.50 GiB
  offloaded 最小 1.11 GiB

MiniMaxH3 diffusion:
  model_size 19.53 GiB
  loaded_size 最大 19.53 GiB

MiniMaxH3AudioVAE:
  model_size 0.56 GiB
  loaded_size 最大 0.41 GiB
  offloaded 最小 0.16 GiB

MiniMaxH3VideoVAE:
  model_size 2.62 GiB
  loaded_size 最大 2.38 GiB
  offloaded 最小 0.24 GiB
```

H3 diffusion 的 loaded_size 比 model_size 大约 5 MB，导致 offloaded 显示为极小负数；这是 model_size 估算与动态加载计数的小差异，报告保留原始值，不把它当成额外可用显存。

### 节点阶段

WebSocket executing 事件和 0.5 秒采样显示：

- `140:131 MiniMaxH3ImageToVideo`（文本编码/条件阶段）约 3.3–355.2 秒，NVML 阶段峰值 15476 MiB。
- `140:125 SamplerCustomAdvanced` 约 357.2–912.6 秒，阶段峰值 36622 MiB。此段包含 diffusion 冷加载与采样，不能把整段都视为纯计算。
- `140:121 VAEDecodeAudio` 约 913.0–922.6 秒。
- `140:122 VAEDecode` 视频解码约 923.2–949.6 秒，**全图 NVML 峰值 38570 MiB 出现在这里**。
- `140:130 CreateVideo` 和 `92 SaveVideo` 随后完成。

因此，T2V 峰值模块是视频 VAE 解码阶段，不是文本编码或采样阶段。

PyTorch `max_allocated=0.63 GiB`、`max_reserved=0.84 GiB`，远小于 NVML。原因是动态权重主要由 ComfyUI/aimdo 管理，不能只用 torch allocator 判断总显存；最终报告会并列展示，不混加。

报告文件：

- `logs/model_card_runs/vram_report.md`
- `logs/model_card_runs/vram_summary.json`
- `logs/model_card_runs/video_minimax_h3_t2v/memory_samples.json`
- `logs/model_card_runs/video_minimax_h3_t2v/events.json`

报告生成器已补充节点 ID 到 class_type 的映射，便于区分 VAEDecode 与 VAEDecodeAudio。

### I2V 分组启动

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run i2v \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/i2v_driver.log 2>&1
```

后台 task `bash-wp51bpfn`。该组先下载 MiniMax T2I 指定的 fp16 video VAE 和输入图片，再依次运行 I2V、continuation、MiniMax T2I；每个模板前仍执行独立清理和峰值重置。

## 25. I2V 分组验收与 Pika 启动

后台 task `bash-wp51bpfn` 已完成，I2V 组 3/3 成功：

| Workflow | 输出 | NVML 峰值 | 总用时 |
|---|---|---:|---:|
| video_minimax_h3_i2v | `video/MiniMax_H3_00003_.mp4` | 39856 MiB | 371.8s |
| video_minimax_h3_i2v_continuation | `video/MiniMax_H3_00004_.mp4` | 39792 MiB | 552.2s |
| image_minimax_h3_t2i | `image_minimax_h3_t2i_00001_.png` | 40004 MiB | 317.2s |

媒体检查：

- 两个视频均为 640×640、24fps、124帧、5.166688秒、32kHz 双声道 AAC；全帧解码和非静音检查通过。
- T2I PNG 为 1296×736 RGB，解码及非空检查通过。
- 两个 MP4 和 PNG 均已用 ReadMediaFile 重新读取。

显存分析：

- I2V/continuation 的权重 loaded 峰值均为 37.19 GiB，峰值时其他占用分别约 1769/1705 MiB。
- 两个视频的峰值都出现在 `VAEDecode`：39856/39792 MiB。
- MiniMax T2I 的权重 loaded 峰值为 37.66 GiB，峰值时其他占用约 1438 MiB；峰值同样出现在 `VAEDecode`，为 40004 MiB。
- T2I fp16 video VAE loaded_size 最大约 4.63 GiB；其文本编码器与 FL2VA 复用，无音频 VAE 常驻需求。

汇总器曾把同一 VAE 类加载早期的较小估算 model_size 保留下来，可能低估 fp16 VAE。已改为取采样期间 model_size 最大值，并重新生成报告；loaded_size 和 NVML 原始数据不受影响。

I2V 新增 fp16 VAE 已通过 SHA256，官方鼠标输入图已下载。随后启动 Pika 组 task `bash-v5b4ab0e`：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run pika \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/pika_driver.log 2>&1
```

该组需要 Ref2VA diffusion、Pika FL2VA/Ref2VA LoRA 和 4 张输入图。素材已下载，Ref2VA 大文件正在下载，后续两个 LoRA 会依次校验并复用已有文件。

## 26. Pika 分组验收与 Ref2VA 启动

后台 task `bash-v5b4ab0e` 已完成，Pika 组 3/3 成功，所有新模型均通过 SHA256：

| Workflow | 输出 | NVML 峰值 | 总用时 |
|---|---|---:|---:|
| Pika T2V | `video/Pika_H3_Flex_T2V_00001_.mp4` | 39622 MiB | 561.6s |
| Pika FL2VA | `video/Pika_H3_Flex_FL2VA_00001_.mp4` | 41848 MiB | 627.2s |
| Pika R2V | `video/Pika_H3_Flex_REF2V_00001_.mp4` | 41888 MiB | 389.0s |

媒体检查：

- T2V：1344×768，124帧，24fps，双声道。
- FL2VA：768×1344，124帧，24fps，双声道。
- R2V：1344×768，124帧，24fps，双声道。
- 三者均全帧解码、非空和非静音检查通过，并已用 ReadMediaFile 重新读取。

显存与 LoRA：

- 三个 H3 patcher 都显示 `patch_count=208`，确认 Pika LoRA 已应用。
- Pika T2V 的峰值首次移动到 `SamplerCustomAdvanced`：39622 MiB；视频 VAE 解码为 38768 MiB。
- Pika FL2VA 的采样峰值为 41848 MiB，视频 VAE 解码为 40082 MiB。
- Pika R2V 的采样峰值为 41888 MiB，视频 VAE 解码为 40146 MiB。
- 峰值时除当前 loaded 权重外的占用约为 5.8–6.4 GiB，高于无 LoRA 的 T2V。这里包含 LoRA/patch、采样激活和运行时开销，不能全部命名为 LoRA 权重。
- ComfyUI `loaded_size` 对基础 H3 仍显示约 19.53 GiB，并未完整计入 patch，因此报告同时保留 LoRA 文件大小（每个约 1.82 GiB）、patch_count 和 NVML 总峰值，不虚构精确的 LoRA 常驻值。

Ref2VA/Multiframe 启动 task `bash-yj71uair`：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run ref2va \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/ref2va_driver.log 2>&1
```

6 张官方参考图已下载，Ref2VA 基础模型复用并已校验，当前下载 Ref2V 4-step Turbo LoRA。后续依次运行 MiniMax R2V 和 Multiframe。

## 27. Ref2VA/Multiframe 验收与 ControlNet 启动

后台 task `bash-yj71uair` 已完成，2/2 成功：

| Workflow | 输出 | NVML 峰值 | 总用时 |
|---|---|---:|---:|
| MiniMax R2V | `video/MiniMax_H3_00005_.mp4` | 40018 MiB | 594.9s |
| Multiframe Reference | `video/MiniMax_H3_00006_.mp4` | 39986 MiB | 595.1s |

两个输出均为 864×480、124帧、24fps、5.166688秒、32kHz 双声道 AAC；全帧解码、非空和非静音检查通过，并已用 ReadMediaFile 重新读取。

显存：

- 两者权重 loaded 峰值均为 37.19 GiB。
- R2V 峰值在 `VAEDecode`，40018 MiB；采样阶段为 38542 MiB。
- Multiframe 峰值在 `VAEDecode`，39986 MiB；采样阶段为 38806 MiB。
- 两者 `patch_count=0`，WebSocket progress 均确认实际为 **20 步**。Ref2V 4-step Turbo LoRA 虽按模板依赖下载并校验，但默认执行没有启用 Turbo，不能把这次显存或效果归因于该 LoRA。

ControlNet 启动 task `bash-imo5ijiy`：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run controlnet \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/controlnet_driver.log 2>&1
```

`dancer_field_pose.mp4` 已下载，Ref2VA 与 Turbo LoRA 复用且已校验，正在下载 Fun ControlNet Union patch，后续还需 SDPose 和 RT-DETR。

## 28. ControlNet 验收、输出收集 bug 与 P-H3 启动

后台 task `bash-imo5ijiy` 返回 failed，但服务器 history 实际为 success。差异发生在报告收集器而非模型执行。

### 实际执行结果

```text
prompt_id: eab627c0-9b9d-43ad-8123-a1fe733411ba
server status: success / completed=true
output: video/MiniMax_H3_00007_.mp4
Prompt executed: 559.98s
```

history 中有 1 个 `type=input` 的控制视频、124 个 `type=temp` 的预览帧和 1 个 `type=output` 的 MP4。原 `output_files` 未按 type 过滤，导致报告器去寻找 `ComfyUI/output/dancer_field_pose.mp4` 并报错。已改为只收集 `type=output`，然后直接复用原 history 和 memory samples 修复 result，不重复执行生成。

最终 MP4 为 864×480、124帧、24fps、5.166688秒、32kHz 双声道；全帧解码、非空、非静音和 ReadMediaFile 均通过。

### 显存与模块

```text
NVML 总峰值: 43632 MiB（SamplerCustomAdvanced）
VAEDecode: 43030 MiB
权重 loaded 峰值: 40.10 GiB
峰值时其他占用: 约 5160 MiB
```

采样到的主要模块：

- MiniMaxH3：loaded 最大 19.53 GiB
- MiniMaxH3TEModel_：loaded 最大 13.50 GiB
- MiniMaxH3FunControl：model_size 2.13 GiB，loaded 最大 2.16 GiB
- MiniMaxH3VideoVAE：loaded 最大 2.63 GiB
- MiniMaxH3AudioVAE：loaded 最大 0.41 GiB
- Lotus（SDPose）：loaded 最大 1.66 GiB
- RT_DETR_v4：loaded 最大约 0.13 GiB
- AutoencoderKL：loaded 最大约 0.09 GiB

Pose 预处理还产生 124 张 temp 关键帧图，已排除在最终输出验收之外，但相关模型计入显存分析。

### P-H3 启动

启动 task `bash-gkr5l8zj`：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python \
  /home/kaokaolv/comfyui_h3/h3_batch_runner.py run p-h3 \
  > /home/kaokaolv/comfyui_h3/logs/model_card_runs/p-h3_driver.log 2>&1
```

该组下载 P-H3 text encoder、prompt upsampler、VAE、LoRA 和输入图，然后依次运行 P-H3 T2I 与 Image Edit。

## 29. P-H3 验收结果

后台 task `bash-gkr5l8zj` 已完成，2/2 成功，4 个 P-H3 专用模型均通过 SHA256：

| Workflow | 输出 | NVML 峰值 | 总用时 |
|---|---|---:|---:|
| P-H3 T2I | `image_p_h3_t2i_00001_.png` | 34316 MiB | 220.0s |
| P-H3 Image Edit | `image_p_h3_image_edit_00001_.png` | 35310 MiB | 333.9s |

媒体检查：

- T2I：2368×1760 RGB PNG，7,146,100 bytes，解码和非空检查通过。
- Image Edit：1024×1024 RGB PNG，1,659,134 bytes，解码和非空检查通过。
- 两张图片均已用 ReadMediaFile 重新读取。
- 两者实际采样均为 8 步，patch_count=208，P-H3 LoRA 生效。

显存：

- P-H3 T2I 权重 loaded 峰值 32.22 GiB，峰值时其他占用约 1321 MiB。
- Image Edit 权重 loaded 峰值 33.22 GiB，峰值时其他占用约 1291 MiB。
- 两者峰值均在 `VAEDecode`，分别为 34316/35310 MiB。
- 专用 MiniMaxH3TEModel_ 的 model_size 约 4.77 GiB；Qwen35TEModel_ prompt upsampler 约 6.26 GiB；PH3ImageVAE 约 2.93 GiB。相比标准 H3 的 14.61 GiB 文本编码器，P-H3 总显存明显更低。
- H3 基础模型 loaded 仍约 19.53 GiB，P-H3 LoRA patch_count=208；loaded_size 不包含完整 patch/激活，因此不能把基础模型计数当成 LoRA 后总占用。

## 30. 全部 12 个 workflow 汇总

全部模板已实际提交执行，服务器 history 成功，媒体验证通过：

| Workflow | 实际步数 | NVML 峰值(MiB) | 峰值节点 |
|---|---:|---:|---|
| MiniMax T2V | 20 | 38570 | VAEDecode |
| MiniMax I2V | 20 | 39856 | VAEDecode |
| I2V continuation | 20 | 39792 | VAEDecode |
| MiniMax T2I | 30 | 40004 | VAEDecode |
| Pika T2V | 6 | 39622 | SamplerCustomAdvanced |
| Pika FL2VA | 6 | 41848 | SamplerCustomAdvanced |
| Pika R2V | 6 | 41888 | SamplerCustomAdvanced |
| MiniMax R2V | 20 | 40018 | VAEDecode |
| Multiframe | 20 | 39986 | VAEDecode |
| Fun ControlNet | 20 | 43632 | SamplerCustomAdvanced |
| P-H3 T2I | 8 | 34316 | VAEDecode |
| P-H3 Image Edit | 8 | 35310 | VAEDecode |

汇总器最初把所有 progress 的 max 当成采样步数，误把 ControlNet 的 124 个 Pose 帧和 P-H3 的 1536 token 上限当成步数。已改为只统计节点类型为 SamplerCustomAdvanced/KSampler 的 progress，得到上表的真实采样步数。

### 主要结论

1. 单张 B200 足够运行全部模板；最大实测为 ControlNet 的 43632 MiB，仍远低于单卡容量。
2. 标准 H3 T2V/I2V/R2V/Multiframe 和图像模板的峰值多在视频 VAE 解码；Pika 与 ControlNet 的峰值在采样阶段。
3. T2V 基线权重 loaded 峰值 35.82 GiB，NVML 峰值时其他占用约 1891 MiB；Pika/ControlNet 因 patch 和采样激活，运行时开销更高。
4. P-H3 使用较小的专用文本编码器，峰值最低。
5. Torch allocator 不能完整反映 ComfyUI/aimdo 动态权重，必须结合 ComfyUI loaded_size 和 NVML，不能只看 `torch.cuda.max_memory_allocated`。
6. 各模板总耗时包含 NFS 冷加载，不能作为纯推理 benchmark。采样节点起止时间也包含权重加载，报告已明确这一限制。
7. LoRA/patch 的精确常驻占用不能仅由 loaded_size 或 0.5 秒采样分离；报告保留 patch_count、文件大小和 NVML 总峰值，不把差额全部称为 LoRA 权重。

### 交付文件

- `logs/model_card_runs/vram_report.md`：人类可读的逐 workflow、逐模型、逐节点报告。
- `logs/model_card_runs/vram_summary.json`：结构化汇总。
- 每个模板目录下的 `api_prompt.json`、`history.json`、`result.json`、`events.json`、`memory_samples.json`。
- `logs/model_card_runs/download_manifest.json`：模型 revision、SHA256 和输入素材校验。
- `h3_batch_runner.py`：分组下载、转换、执行、媒体检查和显存采样。
- `h3_memory_report.py`：汇总报告生成。
- `run_book.md`：全过程问题、证据与修复记录。

## 31. 最终健康检查与完成审计

最终检查：

```text
12/12 result status=success
12/12 媒体文件存在且 validation=PASS
新增模型 manifest=12，输入素材 manifest=13
12 个官方 workflow 均已发布到用户 workflows 目录
queue_running=[]，queue_pending=[]
服务仍监听 0.0.0.0:8188
GPU 2 清理后显存约 1086 MiB，utilization=0%
```

验证脚本最初把 `model_card_workflows/source.json` 也当成需发布的 workflow，产生 `AssertionError: {'source'}`。已排除该元数据文件后重新验证通过；这不影响任何生成结果或输出文件。

最终报告位于：

- `logs/model_card_runs/vram_report.md`（总览、输出验证、各模型、各节点）
- `logs/model_card_runs/vram_summary.json`
- `logs/model_card_runs/download_manifest.json`

服务保持运行，未停止；profiling 接口和全部日志保留供后续复核。

---

# T2V 四个基础模型溯源

## 32. Comfy-Org/MiniMax-H3 从哪里来

用户追问前四个基础模型的上游来源与生成过程。读取了：

- `https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/README.md`
- `https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/README.md`
- `https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4`
- 三个 HF API 的文件/revision/LFS 元数据
- 本地 4 个 safetensors 的 metadata、tensor key、shape、dtype

### 总结论

`Comfy-Org/MiniMax-H3` 不是重新训练的模型，而是为 ComfyUI 做的**单文件化、推理专用裁剪和量化重打包**。原始 H3 权重来自 `MiniMaxAI/MiniMax-H3`；本workflow选择的 NVFP4 文本编码器另有 Qwen 量化上游。

### 1. FL2VA 主生成模型

上游：

```text
MiniMaxAI/MiniMax-H3
FL2VA/transformer/model-00001-of-00013.safetensors ... model-00013-of-00013.safetensors
BF16，合计约 66.28 GB
```

MiniMax 官方说明：H3-Omni-Transformer 是 33B dense Transformer，约 13B 参数在 AdaLN 相关分支；这些调制输出可以预计算和缓存，因此推理专用部署不需要加载这部分参数，完整权重则保留用于 fine-tuning。

Comfy-Org 的形态变化：

```text
完整 BF16：约 66.28 GB
  → 推理专用 pruned BF16：40,225,724,176 bytes（约 40.23 GB）
  → pruned int8_convrot：20,970,379,616 bytes（约 20.97 GB）
```

约 26 GB 的 BF16 裁剪量与官方约 13B AdaLN 参数的解释一致。本地文件还包含 `comfy_quant`、`weight_scale` 等 Comfy 量化标记，以及 `adaln_t_table`，说明它不是原始 HF 分片的简单改名，而是适配 ComfyUI 的量化/打包产物。

### 2. Qwen3-VL-32B 文本编码器

MiniMax 原始 H3-Encoder 使用完整预训练 Qwen3-VL-32B，并把第 50 层 hidden states 交给 H3-Omni-Transformer。

本workflow使用的 NVFP4 变体，Comfy-Org model card 明确写为转换自：

`cybermotaz/Qwen3-VL-32B-Instruct-NVFP4`

该上游 model card 记载的链路为：

```text
Qwen/Qwen3-VL-32B-Instruct
  → NVIDIA Model Optimizer，NVFP4 AWQ_FULL
  → 512 个 WikiText-2 calibration samples
  → 视觉编码器、visual merger、embeddings 保留 BF16
  → cybermotaz NVFP4
  → Comfy-Org 为 H3 做结构适配/层裁剪/单文件重打包
```

本地文件实测只有 `model.layers.0` 到 `model.layers.49`，共 50 层，49 层之后没有 tensor，正好对应 MiniMax 使用第 50 层 hidden states。其 15.69 GB 大小不能只用 NVFP4 解释，还包含去除后续无用层的裁剪。

### 3. 视频 VAE

上游原始文件：

`MiniMaxAI/MiniMax-H3/FL2VA/video_vae/source/model.safetensors`，10,415,548,320 bytes。

本workflow使用 Comfy-Org 的 int8_convrot 重打包文件，2,811,065,184 bytes。其 metadata 保留了 `source_config`、`AutoencoderKLLegacy`、24 个 latent channels、latents_mean/std、16× 空间压缩和 4× 时间压缩配置，能对应 MiniMax 的 H3-VisualVAE，而不是其他 VAE。

### 4. 音频 VAE

上游原始文件：

`MiniMaxAI/MiniMax-H3/FL2VA/audio_vae/model.safetensors`，605,429,308 bytes。

Comfy-Org 文件为 605,254,808 bytes、FP32，包含 32kHz、双声道输出、32 个 latent channels 及 latents_mean/std 等元数据。它属于 FP32 重打包/结构适配，没有主动做 int8 量化；文件大小和哈希不要求与原始文件逐字节一致。

### 证据边界

可以确认：上游仓库、各组件的转换关系、pruned 的架构依据、Qwen 量化链、文本编码器保留前 50 层，以及本地文件的量化/结构元数据。

不能直接确认：Comfy-Org 使用的具体上游 commit、int8_convrot 的完整转换脚本/校准配置，以及原始 tensor 到重打包 tensor 的逐项映射。model card 没有提供这些信息，因此不能把上述来源链进一步夸大为“可逐步复现转换过程”。MiniMax 原始 model card 也未说明 Comfy-Org 的具体裁剪清单。

尝试查询 MiniMax-AI/MiniMax-H3 GitHub tree 中的转换脚本时遇到未认证 GitHub API 403 rate limit；没有凭猜测补写转换脚本，以上结论均来自可读取的官方 model card、HF API 和本地文件证据。

---

# vLLM-Omni 运行 H3 T2V 与量化路线调研

## 33. 任务与结论

用户询问能否用 vLLM-Omni 跑 `video_minimax_h3_t2v.json`，以及把当前 int8 对应模型从 BF16 转为 MXFP8、文本编码器继续使用 NVFP4 是否可行。要求详细记录命令、分析、问题与解决。

本轮只做源码、model card、配置和本机环境调研，没有安装新环境、转换模型、启动 vLLM 服务或生成视频。完整报告见 [vllm_omni_h3_research.md](./vllm_omni_h3_research.md)。

结论：

- 可以用 vLLM-Omni 做 H3 T2V，但要使用官方远程 WF-01 模板，不是直接执行原始 ComfyUI JSON。
- 最稳妥路线是原始 BF16 H3 + 启动时在线 FP8。
- BF16 → 在线 MXFP8 在源码和 B200 上可实验，但 H3 视频没有官方验证。
- 当前 Comfy-Org int8/nvfp4 文件不能直接用于 vLLM-Omni；NVFP4 应使用 FastVideo QAD 的兼容目录。
- QAD 是 8-step 蒸馏模型，混合其文本编码器与原始 BF16 DiT 需要单独验证，不能宣称等价。

## 34. 官方集成与 workflow 证据

读取：

```text
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/apps/ComfyUI-vLLM-Omni/README.md
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/apps/ComfyUI-vLLM-Omni/docs/h3-03-music-video.md
https://docs.vllm.ai/projects/vllm-omni/en/latest/features/comfyui/
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/README.md
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/recipes/MiniMaxAI/MiniMax-H3.md
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md
https://raw.githubusercontent.com/vllm-project/vllm-omni/main/apps/ComfyUI-vLLM-Omni/example_workflows/MiniMax_H3_Text_to_Video.json
```

确认：ComfyUI-vLLM-Omni 是在线 serving API 集成，模型加载、采样、VAE 解码在服务端；ComfyUI 不加载 H3 权重。官方 WF-01 正是基于原 ComfyUI T2V 模板的远程版本，默认 1344×768、24fps、5.167 秒、50 个采样点。

原 ComfyUI 默认 864×480、20 步，与 WF-01 的 50 points 不应直接等同。H3-03 是 Ref2VA + 外部音轨，不是本 T2V。

## 35. 量化支持与源码核查

### FP8 与 MXFP8

H3 recipe 明确：`--quantization fp8` 支持 DiT 和 Qwen3-VL 文本编码器的在线 FP8，checkpoint 在磁盘上仍是 BF16，运行时生成 FP8 并做动态激活缩放。单 96 GB 案例整卡首请求峰值约 92946 MiB，可作 B200 容量参考但不是本机实测。

`docs/user_guide/quantization/mxfp8.md` 明确：BF16 可在线转 MXFP8，不需要预处理；CUDA 不支持 native serialized MXFP8。支持硬件包括 SM100/103/107，但验证模型是 MiniMax-H3-Image，不是 H3 视频。

为核对实际实现，执行：

```bash
mkdir -p /home/kaokaolv/comfyui_h3/research_repos
git clone --depth 1 https://github.com/vllm-project/vllm-omni.git \
  /home/kaokaolv/comfyui_h3/research_repos/vllm-omni
git -C /home/kaokaolv/comfyui_h3/research_repos/vllm-omni rev-parse HEAD
git -C /home/kaokaolv/comfyui_h3/research_repos/vllm-omni log -1 --format='%cI %s'
```

结果：

```text
commit d64649c41a3a2dce6da3cd97ae48b754def4eb25
2026-11-16T11:01:43-05:00 [Bugfix][Wan2.2] ... (#10404)
```

已读取该仓库 AGENTS.md。源码确认：

- `quantization/factory.py` 注册 mxfp8。
- `DiffusionMXFP8Config` 在 CUDA 上调用 `CUDAMxfp8OnlineLinearMethod`。
- `mxfp8_cuda.py` 接受 SM100/103/107。
- `pipeline_minimax_h3.py` 将 config 传入原始 H3 DiT。

因此 BF16 → 在线 MXFP8 有源码和 B200 支持，但仍是 H3 视频未验证的实验路线。

### 当前 NVFP4 文件为何不能直接复用

本地 Comfy-Org 文件使用 `comfy_quant`、`weight_scale` 等字段。

vLLM-Omni 的 `FastH3NVFP4Config.from_encoder` 要求：

- config: `quant_method=nvfp4`、dynamic、e2m1、group_size=16、e4m3、128x4
- shard index
- `.weight_packed`、`.weight_scale`、`.weight_global_scale`

Comfy-Org 不满足该目录和字段契约。

进一步检查 cybermotaz 的 config/index：其 config 为 `quant_method=modelopt`、`quant_algo=NVFP4_AWQ`，也不是 H3 专用 loader 要求的 `quant_method=nvfp4` 和 `.weight_packed` 形式，不能直接替代。

### 找到兼容的 QAD 产物

检查 `FastVideo/FastH3-Omni-NVFP4-QAD` 的 model card、HF API、text_encoder/config.json 和 index：

```text
quant_method: nvfp4
activation_scheme: dynamic
fmt: e2m1
group_size: 16
scale_fmt: e4m3
scale_layout: 128x4
num_hidden_layers_override: 50
producer.converter: convert_minimax_h3_text_encoder_nvfp4.py
kept_language_layers: 50
index 中存在 weight_packed / weight_global_scale
transformer/nvfp4_weights.safetensors 存在
```

这与 loader 完全对应，但整包是 8-step QAD 蒸馏模型，不是原始 H3 Base。混合其 text_encoder 与原始 BF16 DiT 在源码结构上可行，必须单独验证；混合时不能让 QAD 的 `nvfp4_weights.safetensors` 进入 transformer，以免与显式 MXFP8 冲突。

## 36. 本机环境探测与发现的问题

### 版本与模块

检查三个旧环境：

```text
.venv-vllm-omni:   vllm 0.28.0 / vllm-omni 0.28.0 / torch 2.13.0+cu132
.venv-vllmomni029: vllm 0.29.0 / vllm-omni 0.28.1.dev163...
.venv-omini-latest: vllm 0.29.0 / vllm-omni 0.29.0rc2...
```

三者均有 H3 pipeline 和 mxfp8 config，但没有新 `diffusion/models/minimax_h3/checkpoint_quantization.py`。其中 `.venv-vllmomni029` 还明确警告 vLLM 与 vLLM-Omni 版本不匹配。

FlashInfer 探测均显示 `mm_fp4=True`、`nvfp4_quantize=True`，但 `mxfp8_cuda.py` 均不存在。

### MXFP8 实例化测试

执行：

```bash
CUDA_VISIBLE_DEVICES=2 /home/kaokaolv/.venv-vllm-omni/bin/python - <<'PY'
from vllm_omni.quantization import build_quant_config
config=build_quant_config('mxfp8')
print(type(config).__name__,config.get_name(),config.is_checkpoint_mxfp8_serialized)
from vllm_omni.quantization.mxfp8_cuda import CUDAMxfp8OnlineLinearMethod
method=CUDAMxfp8OnlineLinearMethod()
print(type(method).__name__)
PY
```

结果：config 能创建，method=mxfp8、serialized=False；随后报：

```text
ModuleNotFoundError: No module named 'vllm_omni.quantization.mxfp8_cuda'
```

解决决定：不能仅凭 config 判断可运行；新环境必须包含匹配的 CUDA kernel。不能沿用旧 venv 强行做实验，也不能只复制一个 Python 文件掩盖版本差异。使用调研时固定的新 checkout 建独立环境。

## 37. 建议实施顺序与未验证边界

1. 先用原始 BF16 FL2VA + 在线 FP8 + 官方 WF-01 验证真实视频和音轨。
2. 若需要 MXFP8，保持原始 BF16 DiT，在相同服务上改为在线 MXFP8 并做兼容性/质量对比。
3. 若必须保留 NVFP4 文本编码器，再构建“原始 BF16 DiT + QAD text_encoder + 原始 VAE/tokenizer/processor”的混合目录，验证服务端确实分别走 MXFP8 与 NVFP4。
4. 完整 QAD 只作为独立 8-step 模型变体验证，不与原始 Base 结果混为一谈。

原始 BF16 下载需要约 134 GiB 和 HF 授权，因此本轮没有进行真实 vLLM 推理。报告中的启动命令是依据官方 recipe 给出的实施建议，不是本轮已执行成功的命令。详细命令、目录结构和验收项均在 `vllm_omni_h3_research.md` 中。


---

# T2V 函数与 Kernel 调用栈分析

## 38. 任务、范围与本轮约束

用户要求回到原生 ComfyUI 已成功执行的 `video_minimax_h3_t2v.json`，严谨分析模型加载、推理函数调用栈、kernel/op、实现逻辑与原理，写成详尽文档，并完整记录分析、命令、问题和解决办法。

本轮是只读源码与既有运行证据分析，不安装环境、不修改模型、不重新执行大规模推理。固定基线为 ComfyUI v0.45.0、commit `989d93d0d968e704c589ce0146bb18c77f6b50cc`，实际运行 prompt `d16467d5-c6a1-47c1-97d7-a95a616aa1dc`，输出 `ComfyUI/output/video/MiniMax_H3_00002_.mp4`。默认 20 步，无 LoRA，无图像条件，864x480、124 帧、24 fps。

证据分三级：既有日志/媒体是实际运行证据；当前 checkout 源码给出静态调用路径；Comfy Kitchen 注册或启动日志的 backend 支持信息只能说明可用，不能代替 CUPTI 或 torch profiler 的逐 kernel 实际执行证据。

## 39. 补读音频 VAE、解码入口、量化与保存路径（第一批）

使用 Read 读取 `ComfyUI/AGENTS.md`、`ComfyUI/comfy/ldm/minimax/audio_vae.py`、`ComfyUI/comfy_extras/nodes_audio.py` 全文，及 `ComfyUI/comfy/model_sampling.py:320-474`。使用 Grep 对以下源码做定位：

```text
ComfyUI/comfy/sd.py
  pattern: ^    def (encode|decode|__init__)|MiniMaxH3|def load_diffusion_model|^def load_clip
ComfyUI/comfy/ops.py
  pattern: _try_quantized_linear_input_act|CastBiasWeightContext|_load_quantized_module|def linear_input_act
.venv/lib/python3.12/site-packages/comfy_kitchen
  pattern: dequantize_int8_convrot_weight|int8_attention|scaled_mm_nvfp4|fp16_conv3d|convrot_w4a4
ComfyUI/comfy_extras/nodes_video.py
  pattern: VideoFromComponents|save_to|class (SaveVideo|CreateVideo)|def save|def execute
```

这些均为工具只读操作，未在 shell 中另跑 Python 或启动服务。

已确认：

1. `VAEDecodeAudio.execute` → `vae_decode_audio`，nested latent 通过 `unbind()[-1]` 取 audio，然后调用 `vae.decode(latent)`，再做峰值不额外放大的 `std*5` 归一化（低于 1 则限制为 1），并转为 ComfyUI audio waveform 布局。
2. `MiniMaxH3AudioVAE.decode` 将 `[B,32,2,T]` 的两声道展平为独立 mono 解码，反归一化后经 `dec_in_proj` 和 BigVGAN，最终输出 `[B,2,L]`；每 latent frame 为 800 samples、40 Hz、32 kHz。主要 op 为 Conv1d、ConvTranspose1d、SnakeBeta、抗混叠上/下采样和逐点算子。
3. `ModelSamplingAV.audio_scale = shift/audio_shift`，默认 H3 为 12/3=4。它是将 audio latent 放到 video schedule 的尺度，不是额外执行 4 次推理。
4. `comfy/sd.py` 的 video VAE 配置设置 `handles_tiling=True`、chunked I/O 和内存估计；audio VAE 限定 FP32。需要继续读 `VAE.decode` 的分支，避免把内部 tiling 和通用 tiled fallback 混淆。
5. `SaveVideo.execute` 调 `video.save_to(...)`；`CreateVideo.execute` 构造 `VideoFromComponents`。还需从 `_io_impl.py` 读实际编码器，而不是只依据文件扩展名推断编码。
6. `ops.CastBiasWeightContext` 负责权重/bias 的 cast、offload 和清理。量化 tensor 走 `__torch_dispatch__`，后端选择包含回退；因此日志“backend registered”不能写成“每个 op 均已实际执行”。


## 40. DiT、VAE、采样与 checkpoint 量化核查（第二批）

### 源码和运行证据读取

Read 了以下实现（括号为关键行段，不是修改操作）：

```text
ComfyUI/comfy/ldm/minimax/model.py:1-823
ComfyUI/comfy/ldm/minimax/vae.py:1-428,565-1013
ComfyUI/comfy/ldm/minimax/audio_vae.py:1-443
ComfyUI/comfy/model_base.py:173-263,2226-2307
ComfyUI/comfy/model_detection.py:401-435
ComfyUI/comfy/supported_models.py:986-1012
ComfyUI/comfy/model_sampling.py:86-100,289-357
ComfyUI/comfy/samplers.py:208-356,592-652,937-1007,1188-1398
ComfyUI/comfy/k_diffusion/sampling.py:1461-1528
ComfyUI/comfy/sampler_helpers.py:165-204
ComfyUI/comfy_extras/nodes_custom_sampler.py:716-819,1003-1072
ComfyUI/comfy_extras/nodes_minimax_h3.py:50-179
ComfyUI/comfy/text_encoders/minimax.py:1-259
ComfyUI/comfy/text_encoders/qwen3vl.py:51-114
ComfyUI/comfy/text_encoders/llama.py:391-424,556-666,699-859,1005-1134
ComfyUI/comfy/sd.py:1050-1194,1360-1570,2460-2559
ComfyUI/comfy/sd1_clip.py:172-303
ComfyUI/comfy/ops.py:144-528,1064-1217,1342-1679,1857-1956
ComfyUI/comfy/quant_ops.py:1-284
ComfyUI/comfy/model_patcher.py:1786-2010,2282
ComfyUI/comfy/model_prefetch.py:31-346
ComfyUI/comfy/nested_tensor.py:1-97
ComfyUI/comfy/utils.py:130-238,1445-1532
ComfyUI/comfy_api/latest/_input_impl/video_types.py:1318-1416
.venv/lib/python3.12/site-packages/comfy_kitchen/registry.py
.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/int8.py
.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/nvfp4.py
.venv/lib/python3.12/site-packages/comfy_kitchen/backends/cuda/__init__.py
```

Grep 定位了 `optimized_attention`、`attention_pytorch`、`rms_rope_split_half_`、`int8_linear`、`fp16_conv3d`、`group_norm_silu_pad3d`、`VAEDecode`、`VAELoader`、`UNETLoader`、`CLIPLoader`、`CoreModelPatcher` 和 `save_to`。没有改动 ComfyUI 中的源码。

### checkpoint 元数据检查（实际 shell 命令）

使用 `.venv/bin/python` 执行 heredoc 脚本，`safe_open(..., framework='pt', device='cpu')` 只读取 header、key 列表、每个很小的 `.comfy_quant` JSON tensor；没有读取大权重 tensor 或发起 GPU 推理。脚本遍历四个模型文件，统计 `format`、`convrot`、`convrot_groupsize`、`full_precision_matrix_mult` 和 scale key 后缀。

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
from safetensors import safe_open
root = Path('/home/kaokaolv/comfyui_h3/ComfyUI/models')
# 对 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors、
# text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors、
# vae/minimax_h3_video_vae_int8_convrot.safetensors、
# vae/minimax_h3_audio_vae_fp32.safetensors：
# 读取 f.keys() 和 f.metadata()，对 key.endswith('.comfy_quant') 的小 tensor
# 使用 json.loads(f.get_tensor(key).numpy().tobytes()) 统计格式。
# 另外统计 .weight_scale/.weight_scale_2/.input_scale/.pre_quant_scale。
PY
```

实际结果：

```text
DiT: 932 tensors，200 个 comfy_quant。
  200 × {format:int8_tensorwise, convrot:true, convrot_groupsize:256,
         full_precision_matrix_mult:false}
  恰为 50 主 blocks × (qkv_proj, out_proj, fc1, fc2)。
  无 input_scale/pre_quant_scale 文件项。

Qwen: 2054 tensors，351 个 comfy_quant。
  350 × {format:nvfp4, full_precision_matrix_mult:true}
  1 × {format:int8_tensorwise}，为 model.embed_tokens。
  每层 7 个 NVFP4 Linear：q/k/v/o、gate/up/down，共 50 层。
  100 个 pre_quant_scale：每层 self_attn.o_proj 和 mlp.down_proj。
  无 input_scale 文件项。

Video VAE: 850 tensors，144 个 comfy_quant。
  144 × {format:int8_tensorwise, convrot:true, convrot_groupsize:256,
         full_precision_matrix_mult:false}
  位于 36 decoder.transformer_blocks × (to_qkv, to_out, ff.w1, ff.w2)。

Audio VAE: 917 tensors，无 comfy_quant，FP32。
```

关键结论：不能把文件名或 backend 列表当作 kernel 执行证明。Qwen 的 NVFP4 权重虽然以 FP4 形式存储，但默认 encode 路径强制全精度矩阵乘：`full_precision_matrix_mult=true`，且 `SDClipModel` 还以 `full_precision_mm=True` 构造 mixed ops。`MixedPrecisionOps.Linear.forward` 会走 dequantize + `F.linear`，不是输入/权重同时 NVFP4 的 `scaled_mm_nvfp4`。CLIP 日志 FP16 是初始化 dtype，hidden 输入/权重在通用 cast 层按实际输入对齐，输出条件显式转 FP32；没有逐 op dtype trace 时不把 Qwen GEMM dtype 武断写成 FP16 或 BF16。

另一个关键结论：Qwen `CombinedLinear` 只接受少数 INT8 layout，不支持当前 NVFP4；因此 `combined_qkv.forward` 和 `combined_gate_up` 返回 `NotImplemented`，分开执行 q/k/v 和 gate/up 的 Linear。`pre_quant_scale` 在 o_proj/down_proj 输入前做逐通道乘法。

### 既有运行记录解析（实际 shell 命令）

用 Python 标准库 `json` 读取 `events.json` 和 `memory_samples.json`，按 `executing` 事件的首次出现顺序打印 node span，并找 NVML 峰值。命令使用 `.venv/bin/python` heredoc，没有启动服务或调用推理。

```text
140:120 audio VAELoader         0.425s
140:119 video VAELoader         0.525s
140:128 CLIPLoader              0.868s
140:131 文本/空 latent 节点      3.207s → 下一节点 355.193s（约 351.986s）
140:125 sampler                 357.110s → 下一节点 912.694s（约 555.584s）
140:121 audio decode            912.694s → 下一节点 923.038s（约 10.345s）
140:122 video decode            923.038s → 下一节点 949.891s（约 26.853s）
140:130 CreateVideo             949.891s
92 SaveVideo                    949.994s → 结束 952.067s
```

NVML 峰值为 38570 MiB，在 `140:122` 视频解码中约 949.007s。节点间的 next-delta 包括前一节点执行以及调度/事件间隔，不能误写成纯 GPU kernel 时间；本记录没有 CUPTI trace。

`logs/comfyui_server_profiled.log:90` 显示 sampler 第一次初始化约 482.20s，后续单步约 1.71s，20 步主体约 34s；其余时间主要是文本、初始权重准备和首次执行开销。不能把 555s 解释为每步 27.8s 的稳定 DiT 时间。

### BasicGuider 的疑点与解决

阅读基类 `CFGG uider.predict_noise` 时，看到 `cfg_function(model, out[0], out[1], ...)`，初看只有 positive 可能出现 index/None 问题。继续读 `_calc_cond_batch` 后确认：它先为输入 conds 列表的每个位置（包括 None）创建 zero tensor 和 `1e-37` 计数；BasicGuider 的 negative 位置没有模型调用，输出为零；cfg 默认 1，因此 `0+(positive-0)*1=positive`。这是单次正向去噪，不是两次 CFG。无需修复，也不能把它当作实际运行错误。

### 采样、tile 和版本的派生计算

用独立、轻量 Python 按源码公式计算 20 个 video sigma、audio sigma 与空间 tile 边界，未实例化模型：

```text
video sigma: 1, 0.995633188, 0.990825688, ..., 0.571428571, 0.387096774, 0
 audio sigma: 1, 0.982758621, 0.964285714, ..., 0.250000000, 0.136363636
```

实际 scheduler 是 1000 个 shift=12 的网格中每 50 个取一个，末尾补 0。`simple_20_step_schedule` 脚本用 `t=1-0.05*x` 和 `sigma=12*t/(1+11*t)`，audio 用 `time_shift_sigma(...,12,3)`。

864x480 在 `tile_size=256`、`tile_overlap_min=64`、16 倍空间压缩下：

```text
H=480: starts=[0,112,224], overlaps=[144,144]
W=864: starts=[0,144,288,448,608], overlaps=[112,112,96,96]
```

即每个 temporal chunk 3×5=15 个空间 tile。宽度最后一片 608+256=864，高度最后一片 224+256=480。37 latent frames 分 7 个 temporal chunks，每个读取 5+2=7 latent，最后按 7×17+5=124 帧输出。tile 的 decoder batch 仍受实时 free VRAM 影响，不把 15 个 tile 直接等同于 15 次独立 kernel 调用。

版本核验命令：

```bash
git rev-parse HEAD
git log -1 --format='%cI %s'
git status --porcelain=v1
```

工作目录为 `/home/kaokaolv/comfyui_h3/ComfyUI`，结果 commit `989d93d0d968e704c589ce0146bb18c77f6b50cc`、v0.45.0，status 为空，没有源码修改。

### 其他必须写入报告的证据边界

1. 无首帧/尾帧：`MiniMaxH3ImageToVideo` 不调用 video VAE.encode，不运行其 3D CNN encoder；视频 decode 是 post_quant_conv + ViT3D decoder，不应把 CNN encoder 的 group_norm/pad 融合写成实际调用。
2. 无参考音频：`VAEDecodeAudio` 只运行 BigVGAN decoder，不运行 audio encoder 的 CausalAttention/AttnProjection。
3. 无图像 token：Qwen visual tower 不参与本次条件编码，不用 chat template，不生成 token，不执行 lm_head；读取第 50 层的未 final-norm hidden states。
4. attention：日志只证明选择了 `attention_pytorch` Python dispatcher，底层 SDPA 是 Flash、cuDNN 还是 mem-efficient 后端仍需 profiler/CUPTI；不能用“Pytorch attention”断言数学展开 backend。
5. DynamicVRAM 由 main.py 把 `CoreModelPatcher` 换成 `ModelPatcherDynamic`。`malloc_graph` 与 CUDA Graph 不是同一件事；H3 `_run_blocks` 没开启 `enable_graph`，不能说 50 blocks 被整体 CUDA Graph replay。
6. 报告将专门区分：已证实执行路径、静态 fast-path 候选、明确不会走的 encoder/量化分支、未采集的底层 CUDA kernel 名称和耗时。


## 41. 正式文档、修正与完整复核记录

### 交付文件

已创建 `/home/kaokaolv/comfyui_h3/comfyui_h3_t2v_kernel_analysis.md`，包含 17 个一级编号章节：分析基线、端到端栈、四个模型加载、Qwen、latent、scheduler/sampler、DiT、量化 kernel/op、video VAE、audio VAE、DynamicVRAM、视频保存、op 汇总、性能解释、未实测边界、源码索引。

本轮只新增上述文档并追加本 `run_book.md`，未修改 ComfyUI 源码、模型、服务配置或 profiler custom node，未重新生成视频。

### 文档撰写阶段补读

继续 Read 了 `ComfyUI/comfy/rmsnorm.py`、`comfy_kitchen/backends/cuda/__init__.py:2287-2392,4020-4141`、`comfy_kitchen/sage_attention.py:208-317,597-636`。Grep 确认 `comfy.ops.scaled_dot_product_attention` 最终调用 `torch.nn.functional.scaled_dot_product_attention`。

由此补齐：INT8 大 M 优先 CUTLASS、失败后 cuBLAS INT32 + dequant；`_C.rms_rope` 的 paired、分离和 eager 路径；video VAE `sage_sdpa` 的 INT8 Q/K/V、unsigned INT8 probability 和 FP32 softmax 统计。

### 交叉核验发现的问题和修正

1. 初稿写“首个 step Euler，后续都二阶”，不够准确。源码 `if sigma_down == 0 or old_denoised is None` 表明最后一个非零 sigma→0 的 step 也走 Euler。已改为“首个有效 step 和 `sigma_down==0` 的 step 用 Euler，中间其余 step 用二阶 multistep”。
2. 音频缩放改写为与源码完全一致的 `scale=std(audio)*5; scale=max(scale,1); audio/=scale`。避免将等价的简化表达与 peak normalization 混淆。
3. 检查了量化最容易误判的边界：350 个 NVFP4 权重全带 full-precision 标志；当前不是 `scaled_mm_nvfp4` 双 FP4 operand 路径。CLIP 初始化 FP16 也不能单独用来确定实际 GEMM dtype。
4. 复核了本 T2V 不执行 video CNN encoder 和 audio encoder attention，因此不能把这些模块的可选融合 op 计入本 workflow 的实际 kernel 列表。
5. 检查了普通 VAEDecode 仍有模型内部 tiling、video temporal chunks=7、空间 3×5 tiles、tile 合 batch 随 free VRAM 改变，未将逻辑 tile 数冒充 kernel launch 数。

### 实际 shell 命令的完整记录

前面的 §39-40 已列出所有源码 Read/Grep 的路径和 pattern。以下是本轮实际执行过的 shell 检查命令。Read/Grep 是工具只读操作，不是 shell 执行；这里仅列真正的 Bash 命令。

#### A. 第一次 checkpoint 量化格式统计

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
import numpy as np
from safetensors import safe_open
root = Path('/home/kaokaolv/comfyui_h3/ComfyUI/models')
paths = [
    root / 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
    root / 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    root / 'vae/minimax_h3_video_vae_int8_convrot.safetensors',
    root / 'vae/minimax_h3_audio_vae_fp32.safetensors',
]
for path in paths:
    print('\nFILE', path.name, 'bytes', path.stat().st_size)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        md = f.metadata() or {}
        print('header_metadata_keys', sorted(md.keys()))
        print('tensor_count', len(keys), 'quant_metadata_keys', sum(k.endswith('.comfy_quant') for k in keys))
        for k in ['qwen3vl_32b.transformer.model.norm.weight', 'model.norm.weight']:
            if k in keys:
                print('final_norm_present', k)
        formats = Counter()
        groups = Counter()
        for key in keys:
            if not key.endswith('.comfy_quant'):
                continue
            raw = f.get_tensor(key).numpy().tobytes()
            meta = json.loads(raw)
            fmt = meta.get('format') or meta.get('quant_format')
            formats[fmt] += 1
            parts = key.split('.')
            prefix = '.'.join(parts[:2]) if key.startswith('model.layers') else parts[0]
            if key.startswith('blocks.'):
                prefix = 'blocks'
            elif key.startswith('token_refiner.'):
                prefix = 'token_refiner'
            elif key.startswith('decoder.'):
                prefix = 'decoder'
            groups[(prefix, fmt)] += 1
        print('quant_format_counts', dict(formats))
        print('quant_by_component', {f'{a}:{b}': n for (a, b), n in sorted(groups.items(), key=lambda x: str(x[0]))})
        print('example_quant_keys', [k for k in keys if k.endswith('.comfy_quant')][:3])
PY
```

输出已完整概括在 §40：DiT 200 个 int8_tensorwise+convrot256，Qwen 350 个 nvfp4 + 1 个 int8 embedding，video VAE 144 个 int8_tensorwise+convrot256，audio VAE 无量化标记。此处 `import numpy as np` 是实际脚本的一部分，但统计只用 `.numpy().tobytes()`，没有加载大 tensor。

#### B. 第二次 flags 统计、events/memory 结构核验

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
from safetensors import safe_open
root = Path('/home/kaokaolv/comfyui_h3/ComfyUI/models')
paths = [
    root / 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
    root / 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    root / 'vae/minimax_h3_video_vae_int8_convrot.safetensors',
]
for path in paths:
    print('\nFILE', path.name)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        print('key_suffix_counts', {s: sum(k.endswith(s) for k in keys) for s in ['.comfy_quant', '.weight_scale', '.weight_scale_2', '.input_scale', '.pre_quant_scale']})
        flags = Counter()
        for key in keys:
            if key.endswith('.comfy_quant'):
                meta = json.loads(f.get_tensor(key).numpy().tobytes())
                flags[(meta.get('format'), bool(meta.get('convrot', meta.get('params', {}).get('convrot', False))), meta.get('convrot_groupsize', meta.get('params', {}).get('convrot_groupsize')), bool(meta.get('full_precision_matrix_mult', False)))] += 1
                if key.startswith('blocks.0.'):
                    print('dit_layer0', key, meta)
                if key == 'model.layers.0.mlp.down_proj.comfy_quant':
                    print('qwen_down0', meta)
        print('format_convrot_groupsize_fullprecision_counts', dict(flags))
for name in ['video_minimax_h3_t2v/events.json', 'video_minimax_h3_t2v/memory_samples.json']:
    path = Path('/home/kaokaolv/comfyui_h3/logs/model_card_runs') / name
    data = json.loads(path.read_text())
    print('\nJSON', name, 'type', type(data).__name__, 'len', len(data) if hasattr(data, '__len__') else '-')
    if isinstance(data, dict):
        print('top_keys', list(data))
    elif data:
        print('first_keys', list(data[0]) if isinstance(data[0], dict) else type(data[0]).__name__)
    if name.endswith('events.json'):
        starts = {}
        for item in data:
            event = item['event']
            if event['type'] == 'executing' and event['data'].get('node'):
                node = event['data']['node']
                if node not in starts:
                    starts[node] = item['elapsed']
                    print('first_executing', f'{item["elapsed"]:.3f}', node)
            elif event['type'] == 'executing' and event['data'].get('node') is None:
                print('end_executing', f'{item["elapsed"]:.3f}', event['data'])
    else:
        for item in data[:3]:
            print('memory_sample_example', item)
PY
```

此步得到 `qwen_down0 {'format':'nvfp4','full_precision_matrix_mult':True}`，随后对全部 350 个同类 key 计数确认全部为 True。event 最终结束时间为 952.067 秒，peak 38570 MiB。

#### C. AWQ pre-scale 位置与节点 span

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open
path = Path('/home/kaokaolv/comfyui_h3/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
with safe_open(str(path), framework='pt', device='cpu') as f:
    print('AWQ_pre_quant_scale_keys')
    print('\n'.join(k for k in f.keys() if k.endswith('.pre_quant_scale') and ('.0.' in k or '.49.' in k)))
root = Path('/home/kaokaolv/comfyui_h3/logs/model_card_runs/video_minimax_h3_t2v')
data = json.loads((root / 'events.json').read_text())
starts = {}
for item in data:
    event = item['event']
    if event['type'] == 'executing':
        node = event['data'].get('node')
        if node is not None:
            starts.setdefault(node, item['elapsed'])
        elif starts:
            print('last_node_before_end', next(reversed(starts)), 'end', item['elapsed'])
ordered = list(starts.items())
for i, (node, start) in enumerate(ordered):
    end = ordered[i + 1][1] if i + 1 < len(ordered) else data[-1]['elapsed']
    print(f'NODE_SPAN {node:10s} start={start:8.3f}s next_delta={end-start:8.3f}s')
mem = json.loads((root / 'memory_samples.json').read_text())
peak = max(mem, key=lambda x: x['nvml_used_mib'])
print('NVML_PEAK', peak['elapsed'], peak['label'], peak['node'], peak['nvml_used_mib'])
for label in ['sampler_start', 'sampler_end', 'vae_audio_start', 'vae_video_start', 'vae_video_end']:
    hits = [x for x in mem if x['label'] == label]
    for x in hits[:2]:
        print('MEM_PHASE', label, x['elapsed'], x['nvml_used_mib'])
PY
```

`pre_quant_scale` 确认在每层 `self_attn.o_proj` 和 `mlp.down_proj`。后两个 `MEM_PHASE` 标签在该文件不存在，所以没有输出；节点与 peak 信息直接由 `executing:*`/`poll` 样本得出，未虚构缺失标签。

#### D. 版本与本地改动核验

```bash
cd /home/kaokaolv/comfyui_h3/ComfyUI
git status --short && git diff -- comfy/samplers.py comfy_extras/nodes_custom_sampler.py
git rev-parse HEAD
git log -1 --format='%cI %s'
git status --porcelain=v1
```

第一条没有输出；后续显示 commit `989d93d0d968e704c589ce0146bb18c77f6b50cc`、`2026-11-10T10:59:14-05:00 ComfyUI v0.45.0`。没有本地 ComfyUI 源码 diff。

#### E. scheduler、tile 与 memory 标签的派生检查

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/home/kaokaolv/comfyui_h3/logs/model_card_runs/video_minimax_h3_t2v/memory_samples.json')
data = json.loads(p.read_text())
labels = sorted({x['label'] for x in data})
print('memory_labels', labels)
for x in data:
    if x['label'] not in ('poll', 'status', 'baseline'):
        print(x['elapsed'], x['label'], x['node'], x['nvml_used_mib'])
print('simple_20_step_schedule')
for x in range(20):
    t = (999 - 50 * x + 1) / 1000
    sigma = 12 * t / (1 + 11 * t)
    base = sigma / (12 + sigma * (1 - 12))
    sigma_audio = 3 * base / (1 + 2 * base)
    print(x, f'{sigma:.9f}', f'{sigma_audio:.9f}', 'next=0.0' if x == 19 else '')
print('tile_plan_864x480')
for input_len in (480, 864):
    tile_size, overlap_min, ratio = 256, 64, 16
    n = (input_len + tile_size - 1) // tile_size
    while tile_size * n - overlap_min * (n - 1) - input_len < 0:
        n += 1
    overlaps = [overlap_min] * (n - 1)
    remaining = tile_size * n - sum(overlaps) - input_len
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    print(input_len, 'starts', starts, 'overlaps', overlaps, 'tile_size', tile_size)
PY
```

这是按源码独立推导，不是一次新的模型执行。输出的 sigma 和 tile 坐标已经写入正式文档。

#### F. 第一次文档完整性检查

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
from pathlib import Path
import re
p = Path('/home/kaokaolv/comfyui_h3/comfyui_h3_t2v_kernel_analysis.md')
s = p.read_text()
print('doc_lines', len(s.splitlines()), 'doc_chars', len(s), 'doc_bytes', p.stat().st_size)
print('code_fences', s.count('```'), 'balanced', s.count('```') % 2 == 0)
print('h2_sections', re.findall(r'^## (.+)$', s, re.M))
print('unfinished_markers', re.findall(r'TODO|待补|待完成|FIXME', s))
paths = sorted(set(re.findall(r'`((?:ComfyUI|logs|\.venv)/[^`\n]+?)`', s)))
missing = []
for item in paths:
    source = item.split(':')[0]
    if any(c in source for c in '*{},') or ' → ' in source:
        continue
    candidate = Path('/home/kaokaolv/comfyui_h3') / source
    if not candidate.exists():
        missing.append(source)
print('checked_source_paths', len(paths), 'missing', missing)
assert s.count('```') % 2 == 0
assert not missing
PY
```

结果：1282 行（修正末步后 1283 行）、106 个 fence 标记且配对、17 个编号章节、37 个被检查的源码路径全部存在、没有 TODO/待补标记。

#### G. 修正后的最终检查

```bash
cd /home/kaokaolv/comfyui_h3/ComfyUI
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
from pathlib import Path
import re
p = Path('/home/kaokaolv/comfyui_h3/comfyui_h3_t2v_kernel_analysis.md')
s = p.read_text()
assert s.count('```') % 2 == 0
assert len(re.findall(r'^## \d+\.', s, re.M)) == 17
assert '当 `sigma_down == 0`（包括最后的非零 sigma→0 step）' in s
assert 'scale = std(audio) * 5' in s
assert 'full_precision_matrix_mult=true' in s
assert '没有 CUPTI' in s or '没有采集 CUPTI' in s
assert not re.findall(r'TODO|待补|待完成|FIXME', s)
for relative in [
    'video_minimax_h3_t2v.json',
    'logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json',
    'logs/model_card_runs/video_minimax_h3_t2v/events.json',
    'logs/model_card_runs/video_minimax_h3_t2v/memory_samples.json',
    'ComfyUI/comfy/ldm/minimax/model.py',
    'ComfyUI/comfy/ldm/minimax/vae.py',
    'ComfyUI/comfy/ldm/minimax/audio_vae.py',
    'ComfyUI/comfy/text_encoders/minimax.py',
    'ComfyUI/comfy/ops.py',
    '.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/nvfp4.py',
]:
    assert (p.parent / relative).is_file(), relative
sigmas = [12 * (1 - 0.05 * i) / (1 + 11 * (1 - 0.05 * i)) for i in range(20)] + [0]
assert len(sigmas) == 21 and sigmas[0] == 1 and sigmas[-1] == 0
assert '0.995633188' in s and '0.387096774' in s
assert 'W=864：start=[0,144,288,448,608]' in s
print('all_document_checks_passed')
print('lines', len(s.splitlines()), 'bytes', p.stat().st_size)
PY
git status --short
```

Python 检查输出 `all_document_checks_passed`，最终文档为 1283 行、63664 bytes。命令随后进入 `git status --short`，后台任务为 `bash-im5o138r`；在该行追加时只确认 Python 部分通过，不提前宣称后台 git 检查已返回。

### 最终限制

本轮没有标准构建/单元测试可针对这份纯文档运行，也没有为写文档再执行一次大模型。核验是对实际交付文档本身做的：Markdown 完整性、引用路径存在、关键数值/文字与源码和既有运行证据一致。底层逐 kernel 名称、实际 SDPA backend、量化 binding 命中率和耗时仍是未动态 trace 的内容，正式文档已明确标为未验证而非伪造实测。


## 42. 本轮收尾

后台任务 `bash-im5o138r` 已完成。其 output log 只有：

```text
all_document_checks_passed
lines 1283 bytes 63664
```

后续 `git status --short` 没有任何输出，因此再次确认 ComfyUI 源码工作树无修改。`Read` 工具读取了该 output log。至此正式文档、文档自检、源码交叉核验和过程记录全部完成；没有遗留本任务需要运行的大模型或后台分析任务。


---

# CUPTI 逐 Kernel 实测：任务与工具检查

## 43. 新任务与采集原则

用户要求“再去实测 CUPTI 逐 kernel trace 看一下”。这是对前一份静态分析的补充，必须产生真实 trace，不能继续以 backend capability 代替执行证据。

采集原则：

1. 保留当前对外 `8188` 服务，不杀正在使用的进程；如需新实例，使用独立端口与独立日志。
2. 先检查 Nsight Systems、PyTorch CUDA profiler 和 NVTX 支持。
3. 用 `cudaProfilerStart/Stop` 控制 nsys 的有效采集区间，避免把进程启动和首次权重加载混入目标窗口。
4. 尽量对热执行的同一 API prompt 分段加 NVTX，记录文本、采样、video/audio decode 的实际 kernel；明确是否包含 cold initialization。
5. 最终用 trace 数据修正文档中 SDPA backend、INT8/NVFP4 binding、launch 次数、kernel 时间和融合命中情况。

实际执行的环境检查命令：

```bash
cd /home/kaokaolv/comfyui_h3
which nsys || true
which ncu || true
which nvprof || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
pgrep -af 'ComfyUI/main.py' || true
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_profiler_start', hasattr(torch.cuda.profiler, 'start'))
print('cuda_profiler_stop', hasattr(torch.cuda.profiler, 'stop'))
print('nvtx_range', hasattr(torch.cuda.nvtx, 'range'))
try:
    import torch.profiler
    print('torch_profiler_available', True)
except ImportError as exc:
    print('torch_profiler_available', False, exc)
PY
```

结果：`/usr/local/bin/nsys` 存在，`/usr/bin/nvprof` 存在，`ncu` 未找到。PyTorch 2.13.0+cu130 支持 profiler start/stop 和 NVTX。物理 GPU 2 为 B200，现有显存约 37828 MiB，卡总容量 183359 MiB；其他 GPU 大多空闲。`pgrep -af 'ComfyUI/main.py'` 只匹配到检查命令本身，因为启动脚本实际以 `main.py` 相对路径执行，不能据此误判服务已停止。TaskList 中原服务任务 `bash-qooj8jlr` 仍在运行。

当前没有 `custom_nodes/**/AGENTS.md` 的额外约束；ComfyUI 根目录的 AGENTS.md 已读取。没有修改任何模型或执行推理。


## 44. Nsight smoke test 与完整采集准备

### Nsight smoke test（已实际执行）

```bash
cd /home/kaokaolv/comfyui_h3
mkdir -p /home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v
CUDA_VISIBLE_DEVICES=2 nsys profile \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-graph-trace=node \
  --export=sqlite \
  --output=/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke \
  /home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
torch.cuda.init()
torch.cuda.profiler.start()
a = torch.randn((1024, 1024), device='cuda', dtype=torch.bfloat16)
with torch.cuda.nvtx.range('smoke.matmul'):
    b = a @ a
    c = b @ a
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('smoke_result', c.shape, c.dtype, float(c[0, 0]))
PY

nsys stats --report cuda_kern_exec_sum \
  /home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke.sqlite
nsys stats --report nvtx_pushpop_trace \
  /home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke.sqlite
```

结果成功生成 `smoke.nsys-rep` 和 `smoke.sqlite`。Kernel summary 中两次矩阵乘的实际 kernel 为 `nvjet_sm100_tst_128x64_64x10_2x2_2cta_h_bz_NNT`，另有 `distribution_elementwise_grid_stride_kernel<float>`；NVTX 中有 `smoke.matmul` 区间。由此证明 CUPTI 可采集、CUDA kernel 名称可读、SQLite 可解析，且 NVTX 与 kernel 可以关联。

Nsight 版本：`NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0`。启动时 CPU IP/backtrace 和 CPU context switch tracing 在当前环境不支持，自动禁用；不影响 CUDA/CUPTI 和 NVTX。

### 当前服务核验与隔离方案

`start_comfyui_h3.sh` 的真实命令是相对路径 `main.py --listen 0.0.0.0 --port 8188 --disable-partner-nodes`，所以之前 `pgrep ComfyUI/main.py` 找不到不是服务停止。实际 Python PID 为 `1272402`，`/h3/memory` 返回正常。新实例使用 `8189`，输出定向到 `logs/cupti_h3_t2v/output`，不写回原 `ComfyUI/output`。没有停止、重启或修改 8188 的已加载模型。

新采集用 `--capture-range=cudaProfilerApi --capture-range-end=stop`：服务启动阶段不采集；客户端 `POST /h3/cupti/start` 后提交原始 `api_prompt.json`，完成后 `POST /h3/cupti/stop`。这样包含模型冷执行，但不含 ComfyUI 启动初始化。

### 采集辅助文件与检查

新增：

- `ComfyUI/custom_nodes/h3_cupti_profiler.py`：仅添加 profiler HTTP 控制和 NVTX range，不改数学逻辑。标注 `H3.text.qwen`、`H3.text.refiner`、`H3.sampler.res_multistep`、`H3.dit.step.01..20`、`H3.video_vae.decode`、`H3.video_vae.vit.NN`、`H3.audio_vae.decode`、`H3.save_video`。
- `profile_h3_cupti.sh`：独立 nsys 服务 launcher。
- `run_h3_cupti.py`：等待 custom node 就绪、启动采集、提交原 prompt、轮询 history、停止采集并检查 `.nsys-rep/.sqlite`。

启动前修正两处辅助代码问题：

1. DiT step 和 ViT 调用改为两个独立 WeakKeyDictionary 计数，避免总和/最大值误报。
2. 即使 `/prompt` 提交失败，只要 profiler 已启动就执行 stop，避免 CUPTI 一直处于采集状态。

实际语法检查：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import ast
from pathlib import Path
for path in [
    Path('/home/kaokaolv/comfyui_h3/ComfyUI/custom_nodes/h3_cupti_profiler.py'),
    Path('/home/kaokaolv/comfyui_h3/run_h3_cupti.py'),
]:
    ast.parse(path.read_text(), filename=str(path))
    print('syntax_ok', path)
PY
bash -n /home/kaokaolv/comfyui_h3/profile_h3_cupti.sh
```

两个 Python 文件 `syntax_ok`，shell `bash -n` 无错误。另为 sampler 添加 `H3.sampler.res_multistep` range，使 DiT 外的少量 schedule tensor op 也包含在采样区间。

拟执行（后台）：

```bash
cd /home/kaokaolv/comfyui_h3
TRACE_NAME=h3_t2v_cold PORT=8189 bash profile_h3_cupti.sh \
  > logs/cupti_h3_t2v/server_nsys.log 2>&1
/home/kaokaolv/comfyui_h3/.venv/bin/python run_h3_cupti.py \
  > logs/cupti_h3_t2v/client.log 2>&1
```

此轮尚未生成完整 T2V trace；实际运行结果将在后续记录，不把计划写成已完成。


## 45. Nsight 临时目录问题与解决

完整采集前执行 `df -h /tmp`，发现根分区仅约 4.8 GB 可用，完整 T2V CUPTI 数据可能导致系统盘写满。未贸然启动大模型采集。

使用相同小规模 CUDA 测试，但显式设置项目目录：

```bash
TMPDIR=/home/kaokaolv/comfyui_h3/.tmp CUDA_VISIBLE_DEVICES=2 nsys profile \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-graph-trace=node \
  --export=sqlite \
  --output=/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke_home_tmp \
  /home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
torch.cuda.init()
torch.cuda.profiler.start()
a = torch.ones((256, 256), device='cuda', dtype=torch.float32)
with torch.cuda.nvtx.range('smoke.home_tmp'):
    b = a @ a
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('home_tmp_smoke_ok', float(b[0, 0]))
PY
```

结果日志明确为 `Generating '/home/kaokaolv/comfyui_h3/.tmp/nsys-report-4083.qdstrm'`，并正常生成 `smoke_home_tmp.nsys-rep/.sqlite`。`profile_h3_cupti.sh` 同样预设 `TMPDIR=$HERE/.tmp`，因此完整采集使用项目盘（约 4.9 TB 可用）而不是系统 `/tmp`。未删除任何用户文件或改变系统分区配置。


## 46. 完整采集启动与解析器 smoke 验证

### 实际完整采集任务

独立 nsys 服务后台任务 `bash-3v678xbz`，端口 8189；客户端后台任务 `bash-zw3ich08`。实际提交的 prompt ID：

```text
24fe3352-d0a1-4dd7-ac4a-4b1b46b1f09f
```

客户端日志已显示 `profiler_status_before_start active=False`、`prompt_submitted` 和连续 `profile_running`，确认不是仅有启动脚本而未运行。Nsight 服务日志也显示 `h3_cupti_profiler.py` 正常导入、server 启动、`got prompt`、四个模型按原配置加载。采集窗口从 prompt 提交前 `cudaProfilerStart` 开始，到 prompt 完成后 `cudaProfilerStop` 结束。

### 解析器设计

新增 `analyze_h3_cupti.py`，只依赖 Python 标准库 `sqlite3/csv/gzip/json`。输入 nsys SQLite，输出：

- `manifest.json`：kernel 总数、distinct kernel、NVTX 区间数、memcpy、prompt 结果和映射方法；
- `global_kernel_summary.csv`：全局每类 kernel 的次数、总/平均/最小/最大 GPU 时间；
- `phase_kernel_summary.csv`：文本、全部 DiT steps、全部 ViT calls、两个 VAE 等分组；
- `range_kernel_summary.csv`：每个 `H3.dit.step.NN` 和 `H3.video_vae.vit.NN` 分别统计；
- `range_summary.csv`：每个 NVTX 的 wall time、kernel 数和 memcpy；
- `runtime_api_summary.csv`：CUDA API 的名称、次数、host 时间；
- `kernel_events.csv.gz`：逐 kernel 原始时间、名称、stream、grid/block、寄存器、shared/local memory、graphNodeId 和对应 runtime API；
- `summary.md`：便于阅读的摘要。

阶段归属优先通过 kernel `correlationId` 找到 CUDA runtime API，再用 API 的 start timestamp 找最小包含它的 H3 NVTX 区间；没有 runtime 匹配时退回 GPU kernel start。这样比仅比较异步 GPU 执行时间更可靠。`kernel_time_ns` 是各 kernel 时间之和，多 stream 可能重叠，**不等于阶段 wall time**；NVTX duration 是 host 区间时间，包含初始化、拷贝、同步和 CPU 工作。

首次自查发现 runtime 字典把 `name_id` 写成了 correlationId，已修正为 SQLite 第 4 列（`nameId`），并用 smoke 数据验证。

### 实际解析器测试命令

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python analyze_h3_cupti.py \
  logs/cupti_h3_t2v/smoke.sqlite --out logs/cupti_h3_t2v/smoke_analysis
```

结果 `kernel_count=3, distinct_kernels=2`，与 smoke 的 1 个随机数 kernel + 2 个 GEMM 一致。

随后为嵌套区间和 memcpy 新增独立 smoke：

```bash
TMPDIR=/home/kaokaolv/comfyui_h3/.tmp CUDA_VISIBLE_DEVICES=2 nsys profile \
  --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop --cuda-graph-trace=node --export=sqlite \
  --output=/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke_nested \
  /home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
torch.cuda.init()
torch.cuda.profiler.start()
with torch.cuda.nvtx.range('H3.test.outer'):
    a = torch.randn((128, 128), dtype=torch.bfloat16).cuda()
    with torch.cuda.nvtx.range('H3.test.inner'):
        b = a @ a
    c = b + 1
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('nested_smoke_ok', float(c[0, 0]))
PY

/home/kaokaolv/comfyui_h3/.venv/bin/python analyze_h3_cupti.py \
  logs/cupti_h3_t2v/smoke_nested.sqlite \
  --out logs/cupti_h3_t2v/smoke_nested_analysis

/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import csv
from pathlib import Path
path = Path('/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/smoke_nested_analysis/range_summary.csv')
rows = {row['label']: row for row in csv.DictReader(path.open())}
assert len(rows) == 2, rows
assert int(rows['H3.test.inner']['kernel_count']) >= 1, rows
assert int(rows['H3.test.outer']['kernel_count']) >= 1, rows
assert int(rows['H3.test.outer']['memcpy_count']) >= 1, rows
print('nested_mapping_checks_passed', rows)
PY
```

结果：inner 有 1 个 GEMM kernel，outer 有 1 个 elementwise kernel 和一次 32768-byte H2D memcpy；没有父子区间重复统计。输出 `nested_mapping_checks_passed`，证明映射与聚合逻辑可用。

此记录写入时，完整 T2V 仍在执行，后续再解析正式 trace，不提前宣称采集完成。


## 47. 完整 CUPTI 采集结果与 kernel 映射

### 成功采集和输出

完整任务成功结束，prompt ID `24fe3352-d0a1-4dd7-ac4a-4b1b46b1f09f`，`run_result.success=true`。profiler 已 stop。实际文件：

```text
logs/cupti_h3_t2v/h3_t2v_cold.nsys-rep  12,178,305 bytes
logs/cupti_h3_t2v/h3_t2v_cold.sqlite    26,533,888 bytes
logs/cupti_h3_t2v/output/video/MiniMax_H3_00001_.mp4
```

`run_result.elapsed_seconds=394.808`，服务日志 `Prompt executed in 379.21 seconds`。两者计时边界不同，不与原 952 秒冷启动直接作性能对比。本次包含模型冷执行、首次 kernel 初始化，但不包含 ComfyUI 启动阶段。

解析命令：

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python analyze_h3_cupti.py \
  logs/cupti_h3_t2v/h3_t2v_cold.sqlite \
  --out logs/cupti_h3_t2v/analysis
nsys stats --report nvtx_kern_sum \
  logs/cupti_h3_t2v/h3_t2v_cold.sqlite \
  > logs/cupti_h3_t2v/analysis/nvtx_kern_sum.txt
```

实际总量：95,226 kernel、155 种 kernel、67 个 NVTX 区间、5,596 次 memcpy、46,506,971,601 bytes memcpy、421,188 条 runtime API。

区间分解：

```text
Qwen 1 + text refiner 1 + res_multistep 1 + DiT steps 20
+ video decode wrapper 1 + ViT calls 42 + audio decode 1 = 67
```

CPU SaveVideo 没有对应 model kernel/NVTX 记录；另有 19 个很小的 unlabeled kernel，合计约 0.078ms，未强行归属到模型阶段。

### 阶段实测

```text
H3.text.qwen:          host range 113.043s，2366 kernels，GPU kernel work 444.075ms，memcpy 15.237GB
H3.text.refiner:        host range 14.275s，32 kernels，work 0.831ms，memcpy 1.596GB
H3.dit.step.01:        host range 178.617s，2990 kernels，work 1708.844ms，memcpy 19.626GB
H3.dit.step.02..20:    host range约1.16s/step，各2986 kernels，work约1700.5ms，memcpy各175.701MB
H3.sampler.res_multistep: 覆盖整个211.329s采样区间；DiT外另有1067个调度kernel，work 2.742ms
H3.audio_vae.decode:   host range 3.050s，2679 kernels，work 34.288ms，memcpy 259.412MB
H3.video_vae.decode:   host range 25.427s，1979 kernels，work 10.228ms，memcpy 617.103MB
H3.video_vae.vit.01:   host range 22.379s，792 kernels，memcpy 4.869GB
H3.video_vae.vit.02..42: 各648 kernels；约35-63ms/ViT call，常规memcpy 16KiB/call
```

GPU kernel work 是不同 stream 上 kernel 时间之和，可能大于 host range，不能当作 wall time。NVTX 是 host 区间，包含初始化、同步、offload、Python、CPU 和异步等待。

### 真实 kernel 与模型层的对应

#### Qwen

- 350 个 `dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, float, __nv_fp8_e4m3,...>`，合计 26.807ms，平均 76.593us。
- 350 个 FP32 GEMM：`cutlass3x_sm100_simt_sgemm_f32...`。
- 50 个 `fmha_cutlassF_f32_aligned_64x128_rf_sm80`，合计 10.115ms，平均 202.295us。
- 200 个 FP32 layer_norm kernel（每层 input/post-attention/Q/K norm 共 4 个）。

按 grid 和源码 shape 对应：

```text
q_proj:       50个，FP32 64x32 GEMM，grid (128,13)
k/v_proj:    100个，FP32 64x32 GEMM，grid (16,13)
o_proj:       50个，FP32 128x64 GEMM，grid (40,7)
gate/up_proj:100个，FP32 128x64 GEMM，grid (200,7)
down_proj:    50个，FP32 128x64 GEMM，grid (40,7)
```

dequant grid 计数：q/o 共 100 个 grid81920，k/v 共 100 个 grid10240，gate/up/down 共 150 个 grid256000。`o` 与 `down` GEMM output shape 相同，单个 kernel instance 不能只凭 grid 再细分，但总数/阶段/源码顺序与 50×7 完全吻合。

结论升级：不仅“没有 FP4 scaled_mm 的静态证据”，而是实测确认 NVFP4 反量化输出 FP32、后续 FP32 GEMM 和 FP32 mem-efficient attention。全局 155 种 kernel 中没有 `scaled_mm_nvfp4` 或 `cublas_gemm_blockwise_fp4`。

#### DiT

activation quant kernel 的 grid M=15787，结合已知的 video 14985 + audio 414 行，反推本次 text_len=388，packed S=388+14985+414=15787。

- 1000 个主 DiT `pytorch_flash::flash_fwd_kernel<...cutlass::bfloat16_t...>`，总计 17.660879s，平均 17.660879ms。文本 refiner 另有 2 个同 kernel，合计全局 1002 个。
- 主 DiT INT8 GEMM 共 4000 个：
  - `qkv_proj+fc1`：2000 个 identity-swizzle BF16-output CUTLASS，合计 8.776803s，平均 4.388402ms；gridY 分别 84/112。
  - `out_proj+fc2`：2000 个 LeanStreamKT BF16-output CUTLASS，合计 3.630500s，平均 1.815250ms。
- 在线 ConvRot 共 4000 个：
  - K=5376 的 qkv/fc1：2000 个 block_threads=256/code0，合计 498.528ms。
  - K=7168 的 out：1000 个 block_threads=512/code0，合计 337.920ms。
  - K=14336 的 fc2 SwiGLU：1000 个 block_threads=1024/code2，合计 911.173ms。
- 1000 个 BF16 `rope_kernel`（对应 fused RMSNorm+RoPE），合计 720.691ms。
- 6000 个分段 `addcmul`（20×50×2×3），合计 452.751ms。

按 20 步总 kernel work 34.030s 计算，FlashAttention 约 51.9%，INT8 GEMM 约 36.5%，ConvRot 约 5.1%，RoPE 约 2.1%，其他约 4.4%。这是 work 占比，不是 wall-time 占比。

#### Video VAE

42 次 ViT decoder = 7 temporal chunks × 3 高度 tile rows × (4+1) 宽度 batch。每 call 有 36 层，因此：

- `to_qkv/to_out/ff.w1/ff.w2` 各 1512 次，合计 6048 个 INT8 GEMM。
- `qk_int_sv_i8_attn_kernel` 1512 次，合计 284.758ms，平均 188.332us。
- `quant_qk_fused`、`quant_v_int8_kernel`、BF16/FP16 RoPE 各 1512 次。
- ConvRot：rms_norm 输入 3024 次、无输入激活 1512 次、SwiGLU 1512 次，合计 6048 次。
- residual 融合实测：3024 个 CUTLASS 带 `VisitorAuxLoad`，3024 个不带；与 `to_out/ff.w2` 的 residual epilogue 和 `qkv/ff.w1` 的非 residual 分支吻合。

`ViT.01` 比常规多 144 个 kernel（792 vs 648），是冷执行差异，不能将每 call 固定为 792。

#### Audio VAE

实测 2679 kernels，主要是 FP32 Conv1d/ConvTranspose1d、cudnn、Snake、replication pad 和 pointwise op，无 INT8/NVFP4 kernel。

`dgrad2d_grouped_direct_kernel` 虽名称含 dgrad，但它服务于 BigVGAN 的 ConvTranspose1d 前向，不表示发生训练或 backward。

#### 其他

全部 95,226 个 kernel 的 `graphNodeId` 都为空，未观察到 CUDA Graph node kernel；不能把 malloc graph 说成整个 H3 被一次 CUDA Graph replay。

### runtime API 与冷启动边界

为准确分析冷启动，将解析器从“仅 kernel launch API”扩展为全部 runtime API，并补充 memcpy time 和 memset。重新执行同一解析命令后，主要 host API 为：

```text
cudaMemsetAsync       4293 次，host total 13.764s
cudaStreamSynchronize 1038 次，host total 10.573s
cudaHostRegister       852 次，host total  8.194s
cudaLaunchKernel     93597 次，host total  6.718s
cuMemCreate           1153 次，host total  3.140s
cuMemSetAccess        1153 次，host total  1.453s
cuMemGetInfo_v2         86 次，host total  1.234s
cudaMemcpyAsync       4275 次，host total  1.077s
cudaMallocAsync      99972 次，host total  0.692s
cudaFreeAsync        98443 次，host total  0.073s
```

这些 host time 也不能直接相加成 wall time。Qwen 的 `cudaHostRegister` 为 2.979s，DiT step.01 为 4.034s；而 Qwen host range 为 113s，step.01 为 178.6s，说明不能把冷启动全部归因于 memcpy 或 CUDA API。磁盘/mmap page fault、Python、模型结构处理、权重准备和 host 等待不在 CUPTI kernel trace 中，报告必须保留这个边界。

### 媒体验证和服务清理

系统没有 `ffprobe`（`ffprobe: command not found`），改用现有 PyAV 读取，不安装新工具。结果：H.264、864×480、24fps、124 frames、5.166688s；AAC、32kHz、双声道。与原始 API prompt 一致。

独立 8189 profiling 实例在 trace 与 SQLite 导出完成后停止，释放资源。原 8188 `/h3/memory` 随后返回 HTTP 200，确认未受影响。


## 48. 文档更新与最终一致性验证

### 交付文档

- 新增 [cupti_h3_t2v_kernel_report.md](./cupti_h3_t2v_kernel_report.md)：独立 CUPTI 实测报告，包含采集方法、逐模型 kernel、每层映射、时间、runtime API、冷启动边界和原始证据索引。
- 更新 [comfyui_h3_t2v_kernel_analysis.md](./comfyui_h3_t2v_kernel_analysis.md)：新增第 17 章 CUPTI 实测；同时修正前文中 SDPA backend 未知、Qwen GEMM dtype 未定、尚无 CUPTI 等历史限制，并链接独立报告。
- 原始 trace 与解析文件均保留在 `logs/cupti_h3_t2v/`。
- 新增 `verify_h3_cupti.py`，把最终数量与文档检查固化成可重复执行的验证脚本。

### 验证中遇到的问题与处理

1. 初次断言主 DiT `cutlass::Kernel2==4000`，实测为 4020。进一步检查发现 20 个额外的 `cutlass_80_simt_sgemm`，每一步一个非 INT8 单精度 GEMM。真实 INT8 `signed char` GEMM 是 4000，未改变原结论。已在两份文档明确写出“全部 CUTLASS 4020，其中 INT8 4000，其他 SGEMM 20”。
2. 分析这 20 个模板时，曾按“不是 BF16”错误简写为 FP16；完整名称是 `cutlass_80_simt_sgemm`，应描述为额外的单精度 GEMM，不能写 FP16。已纠正，不将非 INT8 kernel 纳入量化数量。
3. 第二次验证把 `signed char` 单独作为条件，匹配到了 GEMM 和量化函数两类签名，改为必须同时满足 `cutlass::Kernel2` 与 `signed char`。
4. 文档检查曾要求独立报告包含自己的文件名，这是验证脚本条件错误，不是报告内容错误；改为只要求主分析文档链接独立报告，并检查独立报告标题。
5. `ffprobe` 不在 PATH，使用现有 PyAV 完成媒体验证，没有安装新依赖。
6. `nsys stats --help-reports` 无参数时返回 exit code 1，但列出了可用报告，不是采集失败；后续使用 `nvtx_kern_sum` 已成功生成统计。

以上问题均通过原始 SQLite/CSV 复核解决，没有用修改数据的方式让断言通过。

### 最终验证命令与结果

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python verify_h3_cupti.py
```

完整断言代码保存在 `verify_h3_cupti.py`，包括：SQLite 与 manifest 总数、NVTX 区间数、20 个 DiT steps、42 次 ViT、各量化 kernel 次数、residual AuxLoad、graphNodeId、文档链接、代码块配对和产物存在性。

实际输出：

```text
all_cupti_evidence_checks_passed
{
  "kernels": 95226,
  "distinct_kernels": 155,
  "nvtx_ranges": 67,
  "dit_steps": 20,
  "video_vit_calls": 42,
  "dit_cutlass_all": 4020,
  "dit_int8_gemm": 4000,
  "other_float_sgemm": 20,
  "vae_int8_gemm": 6048,
  "qwen_nvfp4_dequant": 350,
  "vae_auxload": {
    "no_auxload": 3024,
    "auxload": 3024
  },
  "graph_nodes": 0
}
```

在 `ComfyUI` 目录执行 `git status --short` 与 `git diff --stat` 均无输出，没有 tracked 源码修改。新增分析脚本、文档、custom-node 探针和 trace 都是本任务的交付/证据文件，未删除。

独立 8189 profiling 实例已停止，原 8188 仍可访问。没有遗留正在执行的模型采集任务。


---

# CUPTI 实测回填到各模块（文档深化）

## 49. 新任务与证据核对

用户要求：既然已有真实 kernel trace，就 overview `comfyui_h3_t2v_kernel_analysis.md`，在对应模块（特别点名 3.2 DiT 的识别、构造和量化）更新进去，补充更多函数调用栈/kernel 调用栈细节、模型引用、真实 kernel 的来源与计算原理。其它章节也要详细严谨，使整个推理链路更清晰。

本轮不再重跑模型。工作方式：

1. 以已有 `h3_t2v_cold.sqlite` 和 `analysis/kernel_events.csv.gz` 为准，补齐真实 launch 参数、模板、grid/block、函数绑定。
2. 将 CUPTI 结果直接回填到第 2–16 章相应模块，而不是仅保留第 17 章附录。
3. 保留静态 fallback/其它 workflow 的边界，不把一次 trace 泛化为所有后端和所有模型。
4. 用原 workflow 的 Hugging Face 链接、local model path 和类定义同时标注“模型引用”。

本轮实际读取/查询：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import csv, gzip
p = '/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/analysis/kernel_events.csv.gz'
# 逐行筛选 H3.text.qwen、H3.dit.step.02、H3.video_vae.vit.02、
# H3.video_vae.vit.05，输出 kernel 名称前缀、grid/block、
# registersPerThread、dynamicSharedMemory 和必要的完整模板名。
# 另打印 Qwen 起始 kernel、refiner Flash 和其它 SGEMM 的参数。
PY
```

新增实测 launch 证据：

```text
主 DiT Flash: grid=(124,1,56), block=(128,1,1), registers=255,
              dynamicSharedMemory=65536；ceil(15787/128)=124。
Qwen FMHA:   grid=(7,64,1), block=(32,1,1), registers=251,
              dynamicSharedMemory=86016；ceil(388/64)=7，Q heads=64。
VAE qk_int_sv_i8_attn:
  batch1 grid=(15,32,1), batch4 grid=(15,32,4)，block=32，
  registers=244，dynamicSharedMemory=16384。
```

VAE 每块为 7×16×16 latent tokens + 4 register + 1 suffix = 1797 tokens，因此 1797/7188=1:4 与宽度 tile 合 batch 完全对应。`ceil(1797/128)=15`，head=32。

量化 activation 的代码枚举也已核实：`None=0`、`gelu_tanh=1`、`swiglu=2`、`rms_norm=3`，见 `comfy_kitchen/backends/_activations.py:25-45`。这与 DiT fc2 code2、VAE norm code3 对应，不再只靠名称猜测。

`comfy_kitchen/backends/cuda/__init__.py` 中的 `_C.abi3.so` 加载逻辑、`_C.cutlass_int8_dequant`、`_C.quantize_int8_rowwise_convrot64`、`_C.rms_rope`、`_C.dequantize_nvfp4`、`_C.sage_sdpa` 均已与 trace 对上。PyTorch `at::native`/`pytorch_flash`/`fmha` 是 PyTorch 内置 CUDA 后端，不能误写为 Comfy Kitchen custom op。

原 run_book 的 CUPTI 采集和解析命令均已完成，本轮只做 trace 与源码的再分析和文档更新。


## 50. 各模块深化完成与最终复核

### 已直接更新 `comfyui_h3_t2v_kernel_analysis.md`

本轮不是只追加附录，而是把真实 trace 回填到各模块：

1. **端到端第 2 章**：新增从 Qwen、20 步 DiT、42 次 video ViT、audio VAE 到 CPU SaveVideo 的函数与 kernel 对照图。
2. **3.2 DiT 的识别、构造和量化**：补充 Comfy-Org/上游模型链接、本机文件、`supported_models.MiniMaxH3`、`model_base.MiniMaxH3` 和 `ldm.minimax.model.MiniMaxH3Model` 的对应关系；展开 UNETLoader → model detection → model factory → ModelPatcherDynamic → load weights 的完整链路。
3. **量化对象到真实 kernel**：解释 `.comfy_quant` → `TensorWiseINT8Layout` → `QuantizedTensor.__torch_dispatch__` → `torch.ops.comfy_kitchen.int8_linear` → `_C.quantize_int8_rowwise_convrot64` + `_C.cutlass_int8_dequant`。明确这些 binding 来自 `comfy_kitchen/backends/cuda/_C.abi3.so`（已用 Glob 确认实际存在）。
4. **文本编码第 3.3、4 章**：补充 Qwen 的模型文件和类定义、50 层完整 forward、7 个 Linear 的 M/N/K、350 次 NVFP4 反量化 + 350 次 FP32 GEMM，以及 FP32 FMHA。解释 `scaled_mm_nvfp4` 不成立是 checkpoint flag、loader 和双 operand 前提共同决定，而不是 B200 缺 kernel。
5. **采样第 6 章**：补齐一次 sigma 调用从 `KSAMPLER.sample` 到 `res_multistep`、`KSamplerX0Inpaint`、guider、`calc_cond_batch`、`BaseModel.apply_model` 的调用链；1067 个 DiT 外调度 kernel 与公式逐项对应。
6. **DiT 第 7、8 章**：新增四类主 Linear 的 M/N/K、identity/LeanStreamKT、code0/2、每类调用次数和 ConvRot/INT8 计算公式；主 attention 实测 BF16 Flash，grid=(124,1,56)，对应 ceil(15787/128) 和 56 heads。
7. **final head**：明确 20 个 `cutlass_80_simt_sgemm` 是额外的非 INT8 单精度 GEMM。全部 CUTLASS 为 4020，其中 4000 是 INT8；不再仅凭没有 BF16 字样误判为 FP16。
8. **video VAE 第 9 章**：补齐 to_qkv/to_out/ff.w1/ff.w2 的 K/N、RMSNorm/SwiGLU code、42 次 ViT 的来源、batch=1797/7188 的 grid 证明，以及 3024 个 `VisitorAuxLoad` residual epilogue。
9. **Sage attention**：补充 `quant_qk_fused`、`quant_v_int8_kernel`、`detect_k_anchor` 和 `qk_int_sv_i8_attn_kernel` 的来源、FP32 softmax 计算、grid=(15,32,1/4) 与 tile token 数的对应。
10. **audio VAE 第 10 章**：补充 FP32 BigVGAN 的真实 kernel。127 个 `sin_kernel` 对应 `7×3×6+1` 次 SnakeBeta；254 个 replication pad 对应 activation 两侧 padding。`dgrad` 是转置卷积前向实现，不是训练 backward。
11. **内存与 graph 第 11 章**：加入每阶段 memcpy、host API 和首次/热执行的差异；95,226 个 kernel 的 `graphNodeId` 全部为空，未观察到 CUDA Graph node kernel。
12. **保存与汇总第 12–14 章**：加入 PyAV 实测输出、kernel 来源总表、20 步 kernel work 占比，以及 GPU work 与 wall time 的区别。
13. **模型引用与证据索引**：保留原始 trace、CSV、runtime API、Nsight 交叉统计、独立报告和验证脚本的链接。

最终文档共 1997 行。没有重新执行模型，只复用之前成功采集的真实 trace。

### 本轮使用的 trace 补充查询

使用 `.venv/bin/python` 读取 `analysis/kernel_events.csv.gz`，筛选：

- `H3.text.qwen` 的 NVFP4、FMHA 和开头 kernel；
- `H3.dit.step.02` 的 FlashAttention、RoPE 和其它 SGEMM；
- `H3.video_vae.vit.02` 与 `H3.video_vae.vit.05` 的 INT8 attention 和 ConvRot；
- 输出 `gridX/Y/Z`、`blockX`、`registersPerThread`、`dynamicSharedMemory`、必要的完整模板名。

还使用 Glob 确认 `.venv/lib/python3.12/site-packages/comfy_kitchen/backends/cuda/_C.abi3.so`，Grep 读取 `backends/_activations.py` 的 input_act 枚举和 CUDA backend 的 dequantize/int8/rms_rope/scaled_mm 函数。

### 复核发现的问题与修正

- 一处辅助文本混入英文 `further`，已改为中文“进一步”。
- 文档自检曾把“不能仅凭初始化 FP16 就写成 FP16 GEMM”误判为错误结论。没有删除这条正确的 dtype 边界，而是修正验证条件。
- `verify_h3_cupti.py` 已扩充检查：除原始 kernel/量化/graph 数量外，还检查四个 HF 模型链接、Flash/FMHA/Sage grid、VAE batch 形状、额外 SGEMM、Comfy Kitchen `_C.abi3.so` 来源和独立报告链接。

最终执行：

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python verify_h3_cupti.py
```

输出 `all_cupti_evidence_checks_passed`。在 `ComfyUI` 目录执行 `git status --short` 和 `git diff --stat`，均无输出，确认没有 tracked 源码改动。没有遗留新启动的服务或模型推理任务。


---

# MXFP8/MXFP4 与 compressed-tensors 接入设计调研

## 51. 任务和边界

用户确认 INT8、NVFP4（包括若走真实 GEMM）和 Sage INT8 attention 的 kernel 是否都在 Comfy-Org/comfy-kitchen，要求深入调研能否补充 MXFP8/MXFP4 计算和 GEMM，参考 vLLM/FlashInfer 等推理库，并研究 compressed-tensors 的 MXFP4/MXFP8 权重能否在 ComfyUI 加载后 dispatch 到 Comfy-Kitchen。进一步要求参考 vLLM 的 kernel select/dispatch 策略，按不同硬件选择不同 kernel。最终交付详细 proposal 设计文档（接口、kernel 实现、dispatch、加载映射、验证方案），先审阅，不直接实现。

本轮只做源码/官方文档调研和 proposal，不下载大模型、不修改 ComfyUI/comfy-kitchen 代码、不宣称 MXFP4/MXFP8 已经跑通。

首先确认已有事实：

- INT8 ConvRot/GEMM、fused RMS/RoPE、NVFP4 dequant/GEMM、Sage INT8 attention 都在 comfy-kitchen 中；Python 入口在 `comfy_kitchen` 包，CUDA C++ 实现由 `backends/cuda/_C.abi3.so` 加载。
- ComfyUI 的 Python model/quant dispatch 在 `ComfyUI/comfy/ops.py`、`comfy/quant_ops.py`，不是 comfy-kitchen 仓库本身。
- 当前 Qwen 没走 NVFP4 真 GEMM 是 `full_precision_matrix_mult=true` 和 loader 策略，不是 NVFP4 kernel 不存在。
- 现有 MXFP8 与 MXFP4 不能混同：MXFP8 是 FP8 E4M3 + 32 元素 E8M0 block scale；MXFP4 是 FP4 E2M1 + 32 元素 E8M0 block scale；NVFP4 是 FP4 E2M1 + 16 元素 E4M3 block scale + tensor scale。需要分别实现 layout、quant/dequant、GEMM 和 loader 映射。

开始核对本机 comfy-kitchen、vLLM 和 vllm-omni 源码，随后补官方来源并写 proposal。
