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


### 51.1 用户补充确认：H3 是否都是 dense

用户追问当前 workflow 中的模型是否都是 dense。复核 `video_minimax_h3_t2v.json`、既有 CUPTI 结果和 `ComfyUI/comfy/ldm/minimax/model.py` 后确认：本方案关注的 Linear 全是稠密 Linear，没有 MoE 路由专家；但整个网络并非只有 Linear，还含 BF16 FlashAttention、Sage INT8 attention、RMSNorm、RoPE、卷积等。

- 主 DiT：50 blocks × `qkv_proj/out_proj/fc1/fc2`，200 个目标 Linear。
- Qwen 文本编码器：50 层 × 7 个 Linear，350 个目标 Linear。
- 视频 VAE ViT：36 层 × 4 个 Linear，144 个目标 Linear。
- 音频 VAE：FP32 Conv1d/ConvTranspose1d 为主，不改造。
- `model.py:178` 的可选 `to_gate_compress` 是稀疏 attention 扩展门控，不是 MoE expert 路由；当前 dense forward 不使用它，不能因此误称 MoE。
- proposal 不包含 MoE、专家并行和通用 sparse Linear。文本编码器文件中的非本次 T2V 使用模块也不能未经 manifest 检查就全部量化。

### 51.2 vLLM dense kernel 选择和真实权重格式

本机参考环境：`/home/kaokaolv/.venv-vllmomni029`，vLLM 0.29.0。读取：

```text
vllm/model_executor/kernels/linear/__init__.py
  _POSSIBLE_MXFP8_KERNELS:509-529
  _POSSIBLE_MXFP4_KERNELS:559-574
  init_mxfp8_linear_kernel:845-880
  init_mxfp4_linear_kernel:883-922
  mxfp8/Mxfp8LinearKernel.py
  mxfp8/flashinfer.py
  mxfp8/marlin.py
  mxfp4/base.py
  mxfp4/flashinfer.py
  mxfp4/marlin.py
  layers/quantization/online/mxfp8.py
  layers/quantization/online/mxfp4.py
  layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py
  layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py
```

结论：

1. vLLM 按平台列出候选，依次做 `is_supported()` 和 `can_implement(config)`，支持 backend 过滤、禁用列表和失败原因汇总。该策略适合借鉴，不应只按 GPU 名称写一条 if/else。
2. MXFP8 的 FlashInfer CUTLASS/CuTe-DSL 都是 W8A8；Marlin 是 W8A16，不应被当作 W8A8 的静默替代。
3. MXFP4 的 FlashInfer CuTe-DSL 是真 W4A4，权重 `[N,K/2]` uint8、E8M0 group-32 scale、无 global scale，activation 动态量化；Marlin 是 W4A16。
4. compressed-tensors MXFP8 键为 `weight/weight_scale`，MXFP4 为 `weight_packed/weight_scale`。vLLM 加载后统一成 `layer.weight/layer.weight_scale`，再 `process_weights_after_loading()` 做 scale swizzle 和必要的权重准备。
5. BF16 online quantization 和 compressed-tensors 预量化加载最后进入同一个 kernel，说明 ComfyUI 也应共用运行时格式，而不是复制两套 Linear。

### 51.3 FlashInfer 接口与依赖边界

读取本机包：

```text
flashinfer/gemm/gemm_base.py
  mm_mxfp8:5579-5747
  _cudnn_mm_mxfp8_requirement:5042-5059
  _cute_dsl_gemm_fp4_requirement:6177-6204
  mm_fp4:7080-7234
flashinfer/quantization/fp4_quantization.py
  mxfp4_quantize:1546-1624
  mxfp4_dequantize:1628-1660
```

官方 API 页面也已读取：

- https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_mxfp8.html
- https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html

确认：

- `mm_mxfp8(a, b, a_descale, b_descale, ..., backend=...)` 中 A 是 `[M,K]`，B 是 `[K,N]` 的 column-major 视图；权重可保留 `[N,K]`，调用时 `.t()`，B scale 仍按 `[N,K/32]` 的 128x4 swizzle 语义准备，不把 scale 简单数值转置。
- `mm_fp4(..., block_size=32, use_nvfp4=False, alpha=None)` 才是 MXFP4；packed data 的最后一维是 K/2。默认参数 `block_size=16,use_nvfp4=True` 是 NVFP4，绝不能复用默认值。
- MXFP8 官方接口要求 1D 128x4 scale；当前 Comfy-Kitchen 同物理布局可以 reshape 成 1D view，无需重新量化。
- CuTe-DSL FP4 在本机源码中要求 128x4 scale、N%8==0，以及对应 device capability；输出是 FP16/BF16，不支持直接 FP32 输出。
- MXFP4 quantizer 提供稳定 `backend="cuda"` 和实验性 `backend="cute-dsl"`，实现选择必须单独测试，不能因为 GEMM backend 名称相同就假定量化字节也完全一致。
- FlashInfer 的 kernel 由 FlashInfer/CUTLASS/CuTe 提供。推荐 Comfy-Kitchen 做 adapter 和选择，不复制底层 kernel 源码，也不把新 kernel 错称为 comfy-kitchen `_C` 原生实现。
- ComfyUI 当前虚拟环境探测结果 `comfy_venv_flashinfer_spec: None`，因此 FlashInfer 尚未在 ComfyUI 环境可用。另一环境存在 Python 包不等于当前 8188 服务能调用，不能通过 PYTHONPATH 混用两个 PyTorch 环境。

### 51.4 ComfyUI 加载、dtype 和融合边界

读取本机 ComfyUI（commit `989d93d0d968e704c589ce0146bb18c77f6b50cc`）：

```text
comfy/quant_ops.py:72-147,201-257
  已有 TensorCoreMXFP8Layout 子类和 mxfp8 QUANT_ALGOS，没有 mxfp4。
comfy/ops.py:1342-1485
  _load_quantized_module 读取 .comfy_quant、weight 和 scale。
comfy/ops.py:1542-1670
  mixed_precision_ops、activation 动态量化、weight-only 分支。
comfy/utils.py:1467-1532
  detect_layer_quantization 和 convert_old_quants 的既有入口。
comfy/sd1_clip.py:109-120
  文本编码器使用 full_precision_mm=True 构造 mixed_precision_ops。
comfy/ldm/minimax/vae.py:289-375
  RMSNorm、SwiGLU、residual 的 linear_input_act 调用。
comfy/ldm/minimax/model.py:167-218
  主 DiT dense qkv/out/fc1/fc2，fc2 前有 SwiGLU。
```

设计决策：

1. 新 MXFP 文件必须显式写 `full_precision_matrix_mult=false`，在 `_load_quantized_module` 中只对 MXFP8/MXFP4 让“文件显式设置”覆盖父级默认，未写该字段的文件继续继承父级默认。其他格式维持原有分支。这样不必全局修改 `sd1_clip.py`，也不会意外把现有 NVFP4 文本编码器切换到真 GEMM。
2. 第一版 MX layout 不实现 INT8 ConvRot 的融合语义；`linear_input_act` 对可选融合返回 `NotImplemented`，由现有 `ComfyUI/comfy/ops.py` 在模块层先执行 RMSNorm/SwiGLU，再调用 MX GEMM，最后执行 residual，保持数学顺序。
3. FlashInfer 只输出 FP16/BF16。当前 Qwen 实测为 FP32 GEMM，因此 proposal 必须定义 MX 层的 BF16 输出/后续 FP32 residual 策略和精度验证，不能宣称 dtype 完全不变。
4. VAE 保留现有 Sage INT8 attention；主 DiT BF16 FlashAttention 和音频 FP32 Conv 不改。
5. 原生 `.comfy_quant` 单文件是首要交付格式；不要求 ComfyUI 运行时导入 vLLM 或 compressed-tensors。标准 compressed-tensors 的目录 `config.json` 不会自动被当前单文件 loader 发现，需要离线转换时显式提供 recipe，不能把缺失 recipe 当作未量化 BF16。

### 51.5 源码量化细节和静态 shape/存储测算

已核实：Comfy-Kitchen eager MXFP8 以每 32 个元素为一组，E8M0 exponent 使用 `ceil(log2(amax/448))`，再转 E4M3；vLLM 的 MXFP8 源码也采用向上取 exponent 的算法。vLLM MXFP4 Triton 则按 OCP 规则先将 block amax 做 round-to-nearest-even，再 `floor(log2(amax))-2` 得 scale exponent，E2M1 为 RNE、低 nibble 存偶数元素、高 nibble 存奇数元素。不同 FP4 quantizer 的 scale/舍入规则必须记录并逐字节验证，不能把“都是 MXFP4”当成 bitwise 相同。

读取的额外源码：

```text
comfy_kitchen/backends/eager/quantization.py:280-406,607-718
comfy_kitchen/float_utils.py:292-382
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:14-85,181-237
vllm/model_executor/layers/quantization/utils/mxfp4_utils.py:188-277,470-496
```

静态 shape 测算实际执行命令：

```bash
python3 - <<'PY'
# 用已实测的 50×4 DiT、50×7 Qwen、36×4 Video-VAE-ViT 形状，
# 输出每类层的 N/K、是否均被 128 整除、Linear 数量、总元素，
# 以及 BF16/MXFP8(8.25bit)/MXFP4(4.25bit) 的 payload GiB。
PY
```

结果：全部目标 Linear 的 K、N 都能被 128 整除；总数分别 200、350、144。只计这些 Linear 的 qdata+未对齐线性 scale 的理论 payload（不含其他权重、框架和激活）：

```text
DiT:          19,267,584,000 elements; BF16 35.889 GiB; MXFP8 18.505 GiB; MXFP4 9.533 GiB
Qwen:         24,379,392,000 elements; BF16 45.410 GiB; MXFP8 23.415 GiB; MXFP4 12.062 GiB
Video VAE ViT: 2,415,919,104 elements; BF16  4.500 GiB; MXFP8  2.320 GiB; MXFP4 1.195 GiB
```

上述不是完整 checkpoint 大小，也不是实测 VRAM。实际显存仍需通过 CUPTI 和显存探针测量。

### 51.6 查询中遇到的问题及修正

1. 初次按旧布局猜 `mxfp8/flashinfer_cutedsl.py`、`mxfp4/MxFp4LinearKernel.py`，文件不存在；Glob 后确认真实文件是 `mxfp8/flashinfer.py`、`mxfp4/base.py`。没有继续猜接口，改读实际源码。
2. 初次猜 `flashinfer/quantization/mxfp4_quantize.py`，文件不存在；真实公开入口在 `fp4_quantization.py:1546`，CuTe 实现在其 `quantization/kernels/` 下。
3. 一个读取 Comfy-Kitchen eager 文件的路径漏写 `comfyui_h3`，返回 file-not-found；已使用本任务环境 `.venv` 的正确路径重新读取，没有更换环境。
4. 两次宽泛 Glob/Grep 超时或输出过大；改为固定子目录、精确类名/函数名和限定行段。未依赖超时结果作结论。
5. `pip show` 与 git 查询的命令转入后台，207.683s 后正常结束；确认 comfy-kitchen 0.2.41、vLLM 0.29.0 和 ComfyUI commit。没有把等待时间误写成 kernel 或模型运行时间。
6. 早期摘要中 MXFP4 的 Triton/FlashInfer 文件名不是最终调用入口；proposal 区分 vLLM 权重量化、FlashInfer activation 量化与 FlashInfer GEMM 三部分，避免混为一谈。
7. 查询期间读取了 `/home/kaokaolv/AGENTS.md` 和 `ComfyUI/AGENTS.md`；后者要求保持最小改动、复用原生 loader、失败明确而非静默降质。方案据此采用单文件、明确 metadata 校验、显式 fallback policy，不增加 H3 专用节点。

本轮没有启动/停止 8188 服务，没有安装依赖、修改模型、转换权重或运行新的 GPU 推理。proposal 中的 API 和阶段测试均为待实施设计，不能视为已跑通结果。


### 51.7 Proposal 初稿与 kernel 内部实现补证

已新建 `comfyui_h3_mxfp8_mxfp4_proposal.md`，内容包括：真实 H3 目标层、MX 数学格式、公共接口、原生文件格式、两层 backend dispatch、FlashInfer adapter、ComfyUI loader、compressed-tensors 无损映射、分阶段实施和 CUPTI/质量/显存验收。该文件是设计稿，没有修改 ComfyUI/comfy-kitchen 源码。

为把“kernel 实现”写清楚，进一步读取：

```text
flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py:50-180,193-371,467-573,1127-1144
flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_common.py:84-220
```

确认 B200 CuTe 实现直接接收 A/B/SFA/SFB，先构建 block-scaled tiled MMA 和 TMA descriptor，使用 warp specialization、`PipelineTmaUmma`、`PipelineUmmaAsync`、tensor-memory FP32 accumulator，SFA/SFB 设置后调用 `cute.gemm`，最后 epilogue 输出。MMA instruction K=256 bit（FP8 32 个元素、FP4 64 个元素），tile K 包含 4 个 instruction K step；scale group 仍为 32 元素，两者不是同一概念。底层类有 FP32 accumulator，但公开 FlashInfer wrapper 仍限 FP16/BF16 输出，proposal 已明确这一边界。

### 51.8 128x4 scale 的纯 CPU 实测

为避免只凭“128x4”名称假定 Comfy-Kitchen 和 vLLM 布局相同，执行：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
from comfy_kitchen.float_utils import to_blocked, from_blocked
r, g = 128, 8
s = torch.arange(r * g, dtype=torch.uint8).reshape(r, g)
ck = to_blocked(s).contiguous()
vllm = s.view(1, 4, 32, 2, 4).transpose(1, 3).contiguous().view(-1)
print('byte_equal:', torch.equal(ck, vllm))
print('comfy_inverse_exact:', torch.equal(from_blocked(ck, r, g), s))
print('num_mismatches:', int((ck != vllm).sum()))
PY
```

命令进入后台，157.579s 后正常结束。实际输出：

```text
byte_equal: True
comfy_inverse_exact: True
num_mismatches: 0
```

两套 Python scale 重排在这个 `[128,8]` 样例上逐字节相同，逆变换正确。该结论仅覆盖 CPU 索引变换，未声称 CUDA `quantize_mxfp8` 或 FlashInfer GEMM 已验证；这些仍列入 P0。

### 51.9 文档验证脚本的问题与修正

第一次执行 Markdown 结构检查：

```bash
python3 - <<'PY'
# 检查 code fence、第1至18节标题、必要术语和本地证据路径。
PY
```

首次在章节1处触发 `AssertionError: 1`。原因是检查脚本把 raw regex 写成了 `rf'^## {heading}\\.'`，匹配了多余的反斜杠，而不是文档缺少第1节。应改成 `rf'^## {heading}\.'`，不修改文档来迎合错误的测试。后续重新运行修正后的检查并记录结果。


### 51.10 eager 语义、兼容性和最终一致性修正

继续读取 `comfy_kitchen/scaled_mm_v2.py:101-126`，发现 eager MXFP8 在 PyTorch 有能力时调用 `torch.nn.functional.scaled_mm`，并非必然是 dequantize+普通 GEMM。已修正 proposal：

- 将现有 `torch_scaled_mm_v2` 单独列为 P0/CUPTI 待验证候选，不称为 fake，也不未经实测宣称可用。
- 只有明确写出来的 dequantize+FP32 计算才叫 `eager_reference`。
- fast path 不能因 CUDA backend 失败自动落到未验证的 eager；显式 reference 由 layout 直接 dequantize，不借用普通 registry fallback 冒充 fast path。
- full-precision 覆盖规则只对 MXFP8/MXFP4 生效；其他量化格式保留原分支，避免意外改变旧模型。
- `disabled_formats` 的禁止 fast path 优先级最高。

MXFP4 scale 的来源也已补查 `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:964-1005`。其 `swizzle_mxfp4_scales` 明确使用 `[numMTiles,numKTiles,32,4,4]`，输入 `[N,K//32]`，与 MXFP8 的 32 元素分组及 128x4 重排规则一致；这里只借用 dense 的 scale 工具，不引入 MoE 实现。

### 51.11 已执行 Bash 命令的完整记录

环境版本和仓库状态查询（后台正常完成）：

```bash
git -C /home/kaokaolv/comfyui_h3/ComfyUI rev-parse HEAD && \
git -C /home/kaokaolv/comfyui_h3/ComfyUI status --short && \
/home/kaokaolv/comfyui_h3/.venv/bin/python -m pip show comfy-kitchen && \
/home/kaokaolv/.venv-vllmomni029/bin/python -m pip show vllm
```

输出：ComfyUI `989d93d0d968e704c589ce0146bb18c77f6b50cc`，`git status --short` 无输出，comfy-kitchen 0.2.41，vLLM 0.29.0。

当前 ComfyUI 环境的 FlashInfer 探测：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import importlib.util
print('comfy_venv_flashinfer_spec:', importlib.util.find_spec('flashinfer'))
PY
```

输出 `comfy_venv_flashinfer_spec: None`。

H3 shape 和理论 payload 测算的完整代码：

```bash
python3 - <<'PY'
from math import ceil
shapes = {
    'DiT': (50, [('qkv_proj', 5376, 21504), ('out_proj', 7168, 5376), ('fc1', 5376, 28672), ('fc2', 14336, 5376)]),
    'Qwen': (50, [('q_proj', 5120, 8192), ('k_proj', 5120, 1024), ('v_proj', 5120, 1024), ('o_proj', 8192, 5120), ('gate_proj', 5120, 25600), ('up_proj', 5120, 25600), ('down_proj', 25600, 5120)]),
    'Video_VAE_ViT': (36, [('to_qkv', 2048, 6144), ('to_out', 2048, 2048), ('ff.w1', 2048, 16384), ('ff.w2', 8192, 2048)]),
}
for name, (layers, specs) in shapes.items():
    print(name, 'all_K_divisible_by_128=', all(k % 128 == 0 for _, k, n in specs), 'all_N_divisible_by_128=', all(n % 128 == 0 for _, k, n in specs))
    print('count=', layers * len(specs))
    total = 0
    for layer, k, n in specs:
        elements = layers * n * k
        total += elements
        print(f'  {layer}: N={n} K={k} layers={layers} elements={elements:,}')
    print(f'  total_elements={total:,} bf16_GiB={total * 2 / 2**30:.3f} mxfp8_payload_GiB={total * 33 / 32 / 2**30:.3f} mxfp4_payload_GiB={total * 17 / 32 / 2**30:.3f}')
PY
```

128x4 scale 的完整 CPU 验证代码：

```bash
/home/kaokaolv/comfyui_h3/.venv/bin/python - <<'PY'
import torch
from comfy_kitchen.float_utils import to_blocked, from_blocked
r, g = 128, 8
s = torch.arange(r * g, dtype=torch.uint8).reshape(r, g)
ck = to_blocked(s).contiguous()
num_m_tiles = (r + 127) // 128
num_k_tiles = (g + 3) // 4
vllm = s.view(num_m_tiles, 4, 32, num_k_tiles, 4).transpose(1, 3).contiguous().view(-1)
print('shape:', tuple(s.shape))
print('byte_equal:', torch.equal(ck, vllm))
print('comfy_first_32:', ck[:32].tolist())
print('vllm_first_32: ', vllm[:32].tolist())
print('comfy_inverse_exact:', torch.equal(from_blocked(ck, r, g), s))
print('num_mismatches:', int((ck != vllm).sum()))
PY
```

结果见 51.8：两者 byte 完全相同，逆变换相同，0 个 mismatch。这是 CPU 元数据/layout 测试，不是 GPU 模型推理。

最终 Markdown 和本地证据检查（已修正 51.9 所述测试正则）：

```bash
python3 - <<'PY'
from pathlib import Path
import re
p = Path('/home/kaokaolv/comfyui_h3/comfyui_h3_mxfp8_mxfp4_proposal.md')
s = p.read_text()
assert s.count('```') % 2 == 0, 'unpaired code fence'
for heading in range(1, 19):
    assert re.search(rf'^## {heading}\.', s, re.M), heading
for text in ['尚未实现', 'scaled_mm_mxfp8', 'scaled_mm_mxfp4', 'full_precision_matrix_mult', 'compressed-tensors', 'FlashInfer', 'CUPTI', 'dense', '128x4', 'tcgen05', 'torch_scaled_mm_v2']:
    assert text in s, text
for rel in ['video_minimax_h3_t2v.json', 'logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json', 'comfyui_h3_t2v_kernel_analysis.md', 'logs/cupti_h3_t2v/h3_t2v_cold.sqlite']:
    assert (p.parent / rel).exists(), rel
for link in re.findall(r'\]\((\./[^)]+)\)', s):
    assert (p.parent / link).exists(), link
print('proposal_structure_and_local_evidence_ok')
print('lines:', len(s.splitlines()), 'bytes:', len(s.encode()))
print('code_fences:', s.count('```'))
PY

git -C /home/kaokaolv/comfyui_h3/ComfyUI status --short
git -C /home/kaokaolv/comfyui_h3/ComfyUI diff --check
```

最终实际输出：

```text
proposal_structure_and_local_evidence_ok
lines: 890 bytes: 46719
code_fences: 42
```

`git status --short` 和 `git diff --check` 均无输出。没有修改 ComfyUI tracked 文件，没有安装依赖、下载/转换模型、启动新服务、停止原 8188 服务或执行新的 GPU 推理。

最终交付：[`comfyui_h3_mxfp8_mxfp4_proposal.md`](./comfyui_h3_mxfp8_mxfp4_proposal.md)。详细调研、问题与修正均记录在第 51 节；待用户审阅 proposal 后，才进入 P0 实现前验证。

---

## 52. Comfy-Org MiniMax-H3 五个模型文件的结构/精度/参数量分析

### 52.1 任务

用户要求分析 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 中 5 个文件（DiT int8_convrot、TE nvfp4_awq、video VAE int8_convrot、audio VAE fp32、8-step turbo LoRA）的模型结构、文件大小、各模块精度与参数量占比，要求同时核对 model card、config 与 safetensors 实际内容，输出到一个 markdown 文档。

### 52.2 方法

1. 拉取 HF model card（`README.md`）、HF API 的仓库文件清单与 LFS oid、上游 `MiniMaxAI/MiniMax-H3` 的 `FL2VA/{transformer,text_encoder,video_vae,audio_vae}/config.json`。
2. 写 `analyze_h3_safetensors.py`：只读 safetensors header（8B 长度 + JSON）与 `comfy_quant` blob，不加载权重；按"逻辑参数量"统计（NVFP4 U8 权重 ×2，int8 ×1），量化 side-car 不计入参数。
3. 读取上游 13 个 transformer shard 的 header（HTTP Range，每次只取前 ~6KB）用于 full vs pruned 对比。
4. 对照 ComfyUI 源码 `comfy/ldm/minimax/{model,vae,audio_vae}.py`、`comfy/model_detection.py`、`comfy/quant_ops.py`、`comfy/ops.py`、`comfy_kitchen/tensor/int8.py` 确认加载/计算路径。
5. 抽样读取张量验证数值（INT8 值域、per-channel scale、NVFP4 三层 scale、LoRA alpha）。

关键命令：

```bash
cd /home/kaokaolv/comfyui_h3
.venv/bin/python analyze_h3_safetensors.py
python3 - <<'PY'   # 读取 13 个上游 shard header 求和
...urllib Range 请求...
PY
# 抽样验证
.venv/bin/python - <<'PY'
from safetensors import safe_open
...读取 blocks.0.* / model.layers.0.* / decoder.* / lora_* ...
PY
```

### 52.3 关键结论

- DiT：932 tensor，20.111 B 参数，20.970 GB；int8 占 95.80% 参数（19.268 B），bf16 3.97%，fp16 0.22%，fp32 0.01%。200 个 int8 层 = 50 block × 4 线性层；token_refiner、condition_proj、norm 为 bf16；AdaLN 为 fp16；patch/final head 为 fp32。
- **"pruned" = AdaLN 曲线低秩化**：上游 full 的 `adaln_proj [96768,2688]` + `time_embedder` → `[96768,8]` + `adaln_t_table [1025,8]`。上游 full 33.123 B / 66.28 GB，pruned 20.111 B / 40.23 GB，差 13.01 B 与 26.05 GB 对得上；50 层一层未删。
- TE：2054 tensor，25.753 B 参数，15.687 GB；50 层 Qwen3-VL（截断，无 final norm/lm_head）+ 27 层 ViT；350 个线性层 nvfp4（4.5 bit/参数，含 FP8 block scale），embedding int8 per-row，ViT bf16。`full_precision_matrix_mult=true` → 反量化 BF16 计算，无需 Blackwell。
- Video VAE：850 tensor，2.604 B 参数，2.811 GB；decoder 36 层 ViT 的 144 个线性层 int8（2.416 B），encoder 全部 fp16（0.180 B），bias/norm/proj fp32。
- Audio VAE：917 tensor，151.29 M 参数，0.605 GB，全 fp32；encoder 48.72%、decoder 42.86%、pre_block 8.37%。
- LoRA：624 tensor，978.06 M 参数，1.956 GB，全 bf16；rank 128（qkv 融合 384）、alpha 8（qkv 24），52 block × 4 模块；占 DiT 参数 4.86%。
- 文件完整性：`setup_comfyui_h3.sh` 在 revision `4b53a35…` 下载时已按 HF LFS 清单 `sha256sum --check`，`logs/setup_comfyui_h3.log` 全部 OK；本次复核文件大小与 HF API 一致。

### 52.4 交付与验证

- 新增分析脚本：`analyze_h3_safetensors.py`
- 新增分析文档：`minimax_h3_comfyorg_files_analysis.md`（初版 536 行；追加 §10 证据来源后 714 行 / 40,590 字节）
- 文档结构自检：所有 §8.2 oid 与 `setup_comfyui_h3.sh` 清单一致；参数总量与 `analyze_h3_safetensors.py` 输出一致（DiT 20,111,438,744；TE 25,753,095,920；video VAE 2,603,871,080；audio VAE 151,287,320；LoRA 978,059,264）。
- 未修改 ComfyUI tracked 文件，未启动/停止服务，未跑 GPU 推理。

### 52.5 备注

- 分析期间曾启动 5 个文件的全量 sha256（41 GB），因磁盘 I/O 过慢（`sha256sum` 长期 D 状态）中止；改用 setup 日志里的下载时校验记录，并复核了文件大小。临时输出 `.tmp/local_sha256.txt` 已删除。
- 下载的 config 缓存在 `.tmp/h3cfg/`，供文档引用。

### 52.6 用户追问：结论来源

用户追问“几个值得注意的结论”的依据。已把每条结论拆成三类证据（实测 / 文档 / 推断）并追加为分析文档 §10：

- §10.1 `pruned` 证据链：本地 header、上游 13 shard header、参数量/文件大小对账、`model.py:525` 注释与 `model_detection.py:421-425`；
- §10.2 `int8_convrot` = W8A8：`comfy_quant` blob、`quant_ops.py:238-243`、`comfy_kitchen/backends/cuda/__init__.py` 的 `int8_linear` convrot 分支；
- §10.3 nvfp4 存储量化：`full_precision_matrix_mult=true`、`ops.py:1160`、`pre_quant_scale` 的 AWQ 用法、model card 原文；
- §10.4 量化覆盖面：按 `.weight` dtype 普查；
- §10.5 LoRA：文件 metadata、alpha 张量实测值、A/B 形状、模块计数；
- §10.6 文件完整性：setup 脚本清单 + 日志 + HF API；
- §10.7 明确唯一属于“推断”的是 `pruned` 一词的语义，其余均为实测或源码/文档明文。

文档现为 714 行、40,590 字节；代码围栏 24 个（成对）。

### 52.7 用户反馈：§3.2/§3.3 的 pre_quant_scale 不一致与“weight std”

用户指出三处问题：

1. §3.2 的每层表只列了 `o_proj.pre_quant_scale`、`down_proj.pre_quant_scale`，但 §3.3 用 `q_proj` 举例时却列了 `pre_quant_scale`；
2. §3.3 没有解释为什么只有 `o_proj` / `down_proj` 有 `pre_quant_scale`；
3. “与权重 std 吻合”这句话里的“权重 std”含义不清、且未验证。

核对与修正：

- 实测 `pre_quant_scale` 键：全文件 100 个（50 层 × 2），模块名只有 `self_attn.o_proj` 和 `mlp.down_proj`；`q/k/v/gate/up` 均无。
- 上游源仓库 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` 第 1 个 shard（layer 0..8）同样只有 `o_proj`（9 个）与 `down_proj`（8 个）共 17 个，说明这是 modelopt NVFP4_AWQ 的导出层级，不是 Comfy-Org 转换新增。
- §3.3 示例改为 `self_attn.o_proj`（同时具备 weight/block scale/tensor scale/pre_quant_scale 四个 side-car），并补充 `pre_quant_scale` 的分布、作用（`ops.py:1618-1620` 乘到输入）与源仓库对照。
- “与权重 std 吻合”：原意是反量化后权重的标准差，但文件内没有 BF16 原权重，无法只凭 scale 证明逐值一致；已删除该断言，改为陈述可测量的 scale 算术（block scale × scale_2 = 每 16 权重的步长，o_proj row 0 约 8.8e-4..5.9e-3）。
- 复核时用 Qwen3-VL-32B-Instruct BF16 的 q_proj 行做过对照：反量化值与 BF16 行高度相关，但不是逐值相等；该对照只用于确认 pack 顺序（hi_first=True），未作为精度结论写入文档。
- 文档新增 §10.3(e) 记录这次修正；当前 726 行 / 42,852 字节，代码围栏 24 个（成对）。

### 52.8 追问：DiT/ViT 全称与重新量化的 base 来源

**全称**

- DiT = Diffusion Transformer（扩散 Transformer）。此处指 MiniMax H3 的 video+audio 扩散主干（`MiniMaxH3DiTModel`，50 层，hidden 5376）。
- ViT = Vision Transformer（视觉 Transformer）。文档里有两处：视频 VAE 的 36 层 3D ViT decoder（`ViT3DDecoder`），以及 Qwen3-VL 文本编码器里的 27 层 visual tower（`visual.*`）。

**重新量化的 base（均已核对 header/大小）**

| 目标文件 | BF16/FP16 base | 位置 | 规模 |
|---|---|---|---|
| DiT pruned int8 | pruned_bf16 | `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40.23 GB / 20.111 B，532 tensor |
| DiT 完整版 | bf16 | `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 66.28 GB / 33.123 B，535 tensor |
| DiT 上游原始 | Diffusers shards | `MiniMaxAI/MiniMax-H3/FL2VA/transformer/*`（13 shards） | 66.28 GB，含 time_embedder |
| TE nvfp4 | bf16（50 层截断版） | `Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51.51 GB / 25.753 B，902 tensor |
| TE 原始完整 | 64 层 Qwen3-VL | `Qwen/Qwen3-VL-32B-Instruct` 或 `MiniMaxAI/MiniMax-H3/FL2VA/text_encoder` | 66.71 GB，64 层 + lm_head |
| TE nvfp4 的直接来源 | NVFP4_AWQ 权重 | `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` | 21.861 GB，5 shards，64 层 + lm_head + visual，448 个 input_scale |
| Video VAE int8 | fp16 | `Comfy-Org/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors` | 5.21 GB / 2.604 B，562 tensor |
| Video VAE 上游 | fp32 | `MiniMaxAI/MiniMax-H3/FL2VA/video_vae/source/model.safetensors` | 10.42 GB |
| Audio VAE fp32 | 未量化（本身即 base） | `Comfy-Org/MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors`；上游 `FL2VA/audio_vae/model.safetensors` | 605 MB / 151.3 M |
| LoRA | 不量化；merge 到 bf16 base | base 见 LoRA metadata（`minimax_h3_fl2va_bf16`）；训练源 `lightx2v/Minimax-h3-Turbo` | 1.96 GB / 978 M |

**注意**

- pruned 与 full 的差异是 AdaLN rank-8 曲线蒸馏；复现 int8 文件用 pruned_bf16，量化完整模型用 full bf16。
- Comfy-Org 的 TE bf16 已做 50 层截断（0..49）、去掉 lm_head/model.norm、保留 visual；上游 Qwen3-VL-32B 是 64 层 + lm_head，需要自行截断与 key 映射（上游 key 为 `model.language_model.*`，Comfy-Org 为 `model.layers.*`）。
- 现有 nvfp4 TE 的直接来源是 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4`（model card 原文；实测 2530 tensor / 21.861 GB / 64 层 / lm_head + visual / 448 input_scale）；Comfy-Org 在其基础上做 64→50 截断、去 lm_head/model.norm、key 改名、去掉 input_scale。若要复现该 nvfp4 文件，cybermotaz 是权重来源；若要从 BF16 从零重量化，用 Comfy-Org 的 50 层 bf16 更省事。详见文档 §3.5。
- Ref2VA 也有对应 base：Comfy-Org `minimax_h3_ref2va_bf16` / `pruned_bf16`，上游 `Ref2VA/transformer`（同样 50 层、time_embed_dim 2688）。
- ComfyUI 侧量化格式对应 `comfy.quant_ops`（int8_convrot / nvfp4），TE 的 NVFP4 AWQ 原始由 NVIDIA modelopt 0.40.0 导出。

### 52.9 文档复核：对照 CUPTI / kernel 运行记录

用户要求把“重新量化的 base 来源”补进分析文档，并用本目录的真实运行记录 `comfyui_h3_t2v_kernel_analysis.md` 与 `cupti_h3_t2v_kernel_report.md` 校验文档正确性。

**先做的一致性核对**

- `kernel_analysis` §1.1 的 4 个模型文件字节数（TE 15,687,142,551；DiT 20,970,379,616；video VAE 2,811,065,184；audio VAE 605,254,808）与本机文件完全一致，确认运行记录分析的就是本文档这 4 个文件。
- 运行记录：B200（sm100）、ComfyUI v0.45.0、20 steps、864×480×124 帧、turbo_mode=false、0 patches。

**实测确认**

- DiT：4000 次 INT8 GEMM（20×50×4）+ 4000 次在线 ConvRot；1000 次 BF16 FlashAttention；200 个 int8 层；与文档 §2 一致。
- TE：350 次 NVFP4 反量化（50×7）+ 350 次 FP32 GEMM + 50 次 FP32 attention + 200 次 FP32 layer_norm；与文档 §3 的 50 层×7 结构一致。
- Video VAE：42 次 ViT × 36 层 × 4 = 6048 次 INT8 GEMM；1512 次 `qk_int_sv_i8_attn_kernel`；与文档 §4 的 36 层/144 个 int8 线性层一致。
- Audio VAE：2,679 个 FP32 kernel，无 INT8/NVFP4；与文档 §5 一致。

**发现并修正的错误**

1. TE 计算 dtype：初版写“反量化成 BF16 后 F.linear”。CUPTI 实测是 `dequantize_nvfp4_kernel<...,float>` + `cutlass3x_sm100_simt_sgemm_f32`，即 FP32。已改 §0、§3.3、§7.3、§9、§10.3、§10 表格。
2. Video VAE attention：初版只写了 144 个 int8 线性层，未提 attention 也是 INT8。CUPTI 实测 1512 次 `qk_int_sv_i8_attn_kernel`，已在 §4.2 补注。
3. 参考文档 `kernel_analysis` §3.2/§7.4 写的是 `TimeEmbedder(256→5376→2688)`；但同一 DiT 文件没有 `time_embedder.*`，而是 `adaln_t_table [1025,8]` + `[96768,8]` AdaLN。`model_detection.py:421-425` 与 `model.py:524-527` 证明走 `use_adaln_curves` 分支。该差异不影响 CUPTI 的 4000 次 INT8 计数（AdaLN 不属于 200 个 int8 层），但参考文档这两段应视为沿用了 full 模型描述。已在分析文档 §12.2 记录。

**新增内容**

- §11 重新量化的 base 来源（五个文件）：DiT pruned/full、TE 50 层 BF16 / cybermotaz NVFP4 / 原始 Qwen、video VAE fp16/fp32 source、audio VAE fp32、LoRA 合并 base。
- §12 与真实运行记录的交叉校验：12.1 确认表、12.2 修正项、12.3 未覆盖范围（LoRA 未加载、两个 VAE encoder 未执行、Ref2VA 未跑、B200 trace 的 GPU 边界）。

**逐表复算又修正的内部不一致**（写入文档 §12.4）：

- §2.4 功能归类表重复统计 AdaLN（0.072 B/0.36% → 0.029 B/0.14%，AdaLN 归 blocks）；
- §3.2 层参数把 pre_quant side-car 计入（24.382 B → 24.380 B），并澄清 33,792 是每层元素数、全文件 1,689,600 个；
- §2.5 fc1 scale 范围误用 qkv 的实测值（0.0006..0.0071 → 2.37e-4..7.89e-3）；
- §4.3 decoder bias/scale 行少算 norm/scale（0.0014 B/2.9 MB → 0.0022 B/8.85 MB）；
- §6.3 LoRA 表混用 50-block 与 52-block 统计并重复 token_refiner（改为四类 52-block 合计：qkv 536.74 M、fc1 226.62 M、fc2 131.20 M、out 83.49 M）；
- §9 视频 VAE 92.9% → 92.8%。

同时修正 `analyze_h3_safetensors.py`：`.alpha` 只在 LoRA 里当 side-car，Audio VAE 的 Snake alpha 计入参数；修正后五个文件的总参数为 20,111,438,744 / 25,753,095,920 / 2,603,871,080 / 151,287,320 / 978,059,264，与文档一致。

文档现为 862 行 / 57,738 字节，代码围栏 24 个（成对）。

### 52.10 追问：复现 nvfp4 权重时 AWQ scale 怎么办

用户问：文档说 cybermotaz 是 nvfp4 的直接来源，那 AWQ 的 scale 怎么处理。

核对结果（实测）：

- `pre_quant_scale`（AWQ 输入侧平滑，BF16，仅 o_proj/down_proj）：**源仓库与 Comfy-Org 文件逐位相同**。抽查 `layers.0.self_attn.o_proj.pre_quant_scale`，`torch.equal` 为 True（范围 0.293..3.406、mean 2.045）。
- `input_scale`（ModelOpt 静态激活 scale，F32 标量）：源仓库 448 个（64 层 ×7），Comfy-Org 文件 0 个；ComfyUI 的 `full_precision_matrix_mult` 路径不用它，ModelOpt/vLLM 路径才需要。
- `weight_scale` / `weight_scale_2`（NVFP4 block/tensor scale，不是 AWQ scale）：`weight_scale_2` 抽查相同（5.958e-4）；但 `weight` 的 U8 字节顺序和 `weight_scale` 的数值/顺序与源不同。
  - 实测 q_proj row0：源与目标的 4-bit code 直方图完全一致（同一批 code 的排列），但 U8 字节不同；block scale 多重集不同（源有 14.0，目标有 13.0）。
  - `to_blocked`/`from_blocked` 不能直接复现差异，说明 Comfy-Org 的转换还包含 block 分组重排 + block scale 重算，不是简单 swizzle。
- 结论：cybermotaz 是“量化权重来源”，不是“可直接拷贝的 ComfyUI 布局”。复现现有 nvfp4 文件最稳的是以 Comfy-Org 的 nvfp4 文件本身为 base；从 BF16 重做则用 Comfy-Org 的 50 层 bf16 重新量化，AWQ 的 `pre_quant_scale` 可复用（同 base/同层）或重新校准，`weight_scale`/`weight_scale_2` 必须由新量化器重新生成。

已更新分析文档 §3.5（AWQ scale 与 block scale 处理）和 §11 note 3（cybermotaz 不能直接拷贝）。文档现为 870 行 / 59,384 字节。


---

# 独立校验 `minimax_h3_comfyorg_files_analysis.md`

## 53. 任务、边界与最终结论

用户要求校验 `minimax_h3_comfyorg_files_analysis.md` 的内容。该文件此前已有第 52 节的编写/修订记录，但本轮不沿用“已校验”的结论，重新对照本地 safetensors header、HF API/model card、上游 config 和必要的小范围张量数据。

本轮边界：

- 不修改被校验的原文，不修改 ComfyUI/comfy-kitchen 源码。
- 不下载完整大模型、不启动/停止服务、不运行新的 GPU 推理。
- 远程权重访问只用 HTTP Range 读取 header、scalar、scale 或单个 q_proj tensor。
- CUPTI 部分复用此前已经验证的 `logs/cupti_h3_t2v/h3_t2v_cold.sqlite` 和分析报告，没有重复采集。
- 完整过程写入本节，最终给用户错误清单和正确性边界。

最终结论：**文档主体结构、五个文件的静态统计和运行时量化结论大部分正确，但不能原样视为完全正确。** 两处需要实质修正：

1. cybermotaz → Comfy-Org 的 NVFP4 转换被错误解释为“重新计算 block scale、不能由布局重排复现”。完整 q_proj 实测证明：packed 权重仅高低 nibble 互换，block scale 仅标准 128x4 重排，二者都逐字节吻合。
2. `pruned` 的分项参数差和 time_embedder 参数有算术错误。准确的 full/pruned 总差是 13,011,554,168，逐组对账后剩余差为 0。

另有 license 过度概括、音频配置来源不精确、TE full-precision 引用了错误函数、少量 dtype 字节归属问题，详见 53.7。

## 53.1 本地五个文件的独立复算

先读 `analyze_h3_safetensors.py:1-157`。它只读 8-byte header length、JSON header 和小 `comfy_quant` blob，量化权重按 `weight_params()` 的逻辑 shape 计数；`.alpha` 只在 LoRA 中作为 side-car，Audio VAE 的 Snake alpha 不扣除。没有整模型 tensor 加载行为。

实际执行：

```bash
cd /home/kaokaolv/comfyui_h3
/home/kaokaolv/comfyui_h3/.venv/bin/python analyze_h3_safetensors.py
```

五个文件的全部汇总与原文一致：

```text
DiT:       20,970,379,616 B;  932 tensors; 20,111,438,744 params
TE:        15,687,142,551 B; 2054 tensors; 25,753,095,920 params
Video VAE:  2,811,065,184 B;  850 tensors;  2,603,871,080 params
Audio VAE:    605,254,808 B;  917 tensors;    151,287,320 params
LoRA:       1,956,193,000 B;  624 tensors;    978,059,264 params
```

量化计数也一致：DiT 200 × int8/ConvRot；TE 350 × NVFP4 full-precision-mm + 1 × int8 embedding；Video VAE 144 × int8/ConvRot；Audio VAE 全 F32；LoRA 全 BF16 + 208 个 alpha。

随后独立使用 `weight_params()` 再算总参数，并验证 header 长度，避免直接信任汇总脚本：

```bash
python3 - <<'PY'
import math, os, struct
from analyze_h3_safetensors import load, weight_params, module_of
expected = {
    'diT': (932, 95416, 20111438744),
    'TE': (2054, 231400, 25753095920),
    'video_vae': (850, 99512, 2603871080),
    'audio_vae': (917, 105520, 151287320),
    'lora': (624, 73632, 978059264),
}
for key, (count, header_bytes, params) in expected.items():
    path, hdr, meta, qinfo = load(key)
    data_start = 8 + struct.unpack('<Q', open(path, 'rb').read(8))[0]
    total = 0
    for name, info in hdr.items():
        if name.endswith(('.weight_scale', '.weight_scale_2', '.pre_quant_scale', '.comfy_quant')):
            continue
        if key == 'lora' and name.endswith('.alpha'):
            continue
        total += weight_params(module_of(name), info, qinfo)
    assert len(hdr) == count
    assert data_start - 8 == header_bytes
    assert total == params
    print(key, 'header_and_params_ok', os.path.getsize(path), data_start)
PY
```

实际五个 `header_and_params_ok` 均通过。这里第一次检查脚本曾错误把 TE 所有非 visual `.weight`（包括 RMSNorm）都乘以 2，得到 25,753,620,720；该错误属于本轮检查脚本，不是原文。改用已有的 `weight_params()` 后恢复正确结果，未用这个错误结果评价原文。

分组参数也吻合：

```text
DiT blocks=19,311,680,000; token_refiner=770,730,752;
condition_proj=27,530,496; video_patch=521,472; audio_patch=177,408;
final_layer=790,400; adaln_t_table=8,200; rope=16。

TE model.layers=24,380,441,600; visual=595,266,800;
embed_tokens=777,912,320。三层合计 25,753,095,920。

Video VAE decoder=2,423,532,544; encoder=180,335,536;
post_quant_conv=600; quant_conv=2,352; 两个 latent 统计向量各 24。

Audio encoder=73,696,832; decoder=64,850,424;
pre_block=12,670,304; dec_in_proj=67,584;
mean_proj/logs_proj 各 1,056；latent mean/std 各 32。

LoRA qkv=536,739,840; out=83,492,864;
fc1=226,623,488; fc2=131,203,072。
```

抽样数值：

```text
DiT qkv scale: min=0.0006494268891, max=0.0070543666370
DiT fc1 scale: min=0.0002368853893, max=0.0078861992806
VAE qkv scale: min=0.0002093392104, max=0.0010878733592
TE o_proj pre_quant_scale: min=0.29296875, max=3.40625,
                           mean=2.0445358753204346
LoRA alpha: {8.0: 156, 24.0: 52}
```

这些值与原文给出的四舍五入范围一致。

## 53.2 HF 文件、hash 与 model card

读取 `setup_comfyui_h3.sh:117-139` 和 `logs/setup_comfyui_h3.log:283-298`：固定 revision 为 `4b53a35191437de05370da548e59aeabe2a5a09a`，下载后执行 `sha256sum --check`，日志确有五个 `OK`。

通过 FetchURL 读取 https://huggingface.co/Comfy-Org/MiniMax-H3 ，model card 确实列出四个原始来源、cybermotaz 的 NVFP4 来源、无需 Blackwell 的说明以及 int8 ConvRot 的优先级建议。

FetchURL 对 `/api/models/Comfy-Org/MiniMax-H3?blobs=true` 报“无法提取 meaningful content”。这是 JSON 正文提取限制，不是权限错误。随后改用公开 API 的 Python 只读请求筛选需要的文件，不经过登录、不读取凭据：

```bash
python3 - <<'PY'
import json
from urllib.request import Request, urlopen
repos = (
    'Comfy-Org/MiniMax-H3',
    'MiniMaxAI/MiniMax-H3',
    'cybermotaz/Qwen3-VL-32B-Instruct-NVFP4',
    'Qwen/Qwen3-VL-32B-Instruct',
    'lightx2v/Minimax-h3-Turbo',
)
for repo in repos:
    req = Request(
        f'https://huggingface.co/api/models/{repo}?blobs=true',
        headers={'User-Agent': 'h3-document-verification'})
    with urlopen(req, timeout=45) as r:
        info = json.load(r)
    card = info.get('cardData') or {}
    print(repo, info.get('sha'), card.get('license'),
          card.get('license_name'), card.get('license_link'))
    for f in info.get('siblings', []):
        if f['rfilename'].endswith('.safetensors'):
            # 对 Comfy-Org 输出本文涉及的 8 个目标文件；
            # 对其他仓库累加 safetensors size。
            pass
PY
```

实际输出确认：

- 当前 Comfy-Org revision 与 setup 固定 revision 完全一致。
- 五个本地文件的大小和五个 LFS sha256 都与原文/脚本清单一致。
- `pruned_bf16`：40,225,724,176 B，oid `a32572fb...d541b4`。
- TE BF16：51,506,295,256 B，oid `600d567f...0607d`。
- Video VAE FP16：5,207,808,496 B，oid `7c1f1314...ce5e522`。
- cybermotaz 五个 shard 总字节 21,861,337,872 B。
- Qwen/Qwen3-VL-32B-Instruct  safetensors 总字节 66,714,912,872 B。

本轮没有重新计算约 41 GB 的本地全量 sha256。原文 §10.6 把完整性限定为“下载时校验 + 当前大小一致”，这个证据边界是准确的，不能把它升级表述为本次又做了一次全量 hash。

## 53.3 上游 config 与 base header

通过 FetchURL 读取：

```text
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/transformer/config.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/text_encoder/config.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/video_vae/config.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/video_vae/source/config.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/audio_vae/config.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/audio_vae/config.yaml
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/FL2VA/audio_vae/metadata.json
https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/Ref2VA/transformer/config.json
https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4/resolve/main/config.json
```

结果：

- DiT hidden=5376、50 层、56×128 heads、ffn=14336、time_embed_dim=2688、AdaLN 输出 96768，正确。
- Qwen text_config hidden=5120、64 层、64 Q heads、8 KV heads、intermediate=25600；vision 27 层、1152、16 heads、4304、patch 16、merge 2，正确。
- Video VAE 的 source config、36 层 ViT、32×64、16×/4× downscale 等正确。
- Audio `config.json` 有 32 kHz、32 latent channels、2 声道；`metadata.json` 的 kwargs 与本地 safetensors 中的 `minimax_h3_audio_vae.kwargs` 完全一致，包括两组 rates 和 BigVGAN。
- **上游 `config.yaml` 实际只有 `model_config.sr/decoder_dim/audio_channel/vae_latent_channels` 四项。** 原文 §1.2 把 encoder_rates、decoder_rates、encoder_dim 等笼统归到 `config.json + config.yaml`，证据归属不准确；应引用 `metadata.json`、本地 safetensors metadata 和 ComfyUI `audio_vae.py`。
- Ref2VA config 也确实是 50 层、time_embed_dim=2688。
- cybermotaz `config.json` 确有 `quant_algo=NVFP4_AWQ`、modelopt 0.40.0、`ignore=[lm_head,model.visual*]`。请求的 `hf_quantizer.json` 返回 404，不影响已存在的 `config.json` 证据；没有声称该文件存在。

使用 HTTP Range 只读三份推荐 base 和两个上游 VAE 的 header。`Range: bytes=0-1048575`，仅解析 8B 长度和 JSON header，结果：

```text
pruned_bf16: 532 tensors; 20,111,438,744 params;
             BF16/F16/F32 混合，无 time_embedder。
TE BF16:     902 tensors; 25,753,095,920 params; 50 层;
             无 lm_head/model.norm，metadata=minimax_h3_te。
Video FP16:  562 tensors;  2,603,871,080 params; 全 F16。
Video source:560 tensors;  2,603,871,032 params; 全 F32。
Audio source:1087 tensors; 151,326,585 params; 全 F32。
```

TE BF16 与 NVFP4 文件做了逐 key 对照：非 side-car key 都是 902 个，差集为空；恰好 350 个 Linear 变为 packed half-width，其他 key shape 完全相同。原文“50 层 BF16 是方便的重新量化 base、key 一一对应”成立。

Video source 比 FP16 base 少 `latents_mean/std` 两个向量（48 个元素），结构相符。Audio source 是 1087 tensors，而 Comfy-Org 文件是 917 个，不能把“上游是 base”误解为已经证明逐 key/逐值相同；原文没有明确声称相同，但后续若直接转换上游音频，仍需单独做 key 映射和数值验证。

## 53.4 `pruned` 精确对账：主结论正确，分项算术错误

先按文档 shape 直接计算：

```bash
python3 - <<'PY'
N, D, R = 96768, 2688, 8
block_delta = N * (D - R)
final_delta = 10752 * (D - R)
time_embedder = 5376 * 256 + 5376 + 2688 * 5376 + 2688
print('full_block_adaln_weight', N * D)
print('pruned_block_adaln_weight', N * R)
print('per_block_delta', block_delta)
print('50_block_delta', 50 * block_delta)
print('final_delta', final_delta)
print('time_embedder_including_bias', time_embedder)
print('table', 1025 * 8)
print('total_delta', 50 * block_delta + final_delta + time_embedder - 1025 * 8)
PY
```

正确结果：

```text
full block AdaLN weight       260,112,384
pruned block AdaLN weight         774,144
per-block delta               259,338,240   # 原文写 259,272,704，错误
50-block delta             12,966,912,000   # 原文写 12,963,635,200，错误
final AdaLN delta              28,815,360
TimeEmbedder incl. bias        15,835,008   # 原文写 15,835,584，错误
adaln_t_table                      -8,200
总差                       13,011,554,168
```

随后读取 `FL2VA/transformer/model.safetensors.index.json`，对 13 个 shard 逐一发 Range header 请求。关键脚本逻辑（每个 shard 都只读 8B 长度和 JSON，pop `__metadata__`，合并去重，再与本地 DiT 逐 key 比对）：

```python
index = json.loads(fetch_bytes(BASE + 'model.safetensors.index.json'))
full = {}
for shard in sorted(set(index['weight_map'].values())):
    status, n, hdr = fetch_header(BASE + shard)  # HTTP Range
    hdr.pop('__metadata__', None)
    assert not (set(full) & set(hdr))
    full.update(hdr)
full_total = sum(math.prod(v['shape']) for v in full.values())
# 本地 int8 每元素仍是一个逻辑参数，仅排除 weight_scale/comfy_quant。
local_total = sum(math.prod(v['shape']) for k, v in local.items()
                  if not k.endswith(('.weight_scale', '.comfy_quant')))
```

13 个请求均返回 HTTP 206；JSON header 连同长度字段总计只读取 55,792 B。实测：

```text
full:  535 tensors, 33,122,992,912 params
local: 932 tensors, 20,111,438,744 params
actual_delta = 13,011,554,168

only_full = time_embedder.proj_in.{weight,bias},
            time_embedder.proj_out.{weight,bias}
only_local 共 401 个 = 200 weight_scale + 200 comfy_quant + adaln_t_table
shared shape change = 51 个，恰好 50 个 block AdaLN + final AdaLN；
                      其他 shared shape 无变化。

group_delta blocks         = 12,966,912,000
group_delta final_layer    =     28,815,360
group_delta time_embedder  =     15,835,008
group_delta adaln_t_table  =         -8,200
token_refiner/condition/video/audio patch/rope 的 delta 全为 0。
unexplained_delta = 0
```

因此应把原文“0.004 B 误差来自 bias/对齐”改成：**精确参数差 13,011,554,168，结构分项完全解释，剩余差为 0。**

主结论仍然成立：没有删除 50 个 transformer block；full-only key 只有 time_embedder 四个 tensor，shape 改变只在 AdaLN 投影。`rank-8 曲线蒸馏`的具体训练来源仍是“源码 + 数据推断”，不是 model card 明文，原文 §10 的推断标签应保留。

Comfy-Org `pruned_bf16` 的 532 tensors/20,111,438,744 params 也直接核对通过，与 int8 文件逻辑参数相同。

## 53.5 TE 上游转换：原文结论被完整 q_proj 反证

这是本轮最重要的实质修正。

### 源仓库结构核对

读取 cybermotaz 的 index 和 5 个 shard header：

```text
2530 tensors；64 层；448 input_scale；128 pre_quant_scale；
lm_head.weight 存在；model.language_model.norm.weight 存在。
q_proj weight:       U8 [8192,2560]
q_proj weight_scale: F8_E4M3 [8192,320]
```

与原文这些结构数字一致。

### 完整 packed q_proj 权重

读取源 `model.language_model.layers.0.self_attn.q_proj.weight` 的完整 20,971,520 B 和本地对应 tensor。远程访问为对单个 tensor 的 Range 请求，不是下载 21.86 GB 模型。核心代码：

```python
info = source_header['model.language_model.layers.0.self_attn.q_proj.weight']
lo, hi = info['data_offsets']
source = remote_range(8 + header_len + lo, 8 + header_len + hi - 1)
target = read_blob(local_path, local_data_start,
                   local_header['model.layers.0.self_attn.q_proj.weight'])
assert len(source) == len(target) == 8192 * 2560
swapped = bytes((v >> 4) | ((v & 15) << 4) for v in source)
print('raw_equal', source == target)
print('nibble_swap_equal', swapped == target)
print('nibble_swap_mismatch_count', sum(a != b for a, b in zip(swapped, target)))
```

实际输出：

```text
q_proj_packed_bytes 20971520
raw_equal False
nibble_swap_equal True
nibble_swap_mismatch_count 0
```

即：**源和目标包含完全相同的 4-bit code，只是每 byte 内高低 nibble 顺序不同。** 不需要重新量化。

### 完整 q_proj block scale

读取源/目标全部 `[8192,320]` scale，共 2,621,440 B。按 vLLM/CUTLASS 同样语义计算：

```python
R, G = 8192, 320
blocked = bytearray(len(source_scale))
mt_count, kt_count = (R + 127) // 128, (G + 3) // 4
for r in range(R):
    mt, outer, inner = r // 128, r % 32, (r // 32) % 4
    for g in range(G):
        kt, kg = g // 4, g % 4
        out_index = ((((mt * kt_count + kt) * 32 + outer) * 4 + inner) * 4 + kg)
        blocked[out_index] = source_scale[r * G + g]
print(bytes(blocked) == target_scale)
```

实际结果：

```text
scale_raw_equal False
scale_128x4_relayout_equal True
source_scale_unique_count = target_scale_unique_count = 69
scale_histogram_equal True
```

因此：**目标 scale 是源 scale 的标准 128x4 重排，数值集合与每个值的出现次数全部一致。** 目标第一行 raw scale 与源第一行不同，是重排导致取自其他源位置，不是 scale 被重新计算。

此外：

- `weight_scale_2` 原始 4 bytes 完全相同，值均为 `0.0005958193796686828`。
- `o_proj.pre_quant_scale` 全部 16,384 bytes 完全相同，AWQ pre_quant 确实保留。
- 原文 §3.5、§11 note 3 中“block scale 多重集不同、重新计算、to_blocked/from_blocked 不能复现”的概括与完整 q_proj 数据矛盾，必须修正。

正确表述应为：

> cybermotaz 可以作为量化权重来源。至少对完整 q_proj 已证明，Comfy-Org 的转换是 key/截断处理、4-bit nibble 顺序转换和 block scale 的 128x4 重排，q value 与 scale 数值不变，不需要重新量化或重新校准。若要直接裸拷贝 byte 仍不可行，因为 packing/layout 不同。

证据边界：本轮对 q_proj 的整个 qdata 和整个 scale 做了完整证明，对 o_proj 的 pre_quant 做了完整证明；**没有下载并逐值比较全部 350 层**，所以不能进一步声称全部层已经完成同样的逐值校验。但完整 q_proj 已足以否定原文对整份文件的“scale 重算”概括。

### 本轮行截取错误及处理

最初尝试只读 q_proj row 0 时，脚本误写成 `hi = lo + hi // shape[0]`，没有先减 `lo`，远程读到了 395,257 B 而非一行 2,560 B。该轮的 nibble histogram 不相等结果已丢弃，未用于结论。正确公式为 `hi = lo + (hi-lo)//shape[0]`；随后进一步读取完整 q_proj tensor 做最终证明，避免只凭单行直方图推断。

## 53.6 LoRA、运行链和 base 来源复核

- LoRA 624 tensors = 416 A/B + 208 alpha；alpha 8.0×156、24.0×52；四类参数精确合计 978,059,264，和文档一致。metadata 的 base、rank、scale、QKV block diagonal、SwiGLU 顺序均与原文一致。
- 参考运行未加载 LoRA；文档已明确其运行时边界，没有误把静态分析当成 LoRA kernel 证据。
- 之前已验证的 CUPTI 数据仍支持：DiT 4000 INT8 GEMM + 1000 BF16 FlashAttention；Qwen 350 NVFP4 dequant + 350 FP32 GEMM + 50 FP32 attention；Video VAE 6048 INT8 GEMM + 1512 Sage INT8 attention；Audio VAE 2679 个 FP32 相关 kernel。
- 两个 VAE encoder、LoRA 和 Ref2VA 不在该 T2V trace 中，原文 §12.3 已正确声明，本轮没有扩大运行时覆盖范围。
- `comfyui_h3_t2v_kernel_analysis.md:269,307,865` 仍把当前 pruned checkpoint 描述为 `TimeEmbedder`，与本文件实际 `adaln_t_table` 分支不符；被校验文档 §12.2 指出这一跨文档问题是正确的。本轮没有擅自修改那份旧文档。
- DiT `full_precision_mm`、视频 VAE residual、Sage INT8 attention 和音频 FP32 的既有源码/CUPTI 结论未发现新问题。

## 53.7 其余需要修正或加注的地方

### A. TE full-precision 的源码引用不准确

原文 §3.3、§10.3 用 `ComfyUI/comfy/ops.py:1160` 的 `_v_weight` 代码证明 TE 反量化。该行实际属于 `CombinedLinear.linear_input_act`；`CombinedLinear.prepare` 在 `ops.py:1136-1140` 排除非 INT8 系列格式，不能拿它直接代表 NVFP4 文本编码器。

TE 的正确直接证据是：

- `ComfyUI/comfy/sd1_clip.py:109-115`：文本编码器量化默认 `full_precision_mm=True`。
- `ComfyUI/comfy/ops.py:1381-1385`：读取 checkpoint 的 full-precision flag。
- `ComfyUI/comfy/ops.py:1597-1599`：`MixedPrecisionOps.Linear.forward_comfy_cast_weights` 在 `_full_precision_mm` 时 `weight.dequantize()` 后 `_forward`。

CUPTI 已经独立确认实际 FP32 路径，所以这是**引用函数错误，不是 FP32 结论错误**。

### B. TE 磁盘构成的标签不精确

原文 §3.4 的 U8 字节数 `12,189,715,279` 实际是：

```text
350 个 packed weight = 12,189,696,000 B
351 个 comfy_quant   =         19,279 B
合计                 = 12,189,715,279 B
```

因此“U8（350 个 NVFP4 权重）”的标签漏掉了 comfy_quant 元数据；总数没错，权重和元数据应拆开。

原文 F32 `609,144 B` 实际是：

```text
350 个 weight_scale_2 标量      =   1,400 B
embedding 的 151,936 个 F32 scale = 607,744 B
合计                              = 609,144 B
```

不能只标为“350 个 tensor scale”，应补充 embedding per-row scale。上述不影响 4.87 bit/参数的有效位宽结论。

### C. License 概括错误

原文 §11 note 7 写“这些 base 均在 MiniMax-H3 Community License 下”。HF API 实际为：

- Comfy-Org/MiniMax-H3、MiniMaxAI/MiniMax-H3、lightx2v/Minimax-h3-Turbo：MiniMax H3 Community License。
- Qwen/Qwen3-VL-32B-Instruct：`license=apache-2.0`。
- cybermotaz：`license=other`、`license_name=qwen`，并链接 Qwen LICENSE，不是 MiniMax license。

应改成“分别遵循各仓库 license；MiniMax 系列检查 MiniMax license，Qwen 及其派生权重检查 Qwen/Apache 等对应许可”，不能统一归为 MiniMax。

### D. Audio config 来源

如 53.3 所述，结构值本身正确，但 rates 等应引用 `metadata.json` / 本地 metadata / `audio_vae.py`，不是只有四行的 `config.yaml`。

### E. 不应过度推广的结论

- full/pruned 的文件大小差还受 shard header 数量和 FP32/BF16/F16 混合 dtype 影响，不能把“bf16 文件差”说成逐 byte 都仅由逻辑参数差决定。精确逻辑参数差已由 53.4 完整证明。
- 音频上游与 Comfy-Org 文件 tensor 数不同，后续直接转换时需另做映射。
- LoRA、两个 encoder、Ref2VA 仍无本轮运行时证据。

## 53.8 工具/脚本问题与解决过程汇总

1. 首次合并上游 shard header 把每个文件都有的 `__metadata__` 当成重复 tensor，触发 `AssertionError: {'__metadata__'}`。修正为 `hdr.pop('__metadata__', None)` 后成功，535 个真实 tensor 无重复。
2. 初次远程 q_proj row 截取公式少了 `-lo`，读出 395,257 B；丢弃该轮 histogram，改读完整 q_proj 得最终零差异结论。
3. 初次本地 TE 汇总把 RMSNorm 也乘 2，导致多 524,800 参数；改用已有 `weight_params()` 后五个 header/参数断言全部通过。
4. FetchURL 无法提取 HF API 的 JSON，改用公开 API 的 Python `urlopen` 只读请求，不经过认证。没有访问任何 secret 文件。
5. `hf_quantizer.json` 返回 404；改由实际存在的 `config.json` 核实 modelopt 0.40.0/NVFP4_AWQ，未把 404 当作格式不存在。
6. 根目录宽泛 Glob 多次超时；改用已知文件的 Read、明确的远程 URL 和本地脚本。超时结果未用于证据。
7. `git status --short` 与 `git diff --check` 均无输出，ComfyUI tracked 源码未修改。
8. Python import 在本轮生成了 `__pycache__/analyze_h3_safetensors.cpython-312.pyc`，`stat` 确认是本轮临时产物，已删除；空目录已清理，没有删除其他文件。

## 53.9 交付边界

本轮没有修改 `minimax_h3_comfyorg_files_analysis.md`，因此以上是需要用户/后续编辑采纳的校验意见，不是已写回原文的修订。

已确认可信的主体：五文件大小、tensor/header 统计、逻辑参数、主要结构、量化覆盖、LoRA 参数、既有 CUPTI 精度结论和推荐的三份 BF16/FP16 base。

必须优先修正的原文位置：

- §2.3 / §10.1：`pruned` 的分项数字、time_embedder 和“bias/对齐误差”解释。
- §3.5 / §11 note 3：cybermotaz 的 qdata/scale 转换解释。
- §11 note 7：统一 license 的错误概括。
- §3.3 / §10.3：TE full-precision 的直接源码引用。
- §3.4：U8/F32 字节归属标签。
- §1.2：音频 `config.yaml` 的证据来源。


---

# 面向 Presentation 的汇总文档

## 54. 任务、材料和大纲定位

用户希望在已有技术分析与执行记录的基础上，完善一份用于 presentation 的文档：不要求像源码/kernel 报告那样展开全部细节，但要求介绍清晰、逻辑有条理、要点准确。指定参考：

- `comfyui_h3_t2v_kernel_analysis.md`
- `cupti_h3_t2v_kernel_report.md`
- `comfyui_h3_mxfp8_mxfp4_proposal.md`
- `minimax_h3_comfyorg_files_analysis.md`
- `run_book.md`
- `run_book_quant.md`

用户提到的 `presentation.md` 最初在 `/home/kaokaolv/comfyui_h3` 不存在。读取 `report.md` 后发现它正是用户描述的三段式大纲：

1. `video_minimax_h3_t2v.json` 的执行逻辑、涉及模型和源模型/BF16 模型；
2. 每个模型的函数调用栈、dense Linear、attention 和实际 kernel；
3. 用源模型支持 MXFP4/MXFP8，包括 auto-round model-free 量化、ComfyUI/FlashInfer 与 vLLM-Omni 两条路线及差异。

因此保留 `report.md` 原样，按此大纲**新建 `presentation.md`**。没有因为文件名缺失而向用户反复询问，也没有覆盖 `report.md`。

## 54.1 证据提炼与演示主线

采用以下叙事顺序：

```text
真实输出与执行流程
  → 四个基础模型及源模型/BF16 base
  → 各模型的函数链、dense Linear 和 attention 真实 kernel
  → CUPTI 对性能现象的解释
  → BF16/FP16 到 MXFP8/MXFP4 的量化策略
  → 原生 ComfyUI 与 vLLM-Omni 的路线对比
  → 分阶段验收建议
```

重点区分三类状态：

- **已实测跑通**：原生 ComfyUI 的 T2V、媒体输出、显存采样和 CUPTI kernel 分析。
- **已完成部分量化**：DiT 200 个 Linear 的 MXFP8 权重、文件格式和代表层反量化验证。
- **仍是设计/待验证**：MXFP8/MXFP4 真 GEMM、完整量化 T2V、Qwen MXFP4、视频 VAE MXFP8、vLLM-Omni 实际服务和 XPU 扩展。

没有将 proposal 中的接口或 FlashInfer 路线写成已部署功能。

## 54.2 写入的主要内容

### 执行流程与基线

- 纯文本输入，无首尾帧；`turbo_mode=false`，0 patches，未加载 LoRA。
- 一次 Qwen 编码和 token refiner，20 步联合音视频 DiT 去噪，视频/音频 VAE 解码，CreateVideo/SaveVideo 输出。
- 864×480、24 fps、124 帧、32 kHz 双声道；实际 packed sequence `[388 text | 414 audio | 14,985 video]`，S=15,787。
- 原始冷启动客户端总耗时 953.3 s；独立 CUPTI 的 DiT 第一步 host range 178.617 s，第 2–20 步约 1.16 s，明确两者不能混为稳定吞吐。
- 原始执行 NVML 峰值 38,570 MiB，出现在视频 VAE 解码阶段。

### 模型、来源与 pruned

- DiT 20.111B / 20.970 GB，base 为 40.226 GB 的 pruned BF16；50 层不变。
- Qwen 文件 25.753B / 15.687 GB，base 为 51.506 GB 的 50 层 BF16，含 visual tower 和 embedding。
- 视频 VAE 2.604B / 2.811 GB，base 为 5.208 GB 的 FP16 文件；T2V 只跑 decoder。
- 音频 VAE 0.1513B / 0.605 GB，FP32；T2V 只跑 BigVGAN decoder。
- 补充可选 8-step Turbo LoRA 的来源，但明确本次未加载。
- 使用第 53 节独立核验后的精确数字：full 33,122,992,912，pruned 20,111,438,744，差 13,011,554,168；不是删层，而是 AdaLN 时间分支从全宽输入变成 8 维曲线形式。
- 没有复制原文件中已经被反证的“NVFP4 scale 重算”说法，而是写明 q_proj 实测支持 nibble/scale 布局转换。

### 真实函数/kernel

用一张主表清晰对应模型、函数链、dense Linear 和 attention：

- Qwen：NVFP4 存储 → FP32 dequant/GEMM，350 次；FP32 mem-efficient attention 50 次。
- DiT：ConvRot + INT8 GEMM 4000 次；BF16 FlashAttention 1000 次。
- 视频 VAE：INT8 GEMM 6048 次；Sage INT8 attention 1512 次；42 次 ViT forward 来自时间/空间 tiling。
- 音频 VAE：FP32 Conv/Snake 等，2679 个 kernel，无 INT8/NVFP4。
- 用 34.030 s 的 DiT kernel work 说明 FlashAttention 约 51.9%、INT8 GEMM 约 36.5%，并明确这不是 wall-time 占比。
- 保留 allocation graph 不等于整模型 CUDA Graph replay、Qwen 不走 scaled_mm_nvfp4 等关键防误判结论。

### 量化与两条路线

- DiT 200 Linear、视频 VAE 144 Linear 从 BF16/FP16 转 MXFP8；Qwen 350 Linear 从 BF16 转 MXFP4；其他精度保持不变。
- 单独补充 INT8 embedding 保持现有格式，避免“INT8 模型全部转 MXFP8”的歧义。
- auto-round 初版使用 model-free 逐层量化，后续才考虑 tuning/AWQ/scale search。
- 原生路线保持 `.comfy_quant + QuantizedTensor + mixed_precision_ops`，由 Comfy-Kitchen 做候选选择、FlashInfer 提供 B200 MX GEMM；明确禁止静默回退。
- vLLM-Omni 使用远程 API 和官方远程 workflow；优先官方 BF16 + 在线 FP8，在线 MXFP8 是 H3 视频未验证实验路线；QAD 是不同 8-step 模型变体。
- 对比两条路线在 workflow、模型格式、推理位置、量化方式和当前状态上的差异，不把它们的采样参数或输出直接混比。
- 最后给出 B200 小矩阵、量化文件、垂直切片、同条件对照、CUPTI、综合指标、后续优化/XPU 的实施顺序。

## 54.3 最新量化结果和补充修订

初稿完成后重新读取 `run_book_quant.md`，发现新增第 3 节 INT8 ConvRot 与 BF16 base 的反量化对照。已把以下最新结果补入 presentation：

- INT8 ConvRot 主投影权重 rel_RMSE：0.897%–1.211%。
- MXFP8 主投影权重 rel_RMSE：2.661%–2.662%。
- 这些层的 INT8 权重 RMSE 约为 MXFP8 的 1/2.2–1/3。

特别注明：这只是**同一 BF16 base 的权重误差对比**，不是端到端画质、速度或真实 GEMM 验证结论。没有加入与 presentation 无关的 float32 cosine 累加异常细节。

最终 `run_book_quant.md` 的 mtime 为 `2026-11-17 07:02:51 -0800`，文件末尾仍为第 3.3 节；没有遗漏更新的完整 T2V/MXFP GEMM 结果，因为该记录仍明确写“未做完整 T2V”。

## 54.4 文件、链接和事实检查

实际执行：

```bash
python3 -B - <<'PY'
from pathlib import Path
import re
p = Path('/home/kaokaolv/comfyui_h3/presentation.md')
s = p.read_text()
assert s.count('```') % 2 == 0, 'unpaired code fence'
for title in ['## 一页结论', '## 1. ComfyUI workflow', '## 2. Flow 中的模型',
              '## 3. 函数调用逻辑与真实 Kernel', '## 4. 用源模型支持 MXFP8 / MXFP4',
              '## 5. 总结']:
    assert title in s, title
for value in ['15,787', '13,011,554,168', '95,226', '4000 次', '6048 次',
              '2.66%', '38,570 MiB', '953.3 s', '0.897%–1.211%', '2.661%–2.662%']:
    assert value in s, value
for value in ['尚未完成', '不是 FP4 GEMM', '不能原样执行当前 JSON',
              '禁止静默回退', '这是权重误差对比，不是端到端画质或速度结论',
              'INT8 embedding']:
    assert value in s, value
links = re.findall(r'\]\((\./[^)]+)\)', s)
for link in links:
    path = link.split('#', 1)[0]
    assert (p.parent / path).exists(), link
print('presentation_final_checks_passed')
print('lines:', len(s.splitlines()), 'bytes:', len(s.encode()),
      'local_links:', len(links), 'code_fences:', s.count('```'))
PY

git -C /home/kaokaolv/comfyui_h3/ComfyUI status --short
git -C /home/kaokaolv/comfyui_h3/ComfyUI diff --check
```

实际输出：

```text
presentation_final_checks_passed
lines: 331 bytes: 19586 local_links: 10 code_fences: 14
```

`git status --short` 和 `git diff --check` 均无输出。本轮只新增 `presentation.md` 并追加本记录，没有修改 ComfyUI 源码、`report.md`、proposal 或其他技术文档。没有启动/停止服务、运行新模型或重新采集 CUPTI。

最终交付：[presentation.md](./presentation.md)。文档保留大纲主体，增加一页结论、清晰的模型/来源表、函数/kernel 对照、实测性能解释、两条 MXFP 推理路线和分阶段验收建议。


---

# Presentation 补充：其它模型 Kernel 构成与 INT8 反量化误差

## 55. 任务与证据来源

用户要求在 `presentation.md` 第三节增加其它模型的 kernel 构成，形式参照“3.3 DiT：kernel work 的构成”；并在 4.3 节增加 `int8_convrot` 反量化误差分析表，参考 `minimax_h3_comfyorg_files_analysis.md`。

本轮不重新跑模型，直接使用此前已经验证的 CUPTI 数据：

- `logs/cupti_h3_t2v/h3_t2v_cold.sqlite`
- `logs/cupti_h3_t2v/analysis/phase_kernel_summary.csv`
- `logs/cupti_h3_t2v/analysis/range_summary.csv`
- `logs/cupti_h3_t2v/analysis/manifest.json`

误差数据来自 `run_book_quant.md` 第 3 节，格式与 scale 语义来自 `minimax_h3_comfyorg_files_analysis.md` §2.5。

分析对象是本次 T2V 实际执行的 **Qwen、视频 VAE、音频 VAE**。未加载 LoRA 和两个 VAE encoder 不被虚构为已有运行时 kernel 证据。

## 55.1 原始 kernel 数据提取

使用 Python 标准库 `csv` 读取 `phase_kernel_summary.csv`，按 `range_group` 汇总每个 kernel 的 `count/total_ns`，先打印各阶段总 kernel 数、GPU work 和按时间排序的主要 kernel。

实际命令的核心逻辑：

```bash
python3 -B - <<'PY'
import csv
from collections import defaultdict
path = '/home/kaokaolv/comfyui_h3/logs/cupti_h3_t2v/analysis/phase_kernel_summary.csv'
groups = defaultdict(list)
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        row['count'] = int(row['count'])
        row['total_ns'] = int(row['total_ns'])
        groups[row['range_group']].append(row)
for group in ('H3.text.qwen', 'H3.text.refiner', 'H3.video_vae.all_vit',
              'H3.video_vae.decode', 'H3.audio_vae.decode'):
    rows = groups[group]
    count = sum(r['count'] for r in rows)
    total = sum(r['total_ns'] for r in rows)
    print(group, 'kernels=', count, 'work_ms=', total / 1e6)
    for r in sorted(rows, key=lambda r: r['total_ns'], reverse=True)[:24]:
        print(r['count'], r['total_ns'] / 1e6, r['kernel'])
PY
```

汇总时只合并 kernel duration，不把 host range、memcpy 或 CUDA runtime API time 混进 kernel work。

## 55.2 Qwen 的 kernel work 构成

按以下精确名称分类：

- FP32 GEMM：`simt_sgemm_f32`。
- NVFP4 反量化：`dequantize_nvfp4_kernel`。
- FP32 attention：`fmha_cutlassF_f32`。
- 其余为 RMSNorm、RoPE、copy 和其他 pointwise。

结果：

| 类别 | 次数 | GPU work | 占比 |
|---|---:|---:|---:|
| FP32 GEMM | 350 | 397.036370 ms | 89.4074% |
| NVFP4 dequant | 350 | 26.807478 ms | 6.0367% |
| FP32 attention | 50 | 10.114768 ms | 2.2777% |
| 其他 | 1616 | 10.116786 ms | 2.2782% |
| 合计 | 2366 | 444.075402 ms | 100% |

演示中重点说明：FP32 普通 GEMM 占接近九成 GPU work；但 Qwen host range 为 113.043 s，冷启动的权重准备/offload 仍是端到端延迟的重要部分，不能只看 GEMM。

## 55.3 视频 VAE 的 kernel work 构成与归类修正

ViT 内的精确分类：

- INT8 GEMM：以 `void cutlass::Kernel2` 开头的 kernel。
- ConvRot：`quantize_int8_rowwise_convrot64`。
- Sage attention：`qk_int_sv_i8_attn_kernel`。
- fused RMS+RoPE：`comfy::<unnamed>::rope_kernel`。
- attention 的 Q/K/V 量化：`quant_qk_fused` 或 `quant_v_int8_kernel`。
- 其余 ViT op 单独汇总；`H3.video_vae.decode` 父区间中不属于 42 个 ViT 子区间的 wrapper 再单独汇总。

最终使用的合并逻辑：

```python
vit = groups['H3.video_vae.all_vit']
wrapper = groups['H3.video_vae.decode']
# 先在 vit 内分类，再单独计算 vit_other 和 wrapper，避免按名字跨区间误归。
assert sum(r['count'] for r in vit if r['kernel'].startswith('void cutlass::Kernel2')) == 6048
assert sum(r['total_ns'] for r in vit if r['kernel'].startswith('void cutlass::Kernel2')) == 1130605806
```

过程中曾用“名字是否出现在 wrapper”判断外层 op，但 ViT 内也有相同的 copy/elementwise kernel 名称，会错误归到外层。该结果已弃用，改为严格按 `range_group` 先拆分再合并。最终验证：

```text
完整视频解码：29,339 kernels，2,123.436509 ms。
Other ViT：9,216 kernels，64.093625 ms。
ViT wrapper：1,979 kernels，10.227777 ms。
```

最终表格：

| 类别 | 次数 | GPU work | 占比 |
|---|---:|---:|---:|
| INT8 GEMM | 6048 | 1130.605806 ms | 53.2442% |
| ConvRot | 6048 | 316.196056 ms | 14.8908% |
| Sage attention | 1512 | 284.758450 ms | 13.4103% |
| fused RMS+RoPE | 1512 | 188.217871 ms | 8.8638% |
| attention QKV quant | 3024 | 129.336924 ms | 6.0909% |
| 其他 ViT op | 9216 | 64.093625 ms | 3.0184% |
| ViT wrapper | 1979 | 10.227777 ms | 0.4817% |

次数与 GPU work 加总均严格等于完整视频解码。`to_out/ff.w2` residual 已在 INT8 GEMM epilogue 内，不另列，避免重复统计。

## 55.4 音频 VAE 的 kernel work 构成

分类规则：

- ConvTranspose1d 前向：`dgrad2d_grouped_direct_kernel`。
- Depthwise：`conv_depthwise2d_forward_kernel`。
- 其它 Conv/GEMM：名称包含 `implicit_convolve_sgemm`、`implicit_gemm`、`dgrad_engine`、`xmma_fprop`、`cutlass__5x_cudnn`、`cutlass::Kernel2`。
- padding：`replication_pad_forward_kernel1d`。
- layout：`nchwToNhwcKernel` 或 `nhwcToNchwKernel`。
- Snake 超越函数：`sin_kernel`、`exp_kernel`、`reciprocal_kernel`。
- 剩余为 pointwise/residual/copy。

结果：

| 类别 | 次数 | GPU work | 占比 |
|---|---:|---:|---:|
| ConvTranspose1d 前向 | 127 | 11.756711 ms | 34.2879% |
| Depthwise conv | 127 | 3.626873 ms | 10.5776% |
| 其它 Conv/GEMM | 136 | 4.176538 ms | 12.1807% |
| Padding | 254 | 3.436250 ms | 10.0217% |
| Layout conversion | 286 | 1.671486 ms | 4.8748% |
| Snake sin/exp/reciprocal | 508 | 1.460029 ms | 4.2581% |
| 其他 pointwise/residual/copy | 1241 | 8.160378 ms | 23.7993% |
| 合计 | 2679 | 34.288265 ms | 100% |

演示中保留重要边界：`dgrad` 是 BigVGAN ConvTranspose1d 的前向实现，不是训练 backward；音频 VAE 没有 INT8/NVFP4 kernel。

## 55.5 INT8 ConvRot 反量化误差明细

在 `presentation.md` 4.3 新增 `Detail：int8_convrot 的反量化误差分析`，加入来源链接：

- `minimax_h3_comfyorg_files_analysis.md#L195`：INT8/ConvRot 的 256 组 Hadamard 和 per-channel FP32 scale。
- `run_book_quant.md#L250`：实际反量化与 BF16 base 对照。

展示公式：

```text
rel_RMSE = ||W_dequant - W_bf16||₂ / ||W_bf16||₂
cosine   = <W_dequant, W_bf16> / (||W_dequant||₂ × ||W_bf16||₂)
```

四层表格逐项使用 `run_book_quant` 的实测值：

| 模块 | INT8 rel_RMSE | cosine(float64) | MXFP8 rel_RMSE |
|---|---:|---:|---:|
| blocks.0.attn.qkv_proj | 0.8985% | 0.999960 | 2.6606% |
| blocks.0.attn.out_proj | 1.2113% | 0.999927 | 2.6611% |
| blocks.0.mlp.fc1 | 0.8973% | 0.999960 | 2.6608% |
| blocks.49.mlp.fc2 | 0.9640% | 0.999954 | 2.6620% |

保留解释：INT8 的 per-channel FP32 scale + ConvRot 与 MXFP8 的 E4M3 mantissa 差异能够解释这组权重误差；但这些误差不代表端到端画质、速度或数值稳定性，仍需完整 workflow 验证。

## 55.6 最终文档与一致性检查

第三节现在依次为：

- 3.3 DiT kernel work；
- 3.4 Qwen kernel work；
- 3.5 视频 VAE kernel work；
- 3.6 音频 VAE kernel work；
- 3.7 视频 VAE tile/chunk/fusion；
- 3.8 实测对后续设计的约束。

4.3 新增 INT8 误差表和来源链接。所有表格都沿用 kernel work，不将其写成 wall-time。

最终验证脚本执行了以下断言：

1. 3.1–3.8 标题各出现一次，代码围栏成对。
2. Qwen 的 350 次 FP32 GEMM、350 次 dequant、50 次 attention 和总 GPU work 与原始 CSV 一致。
3. 视频完整解码为 29,339 个 kernel、2,123.436509 ms；INT8 GEMM 为 6048 次、1,130.605806 ms；ConvRot 为 316.196056 ms；Sage 为 284.758450 ms。
4. 音频为 2679 个 kernel、34.288265 ms；最大 ConvTranspose 项为 11.756711 ms。
5. 四层 INT8/MXFP8 误差值逐项出现在 4.3 的 Detail 区域。
6. 全部 12 个本地文档/媒体链接的目标存在。

实际输出：

```text
presentation_new_kernel_tables_and_int8_error_table_ok
lines: 398 bytes: 23443 links: 12
```

本轮仅更新 `presentation.md` 和追加本记录，没有修改 ComfyUI 源码或重新运行 GPU 推理。
