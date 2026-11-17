# video_minimax_h3_t2v 基于 vLLM-Omni 的可行性与量化路线

## 1. 结论

可以把 MiniMax H3 T2V 改为 vLLM-Omni 推理，但不能把原始 `video_minimax_h3_t2v.json` 原样当成 vLLM-Omni workflow。正确方式是：

1. 用 vLLM-Omni 启动原始 H3 模型服务。
2. 在 ComfyUI 安装 `apps/ComfyUI-vLLM-Omni` 扩展。
3. 使用官方的 `MiniMax_H3_Text_to_Video.json`（WF-01）远程模板。
4. 将原 workflow 的 prompt、分辨率、时长和 seed 等参数迁移到远程 Generate Video 节点。

模型加载、采样和 VAE 解码都在 vLLM-Omni 服务端完成；ComfyUI 只负责输入、调用 HTTP API 和保存视频。

量化结论：

- **推荐路线：原始 BF16 H3 + 启动时在线 FP8。** 这是 H3 官方 recipe 明确支持的路线。
- **实验路线：原始 BF16 DiT + 启动时在线 MXFP8。** 源码支持 B200/SM100，但官方只验证了 MiniMax-H3-Image，没有验证 H3 视频，必须实测。
- **当前 Comfy-Org int8_convrot 和 nvfp4 单文件不能直接复用。**
- 若保留 NVFP4 文本编码器，应使用 vLLM 专用 `FastVideo/FastH3-Omni-NVFP4-QAD` 格式，不能直接使用 Comfy-Org 或 cybermotaz 的现有文件。
- QAD 整包是 8-step 蒸馏模型，不是原始 H3 Base。把 QAD 文本编码器与原始 BF16 DiT 混合，源码层面可以成立，但输出质量和兼容性都需要独立验证。

## 2. 为什么不能直接执行原始 JSON

原始 workflow 使用 ComfyUI 原生节点：

- `UNETLoader`
- `CLIPLoader(type=minimax)`
- 两个 `VAELoader`
- `MiniMaxH3ImageToVideo`
- `SamplerCustomAdvanced`
- `VAEDecode` / `VAEDecodeAudio`
- `CreateVideo`
- `SaveVideo`

vLLM-Omni 的 ComfyUI 扩展通过在线 `/v1/videos` API 工作，使用：

- `VLLMOmniGenerateVideo`
- `VLLMOmniDiffusionSampling`
- `VLLMOmniMiniMaxH3Params`
- 可选 `VLLMOmniRemoteLoRA`
- `SaveVideo`

因此应使用官方远程模板，而不是把 ComfyUI 原生 H3 节点逐个替换成同名 vLLM 节点。

官方 T2V 模板默认是 1344×768、24 FPS、5.167 秒（124 帧）、50 个采样点。原 ComfyUI 模板默认是 864×480、20 步。vLLM API 使用 sigma-point 计数，不能直接假设两个数字一一对应；输出不应要求逐像素一致。

用户提供的 H3-03 音乐视频文档属于 **Ref2VA + 外部音轨**，不是本 T2V workflow。它只能证明 ComfyUI-vLLM-Omni 的远程 H3 调用方式，不能证明原始 T2V JSON 可直接执行。

## 3. 模型与量化格式对照

| 模型/格式 | 能否直接用于当前 vLLM-Omni H3 | 结论 |
|---|---|---|
| Comfy-Org `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 否 | ComfyUI 专用 pruned/int8 单文件；应换原始 BF16 目录 |
| Comfy-Org `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 否 | 使用 `comfy_quant` 等字段，不符合 FastH3 NVFP4 loader |
| cybermotaz `Qwen3-VL-32B-Instruct-NVFP4` | 不能直接替代 | `quant_method=modelopt`、`quant_algo=NVFP4_AWQ`，且字段不是 loader 要求的 `.weight_packed` 格式 |
| 原始 `MiniMaxAI/MiniMax-H3/FL2VA` BF16 | 是 | 官方服务路径；包含原始 DiT、文本编码器、两个 VAE、tokenizer/processor |
| 原始 BF16 + `--quantization fp8` | 是，官方支持 | DiT 和文本编码器都可在线 FP8，磁盘仍为 BF16 |
| 原始 BF16 + `--quantization mxfp8` | 实验可行，未验证 | 通用 config 和 CUDA SM100 kernel 存在，但 H3 视频未做官方验证 |
| `FastVideo/FastH3-Omni-NVFP4-QAD` | 是，但属于不同模型变体 | 有完整兼容的 NVFP4 DiT/文本编码器，8-step QAD，不是原始 Base |

## 4. BF16 → MXFP8 应该怎样做

不要先做离线 MXFP8 文件转换。当前文档明确写出：

- 在线模式：BF16/FP16 权重在加载时转换为 MXFP8。
- CUDA 不支持 native serialized MXFP8 checkpoint 模式。
- B200 的 SM100 在允许列表中；`CUDAMxfp8OnlineLinearMethod` 会接受 SM100/103/107。

因此实验路线应使用原始 BF16 目录，并在启动命令中添加：

```bash
--quantization mxfp8
```

源码路径已经确认：

- `vllm_omni/quantization/factory.py` 注册 `mxfp8`。
- `DiffusionMXFP8Config` 在 CUDA 上选择 `CUDAMxfp8OnlineLinearMethod`。
- `pipeline_minimax_h3.py` 把量化配置传给 H3 DiT。
- MXFP8 使用 32 个 K-dimension 元素共享一个 exponent scale，W8A8。

限制：官方 `mxfp8.md` 的模型验证是 MiniMax-H3-Image，而不是 MiniMax-H3 视频。当前只能称为“源码和硬件路径支持的实验配置”，不能称为已验证部署。

## 5. NVFP4 文本编码器的正确来源

当前 Comfy-Org 文件不能直接使用，原因有两个：

1. vLLM-Omni 的 `FastH3NVFP4Config.from_encoder` 要求 `config.json` 中：
   - `quant_method=nvfp4`
   - `activation_scheme=dynamic`
   - `fmt=e2m1`
   - `group_size=16`
   - `scale_fmt=e4m3`
   - `scale_layout=128x4`
2. shard 中要包含 `.weight_packed`、`.weight_scale`、`.weight_global_scale` 等约定字段。

`FastVideo/FastH3-Omni-NVFP4-QAD` 满足这些要求：

- `text_encoder/config.json` 包含上述 NVFP4 字段。
- `num_hidden_layers_override=50`，与 H3 使用第 50 层 hidden states 一致。
- `producer.converter=convert_minimax_h3_text_encoder_nvfp4.py`。
- `kept_language_layers=50`。
- index 中存在 `.weight_packed` 和 `.weight_global_scale`。

这不是 Comfy-Org 当前文件的简单格式转换问题；应使用已生成的兼容产物，而不是手工改几个 JSON 字段。

## 6. 推荐实施方案：原始 BF16 + 在线 FP8

这是当前风险最低的路线。

### 6.1 环境

本次核对的源码 commit：

`d64649c41a3a2dce6da3cd97ae48b754def4eb25`

本机旧环境虽然已有 H3 和 MXFP8 config，但：

- `diffusion/models/minimax_h3/checkpoint_quantization.py` 不存在。
- `quantization/mxfp8_cuda.py` 不存在。
- 一个旧环境还有 vLLM/vLLM-Omni 版本不匹配警告。

因此应在新独立 venv 安装匹配的 checkout，不修改旧环境。官方当前安装文档对应 vLLM 0.33 / PyTorch 2.14。

### 6.2 模型

```bash
hf auth login
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" "FL2VA/*" \
  --local-dir /home/kaokaolv/models/MiniMax-H3
```

原始 FL2VA 约 134 GiB 存储，不能用当前 Comfy-Org 单文件冒充原始目录。官方 recipe 要求 Hugging Face 访问授权。

### 6.3 启动服务

单张 B200 可先使用：

```bash
MODEL=/home/kaokaolv/models/MiniMax-H3/FL2VA
CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "$MODEL" --omni \
  --host 0.0.0.0 --port 8000 \
  --served-model-name MiniMaxAI/MiniMax-H3 \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 1 \
  --tensor-parallel-size 1 \
  --text-encoder-tp-size 1 \
  --quantization fp8 \
  --enforce-eager
```

官方单 96 GB FP8 案例的首次请求整卡峰值约 92,946 MiB，因此本机约 180 GiB 的 B200 有足够余量。这个数值只用于容量参考，不是本机实测。

### 6.4 ComfyUI

把 `apps/ComfyUI-vLLM-Omni` 复制到 `ComfyUI/custom_nodes/ComfyUI-vLLM-Omni` 并重启。加载官方 `MiniMax_H3_Text_to_Video.json`，设置：

- URL：`http://sc09super22-b200.sc.intel.com:8000/v1`
- Model：`MiniMaxAI/MiniMax-H3`
- prompt：可复用原始 T2V prompt
- width/height：1344×768
- duration：5.167 秒
- frame/references/fast_h3：断开
- Turbo：首次保持关闭

先按官方 Base 配置验证，再决定是否对齐原模板的 20-step 参数。不要把 vLLM 的 50 points 与 ComfyUI 的 20 steps 当作同一调度。

## 7. 实验方案：MXFP8 DiT + NVFP4 文本编码器

这是与提出方案最接近、且源码结构能够支持的混合路线，但尚未实测。

### 7.1 模型目录原则

- `transformer/`：使用原始 `MiniMaxAI/MiniMax-H3/FL2VA/transformer` BF16。
- `text_encoder/`：使用 `FastVideo/FastH3-Omni-NVFP4-QAD/text_encoder`。
- `video_vae/`、`audio_vae/`：使用原始 H3 组件。
- `tokenizer/`、`processor/`：使用原始 H3 配置，保留 `<d>` 等专用 token。
- 不要把 QAD 的 `transformer/nvfp4_weights.safetensors` 放进混合目录；否则服务端会改用 FastH3 QAD DiT，且与显式 `--quantization mxfp8` 冲突。

可以新建目录并用 symlinks 组合，避免复制大文件。概念结构：

```text
H3-hybrid/
├── model_index.json        -> 原始 FL2VA/model_index.json
├── transformer/            -> 原始 BF16 transformer
├── text_encoder/           -> FastVideo QAD text_encoder
├── video_vae/              -> 原始 video_vae
├── audio_vae/              -> 原始 audio_vae
├── tokenizer/              -> 原始 tokenizer
└── processor/              -> 原始 processor
```

### 7.2 启动

与第 6.3 节相同，仅把量化参数改为：

```bash
--quantization mxfp8
```

加载器会从 text_encoder 自己的 NVFP4 config 选择 FastH3 NVFP4 路径，同时把 MXFP8 用于原始 BF16 DiT。

### 7.3 必须验证的事项

- 服务是否接受混合目录，且没有把 QAD DiT 一并加载。
- MXFP8 CUDA kernel 是否成功初始化并完成一次请求。
- 文本编码器是否确实走了 NVFP4，而非回退 BF16。
- 输出是否包含 24 FPS 视频和 32 kHz 双声道音频。
- 与原始 BF16/FP8 结果比较时，是否出现文本、画面或音频异常。
- 记录服务端 commit、两个模型的 revision、显存峰值和完整日志。

若此路线失败，优先回退到“原始 BF16 DiT + 原始文本编码器 + 在线 FP8”，不要先尝试手改 Comfy-Org 文件。

## 8. 如果直接使用完整 QAD NVFP4 包

`FastVideo/FastH3-Omni-NVFP4-QAD` 是另一种可选部署，但它是 **8-step quantization-aware distilled FastH3 Omni**，不是原始 H3 Base。

它包含：

- NVFP4 reference-conditioned transformer
- NVFP4 Qwen3-VL text/vision encoder
- video/audio VAE
- 可选 NVFP4 video decoder

这条路线可以作为独立的 FastH3 实验，但不能用来证明原始 T2V Base 的结果或质量等价。应遵循其 model card 和 FastH3 recipe，不套用原始 20-step/50-step 设定。

## 9. 当前未完成的验证

本轮没有启动 vLLM-Omni 真实模型服务，也没有生成新视频。原因：

- 原始 BF16 FL2VA 需要约 134 GiB 下载和 Hugging Face 授权。
- 当前旧环境缺少新 NVFP4 loader 和 MXFP8 CUDA 模块，需要新环境。
- MXFP8 + 原始 DiT 与 QAD 文本编码器的混合没有官方质量验证。

因此本报告给出的是有源码和 model card 支撑的可行性与实施路线，不声称已经跑通。下一步应先做第 6 节官方 FP8 路线，再决定是否投入时间验证第 7 节混合路线。

## 10. 主要来源

- vLLM-Omni ComfyUI 集成：<https://github.com/vllm-project/vllm-omni/tree/main/apps/ComfyUI-vLLM-Omni>
- H3 T2V 模板指南：<https://github.com/vllm-project/vllm-omni/blob/main/apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md>
- H3 recipe：<https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md>
- MXFP8 文档：<https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/mxfp8.md>
- QAD NVFP4：<https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD>
- H3-03 音乐视频：<https://github.com/vllm-project/vllm-omni/blob/main/apps/ComfyUI-vLLM-Omni/docs/h3-03-music-video.md>
