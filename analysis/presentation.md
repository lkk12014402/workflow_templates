# MiniMax H3 T2V：从 ComfyUI 执行与 Kernel 实测到 MXFP 量化路线

> 分析对象：`video_minimax_h3_t2v.json`
>
> 汇报重点：模型与权重来源、端到端执行逻辑、真实函数/kernel 调用，以及 BF16 → MXFP8/MXFP4 的实施路线。
>
> 证据边界：原生 ComfyUI T2V 和 CUPTI 分析已完成；DiT MXFP8 权重量化已完成；MXFP8/MXFP4 真 GEMM 与 vLLM-Omni 路线仍需验证。

## 一页结论

1. **原生 ComfyUI 已跑通音视频联合生成。** 一次文本编码、20 步联合 DiT 去噪、视频/音频 VAE 解码，最终输出 864×480、24 fps、124 帧并带 32 kHz 双声道音频的 MP4。
2. **模型文件名称不等于实际计算精度。** DiT 和视频 VAE 的大 Linear 是 INT8 W8A8，但主 DiT attention 是 BF16 FlashAttention；Qwen 是 NVFP4 存储、FP32 计算，不是 FP4 GEMM；音频 VAE 保持 FP32。
3. **性能分析必须区分冷启动、稳态和 GPU work。** 首次权重准备/offload 占主要时间，不能直接用端到端冷启动时间衡量稳定吞吐；kernel 时间之和也不等于 wall time。
4. **量化与推理是两项独立任务。** auto-round 已完成 DiT 200 层 MXFP8 权重转换及格式验证，但这还不等于 ComfyUI 已执行 MXFP8 GEMM。
5. **首选工程路线是原生 ComfyUI + Comfy-Kitchen + FlashInfer。** vLLM-Omni 更适合独立 serving，需要更换远程 workflow 和模型格式，不能原样执行当前 JSON。

---

## 1. ComfyUI workflow：`video_minimax_h3_t2v.json`

### 1.1 本次实际运行配置

| 项目 | 实际值 |
|---|---|
| 输入 | 纯文本；无首帧、尾帧、参考音频或 negative prompt |
| 输出 | 864×480，24 fps，124 帧，约 5.167 秒 |
| 采样 | `res_multistep` + `simple`，20 steps，denoise=1 |
| 条件分支 | 只有 positive；每次 sigma 调用一次 DiT，无 CFG 双分支 |
| Turbo / LoRA | `turbo_mode=false`，主模型 0 patches，未加载 Turbo LoRA |
| 实测环境 | NVIDIA B200、ComfyUI v0.45.0、PyTorch 2.13.0+cu130、comfy-kitchen 0.2.41 |

虽然模板节点名为 `MiniMaxH3ImageToVideo`，本次 `first_frame` 和 `last_frame` 都未连接，因此实际是纯 T2V，不执行参考图像 VAE encode。

### 1.2 端到端执行逻辑

```text
加载 Qwen / 联合 DiT / 视频 VAE / 音频 VAE
  │
  ├─ Qwen 文本编码（一次）
  │    └─ condition_proj + 2 层 token refiner（一次）
  │
  ├─ 构造空 AV latent 与初始噪声
  │
  ├─ 20 步联合去噪
  │    └─ 同一个 DiT 同时建模视频流和音频流
  │       将 text / audio / video pack 后做 full self-attention
  │
  ├─ 音频 VAE decode → 32 kHz 双声道波形
  ├─ 视频 VAE decode → RGB 帧
  │
  └─ CreateVideo → SaveVideo → 带音轨 MP4
```

**关键点：音视频是在同一个 DiT 中联合生成的，不是“先生成视频，再由另一个音乐模型配音”。** 文本编码和 token refiner 只执行一次，不会在 20 个采样步中重复。

本次实测 packed sequence 为：

```text
[388 text | 414 audio | 14,985 video]，总长度 S = 15,787
```

### 1.3 冷启动、稳态与显存

- 原始冷启动执行的客户端总耗时为 **953.3 s**，包含模型加载、初始化和权重准备，不能当成稳定生成耗时。
- 独立 CUPTI 采集中，DiT 第一步 host range 为 **178.617 s**，第 2–20 步每步约 **1.16 s**。二者必须分开分析。
- 原始执行整卡 NVML 峰值为 **38,570 MiB**，出现在**视频 VAE 解码阶段**，不是文本编码或采样阶段。
- 动态权重主要由 ComfyUI/aimdo 管理，不能只用 PyTorch allocator 判断总显存，也不能把所有模型文件大小简单相加当作运行峰值。

---

## 2. Flow 中的模型、来源与精度

### 2.1 四个实际加载的基础模型

以下文件大小使用十进制 GB；参数量是逻辑参数量，不是量化后的文件字节数。

| 模块 | 结构与作用 | 当前文件 / 参数量 | 推荐的重新量化 base |
|---|---|---|---|
| **联合 DiT** | 50 层，hidden=5376，56×128 heads；联合建模视频和音频 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`；20.970 GB / **20.111B** | Comfy-Org `minimax_h3_fl2va_pruned_bf16.safetensors`；40.226 GB |
| **Qwen3-VL 文本编码器** | 使用 Qwen3-VL-32B 前 50 层生成文本条件；文件还含完整 visual tower 和 embedding | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`；15.687 GB / **25.753B** | Comfy-Org `qwen3vl_32b_minimax_h3_bf16.safetensors`；51.506 GB，902 tensors |
| **视频 VAE** | Conv3D encoder + 36 层 ViT decoder；本 T2V 只执行 decoder | `minimax_h3_video_vae_int8_convrot.safetensors`；2.811 GB / **2.604B** | Comfy-Org `minimax_h3_video_vae_fp16.safetensors`；5.208 GB |
| **音频 VAE** | DAC encoder + attention projection + BigVGAN decoder；本 T2V 只执行 decoder | `minimax_h3_audio_vae_fp32.safetensors`；0.605 GB / **0.1513B** | 当前 FP32 文件或 MiniMaxAI 上游 FP32 文件；保持 FP32 |

补充来源：

- DiT 原始上游是 `MiniMaxAI/MiniMax-H3/FL2VA/transformer`，完整模型为 **33.123B**。
- 当前 Qwen NVFP4 文件源自 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4`。完整 q_proj 校验表明，Comfy-Org 的转换是 nibble 和 scale 布局转换，不是重新量化。
- 可选 8-step Turbo LoRA 来自 `lightx2v/Minimax-h3-Turbo`，为 BF16、1.956 GB / 0.978B；本次没有加载。

### 2.2 `pruned` 到底删减了什么

`pruned` **不是删除 transformer 层，也不是削减 attention/MLP 宽度**。50 个主 block 全部保留，主要变化在 AdaLN 时间嵌入分支：

```text
完整模型：TimeEmbedder + AdaLN [96768, 2688]
pruned：  共享 adaln_t_table [1025, 8] + AdaLN [96768, 8]
```

逐 key 对账结果：

- full：**33,122,992,912** 参数；
- pruned：**20,111,438,744** 参数；
- 精确差值：**13,011,554,168**，约 39.3%；
- 差异可由 AdaLN、final AdaLN、TimeEmbedder 和新增 table 完整解释，没有未解释的参数差。

“曲线蒸馏”是根据 checkpoint 和 ComfyUI 源码得到的结构性解释，不是 model card 对训练过程的直接描述。

### 2.3 不能把所有模块都称为同一种量化精度

| 模块 | Dense Linear | Attention / 其他计算 |
|---|---|---|
| DiT | 200 个 INT8 ConvRot Linear：`qkv_proj/out_proj/fc1/fc2` | 主 attention 为 BF16 FlashAttention；token refiner、AdaLN、patch/final head 等保留 BF16/FP16/FP32 |
| Qwen | 350 个 NVFP4 Linear，但运行时反量化成 FP32 后普通 GEMM | FP32 mem-efficient attention；embedding 为 INT8；visual tower 为 BF16 |
| 视频 VAE | 144 个 INT8 ConvRot Linear | Sage INT8 attention；encoder 为 FP16；norm/bias 等有 FP32 |
| 音频 VAE | 不属于本轮量化对象 | FP32 Conv1d / ConvTranspose1d / Snake 等 |

因此，后续 MXFP 改造针对的是 **dense Linear**，不意味着整个网络都变成 MXFP，也不涉及 MoE 专家路由。

---

## 3. 函数调用逻辑与真实 Kernel

### 3.1 证据来源：真实 CUPTI trace，而非只看注册表

在独立 profiling 实例上，对相同 API prompt 采集 Nsight Systems/CUPTI：

- **95,226** 个 kernel、**155** 种 kernel；
- **67** 个 NVTX 区间；
- kernel 通过 runtime correlationId 映射到最小 NVTX 子区间；
- 最终 MP4 已验证可解码，含预期视频和双声道音轨。

这种证据可以区分“backend 支持某功能”和“本次实际执行了某功能”。

### 3.2 各模型的函数链、GEMM 与 attention

| 模型 | 主要函数路径 | Dense Linear 实际计算 | Attention 实际计算 |
|---|---|---|---|
| **Qwen** | `CLIPLoader → MiniMaxQwen3VL → 50×Qwen layer → MixedPrecisionOps.Linear` | NVFP4 权重 → `dequantize_nvfp4_kernel` → FP32 `cutlass3x_sm100_simt_sgemm`；**350 次** | FP32 `fmha_cutlassF_f32_aligned`；**50 次** |
| **联合 DiT** | `UNETLoader → MiniMaxH3Model → 20×50 blocks → linear_input_act → ck.int8_linear` | 在线 ConvRot + per-row INT8 量化 → CUTLASS INT8 GEMM，BF16 输出；**4000 次** | PyTorch SDPA → `pytorch_flash::flash_fwd_kernel`，BF16；**1000 次** |
| **视频 VAE** | `VAEDecode → 42×ViT3DDecoder → 36×4 Linear` | RMSNorm/SwiGLU → ConvRot → INT8 GEMM；**6048 次** | `ck.int8_attention → _C.sage_sdpa → qk_int_sv_i8_attn_kernel`；**1512 次** |
| **音频 VAE** | `VAEDecodeAudio → dec_in_proj → BigVGAN` | 不以量化 Linear 为主 | FP32 卷积和逐点 op，共 **2679 个 kernel**；本阶段不跑 encoder attention |

几个容易误判的点：

- Qwen 的 `full_precision_matrix_mult=true` 和文本编码器默认策略决定了它没有走 `scaled_mm_nvfp4`。
- DiT 的 INT8 是 **W8A8**：权重 INT8，activation 也在线旋转并量化为 INT8，不是 W8A16。
- `Using pytorch attention` 是 Python 层选择，不代表最终一定是普通 FP32 attention；实际主 DiT 为 BF16 FlashAttention。
- 视频 VAE 与主 DiT 使用不同的 attention 精度和 kernel，不能混为一谈。

### 3.3 DiT：kernel work 的构成

20 步 DiT 共 **59,724 个 kernel**，总 kernel work 为 **34.030 s**：

| 类别 | 实测次数 | kernel work 占比 |
|---|---:|---:|
| BF16 FlashAttention | 1000 | 约 **51.9%** |
| INT8 CUTLASS GEMM | 4000 | 约 **36.5%** |
| ConvRot 在线量化 | 4000 | 约 **5.1%** |
| fused RMSNorm + RoPE | 1000 | 约 **2.1%** |
| 其他 op | — | 约 **4.4%** |

这些比例是 **kernel work 占比，不是 wall-time 占比**。多 stream 并行时，kernel 时间之和可以大于 host range。

### 3.4 Qwen：kernel work 的构成

以下三表继续使用 **kernel work** 口径，不等于各阶段的 wall-time 占比。

Qwen 阶段共 **2,366 个 kernel**，总 kernel work 为 **444.075 ms**：

| 类别 | 实测次数 | GPU work | kernel work 占比 |
|---|---:|---:|---:|
| FP32 GEMM | 350 | 397.036 ms | 约 **89.41%** |
| NVFP4 反量化 | 350 | 26.807 ms | 约 **6.04%** |
| FP32 mem-efficient attention | 50 | 10.115 ms | 约 **2.28%** |
| RMSNorm、RoPE、copy 及其他 pointwise | 1616 | 10.117 ms | 约 **2.28%** |

这里最主要的问题不是 NVFP4 反量化，而是 **FP32 普通 GEMM 占了接近九成 GPU work**。同时，Qwen host range 为 113.043 s，远大于 444.075 ms 的 kernel work，说明首次权重准备和 offload 仍是端到端延迟的主要组成，不能只看 GEMM 优化。

### 3.5 视频 VAE：kernel work 的构成

完整视频解码包括 42 次 ViT 及外层处理，共 **29,339 个 kernel**，总 kernel work 为 **2,123.437 ms**：

| 类别 | 实测次数 | GPU work | kernel work 占比 |
|---|---:|---:|---:|
| INT8 CUTLASS GEMM | 6048 | 1130.606 ms | 约 **53.24%** |
| ConvRot 在线量化 | 6048 | 316.196 ms | 约 **14.89%** |
| Sage INT8 attention | 1512 | 284.758 ms | 约 **13.41%** |
| fused RMSNorm + RoPE | 1512 | 188.218 ms | 约 **8.86%** |
| attention 的 Q/K/V 量化 | 3024 | 129.337 ms | 约 **6.09%** |
| 其他 ViT op | 9216 | 64.094 ms | 约 **3.02%** |
| ViT 外层 copy、拼接与 pointwise | 1979 | 10.228 ms | 约 **0.48%** |

`to_out/ff.w2` 的 residual 已包含在 INT8 GEMM epilogue 中，不另列为独立 kernel work，避免重复统计。与主 DiT 不同，视频 VAE 的量化及 Sage attention 也占据显著比例，不能只优化 GEMM。

### 3.6 音频 VAE：kernel work 的构成

音频解码共 **2,679 个 kernel**，总 kernel work 为 **34.288 ms**：

| 类别 | 实测次数 | GPU work | kernel work 占比 |
|---|---:|---:|---:|
| ConvTranspose1d 前向实现（`dgrad2d_grouped_direct`） | 127 | 11.757 ms | 约 **34.29%** |
| Depthwise convolution | 127 | 3.627 ms | 约 **10.58%** |
| 其他 Conv / implicit GEMM | 136 | 4.177 ms | 约 **12.18%** |
| Replication padding | 254 | 3.436 ms | 约 **10.02%** |
| NCHW/NHWC layout conversion | 286 | 1.671 ms | 约 **4.87%** |
| Snake 相关 sin/exp/reciprocal | 508 | 1.460 ms | 约 **4.26%** |
| 其他 pointwise、residual 与 copy | 1241 | 8.160 ms | 约 **23.80%** |

`dgrad` 在这里是 BigVGAN ConvTranspose1d 的前向实现，不是训练 backward。音频 VAE 没有 INT8/NVFP4 kernel，因此量化 Linear 的改造不会直接改变这部分计算。

### 3.7 视频 VAE：tile、chunk 与融合

视频 decoder 并非一次解整段视频：

```text
7 个时间 chunk × 3 个高度 tile row × 每组宽度 tile 按 (4+1) 合 batch
= 42 次 ViT forward
```

每次有 36 层，因此形成 `42 × 36 × 4 = 6048` 次 INT8 GEMM。

其中：

- `to_qkv/ff.w1`：RMSNorm → ConvRot → INT8 GEMM；
- `to_out`：ConvRot → INT8 GEMM → residual；
- `ff.w2`：SwiGLU → ConvRot → INT8 GEMM → residual；
- **3024** 个 kernel 带 `VisitorAuxLoad`，证实 `to_out/ff.w2` 的 residual 确实进入融合 epilogue。

### 3.8 实测对后续设计的约束

1. 不能只优化 GEMM 而忽略 BF16 FlashAttention，它占当前 DiT kernel work 的一半以上。
2. 量化权重能减少存储，但 activation 量化、ConvRot、offload 和首次初始化同样影响端到端性能。
3. 当前 trace 未观察到 CUDA Graph node kernel；allocation graph 不能当成整模型 CUDA Graph replay。
4. 任何新量化路线都必须重新检查真实 kernel，不能只凭文件加载成功或输出视频就宣称加速成功。

---

## 4. 用源模型支持 MXFP8 / MXFP4

### 4.1 重新量化策略：从干净的 BF16/FP16 base 出发

| 当前量化对象 | 新格式 | 来源与目标层 |
|---|---|---|
| DiT 的 200 个 INT8 Linear | **MXFP8** | `pruned_bf16` → 50×4 个目标 Linear |
| 视频 VAE 的 144 个 INT8 Linear | **MXFP8** | `video_vae_fp16` → 36×4 个目标 Linear |
| Qwen 的 350 个 NVFP4 Linear | **MXFP4** | 50 层 BF16 TE → 50×7 个目标 Linear |
| Qwen 的 INT8 embedding | **保持现有格式** | 不属于这 350 个 Linear 的量化目标 |
| BF16 / FP16 / FP32 模块 | **保持不变** | attention、卷积、norm、patch/final head、音频 VAE 等 |

不要从现有 INT8/NVFP4 文件反量化后再量化，以免引入二次量化误差。原有模型结构和 workflow 不需要改成 MoE 或另一套网络。

### 4.2 auto-round：先 model-free，后优化

初版采用 auto-round model-free 路径，对单个 safetensors 文件逐层处理：

| 格式 | 权重元素 | Scale | 运行时目标 |
|---|---|---|---|
| MXFP8 | E4M3，8 bit | 每 32 个元素一个 E8M0 scale | 动态量化 activation，W8A8 |
| MXFP4 | E2M1，4 bit，两个值打包为 1 byte | 每 32 个元素一个 E8M0 scale | 动态量化 activation，W4A4 |

处理原则：

- 只转换 manifest 中明确列出的 Linear，其他 tensor 原样保留；
- 将 auto-round 的 row-major scale 转成 Comfy-Kitchen 需要的 **128x4 blocked scale**；
- 写出 ComfyUI 原生 `.safetensors + .comfy_quant`，不把后端名称绑定到文件；
- 后续再评估 auto-round 的 tuning、AWQ、scale search 等优化，不与初版格式验证混在一起。

### 4.3 已完成：DiT 200 层 MXFP8 权重量化

auto-round 0.17.0 已生成：

```text
ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors
大小：21.560 GB
```

校验结果：

- 200 个目标模块全部完成；
- 932 个 tensor，dtype 数量符合预期；
- `qkv_proj`、`fc1`、最后一层 `fc2` 三个代表层的反量化相对 RMSE 均为 **2.66%**；
- Comfy-Kitchen 能正确解释 FP8 权重、E8M0 scale 和 `comfy_quant` 元数据。

#### Detail：int8_convrot 的反量化误差分析

格式与 scale 定义见[模型文件分析 §2.5](./minimax_h3_comfyorg_files_analysis.md#L195)，逐层实测数据来自[量化执行记录 §3](./run_book_quant.md#L250)。

INT8 ConvRot 的权重以 256 个元素为一组进行 Hadamard 旋转，并使用每个输出通道一个 FP32 scale。反量化时需要恢复旋转并应用该通道 scale；下面的对比使用 Comfy-Kitchen 的实际 layout 反量化，再与同一 BF16 base 逐元素比较：

```text
rel_RMSE = ||W_dequant - W_bf16||₂ / ||W_bf16||₂
cosine   = <W_dequant, W_bf16> / (||W_dequant||₂ × ||W_bf16||₂)
```

| 模块 | 权重 shape | INT8 rel_RMSE | INT8 cosine（float64） | MXFP8 rel_RMSE |
|---|---|---:|---:|---:|
| `blocks.0.attn.qkv_proj` | 21504×5376 | **0.8985%** | 0.999960 | 2.6606% |
| `blocks.0.attn.out_proj` | 5376×7168 | **1.2113%** | 0.999927 | 2.6611% |
| `blocks.0.mlp.fc1` | 28672×5376 | **0.8973%** | 0.999960 | 2.6608% |
| `blocks.49.mlp.fc2` | 5376×14336 | **0.9640%** | 0.999954 | 2.6620% |

这些主投影的 INT8 权重 rel_RMSE 为 **0.897%–1.211%**，MXFP8 为 **2.661%–2.662%**；即 INT8 权重 RMSE 约为 MXFP8 的 **1/2.2–1/3**。这与 INT8 的 per-channel FP32 scale + ConvRot 抑制 outlier，以及 MXFP8 的 E4M3 有限 mantissa 相符。float64 cosine 均大于 0.9999，说明这些权重的整体方向保持较好。

**这是权重误差对比，不是端到端画质、速度或数值稳定性的结论。** 量化对生成结果的影响仍需通过固定 prompt/seed 的完整 workflow 验证。

**尚未完成：** 该文件的完整 T2V 生成、真实 MXFP8 Tensor Core GEMM、显存/速度/画质对照。Qwen MXFP4 和视频 VAE MXFP8 也不能据此宣称已完成。

### 4.4 路线 A：原生 ComfyUI + Comfy-Kitchen + FlashInfer

目标是保持当前模型结构和原生 workflow，在现有 `QuantizedTensor` 体系下支持真正的 dense W8A8/W4A4：

```text
BF16/FP16 或 compressed-tensors 权重
  → 原生 .comfy_quant 加载
  → QuantizedTensor / mixed_precision_ops.Linear
  → Comfy-Kitchen backend 选择
  → FlashInfer MXFP8/MXFP4 GEMM（B200 优先）
```

设计要点：

1. **kernel 选择参考 vLLM。** 按硬件、依赖、输入格式、shape 和输出 dtype 依次检查候选，记录不可用原因。
2. **FlashInfer 提供实际 MX GEMM。** Comfy-Kitchen 负责统一接口、格式和 dispatch，不复制 CUTLASS/CuTe kernel。
3. **补齐真实缺口。** 当前 CUDA backend 没有 `scaled_mm_mxfp8`；MXFP4 需要新增 layout、量化、加载和 GEMM。PyTorch `scaled_mm_v2` 也先作为待验证候选。
4. **禁止静默回退。** dequantize 后普通 GEMM 只能是显式 reference，不能冒充 MXFP fast path。
5. **保留其他计算。** DiT 继续 BF16 FlashAttention，视频 VAE 继续 Sage INT8 attention，音频 VAE 保持 FP32。
6. **处理 Qwen 特殊策略。** 新 MXFP4 文件需要显式启用量化矩阵乘法；FP32 文本输入到 BF16 MX GEMM 输出的精度变化必须单独评估。
7. **接入 compressed-tensors。** 权重 qdata 不变，只做 key 映射和 scale 重排，与 BF16 转换结果共用运行时。

该路线与现有 ComfyUI 模型、节点、显存管理和 CUPTI 分析方式最一致，但 kernel adapter 和模型加载支持仍需实现与验证。

### 4.5 路线 B：vLLM-Omni serving

vLLM-Omni 不能把当前 JSON 原样当作自己的 workflow。正确结构是：

```text
vLLM-Omni 服务端：加载模型、采样、VAE 解码
        ↑ HTTP /v1/videos
ComfyUI-vLLM-Omni 扩展：输入参数、调用 API、保存视频
```

需要改用官方远程 `MiniMax_H3_Text_to_Video.json`，而不是逐个替换当前原生节点。

量化与模型选择：

- **优先验证：** 原始 BF16 FL2VA + 启动时在线 FP8，这是官方支持的路线。
- **实验验证：** 原始 BF16 DiT + 在线 MXFP8；源码和 B200 路径存在，但官方验证的是 MiniMax-H3-Image，不是 H3 视频。
- **NVFP4：** 需要 FastH3 兼容目录，例如 `FastVideo/FastH3-Omni-NVFP4-QAD`；当前 Comfy-Org 或 cybermotaz 文件不能直接替代。
- QAD 是 **8-step 蒸馏模型变体**，不是原始 H3 Base。混用 QAD 文本编码器与原始 BF16 DiT 必须单独验证质量。
- 官方远程模板为 1344×768、50 个采样点，与当前 864×480、20 steps 的配置不同，不能直接要求逐像素一致。

### 4.6 两条路线对比

| 维度 | 原生 ComfyUI + FlashInfer | vLLM-Omni serving |
|---|---|---|
| Workflow | 保留当前原生执行逻辑 | 改用官方远程 API workflow |
| 推理位置 | ComfyUI 本地模型与 kernel | vLLM-Omni 服务端 |
| MXFP8 | 离线 BF16 → MXFP8，接入统一 dispatch | 原始 BF16 在线 MXFP8，H3 视频需验证 |
| MXFP4 | 按 proposal 新增 W4A4 支持 | 当前没有已验证的同目标 H3 T2V 路线 |
| 现有权重 | 转换后可复用原生格式 | 现有 Comfy-Org int8/NVFP4 文件不能直接用 |
| 工程改动 | 需要 kernel adapter 和量化加载支持 | 需要独立 serving、远程扩展和模型目录 |
| 适合场景 | 保留当前 ComfyUI 系统并做精细量化/kernel 研究 | 独立 API 服务、标准化部署 |
| 当前状态 | DiT MXFP8 文件已完成；真 GEMM 待验证 | 完成源码/官方文档调研，尚未启动真实服务 |

**建议：** 当前目标优先走原生 ComfyUI 路线；vLLM-Omni 先以官方 BF16 + 在线 FP8 路线做独立可行性验证，不把两条路线的输出或采样参数直接混比。

### 4.7 建议实施与验收顺序

1. **B200 小矩阵验证：** 检查 FlashInfer 版本、`mm_mxfp8/mm_fp4`、scale layout 和实际 kernel。
2. **完成量化文件：** 验证 DiT/VAE MXFP8、Qwen MXFP4 的权重和 loader，不把格式验证等同于运行验证。
3. **做垂直切片：** DiT 单层/单步 → Qwen 编码 → 视频 VAE decode → 完整 T2V。
4. **做同条件对照：** 固定 prompt、seed、分辨率和采样参数，比较当前 INT8/NVFP4、MXFP8/MXFP4，以及可用时的 BF16 参考。
5. **验收真实计算路径：** CUPTI 确认 MXFP GEMM、activation 量化和 attention 类型，排除逐层 dequantize 后普通 GEMM。
6. **评估综合结果：** 输出质量、显存峰值、冷启动、稳态耗时和 backend workspace 一起比较。
7. **后续优化：** 再考虑 auto-round tuning/AWQ/scale search、算子融合和 XPU kernel；不在 B200 正确性验证前扩大范围。

---

## 5. 总结

- **已经证明的：** 原生 ComfyUI 能完成 H3 音视频联合生成；模型文件、量化格式和真实 kernel 已形成闭环证据。
- **最关键的实测认识：** DiT 是 INT8 Linear + BF16 attention，Qwen 是 NVFP4 存储 + FP32 计算，视频 VAE 是 INT8 Linear + Sage INT8 attention，音频 VAE 是 FP32。
- **已经完成的量化工作：** DiT 200 层 MXFP8 文件及格式/反量化校验已完成，三个代表层 rel_rmse 为 2.66%。
- **下一阶段的核心：** 不是再写一份权重就结束，而是让 ComfyUI 真正 dispatch 到 MXFP8/MXFP4 GEMM，并用 CUPTI、显存、速度和质量结果共同验收。
- **路线选择：** 保留当前系统时优先原生 ComfyUI + Comfy-Kitchen + FlashInfer；vLLM-Omni 作为独立 serving 路线另行验证。

## 参考资料与证据入口

- [Workflow 与函数/kernel 深度分析](./comfyui_h3_t2v_kernel_analysis.md)
- [CUPTI 逐 Kernel 实测报告](./cupti_h3_t2v_kernel_report.md)
- [MXFP8/MXFP4 接入 Proposal](./comfyui_h3_mxfp8_mxfp4_proposal.md)
- [五个模型文件的结构、精度与来源](./minimax_h3_comfyorg_files_analysis.md)
- [模型文件独立复核与修正](./run_book.md#L3591)
- [vLLM-Omni 路线调研](./vllm_omni_h3_research.md)
- [量化执行记录](./run_book_quant.md)
- [完整部署、运行与分析记录](./run_book.md)
- [实际 API prompt](./logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json)
- [CUPTI 复测输出视频](./logs/cupti_h3_t2v/output/video/MiniMax_H3_00001_.mp4)
