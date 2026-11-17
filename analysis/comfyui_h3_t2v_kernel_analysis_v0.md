# MiniMax H3 T2V：模型加载、函数调用栈与 Kernel/Op 深度分析

> 分析对象：`video_minimax_h3_t2v.json` 在本机原生 ComfyUI 中的实际运行。
>
> 目的：回答“每个模型怎样加载、怎样推理、经过哪些函数和 op、数学逻辑是什么”，并明确哪些底层 CUDA kernel 尚未通过动态 trace 证实。
>
> 本文不是 vLLM-Omni 路线，不讨论其他 H3 workflow，也不重新估算整套显存。显存只在与权重加载、offload、prefetch 和 tensor 生命周期直接相关处展开。

## 0. 先给出结论

1. **这是纯 T2V，不是实际带首帧的 I2V。** 模板节点叫 `MiniMaxH3ImageToVideo`，但本次 `first_frame` 和 `last_frame` 都未连接。因此不走 keyframe VAE encode，也不向 DiT 注入图像条件 latent。
2. **音视频是在同一个 DiT 中联合生成的。** 每次去噪调用把文本、目标音频、目标视频 pack 成一个序列做 full self-attention，不是先出视频再用另一个音乐模型配音。
3. **默认 20 步只有 positive，没有 negative CFG。** 每次 sigma 调用一次 DiT；20 步共 20 次主模型 forward，主 50 层形成 1000 次 packed attention。文本编码器只编码一次，`condition_proj + 2 层 token refiner` 也只在采样开始前处理一次，而不是重复 20 次。
4. **主 DiT 的大 GEMM 是 INT8 ConvRot 路径，但不是整个模型都 INT8。** 只有 50 个主 block 中的 `qkv_proj`、`out_proj`、`fc1`、`fc2` 共 200 个 Linear 带 INT8 + ConvRot 元数据。AdaLN、patch projection、token refiner、final head 等必须分开看。
5. **Qwen 的 NVFP4 是存储格式，不等于本流程做 FP4 GEMM。** 全部 350 个 NVFP4 Linear 都设置 `full_precision_matrix_mult=true`，文本 encode 又默认 `full_precision_mm=True`。因此默认是反量化后普通全精度 `F.linear`，不是 `scaled_mm_nvfp4`。
6. **视频 VAE 的 3D CNN encoder 在本 T2V 中完全不执行。** 实际 decode 是 `post_quant_conv` 加 36 层 ViT3D decoder，并且内部同时做空间 tile、时间 chunk。音频 VAE 也只跑 BigVGAN decoder，不跑 encoder 的 CausalAttention。
7. **日志中的 `Using pytorch attention` 是 Python dispatcher 的选择，不是最底层 kernel 名。** H3 主 attention 经 `attention_pytorch` 到 PyTorch SDPA；SDPA 最终是 Flash、cuDNN 还是 mem-efficient 后端，必须有 profiler/CUPTI 才能断言。
8. **本报告不伪造 kernel trace。** 下文会给出静态可确定的 PyTorch op、Comfy Kitchen custom op 和 C++ binding，以及各回退条件；不把 backend “注册可用列表”冒充实际执行列表。

## 1. 分析基线与证据等级

### 1.1 固定版本和实际输入

| 项目 | 本报告基线 |
|---|---|
| ComfyUI | v0.45.0 |
| Git commit | `989d93d0d968e704c589ce0146bb18c77f6b50cc` |
| Git 状态 | 核验时 `git status --porcelain=v1` 为空 |
| Workflow | `/home/kaokaolv/comfyui_h3/video_minimax_h3_t2v.json` |
| 实际 API prompt | `logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json` |
| Prompt ID | `d16467d5-c6a1-47c1-97d7-a95a616aa1dc` |
| GPU | 物理 GPU 2，NVIDIA B200；进程内为 `cuda:0` |
| PyTorch / Kitchen | PyTorch 2.13.0+cu130；comfy-kitchen 0.2.41；comfy-aimdo 0.6.0 |
| 实际分辨率 | 864×480；由外层 `ResolutionSelector` 覆盖 subgraph 内的旧 widget 值 |
| 时长与帧率 | 124 帧，24 fps，约 5.1667 秒 |
| 采样 | `res_multistep` + `simple`，20 steps，denoise=1 |
| 条件 | 纯文本；无首帧、尾帧、参考音频、ControlNet、negative prompt |
| Turbo/LoRA | `turbo_mode=false`；本次主模型 0 patches，不执行 Turbo LoRA 分支 |
| 输出 | `ComfyUI/output/video/MiniMax_H3_00002_.mp4`，H.264 视频和 32 kHz 双声道 AAC 音频 |

注意：原始 JSON 的 subgraph 里仍能看到 1344×768、另一个示例 prompt 和 LoRA 文件名。**分析必须以实际提交的 `api_prompt.json` 为准，而不能只看这些未生效的 widget 值。**

四个实际模型文件：

| 模块 | 文件 | 本机字节数 | 本流程计算角色 |
|---|---|---:|---|
| 文本编码器 | `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | Qwen3-VL-32B 前 50 层，文本条件 |
| 联合 DiT | `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | 20 步联合音视频 flow 去噪 |
| 视频 VAE | `models/vae/minimax_h3_video_vae_int8_convrot.safetensors` | 2,811,065,184 | 视频 latent → RGB frames |
| 音频 VAE | `models/vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | 音频 latent → 32 kHz 双声道波形 |

### 1.2 三类证据标记

本文使用以下措辞严格区分证据强度：

- **[实测]**：来自实际 API prompt、执行事件、日志、内存采样、媒体检查或 checkpoint 元数据。
- **[静态路径]**：由固定版本源码可确定的调用关系。若不存在运行时 shape/backend 限制，它就是对本流程的高可信解释。
- **[候选/回退]**：依赖运行时 dtype、shape、设备能力、环境变量、tensor 对齐、异常或 backend 注册的路径。没有动态 trace 时不把候选写成已执行。

实际没有采集 CUPTI、Nsight Systems、`torch.profiler` kernel 表或 CUDA Graph launch trace。所以：

- 可以严谨列出 `F.linear`、`scaled_dot_product_attention`、`ck.int8_linear`、`ck.rms_rope_split_half_` 等调用；
- 可以列出 `_C.cutlass_int8_dequant`、`_C.cublas_gemm_blockwise_fp4` 等源码 binding 的选择条件；
- **不能**给出实际 CUDA kernel 的精确 launch 次数、耗时、SM 占用、寄存器数或某次 SDPA 的模板实例名。

### 1.3 实际节点时序

**[实测]** `events.json` 中第一次 `executing` 事件顺序如下。`next-delta` 是到下一节点首次执行事件的间隔，包含调度和事件间隔，不等于纯 kernel 时间。

| 节点 | 作用 | 开始时刻（秒） | 到下一节点的间隔 |
|---|---|---:|---:|
| `140:120` | Audio VAE Loader | 0.425 | 0.100 |
| `140:119` | Video VAE Loader | 0.525 | 0.344 |
| `140:128` | CLIP Loader | 0.868 | 1.826 |
| `140:133/132`、`115` | 时长、帧数、分辨率计算 | 2.694–3.207 | 小计算节点 |
| `140:131` | 文本编码 + 空 AV latent | 3.207 | 约 351.986 |
| `140:125` | `SamplerCustomAdvanced` | 357.110 | 约 555.584 |
| `140:121` | 音频 VAE decode | 912.694 | 约 10.345 |
| `140:122` | 视频 VAE decode | 923.038 | 约 26.853 |
| `140:130` | `CreateVideo` | 949.891 | 0.102 |
| `92` | `SaveVideo` | 949.994 | 约 2.073 |

`logs/comfyui_server_profiled.log:90` 进一步说明：sampler 第一次初始化约 482.20 秒，之后单步约 1.71 秒，20 步主体约 34 秒。**不能把 sampler 的约 556 秒理解为 20 个稳定 DiT step 的时间，也不能把整次 952 秒当作稳定吞吐。**

## 2. 端到端调用栈

下面先给出全流程，再逐层展开。包装器、hook、patch 分支保留在关键位置；本次没有模型 patch，因此不会改写主 forward。

```text
API 提交展开后的 prompt graph
  ├─ VAELoader(audio)
  │    └─ nodes.VAELoader.load_vae
  │       └─ comfy.sd.VAE → MiniMaxH3AudioVAE → CoreModelPatcher
  ├─ VAELoader(video)
  │    └─ comfy.sd.VAE → MiniMaxH3VideoVAE（mixed_precision_ops）
  ├─ CLIPLoader(type=minimax)
  │    └─ comfy.sd.load_clip
  │       └─ load_text_encoder_state_dicts
  │          └─ MiniMaxH3Tokenizer + MiniMaxH3TEModel_
  ├─ UNETLoader
  │    └─ comfy.sd.load_diffusion_model
  │       └─ model_detection → supported_models.MiniMaxH3
  │          └─ model_base.MiniMaxH3 → ldm.minimax.model.MiniMaxH3Model
  ├─ MiniMaxH3ImageToVideo.execute
  │    ├─ _empty_av_latent
  │    ├─ clip.tokenize(prompt, images=[])
  │    └─ clip.encode_from_tokens_scheduled
  │       └─ Qwen 前 50 层 + modality tags
  ├─ BasicScheduler / KSamplerSelect / RandomNoise / BasicGuider
  └─ SamplerCustomAdvanced.execute
       └─ Guider_Basic.sample
          ├─ pack_latents(video, audio)
          ├─ prepare_sampling / process_conds
          │    └─ MiniMaxH3.extra_conds（文本 refiner 与 PackedLayout 仅一次）
          └─ KSAMPLER.sample → sample_res_multistep（20 次）
               └─ BaseModel.apply_model
                  └─ MiniMaxH3Model.forward / _forward
                     ├─ 解开 audio schedule scale
                     ├─ video/audio patchify + pack
                     ├─ time embedding + AdaLN segments
                     ├─ 50 × DiTBlock
                     └─ final video/audio heads + unpack

       sampler 输出 NestedTensor(video, audio)
          ├─ VAEDecodeAudio
          │    └─ sd.VAE.decode → MiniMaxH3AudioVAE.decode → BigVGAN
          ├─ VAEDecode
          │    └─ sd.VAE.decode → MiniMaxH3VideoVAE.decode
          │       └─ temporal chunks → spatial tiles → ViT3DDecoder
          ├─ CreateVideo → VideoFromComponents
          └─ SaveVideo → VideoFromComponents.save_to → PyAV/FFmpeg MP4
```

## 3. 模型是怎样加载的

### 3.1 共同的 safetensors 与 DynamicVRAM 路径

`comfy/utils.py:192-238` 的 `load_torch_file` 对 `.safetensors` 有两条读取方式：

1. 启用 AIMDO 时，`load_safetensors` 通过 `comfy_aimdo.model_mmap.ModelMMAP` 映射文件，解析 header 和 dtype/shape/offset，让 tensor storage 带文件 slice 与 mmap 引用；
2. 非 AIMDO 时，用 `safetensors.safe_open` 逐 tensor 读取。

本机日志证明 DynamicVRAM 已启用：`main.py:318-320` 将 `CoreModelPatcher` 替换为 `ModelPatcherDynamic`。因此“loader 返回模型”不等于“全部权重已常驻 GPU”。实际流程是：

```text
state_dict / mmap-backed tensor
  → 根据 key 识别模型结构
  → 建立 nn.Module 树（不主动初始化大权重）
  → ModelPatcherDynamic 注册权重、虚拟地址范围、host buffer
  → load_state_dict(..., assign=True)
  → 执行某个节点时 load_models_gpu
  → 首次使用时 fault / host-to-device / cast / dequant
  → 按 block 或权重需求驻留，必要时释放、换出或重用
```

`ModelPatcherDynamic._vbar_get` 中 VBAR 的地址空间可以按 model size 的 10 倍预留，**这是虚拟地址空间策略，不是物理显存直接分配 10 倍**。`fast_disk=False` 是本机日志中的实际存储策略。

通用权重访问入口为：

```text
CastBiasWeightContext
  → cast_bias_weight
     ├─ VBAR fault / cast_modules_with_vbar
     ├─ resolve_cast_module_with_vbar
     ├─ weight/bias function（LoRA、patch；本次为空）
     └─ 按输入 dtype/device 对齐
  → Linear / Conv / RMSNorm 等计算
  → uncast_bias_weight + stream 同步/解 pin
```

这套路径解释了为什么“模型 dtype”“checkpoint 存储 dtype”“某次 GEMM 的输入/权重 dtype”不是完全相同的概念。

### 3.2 DiT 的识别、构造和量化

调用链：

```text
nodes.UNETLoader.load_unet                         nodes.py:1003
  → comfy.sd.load_diffusion_model                 sd.py:2557
     → comfy.utils.load_torch_file
     → load_diffusion_model_state_dict            sd.py:2460
        → convert_old_quants
        → model_detection.model_config_from_unet
        → supported_models.MiniMaxH3.get_model
        → model_base.MiniMaxH3
        → MiniMaxH3Model
        → CoreModelPatcher / ModelPatcherDynamic
        → model.load_model_weights(..., assign=True)
```

H3 的检测签名在 `comfy/model_detection.py:401-435`：

- 同时有 `video_patch_proj.weight`、`condition_proj.weight`；
- 有 `audio_patch_proj.weight` 时是音视频 `minimax_h3`，否则是 image 版本；
- 从权重 shape 读出 hidden size、block 数、head 数、FFN 宽度、text dim、time embedding 等；
- 从 `final_layer.*_out` 的形状反推 latent channels 和 PDD head 数。

`comfy/supported_models.py:986-1012` 指定：

- `latent_format=MiniMaxH3AV`；
- `shift=12.0`、`audio_shift=3.0`；
- `memory_usage_factor=0.114`；
- 推理 dtype 候选为 BF16/FP32。

**[实测]** 本机日志为 `model weight dtype torch.bfloat16, manual cast: torch.bfloat16`、`model_type FLOW_AV`，主模型 staged 约 19995 MB，0 patches。

checkpoint 元数据是逐层 uint8 JSON `.comfy_quant`，由 `_load_quantized_module`（`comfy/ops.py:1342-1485`）消费。当前 DiT：

| 位置 | 数量 | 格式 | ConvRot | full-precision GEMM |
|---|---:|---|---|---|
| `blocks.{0..49}.attn.qkv_proj` | 50 | `int8_tensorwise` | true，group 256 | false |
| `blocks.{0..49}.attn.out_proj` | 50 | 同上 | 同上 | false |
| `blocks.{0..49}.mlp.fc1` | 50 | 同上 | 同上 | false |
| `blocks.{0..49}.mlp.fc2` | 50 | 同上 | 同上 | false |

`mixed_precision_ops` 将这些权重包装为 `QuantizedTensor(TensorWiseINT8Layout)`，其他普通权重仍由普通 tensor/cast 逻辑管理。不能用文件名推断 `token_refiner` 或 AdaLN 也走了 INT8。

主结构（`MiniMaxH3Model.__init__`，`model.py:507-550`）：

- hidden size 5376，50 层；
- 56 heads，head_dim=128，inner dim=7168；
- FFN hidden 14336，`fc1` 输出两份共 28672；
- video latent 24 channels，patch `(1,2,2)`，投影输入宽度 96；
- audio latent 32 channels、双声道、40 Hz；
- text dim 5120；
- `TimeEmbedder(256 → 5376 → 2688)`；
- `condition_proj + 2 层 TokenRefiner`；
- RoPE inverse-frequency 长度 16；
- `pdd_steps=1`，本 workflow 不用 PDD 多头路径。

### 3.3 文本编码器的识别与构造

```text
nodes.CLIPLoader.load_clip(type='minimax')
  → comfy.sd.load_clip
     → load_torch_file / convert_old_quants
     → load_text_encoder_state_dicts
        → detect_te_model → TEModel.QWEN3VL_32B
        → comfy.text_encoders.minimax.te(...)
        → MiniMaxH3TEModel_
        → MiniMaxH3ClipModel
        → MiniMaxQwen3VL / Qwen3VL
        → Llama2_(Qwen3VL_32BConfig)
```

识别依据是视觉 DeepStack key 与 `model.layers.49.self_attn.q_proj.weight`（`comfy/sd.py:1841-1843`）。`Qwen3VL_32BConfig`（`text_encoders/llama.py:415-424`）明确：

- hidden size 5120；intermediate size 25600；
- 64 query heads、8 KV heads，head_dim=128；
- 50 层而不是原始 64 层；
- `final_norm=False`、`lm_head=False`。

因此这里不是拿 Qwen 做文本续写，而是取第 50 层 hidden state 作为 H3 条件。日志确认初始化 dtype 为 FP16，但 `SDClipModel.forward` 会对 token embedding 请求 FP32，最后 hidden condition 也 `.float()`；权重与激活实际 GEMM dtype 由 cast 路径随输入对齐。**不能仅凭“CLIP dtype FP16”就写成“每次 NVFP4 反量化后 FP16 GEMM”。**

### 3.4 两个 VAE 的识别

`VAELoader.load_vae`（`nodes.py:844-873`）调用 `comfy.sd.VAE`。`sd.py:1050-1147` 根据 state-dict 签名区分：

- `MiniMaxH3VideoVAE`：24 latent channels，16×空间压缩，特殊 17k+5 ↔ 5k+2 时间映射；检测 `.comfy_quant` 后使用 mixed precision ops；`working_dtypes=[FP16, FP32]`，本机选择 FP16；
- `MiniMaxH3AudioVAE`：由 `pre_block.attn.zero_k_bias` 识别，32 channels、双声道、32 kHz、800 samples/latent frame，只使用 FP32。

二者都创建 patcher 并 `load_state_dict(..., assign=is_dynamic)`。本流程加载 video VAE 时包含 encoder 和 decoder，但“加载”不意味着 encoder 会在 T2V 中执行。

## 4. 文本条件：Qwen 前 50 层到底做了什么

### 4.1 T2V tokenize 不是 chat prompt

`MiniMaxH3Tokenizer.tokenize_with_weights`（`text_encoders/minimax.py:187-241`）在本次 `images=[]`、无 ref items 时，只对原始 prompt 调 tokenizer：

- 不套 `<|im_start|>user...` chat template；
- 不插入 vision block；
- 不追加文本生成任务；
- 返回 `{qwen3vl_32b: [[(token_id, weight), ...]]}`。

`token_tags_from_embeds_info` 先建立全 1 的 tags。本流程无 image embed，因此所有文本 token 的 modality tag 都是 **1**。该 tags 后续不是送入 Qwen 的 embedding，而是交给 H3 DiT 的 AdaLN 调制。

### 4.2 文本编码函数链

```text
MiniMaxH3ImageToVideo.execute
  → CLIP.encode_from_tokens_scheduled              sd.py:345
     → CLIP.encode_from_tokens                     sd.py:408
        → CLIP.load_model(tokens)
        → MiniMaxH3ClipModel.encode_token_weights
           → SDClipModel.process_tokens
           │    ├─ token ids → int8 embedding lookup
           │    └─ 无视觉 embed 可插入
           ├─ MiniMaxQwen3VL.forward
           │    ├─ 记录 last_token_tags
           │    └─ Qwen3VL.forward → Llama2_.forward
           └─ 返回 hidden state + minimax_token_tags
```

int8 embedding 不会把整张词表先反量化。`mixed_precision_ops.Embedding.forward_comfy_cast_weights`（`comfy/ops.py:1927-1953`）走 `dequantize_embedding(qdata, params, input)`，即按输入索引取行并反量化。

### 4.3 每一层 Qwen 的函数和 op

`Llama2_.forward`（`text_encoders/llama.py:1005-1134`）：

1. 使用已提供的 embeddings；
2. 计算 causal RoPE 频率；
3. prompt 长度大于 1 时创建上三角 causal mask；
4. 顺序执行 50 个 `TransformerBlock`；
5. `self.norm is None`，所以最后不做 final RMSNorm；
6. 不做 LM head 和 next-token sampling。

每个 block（`llama.py:699-859`）为：

```text
residual = x
x = input_layernorm(x)                     RMSNorm
q = q_proj(x); k = k_proj(x); v = v_proj(x)  三个独立 Linear
q/k = q_norm/k_norm(q/k)                  per-head RMSNorm
q/k = apply_rope_split_half(q, k)         Comfy Kitchen RoPE
attn = optimized_attention(q, k, v, mask)  GQA: 64 Q heads / 8 KV heads
x = residual + o_proj(attn)

residual = x
x = post_attention_layernorm(x)
gate = gate_proj(x)
up = up_proj(x)
x = down_proj(silu(gate) * up)
x = residual + x
```

`merged_qkv=False`、`merged_mlp=False`。虽然类中有 `CombinedLinear`，它只支持特定 INT8 layout，不支持这里的 NVFP4；所以 `prepare` 不组合，`forward` 返回 `NotImplemented`，再走分开的 q/k/v 和 gate/up。每层共 7 个 NVFP4 权重 Linear，50 层共 350 个，和 checkpoint 元数据完全对应。

`self_attn.o_proj` 与 `mlp.down_proj` 还带 `pre_quant_scale`，`MixedPrecisionOps.Linear.forward` 在矩阵乘之前先做逐通道乘法，共 100 个此类 scale。

### 4.4 最重要的量化纠正：本流程不是 FP4 GEMM

**[checkpoint 实测]** 350 个 NVFP4 Linear 全部是：

```json
{"format": "nvfp4", "full_precision_matrix_mult": true}
```

同时 `SDClipModel` 在 `comfy/sd1_clip.py:109-117` 以 `full_precision_mm=True` 创建 mixed precision ops。编码路径没有切到 generation 用的 `use_quantized_matmul` context。

所以实际静态选择是：

```text
NVFP4 packed weight + weight_scale(block, FP8 E4M3) + weight_scale_2(tensor, FP32)
  → QuantizedTensor
  → CastBiasWeightContext
  → weight.dequantize()（ck.dequantize_nvfp4 的可用实现）
  → 普通 F.linear，而非 QuantizedTensor 双输入 scaled_mm_nvfp4
```

如果走快速 NVFP4 GEMM，`tensor/nvfp4.py:212-260` 会要求输入和权重都是 NVFP4 QuantizedTensor，然后调用 `ck.scaled_mm_nvfp4`，再进入 `_C.cublas_gemm_blockwise_fp4`。**当前元数据和 full-precision 设置使这条路径不成立。**

NVFP4 仍减少权重存储与 host/device 传输量，但不能因此声称本流程利用 Blackwell FP4 Tensor Core 完成文本大矩阵乘。底层反量化后 `F.linear` 的库和具体 cuBLAS kernel 仍需 profiler 才能精确命名。

## 5. 空 AV latent、噪声和两种“pack”

### 5.1 时间形状

`nodes_minimax_h3.py:50-107` 的规则：

```text
frame_count = align_frame_count(max(5, length))，要求 frame_count % 17 == 5
video_latent_t = 2                            frame_count <= 5
               = ((frame_count-5)//17)*5 + 2  其他情况
audio_t = round((frame_count/24) * 40)
```

本次 124 帧：

```text
124 = 17*7 + 5
video latent T = 5*7 + 2 = 37
audio latent T = round(124/24*40) = 207
```

`_av_latent` 在 intermediate device 上建立零 tensor：

```text
video: [1, 24, 37, 30, 54]   # 480/16=30，864/16=54
audio: [1, 32, 2, 207]       # 32 channels、stereo、40 Hz
samples = NestedTensor((video, audio))
```

这里两个 VAE 的 mean/std 不在空 latent 创建阶段应用。真正的随机初始值由 `Noise_RandomNoise.generate_noise` 调用 `prepare_noise` 生成：对每个 nested stream 在 CPU 上用固定 seed 生成 FP32 `torch.randn`，然后移到 sampler device。本次 seed 是 `757358688076805`。

### 5.2 sampler 的 flat pack 与 DiT 的 token pack 不同

`NestedTensor` 只是持有两个不同 shape tensor 的轻量容器，重载 `+ - * / .to()` 逐 stream 应用，不是 PyTorch native nested tensor。

进入采样时，`CFGG uider.sample`（`comfy/samplers.py:1277-1347`）用 `utils.pack_latents` 把两个 stream 各自 flatten 后在最后一维拼接：

```text
video numel = 24*37*30*54 = 1,438,560
audio numel = 32*2*207    =    13,248
packed tensor shape        = [1, 1, 1,451,808]
```

这是 **latent storage flatten**，不是 Transformer token 序列。`BaseModel._apply_model` 再依据保存的 `latent_shapes` 调 `unpack_latents`，恢复两种原始 shape 后才送进 H3 DiT。采样结束重新恢复成 `NestedTensor`，两个解码节点各自取 stream：

- `nodes.VAEDecode.decode` 用 `unbind()[0]`；
- `vae_decode_audio` 用 `unbind()[-1]`。

## 6. 调度器、采样器与 flow 的数学关系

### 6.1 simple scheduler 的 sigma

`ModelSamplingDiscreteFlow.set_parameters` 在 1000 个网格点上计算：

```text
sigma = time_snr_shift(shift, t)
      = shift*t / (1 + (shift-1)*t)
video shift = 12，audio shift = 3
```

`simple_scheduler`（`comfy/samplers.py:645-652`）从这 1000 个值中均匀取 20 个，并补 0。本次 video sigma 为：

```text
1,
0.995633188, 0.990825688, 0.985507246, 0.979591837,
0.972972973, 0.965517241, 0.957055215, 0.947368421,
0.936170213, 0.923076923, 0.907563025, 0.888888889,
0.865979381, 0.837209302, 0.800000000, 0.750000000,
0.679245283, 0.571428571, 0.387096774,
0
```

`CONST.noise_scaling` 在第一个 step 形成：

```text
x_sigma = sigma * noise + (1-sigma) * empty_latent
```

本次 empty latent 为 0，所以是 `sigma * noise`，没有图像 latent 混合。

### 6.2 audio 的 schedule 不是独立另跑一个 sampler

视频 sigma 通过 `time_shift_sigma` 转到音频（`model.py:38-41`）：

```text
base = sigma_v / (12 + sigma_v*(1-12))
sigma_a = 3*base / (1 + (3-1)*base)
```

`ModelSamplingAV.audio_scale = shift/audio_shift = 4`。H3 在 `model_base.py:2230-2252` 把 packed audio stream 按这个固定比例携带，使外层采样器仍能使用单一 sigma schedule；`MiniMaxH3Model.forward` 再用：

```text
carry = sigma_a / sigma_v
x_audio_own = x_audio_carried * carry
```

恢复 audio 自身的输入。输出端还按 `model.py:710-713` 的导数公式把 audio velocity 转回 carried variable 的导数，最后 `process_latent_out` 除以 4。不要把这个 4 理解为音频步数、采样率或额外四次模型调用。

### 6.3 `res_multistep` 每步做什么

调用链：

```text
SamplerCustomAdvanced.execute
  → Guider_Basic.sample
     → KSAMPLER.sample
        → model_sampling.noise_scaling
        → sample_res_multistep
           → res_multistep(..., eta=0, cfg_pp=False)
```

`comfy/k_diffusion/sampling.py:1462-1528` 中：

- 首个有效 step 使用 Euler：`d=(x-denoised)/sigma`，`x += d*(sigma_next-sigma)`；
- 后续中间 step 使用上一个 `denoised` 做二阶 multistep；当 `sigma_down == 0`（包括最后的非零 sigma→0 step）时，也回到同一个 Euler 分支；
- `t=-log(sigma)`，`h=t_next-t`；
- `phi1=expm1(x)/x`，`phi2=(phi1-1)/x`；
- 非 CFG++ 时 `x = exp(-h)*x + h*(b1*denoised + b2*old_denoised)`；
- `eta=0`，因此 ancestral `sigma_up=0`，**step 中不另加噪声**。随机数只用于初始 noise。

这些是普通 tensor 运算：`log`、`exp/expm1`、乘加、除法、`nan_to_num`、slice/view。不存在名为“res_multistep CUDA kernel”的单一底层算子。

### 6.4 为什么没有 CFG 双分支

`BasicGuider` 构造 `Guider_Basic`，只调用 `set_conds(positive)`，继承 `CFGG uider` 的默认 `cfg=1.0`。基类 `_calc_cond_batch` 对 negative 的 None 位置保留零输出和极小计数，不执行 negative 模型；`0 + (positive-0)*1` 最终等于 positive。

因此：

- 没有空的 negative prompt 编码；
- 没有每步两次 DiT；
- 没有 `uncond + scale*(cond-uncond)` 的非平凡 guidance。

主调用仍是 `predict_noise → sampling_function → calc_cond_batch → model.apply_model`。

### 6.5 DiT 输出怎样回到 x0

`model_base.py:154-156` 选择 `ModelSamplingAV + CONST`。`CONST.calculate_denoised`：

```text
x0 = model_input - sigma * model_output
```

而 `MiniMaxH3Model._forward` 最后返回：

```python
[-video_out, -audio_out]
```

也就是 checkpoint 所表示速度方向的负号适配到 ComfyUI 的 flow 约定。采样器拿到的 `denoised` 是 x0，不是原始网络 head 的直接输出。

## 7. H3 DiT：每次 forward 的完整逻辑

### 7.1 文本 refiner 和 PackedLayout 只在采样前构造一次

`MiniMaxH3.extra_conds`（`model_base.py:2254-2300`）在 `process_conds` 阶段执行：

```text
Qwen hidden [1,L,5120]
  → condition_proj: 5120 → 5376
  → TokenRefiner：2 × RefinerBlock + final RMSNorm
  → refined context [1,L,5376]

同时：
  → 收集 minimax_token_tags
  → 构造 minimax_payload
  → 预建 PackedLayout(text_len,37,30,54,207)
```

`CFGG uider.inner_sample` 在采样前调用 `process_conds`，所以这不是 20 步重复计算。`_embed_and_pack` 中发现 context 最后一维已经是 5376，就不再重复 refiner。

`RefinerBlock` 和主 block 使用相同的 attention/MLP 构件，但没有主 DiT 的 AdaLN：它是 `RMSNorm → attention → residual add → RMSNorm → MLP → residual add`。

### 7.2 真正的 packed token 序列

本次无 cond、无 ref，顺序为：

```text
[text (L tokens) | target audio (414 rows) | target video (14,985 rows)]
```

计算：

```text
video rows = 37 * (30/2) * (54/2)
           = 37*15*27 = 14,985
audio rows = 207*2 = 414
S = L + 414 + 14,985 = L + 15,399
```

`PackedLayout` 为每一行生成：

- `(t,h,w)` 三维位置；
- segment 起止位置和 kind；
- target image/audio 的位置索引；
- `img_update/audio_update`（本 T2V 全为 true）；
- 用于缓存和校验的 signature。

视频时间坐标不是简单的 `frame_index/4`。`FRAME_PER_TOKEN=(1,4,4,4,4)`，五个 latent 周期对应 1+4+4+4+4=17 帧；坐标 span 还乘 `FRAME_RESCALE=5/3`。音频每个 latent frame 前进 1，两个声道分别固定在空间 w 网格的两端。文本位置的 t 是递增序号。

### 7.3 patchify、线性投影与装配

`MiniMaxH3Model._forward` 先对 video 做 `pad_to_patch_size`，本 shape 已满足 `(1,2,2)`，因此无实际新增 padding。模型限制 batch size=1。

```text
video: [1,24,37,30,54]
  → patchify_video
  → [14,985, 96]（24*1*2*2=96）
  → video_patch_proj → [14,985,5376] → 转 compute dtype

audio: [1,32,2,207]
  → pack_audio（channel-major：左声道全时间，再右声道）
  → [414,32]
  → audio_patch_proj → [414,5376] → 转 compute dtype

最后按 segments 写入：
h = torch.empty([S,5376], dtype=context.dtype, device=...)
```

输入 patch projection 的模型构造请求 FP32，且 patchify 前输入转 FP32；完成投影后才转 `context.dtype`。通用 `CastBiasWeightContext` 负责相应权重 dtype/device，不能把所有 projection 一概归类为 INT8 GEMM。

### 7.4 时间嵌入和逐模态 AdaLN

当前 step 有 video sigma 和由其推导的 audio sigma：

```text
t_v = 1 - sigma_v
t_a = 1 - time_shift_sigma(sigma_v,12,3)
```

本 T2V 无条件行、无 denoise mask，通常 `unique_t={t_v,t_a}`。`TimeEmbedder` 对每个唯一 t：

1. FP32 构造 256 维 cos/sin 特征；
2. `proj_in: 256→5376`；
3. SiLU；
4. `proj_out: 5376→2688`。

每个主 block 的 `adaln_proj` 再产生 6 组调制向量，并为 3 种 modality 预留行：

- video / visual：tag 0；
- text：tag 1；
- audio：tag 2。

在本 T2V 的简单情况下，`t_emb` 有 2 个时间值，经 `6×3` 行展开后，每个 block 的 shift/scale/gate 按 text、audio、video 三个 segment 索引。**“条件文本只处理一次”不等于“AdaLN 只计算一次”；50 个主 block 的 AdaLN projection 在每个 step 都会执行。**

### 7.5 50 个 DiTBlock 的函数和 op

`_run_blocks` 先为整个 packed sequence 计算一次 RoPE table，再循环 50 层。每层 `DiTBlock.forward`（`model.py:315-321`）为：

```text
(shift_msa, scale_msa, gate_msa,
 shift_mlp, scale_mlp, gate_mlp) = adaln_proj(t_emb)

h = RMSNorm(x)
h = h * (1 + scale_msa[segment]) + shift_msa[segment]

qkv = attn.qkv_proj(h)                  INT8 ConvRot Linear
q,k,v = split(qkv)
q,k = rms_rope_split_half_(q,k,rope)     fused per-head RMSNorm + partial RoPE
attn = optimized_attention(q,k,v)       full self-attention，无 mask
attn = attn.out_proj(attn)              INT8 ConvRot Linear
x = x + gate_msa[segment] * attn        分段 addcmul_

h = RMSNorm(x)
h = h * (1 + scale_mlp[segment]) + shift_mlp[segment]
h = mlp.fc1(h)                           INT8 ConvRot Linear
h = mlp.fc2(h)，输入经 swiglu            linear_input_act + INT8 ConvRot
x = x + gate_mlp[segment] * h            分段 addcmul_
```

主 attention 的 tensor 形状：

```text
qkv_proj: [S,5376] → [S,21504]  # 3*7168
q/k/v:    [S,7168]
per head: [1,56,S,128]
```

RoPE 由 3 个轴、每轴 16 个 inverse frequency 构成：先得到 48 个角度，再复制为 96 个；`rope_rotation_table` 产生 48 个 2×2 旋转矩阵。`rot_dim=96`，因此只旋转每个 head 前 96 维，**RMSNorm 仍覆盖完整 128 维**。推理用原地 `rms_rope_split_half_`；训练路径才用保留 autograd 的非下划线版本。

`_mod_scale_shift` 和 `_mod_gate` 按 segment 原地执行 `mul/add` 和 `addcmul_`。这里没有调用 Kitchen 的通用 `adaln` custom op；源码用的是 PyTorch pointwise/TensorIterator op。

MLP：

```text
fc1: 5376 → 28672
swiglu: 对半切分后形成 SiLU(gate)*up，宽度 14336
fc2: 14336 → 5376
```

`comfy.ops.linear_input_act` 允许把 SwiGLU 合并进 fc2 的在线激活量化；无法融合时退回显式 SwiGLU + Linear。它是 dispatch helper，不代表永远存在一个同时完成 fc2、SwiGLU、量化、residual 的单独大 kernel。

### 7.6 attention 是单流 full self-attention

H3 的主 `Attention.forward` 传 `mask=None`，所以文本、音频和视频 token 都能直接相互注意。这不是：

- text-to-video cross-attention；
- 独立视频 DiT 加独立音频 DiT；
- 只对音频或视频各自做 attention 后再拼接。

它是一串带 3D RoPE 和 modality AdaLN 的 full self-attention。文本通过 packed token 参与整层计算，因此音频/视频能在同一套网络中学会时间、语义和声学事件的对应关系。

调用链：

```text
optimized_attention
  → attention_pytorch（本机日志明确选择）
  → comfy.ops.scaled_dot_product_attention
  → torch.nn.functional.scaled_dot_product_attention
  → PyTorch 运行时选择 SDPA backend（未 trace，不写死）
```

主 attention 不是 `ck.int8_attention`；后者出现在量化 video VAE decoder 的特定分支。

### 7.7 final layer 与输出

`FinalLayer.forward` 分别截取 target audio 和 target video：

1. 对两个 segment 分别 RMSNorm、shift/scale 调制；
2. `_modulate` 将输入转 FP32；
3. `audio_out` 得到 `[414,32]`，`video_out` 得到 `[14985,96]`；
4. `unpack_audio` 恢复 `[1,32,2,207]`；
5. `unpatchify_video` 恢复 `[1,24,37,30,54]`；
6. 裁掉可能的 patch padding；
7. 返回负的 video/audio 输出。

源码构造上把 patch projection 和 output head 作为 FP32 边界处理，但具体权重访问仍受 mixed precision/cast 管理。因此应避免“全 DiT 都是 BF16 GEMM”或“所有非量化权重始终 FP32 常驻”这两种过度概括。

## 8. Kernel/op 专题：量化、融合与回退

### 8.1 INT8 + ConvRot 的数学意义

当前 DiT 和 video VAE decoder 的量化格式是 `int8_tensorwise`，并额外带 `convrot=true, convrot_groupsize=256`。它不是另一个 `convrot_w4a4` 格式。

ConvRot 在权重离线量化前按 256 维分组施加正交 Hadamard 类旋转；推理时对激活施加相同旋转。理想数学上：

```text
(Hx) · (Hw) = x · w
```

旋转不改变内积，但能分散 outlier，使 INT8 动态量化范围更平稳。`TensorWiseINT8Layout` 的 `convrot` 要求 per-channel 权重 scale；在线激活则按行求 scale 并量化。它同时包含：

- 权重的 INT8 storage 与 scale；
- 激活的旋转 + per-row INT8 quantization；
- INT8 矩阵乘和反量化输出；
- 可选输入预处理（RMSNorm、SwiGLU）和 residual epilogue。

本 checkpoint 离线量化已完成，运行时不重新量化整份权重。**每次 Linear 的激活量化是实际需要的在线工作，不应只统计 GEMM 而忽略它。**

### 8.2 DiT 主 Linear 的分发链

```text
MixedPrecisionOps.Linear.forward
  → quantize_input=False（int8_tensorwise 配置）
  → forward_comfy_cast_weights
  → F.linear(input, QuantizedTensor weight, bias)
  → QuantizedTensor.__torch_dispatch__
  → TensorWiseINT8Layout._handle_int8_linear_tensorwise
  → torch.ops.comfy_kitchen.int8_linear
  → backend registry 根据 shape/dtype/device 选择 CUDA 或 eager
```

带输入预处理时：

```text
comfy.ops.linear_input_act
  → _try_quantized_linear_input_act
  → TensorWiseINT8Layout.linear_input_act
  → ck.int8_linear(..., input_act='swiglu'/'rms_norm', residual=...)
```

本机 registry 默认优先级为 CUDA → eager；Triton 在启动日志中为 `disabled: True`。CUDA backend `int8_linear`（`backends/cuda/__init__.py:2128` 起）的主要分支为：

1. 需要时先做输入激活；
2. 若 ConvRot 与 K 条件合适，使用 `_C.quantize_int8_rowwise_convrot64` 做融合的旋转和行量化；否则回退到显式 Hadamard/rotate + row-wise quantize；
3. M=1/2 可尝试 `_C.int8_linear_m1` 或 `_C.int8_gemv_dequant`；
4. 大 M 在设备支持且未禁用 CUTLASS 时，优先 `_C.cutlass_int8_dequant`；
5. 有 residual 且 dtype/对齐允许时，尝试 `_C.cutlass_int8_dequant_residual`；
6. 不满足时回退 `_C.cublas_gemm_int8`（INT32 accumulator）+ `_C.dequantize_int8_linear`；
7. Python/eager fallback 则分解为普通 PyTorch op。

DiT 的 M 是 15,399+L，远大于 GEMV 分支，因此静态上应看大 M 的量化 + CUTLASS/cuBLAS 分支，而不是 M=1 kernel。**实际这次是否命中某个 C++ binding，仍需要动态 profiler 证实。**

### 8.3 `linear_input_act` 到底能融合什么

`comfy/ops.py:1096-1217` 提供 `linear(act(x)) + residual*scale` 的统一入口：

- 量化权重可用时，交给 layout 的 `linear_input_act`；
- 不满足条件时，先 eager 计算 `input_act`，再普通 Linear，最后 `torch.addcmul`；
- FP16 accumulation fast path 需要用户显式 opt-in，不是看到 FP16 就自动启用；
- LoRA、`comfy_force_forward`、训练状态、`_full_precision_mm` 都可能使融合路径退回普通路径。

本流程可明确关联的调用：

- DiT `fc2`：输入 `swiglu`；
- video VAE `to_qkv`、`ff.w1`：输入 `rms_norm`；
- video VAE `ff.w2`：输入 `swiglu` 并带 residual/scale；
- video VAE `attn.to_out`：带 residual/scale。

“能够融合”与“实际一定融合”仍然不同。尤其 residual 的融合要求 dtype、shape 和 16-byte 对齐满足 epilogue 条件。

### 8.4 fused RMSNorm + RoPE

`ck.rms_rope_split_half_` 在 Kitchen Python 层是 `torch.ops.comfy_kitchen.rms_rope_split_half_`。CUDA backend `_rms_rope_cuda` 最终调用 `_C.rms_rope`，支持：

- Q/K 同 shape 的 paired 路径；
- Q/K 不同 shape 时分别 `_rms_rope1_cuda`；
- 不满足 native layout 时 eager RoPE fallback；
- 原地和非原地两种接口。

H3 DiT 的 Q/K 同 shape，静态走 paired 候选；Qwen 的 Q 64 heads、K 8 heads，且只调 `apply_rope_split_half`，并不是主 DiT 的“Q/K RMSNorm + RoPE”同一调用。

普通 `RMSNorm` 另有 `comfy.rmsnorm.rms_norm` → `torch.nn.functional.rms_norm`，不要把所有 norm 都说成 fused RoPE kernel。

### 8.5 `attention_pytorch` 不等于朴素 S² 矩阵实现

`attention_pytorch` 只负责 reshape、mask 处理和调用 PyTorch SDPA。`comfy/ops.py:58-105` 还会处理 GQA 与部分 backend priority/contiguous 情况。

因此主 DiT 的 attention 算法复杂度仍约为 O(S²)，但不一定在显存中实体化 `[56,S,S]` score matrix。若使用 fused SDPA backend，可通过 tiling 避免完整分数矩阵常驻；若回退 math backend，则可能实体化大矩阵。没有实际 backend trace 就不能断言哪种内存行为和 kernel。

近似规模只用于理解，不是测量：

- 每个 head 的 QK 逻辑分数规模为 S²；
- 56 个 heads 是 `56*S²` 个分数元素；
- 当 L 较小，S 约 1.54 万，单个 head 已约 2.37 亿个分数元素；
- 20 步×50 层共有 1000 次这样的 full attention。

这也解释了为何长文本虽会增加一些 token，主体计算仍由视频 token 主导；但不能仅凭 FLOPs 推断本机各阶段实际时间，因为 INT8 量化、offload 和首次初始化同样重要。

### 8.6 video VAE 特有的 `int8_attention`

`MiniMaxH3VideoVAE` 的 `Attention.attention`（`vae.py:340-348`）在：

```text
to_qkv.weight 是 QuantizedTensor
且 COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
```

时调用 `ck.int8_attention`，否则回退 `optimized_attention`。

当前 36 层 ViT decoder 的 `to_qkv` 都是 INT8 ConvRot，且本机 Kitchen 支持 int8 attention，因此静态条件满足。Kitchen `sage_attention.py:597-636` 描述这条路径为：

- Q/K/V signed INT8，attention probability unsigned INT8；
- Q/K 使用保持内积的 block-Hadamard rotation；
- softmax 的 max、exp、denominator、reciprocal 和 V-scale 使用 FP32；
- 不分配 FP8 tensor、不执行 FP8 MMA；
- C++ binding 为 `_cuda_backend._C.sage_sdpa`。

这和主 DiT 的 SDPA 是两条不同路径，不能混写。

### 8.7 日志“支持某 op”与“本流程实际使用该 op”

本机启动日志列出很多 CUDA capabilities。下列项目**不能直接写成本 workflow 已执行**：

| 名称 | 正确解释 |
|---|---|
| `scaled_mm_nvfp4` | 可用，但 Qwen 全部 full-precision GEMM 标志使默认 encode 不走这里 |
| `gemv_awq_w4a16` | 可用，但不能替代 `full_precision_matrix_mult` 的普通 GEMM 证据 |
| `convrot_w4a4` | 另一种量化 layout；当前四个模型没有该格式 |
| `fp16_linear` | 需 FP16 accumulation opt-in 且满足 shape/dtype，不是默认必走 |
| `fp16_conv3d` | 同样需 opt-in；本 T2V 不执行 video CNN encoder |
| `group_norm_silu_pad3d` | video CNN encoder 的候选，本 T2V 不跑 encoder |
| `scaled_mm_mxfp8` | 当前四个模型无 MXFP8 权重 |
| Triton kernels | 启动时 Triton backend disabled |

这一区分是解释 kernel 调用栈时最容易被忽略、但最影响结论正确性的部分。

## 9. 视频 VAE：从 latent 到 RGB 帧

### 9.1 ComfyUI 包装器调用

`nodes.VAEDecode.decode` 取 nested latent 的 video 部分，进入 `sd.VAE.decode`（`comfy/sd.py:1360-1451`）：

```text
samples_in [1,24,37,30,54]
  → memory_used_decode 估算
  → load_models_gpu(video_vae.patcher)
  → 因 comfy_has_chunked_io=True，预分配完整输出 buffer
  → latent 转 device 和 vae_dtype
  → MiniMaxH3VideoVAE.decode(z, output_buffer=...)
  → movedim(channel) → [124,480,864,3]
```

输出 buffer 默认位于 intermediate device（本流程 CPU），形状为 `[1,3,124,480,864]`，dtype FP32。它的元素数约 1.544 亿，约 588.2 MiB，**不是一整段 GPU activation**。

OOM 时包装器可以进入 tiled fallback，但 `MiniMaxH3VideoVAE.decode_tiled` 直接回到自己的 `decode`，保留模型内部语义 tile，而不是简单套用通用 VAE tile 逻辑。

### 9.2 真正的 decoder 结构

`MiniMaxH3VideoVAE.decode`（`vae.py:992-1013`）先对每个 latent channel 反归一化：

```text
z = z * latents_std + latents_mean
```

本 T>1 走 `decode_temporal`。每个 temporal/spatial chunk 的核心为：

```text
z_chunk
  → post_quant_conv: 24→24，1×1×1 Conv3d
  → ViT3DDecoder.forward
     ├─ x_embedder: 24 → 2048
     ├─ append 4 个 register tokens + 1 个全零 suffix token
     ├─ 三维归一化坐标 → RotaryEmbeddingND
     ├─ 36 × TransformerBlock
     ├─ norm_out + proj_out
     └─ unpatchify: 时间×4、空间×16 → RGB
```

ViT 每层是：

```text
qkv = linear_input_act(to_qkv, x, 'rms_norm', norm1)
q,k = rms_rope_split_half_(q,k,rope)
out = int8_attention(q,k,v)            # 满足本 checkpoint/本机条件
x = linear_input_act(to_out, out, residual=x, residual_scale=scale1)

h = linear_input_act(ff.w1, x, 'rms_norm', norm2)
h = linear_input_act(ff.w2, h, 'swiglu', residual=x, residual_scale=scale2)
x = h
```

维度为 32 heads × 64 = 2048。每个 latent token 对应 `3*4*16*16=3072` 个输出值，但只去掉最前面的 latent patch tokens 做 `proj_out`，register/suffix 不变成像素。

本 decoder 的 144 个 INT8 ConvRot Linear 正好是：

```text
36 × (to_qkv, to_out, ff.w1, ff.w2)
```

`x_embedder`、`proj_out`、`post_quant_conv`、register tokens 不属于这 144 个量化权重。VAE 尾部还执行 `nan_to_num`，用于清理异常值。

### 9.3 时间 chunk：37 latent 如何恢复成 124 帧

默认参数：

```text
vae_ratio_t = 4
clip_length = 17
tokens_chunk_size = ceil(17/4) = 5
token_drop = 3
token_overlap = (-3) % 5 = 2
frame_pre_padding = (-17) % 4 = 3
frame_overlap = max(2*4-3,0) = 5
```

`decode_temporal` 处理 37 个 latent 时：

```text
pseudo_total_tokens = 37 + 3 = 40
pad_tokens = 0
num_chunks = 40/5 - 1 = 7
```

每轮读取 `5+2=7` 个 latent，产生 28 帧的解码片段。首轮裁掉 pre-padding，保留 17 个新帧；下一片保留与上一片重叠的 5 帧用于 blend，再写 17 个新帧；最后补回尾部重叠片段：

```text
7 × 17 + 5 = 124 frames
```

`blend` 对重叠区使用线性权重，不是硬切拼接。`_finalize_pixels` 用 ImageNet mean/std 逆变换并 clamp 到 `[0,1]`，然后逐段写入预分配输出，避免整段 RGB 视频常驻 GPU。

### 9.4 空间 tile：普通 VAEDecode 也不是一次解整帧

`MiniMaxH3VideoVAE.tiling=True`、`tile_size=256`、`tile_overlap_min=64`，`_adaptive_decode` 总是进入 `tiled_decode`。按 `split_tiles` 源码公式，864×480 的空间划分为：

```text
H=480：start=[0,112,224]，overlap=[144,144]，3 个 tile
W=864：start=[0,144,288,448,608]，overlap=[112,112,96,96]，5 个 tile
```

每个时间 chunk 因此有 15 个空间 tile。空间 overlap 使用与 `blend` 相同的线性混合，并直接写 canvas，最后恢复为完整 864×480。

`_decode_tile_row` 还会把宽度方向的多个 latent tile 在 batch 维拼接，一次送入 `_decode_pixels`，batch 上限为 4，同时受 free VRAM 估算约束。因此：

- 15 是逻辑 tile 数；
- 7 是 temporal 循环数；
- **不能**据此宣称固定 105 次 `_decode_pixels` 或某个固定数量的 CUDA launches；
- 实际 batch 取决于每次调用时的可用显存。

这是视频 decode 显存峰值仍高于其他阶段的原因之一：权重按需驻留、较大 ViT attention、多个 tile 的中间激活和 decoder 临时 buffer 同时存在，而不是只由那 2.62 GB 的 VAE 文件决定。

### 9.5 本 T2V 没有执行的 video VAE 代码

`MiniMaxH3VideoVAE` 还包含 `EncoderFCN3D`：

- `CausalConv3d`：时间前向零 padding，空间 reflect padding；
- `TemporalIsolatedGroupNorm`：把时间维并入 batch，每帧单独归一化；
- `ResnetBlock3D`、`Downsample3D`；
- 可选 `group_norm_silu_pad3d` 融合；
- channels-last/NDHWC 和可选 `fp16_conv3d`；
- 17 帧一段的 `encode_temporal`；
- moments 取 mean，再按 latent mean/std 归一化。

只有首帧、尾帧、参考视频或 guide image 需要 VAE encode 时才会执行这些代码。当前 workflow 在采样前直接从零 latent + noise 开始，**这些 encoder op 没有实际执行**。这部分可以作为其他 I2V workflow 的静态参考，但不应计入本 T2V 的 kernel 列表或耗时。

## 10. 音频 VAE：从双声道 latent 到波形

### 10.1 解码调用和形状

`VAEDecodeAudio.execute` 调 `vae_decode_audio`（`nodes_audio.py:99-113`）：

```text
audio latent = samples.unbind()[-1] = [1,32,2,207]
  → sd.VAE.decode（FP32）
  → MiniMaxH3AudioVAE.decode
  → waveform [1,2,165600]
  → movedim → [1,165600,2]
  → 按 std 做缩放
  → AUDIO {waveform:[1,2,165600], sample_rate:32000}
```

`207 × 800 = 165600` 个 samples，约 5.175 秒。它比视频 124/24 秒略长，SaveVideo 会在封装时按视频长度裁切。

### 10.2 `MiniMaxH3AudioVAE.decode`

`audio_vae.py:416-425` 首先把两个声道当成独立 mono batch：

```text
[1,32,2,207]
  → permute/reshape → [2,32,207]
  → z * latents_std + latents_mean
  → dec_in_proj：32→2048，1D pointwise convolution
  → BigVGAN decoder
  → [2,1,165600]
  → reshape → [1,2,165600]
```

因此左右声道共享同一套 decoder 权重，但分别前向。没有额外一个“立体声混音神经网络”。

### 10.3 BigVGAN decoder 的 op 和原理

BigVGAN 配置在 `audio_vae.py:308-369`：

- `conv_pre`：2048→1024，kernel 7；
- 7 级 ConvTranspose1d 上采样，rates=`(5,5,2,2,2,2,2)`；
- 乘积 `5*5*2^5 = 800`，正好对应 40 Hz latent → 32 kHz waveform；
- 每级 3 个 `AMPBlock1` residual blocks，kernel size 分别为 3、7、11；
- 每个 AMPBlock 有 3 组 dilation conv，再各接一个普通 conv；
- 最后 `activation_post + conv_post`，并 clamp 到 `[-1,1]`。

SnakeBeta 激活为：

```text
x + sin(alpha*x)^2 / (beta+1e-9)
```

alpha、beta 在 checkpoint 中以 log scale 保存，前向时 `exp` 后使用。`Activation1d` 采用：

```text
2× 上采样 → Snake/SnakeBeta → 2× 低通下采样
```

上采样是 depthwise ConvTranspose1d 加 Kaiser-windowed sinc 低通滤波，之后裁剪；下采样是 depthwise Conv1d 加同类低通。这是 BigVGAN 的 anti-aliased activation，不是普通直接堆叠 ConvTranspose。

主要 op：

- `F.conv1d`、`F.conv_transpose1d`；
- `F.pad(mode='replicate')`；
- `sin`、`exp`、乘加、clamp；
- residual add、3 个 resblock 的平均；
- reshape、permute、slice。

本模型使用普通 `ops.Conv1d/ConvTranspose1d` 的 FP32 权重，没有 audio VAE 量化元数据，也没有 int8 attention。

### 10.4 输出缩放与未执行的 encoder

`vae_decode_audio` 对 waveform 计算：

```text
scale = std(audio) * 5
scale = max(scale, 1)
audio /= scale
```

也就是说，只有 `std(audio) > 0.2` 时才缩小波形；这不是“峰值归一化到 ±1”。

本流程不执行 `MiniMaxH3AudioVAE.encode`，所以以下只属于 encoder/参考音频路径：

- DAC 风格 `Encoder`、ResidualUnit、Snake1d；
- `AttnProjection` 中的 causal self-attention；
- `mean_proj/logs_proj` posterior；
- posterior mean 归一化。

尤其 `CausalAttention` 中的 `scaled_dot_product_attention(is_causal=True)` **不在本次音频 decode 调用栈里**。

## 11. 权重驻留、offload、prefetch 和 allocation graph

### 11.1 为什么各模型不是简单“全部加载后同时推理”

`prepare_sampling` 先估算 DiT 推理内存并 `load_models_gpu`；文本 CLIP 在自己的 `encode_from_tokens` 前加载；VAE decode 又在各自调用前加载。DynamicVRAM 通过 patcher、VBAR、host buffers 和 stream 管理驻留。

本次日志 staged 值约为：

```text
Qwen text encoder  14956 MB
H3 DiT             19995 MB
Video VAE           2677 MB
Audio VAE            576 MB
```

这些是 patcher 管理的模型规模，不是同一时刻物理 GPU 驻留量。实际 NVML 峰值为 38570 MiB，出现在 video VAE decode；不能把四者 staged 值简单相加当作峰值，也不能把权重文件大小当作推理 activation。

### 11.2 DiT block prefetch

`MiniMaxH3.extra_conds` 最终让 `BaseModel._apply_model` 在 `transformer_options` 中设置 `prefetch_dynamic_vbars=True`。`_run_blocks` 建立：

```text
prefetch_queue = [None, block0, block1, ..., block49, None]

每个 block：
  prefetch_queue_pop
    ├─ 等待上一批 offload stream
    ├─ 清理/解 pin 已消费权重
    ├─ pin/fault 下一批 block 权重
    ├─ host buffer / device / dtype / quant layout 准备
    └─ stream 同步后执行 block
```

本机日志显示 `Using async weight offloading with 2 streams`。预取的意义是让下一批权重的 I/O、cast 与当前计算重叠，而不是降低 DiT 数学计算量。

Qwen 的 `Llama2_.forward` 有相同队列接口，但在本次单次 encode 中 `past_key_values=None`，其 `prefetch_dynamic_vbars and past_key_values is not None` 条件不成立，所以不能用 DiT 的 block 队列行为去解释文本编码；它仍可通过 DynamicVRAM 的按需 fault/cast 机制运行。

### 11.3 `malloc_graph` 不等于整个 DiT 的 CUDA Graph

`MiniMaxH3Model.forward` 在 AIMDO+CUDA 条件满足时调用：

```text
malloc_graph_begin
  → WrapperExecutor → _forward
malloc_graph_end
```

它记录/管理分配，预分配 video/audio 输出，并处理分配图的生命周期。`_run_blocks` 的 `prefetch_queue_pop` 没有传 `enable_graph=True`，因此不能声称 50 个 block 被整体 `torch.cuda.CUDAGraph` 捕获并 replay。

本日志为 `Comfy model compiler graph breaks: 1, rogues: 0`。这也不能自动翻译成“模型经过了 torch.compile”。本分析中没有一个依赖 `torch.compile` 才能成立的调用链。

### 11.4 与 tensor/op 直接相关的内存行为

- `CastBiasWeightContext`：临时取得权重/bias、执行 cast/dequant、流同步和释放；
- `empty` 与预分配：主 packed hidden、视频 VAE 输出、Kitchen workspace；
- `contiguous/channels_last`：可能产生额外拷贝和内存峰值；
- in-place `add_`、`addcmul_`、`mul_`：减少新 activation，但不改变数学含义；
- `del clip_dec/clip_z`：及时释放 video VAE 当前 temporal chunk；
- CPU 预分配最终帧：避免整段 124 帧 RGB 常驻 GPU；
- `soft_empty_cache`：只在 VAE OOM fallback 等特定路径出现，不能当作正常每步行为。

本流程已有的 NVML/loaded-weight 观测与本文函数解释一致：不是单一“权重总和”决定峰值，而是当前驻留权重、activation、量化 workspace、offload buffer 和 VAE 临时块共同决定。

## 12. CreateVideo、SaveVideo 与最终 MP4

`CreateVideo.execute`（`nodes_video.py:259-273`）只把 images、audio、fps 包成 `VideoFromComponents`，并不再跑模型。

`SaveVideo.execute` 解析 auto format/codec，然后调用 `VideoFromComponents.save_to`（`video_types.py:1318-1416`）：

1. 用 PyAV 打开输出容器；
2. 逐帧将 `[0,1]` float tensor 转成 `uint8`：`frame*255 → clamp → byte → cpu().numpy()`；
3. 构造 `av.VideoFrame`，重排为编码器像素格式；
4. 编码 H.264 视频流并 mux；
5. 音频转 `fltp` planar float，按 stereo AAC 编码；
6. flush video/audio encoder，生成 MP4。

实际音频长度裁切为：

```text
ceil((32000/24) * 124) = 165334 samples
```

所以 BigVGAN 产生的 165600 samples 不会全部写入 MP4。音频保持 32 kHz，不因为视频是 24 fps 而重采样成 48 kHz；只有 WebM 分支才默认 48 kHz Opus。

这些主要是 CPU、NumPy、PyAV 和 FFmpeg 操作，不属于 DiT/VAE 的 GPU kernel。实际输出检查确认 124 帧、24 fps、32 kHz 双声道音频，与这条路径吻合。

## 13. 按模块汇总：函数、op 与确定性

### 13.1 本次实际路径的高层调用清单

| 模块 | 主要函数 | 主要 op/后端 | 次数或形状 | 证据边界 |
|---|---|---|---|---|
| Tokenize | `MiniMaxH3Tokenizer.tokenize_with_weights` | Qwen tokenizer，无 chat template | prompt → L tokens | 静态+实际输入 |
| Embedding | `mixed_precision_ops.Embedding` | `dequantize_embedding` | 按 token 行取 embedding | int8，非整表反量化 |
| Qwen | `Llama2_.forward` ×50 | 反量化后 `F.linear`、RMSNorm、RoPE、causal SDPA、SiLU | 64 Q heads / 8 KV heads，50 层一次 | 350 个 NVFP4 Linear 全为 full-precision 标志 |
| 文本 refiner | `preprocess_text_embeds` | 5120→5376 projection，2 个 attention/MLP block | 采样前一次 | 不在每步重复 |
| 空 latent/noise | `_empty_av_latent`、`prepare_noise` | `zeros`、CPU `randn`、`NestedTensor` | V=[1,24,37,30,54]，A=[1,32,2,207] | 实际参数推导 |
| 调度 | `simple_scheduler`、`time_shift_sigma` | 标量/tensor 算术 | 21 个 video sigma（含 0） | 非专用 sampler kernel |
| 主采样 | `sample_res_multistep` | Euler + 二阶 multistep、`exp/expm1`、乘加 | 20 次 DiT | `eta=0`，无额外噪声 |
| DiT packed attention | `Attention.forward` ×50×20 | INT8 QKV/out Linear、RMS+RoPE、PyTorch SDPA | 1000 次；[1,56,S,128] | 具体 SDPA backend 未 trace |
| DiT MLP | `MLP.forward` ×50×20 | INT8 fc1、SwiGLU、INT8 fc2 | 4000 个主 block INT8 Linear（含 attention） | 融合取决于 shape/backend |
| AdaLN/final | `AdalnProj`、`FinalLayer` | 普通 Linear、RMSNorm、分段乘加、unpack | 每步每层均执行 | 不属于那 200 个 INT8 Linear |
| Video VAE | `decode_temporal` + `tiled_decode` | 1×1×1 conv、36 层 ViT、INT8 attention、RMS+RoPE、blend | 7 temporal chunks，每 chunk 15 空间 tiles | decoder batch 随 free VRAM 变化 |
| Audio VAE | `MiniMaxH3AudioVAE.decode` | FP32 Conv1d/ConvTranspose1d、Snake、抗混叠滤波 | 2 声道，207→165600 samples | 无 encoder attention |
| 保存 | `VideoFromComponents.save_to` | CPU cast/NumPy、PyAV H.264/AAC | 124 帧、32 kHz stereo | 非 GPU 推理 kernel |

### 13.2 结构性调用次数，不等于 kernel launch 次数

在无 hook、无异常、默认配置下，可由图和源码确定：

```text
Qwen 主 block forward                  50 次（整次任务一次）
Qwen 大 Linear                         350 个权重调用（不含 embedding）
文本 refiner block                     2 次（整次任务一次）
主 DiT forward                         20 次
主 DiT block                           20×50 = 1000 次
主 DiT packed full attention           1000 次
主 DiT INT8 主 Linear                  20×50×4 = 4000 次
主 DiT fused RMS+RoPE                  1000 次（另有 token refiner）
主 DiT AdaLN projection                1000 次
Video VAE ViT block                    36 × 实际 decoder 调用次数
Video VAE INT8 Linear                  144 × 实际 decoder 调用次数
Audio VAE upsample stage               7 次（每声道共用权重、分别计算）
```

这些次数不能直接相加为“CUDA kernel 数”：

- 一个 `F.linear` 可能映射到一个或多个库 kernel，还可能先反量化、cast、量化；
- 一个 `linear_input_act` 可能融合，也可能分成多个 PyTorch op；
- 一个 Python tile 可以与其他 tile 合 batch；
- DynamicVRAM 还有拷贝、pin/unpin、stream event 和 memory-management work；
- PyTorch SDPA 的具体 kernel 数和内部阶段取决于所选后端。

### 13.3 本流程明确不走或不能声称已走的分支

| 分支 | 本流程结论 |
|---|---|
| Video VAE encoder、keyframe latent | 不走：无首帧/尾帧/参考图 |
| Audio VAE encoder causal attention | 不走：无参考音频 |
| Qwen visual tower / DeepStack | 不走：无图像/视频条件 |
| Qwen lm_head / token generation | 不走：只取 hidden states |
| negative prompt / CFG 双分支 | 不走：BasicGuider positive only |
| Turbo LoRA | 不走：switch=false，主模型 0 patches |
| PDD output head bank | 不走：本模型 `pdd_steps=1` |
| NVFP4 `scaled_mm_nvfp4` GEMM | 默认文本 encode 不走：full-precision 标志和 loader 策略 |
| MXFP8、`convrot_w4a4` | 当前模型没有这些格式 |
| `fp16_conv3d` / CNN encoder 的 pad 融合 | 本 T2V 不执行 video encoder，且 fp16 accumulation 需 opt-in |
| 整个 50-block DiT 的 CUDA Graph replay | `_run_blocks` 未开启 block `enable_graph`，不能这样声称 |
| 朴素 attention 的确定 S² 实体矩阵 | 不确定：取决于 SDPA backend，不能默认一定物化 |

## 14. 对性能现象的正确解释

### 14.1 首次慢与稳定 step 慢不是同一问题

- 文本节点约 352 秒包含大量 32B 权重首次读取、NVFP4 反量化、device/cast 和初始化工作；
- sampler 的第一步还承担主模型首次准备，日志显示约 482 秒；
- 后续 DiT step 约 1.71 秒，才更接近同 shape 稳定执行；
- video VAE 的 NVML 峰值还受到 temporal/spatial tile batch、临时 activation 和权重驻留影响。

因此不能只用“B200 上 50 层 Transformer 的理论 FLOPs”解释 15 分钟总耗时，也不能把首次反量化/加载成本平摊成每个 DiT kernel 的固定开销。

### 14.2 packed self-attention 的规模与内存不能混为一谈

S≈15,399+L 时，attention 的算术复杂度是 O(S²)，但 fused backend 可能只保存 tile 级中间状态。权重压缩也不等价于 activation 压缩：

- INT8 主权重减少存储和传输，但主 hidden/QKV/MLP 中间值仍可为 BF16 或 FP32；
- NVFP4 Qwen 减少权重存储，但 full-precision GEMM 需要反量化权重；
- video VAE 分块控制单个块的大小，但合 batch、overlap 和临时输出仍产生峰值；
- CPU 最终帧缓冲降低 GPU 持续占用，不代表所有 decode 临时块都在 CPU。

### 14.3 最易误判的四点

1. `int8_tensorwise` + ConvRot 不只是 INT8 GEMM，还包含激活旋转和量化。
2. NVFP4 checkpoint 不等于 FP4 Tensor Core 执行；必须读 `full_precision_matrix_mult`。
3. `Using pytorch attention` 不等于明确某个 SDPA 实现，也不等于 `ck.int8_attention`。
4. 普通 `VAEDecode` 节点的“非 tiled 名称”不表示 H3 video VAE 内部没有 tiling。

## 15. 尚未实测的部分与后续验证方法

### 15.1 当前不能下的结论

在没有动态 trace 的情况下，本文刻意不声称：

- 某次 `F.linear` 对应的具体 cuBLASLt/CUTLASS 模板名；
- 主 SDPA 最终选了 Flash、cuDNN、mem-efficient 或 math 哪个后端；
- `_C.cutlass_int8_dequant` 相对 cuBLAS fallback 的实际命中率；
- 每种 op 的精确 launch 数、kernel 时间、GPU 利用率、HBM 带宽；
- 每个量化 Linear 的实际输入/权重 dtype 组合；
- CPU↔GPU 传输在总时间中的精确比例。

### 15.2 若必须形成逐 kernel 证据，建议这样做

这部分是**建议，未在本轮执行**：

1. 固定相同 prompt、seed、shape、20 steps 和模型文件，先做一次不计时的 warm-up，再分别测冷启动和热执行，避免把首次加载混入稳定 kernel 对比。
2. 在文本 encode、第一个 denoise step、后续 step、audio decode、video decode 分段使用 `torch.profiler`，开启 CPU/CUDA activities、`record_shapes`、`profile_memory`、必要的 `with_stack`。
3. 重点筛选：
   - `aten::linear`、`aten::matmul`、`aten::scaled_dot_product_attention`；
   - `comfy_kitchen::int8_linear`、`rms_rope_split_half_`、`int8_attention`；
   - `dequantize_nvfp4`、`quantize_int8_rowwise_convrot64`；
   - `aten::convolution`、`aten::addcmul`、`aten::rms_norm`、memory copy。
4. 需要最终 CUDA kernel 名时，用 Nsight Systems/CUPTI 同时采集 CUDA API、GPU kernels 和 memcpy，并把节点 span 与 GPU timeline 对齐；不能只依赖 Comfy Kitchen backend capability 列表。
5. 输出按“Python op → custom op/C++ binding → CUDA kernel”三级整理，另记 fallback、dtype、shape 和每次调用所在 block，才能将本报告的静态候选转成真实调用栈。

本轮未进行这些 profiling，因为用户要求的是对已经跑通 workflow 的深入源码与既有证据分析，而不是再次承担一次完整 20 步生成及 profiler 的额外开销。

## 16. 关键源码与证据索引

以下行号均对应固定 commit 的本地 checkout；`.venv` 路径均省略前缀 `/home/kaokaolv/comfyui_h3/`。

| 主题 | 文件与位置 |
|---|---|
| 实际 workflow | `video_minimax_h3_t2v.json`；`logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json` |
| 实际节点/峰值 | `logs/model_card_runs/video_minimax_h3_t2v/events.json`、`memory_samples.json` |
| 服务版本/backend/dtype | `logs/comfyui_server_profiled.log:31-96` |
| 通用 loaders | `ComfyUI/nodes.py:321-346,776-873,992-1042` |
| H3 T2V 条件节点 | `ComfyUI/comfy_extras/nodes_minimax_h3.py:50-179` |
| DiT 检测和配置 | `ComfyUI/comfy/model_detection.py:401-435`；`ComfyUI/comfy/supported_models.py:986-1012` |
| 模型加载 | `ComfyUI/comfy/sd.py:2460-2559`；`ComfyUI/comfy/utils.py:130-238` |
| DynamicVRAM | `ComfyUI/main.py:318-320`；`ComfyUI/comfy/model_patcher.py:1786-2010` |
| 权重 cast/offload | `ComfyUI/comfy/ops.py:144-528` |
| 量化 Linear/Embedding | `ComfyUI/comfy/ops.py:1096-1217,1342-1679,1857-1956` |
| 量化格式 | `ComfyUI/comfy/quant_ops.py:150-265` |
| 文本节点与 tags | `ComfyUI/comfy/text_encoders/minimax.py:1-20,69-97,134-148,168-241` |
| Qwen 结构 | `ComfyUI/comfy/text_encoders/qwen3vl.py:51-114`；`ComfyUI/comfy/text_encoders/llama.py:391-424,556-666,699-859,1005-1134` |
| CLIP 前向 | `ComfyUI/comfy/sd1_clip.py:172-303` |
| H3 extra_conds | `ComfyUI/comfy/model_base.py:221-263,2226-2307` |
| DiT 主结构 | `ComfyUI/comfy/ldm/minimax/model.py:38-823` |
| sampler | `ComfyUI/comfy/samplers.py:208-356,592-652,937-1007,1188-1398` |
| res_multistep | `ComfyUI/comfy/k_diffusion/sampling.py:1461-1528` |
| flow/audio schedule | `ComfyUI/comfy/model_sampling.py:86-100,289-357` |
| VAE wrapper | `ComfyUI/comfy/sd.py:1050-1194,1360-1570` |
| Video VAE | `ComfyUI/comfy/ldm/minimax/vae.py:44-428,565-1013` |
| Audio VAE | `ComfyUI/comfy/ldm/minimax/audio_vae.py:24-150,277-443` |
| Audio decode 节点 | `ComfyUI/comfy_extras/nodes_audio.py:99-135` |
| prefetch/allocation | `ComfyUI/comfy/model_prefetch.py:31-346` |
| 保存视频 | `ComfyUI/comfy_extras/nodes_video.py:137-273`；`ComfyUI/comfy_api/latest/_input_impl/video_types.py:1318-1416` |
| Kitchen INT8 layout | `.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/int8.py:60-276,337-411` |
| Kitchen NVFP4 layout | `.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/nvfp4.py:23-115,212-260` |
| Kitchen registry | `.venv/lib/python3.12/site-packages/comfy_kitchen/registry.py:20-307` |
| Kitchen INT8 CUDA | `.venv/lib/python3.12/site-packages/comfy_kitchen/backends/cuda/__init__.py:2128-2400` |
| Kitchen fused RMS/RoPE | 同上文件 `4040-4191` |
| Kitchen INT8 attention | `.venv/lib/python3.12/site-packages/comfy_kitchen/sage_attention.py:208-317,597-636` |

完整的命令、元数据输出、问题与解决过程另见 `run_book.md` 的“T2V 函数与 Kernel 调用栈分析”章节。本文所有“实测”结论均能回指上述本地文件，不以模型名或经验推断替代证据。
