# MiniMax H3 T2V：CUPTI 逐 Kernel 实测报告

> 本报告是对 `comfyui_h3_t2v_kernel_analysis.md` 的动态补充。这里只把实际 Nsight Systems/CUPTI trace 中出现的内容称为“实测”；静态源码仅用于把 kernel 映射回模型层和数学逻辑。

## 0. 实测对象、产物与结论摘要

### 实测配置

| 项目 | 内容 |
|---|---|
| ComfyUI | v0.45.0，commit `989d93d0d968e704c589ce0146bb18c77f6b50cc` |
| Nsight Systems | 2024.6.2.225 |
| GPU | 物理 GPU 2，NVIDIA B200；进程内 `cuda:0` |
| PyTorch / Kitchen | 2.13.0+cu130 / comfy-kitchen 0.2.41 |
| Prompt ID | `24fe3352-d0a1-4dd7-ac4a-4b1b46b1f09f` |
| API prompt | 与此前跑通的 `video_minimax_h3_t2v` 完全相同 |
| 采集窗口 | prompt 提交前 `cudaProfilerStart`，完成后 `cudaProfilerStop` |
| 是否含冷执行 | 含模型首次执行、首次 kernel 初始化；不含 ComfyUI 启动初始化 |
| 运行方式 | 独立 8189 profiling 实例；原 8188 未停止或修改 |
| 客户端计时 | 394.808 秒；服务日志为 379.21 秒，边界不同 |

原始 trace：

- `logs/cupti_h3_t2v/h3_t2v_cold.nsys-rep`：12,178,305 bytes。
- `logs/cupti_h3_t2v/h3_t2v_cold.sqlite`：26,533,888 bytes。
- `logs/cupti_h3_t2v/run_result.json`：`success=true`。
- `logs/cupti_h3_t2v/output/video/MiniMax_H3_00001_.mp4`：H.264、864×480、24 fps、124 帧、32 kHz 双声道 AAC。

总量：

- **95,226** 个 kernel，**155** 种 kernel；
- **67** 个 H3 NVTX 区间；
- **5,596** 次 memcpy，合计 **46,506,971,601 bytes**；
- **421,188** 条 CUDA runtime API 记录。

最重要的实测结论：

1. 主 DiT attention 实际是 **BF16 FlashAttention**，kernel 为 `pytorch_flash::flash_fwd_kernel`。
2. Qwen 实际是 **NVFP4 反量化 → FP32 GEMM + FP32 mem-efficient attention**，没有执行 `scaled_mm_nvfp4`。
3. 主 DiT 的 4000 次 INT8 Linear、video VAE 的 6048 次 INT8 Linear 都真实命中 ConvRot 在线量化和 CUTLASS INT8 kernel。
4. video VAE 的 residual 也实际进入 `VisitorAuxLoad` epilogue，不只是源码中“可能融合”。
5. 全部 95,226 个 kernel 的 `graphNodeId` 都为空，本次没有观察到 CUDA Graph node kernel；不能把 allocation graph 当作整个模型的 CUDA Graph replay。

## 1. 如何采集和映射

### 采集方法

使用：

```text
nsys profile --trace=cuda,nvtx
             --capture-range=cudaProfilerApi
             --capture-range-end=stop
             --cuda-graph-trace=node
             --export=sqlite
```

辅助 custom node 只添加控制接口和 NVTX range，不改变模型数学计算。阶段标签为：

- `H3.text.qwen`
- `H3.text.refiner`
- `H3.sampler.res_multistep`
- `H3.dit.step.01` 至 `H3.dit.step.20`
- `H3.video_vae.decode`
- `H3.video_vae.vit.01` 至 `H3.video_vae.vit.42`
- `H3.audio_vae.decode`

67 个区间的数量关系为：

```text
1 Qwen + 1 text refiner + 1 sampler + 20 DiT
+ 1 video decode wrapper + 42 ViT + 1 audio decode = 67
```

### kernel → NVTX 的归属方式

`analyze_h3_cupti.py` 对每个 kernel：

1. 用 `correlationId` 找对应 CUDA runtime API；
2. 取 API 的 start timestamp；
3. 选择包含该 timestamp 的**最小** H3 NVTX range；
4. 如果没有 runtime 匹配，退回 GPU kernel start；
5. 每个 kernel 只归到一个最小子区间，不重复计算父区间。

这种方法避免异步 GPU 执行将 kernel 错分到后续阶段。`nsys stats --report nvtx_kern_sum` 另用于交叉核验。

### 三种“时间”不能混用

- **kernel time**：某个 GPU kernel 的 `end-start`；多个 kernel 的总和是 GPU work。
- **NVTX host range**：host 端区间，包含初始化、权重准备、Python、同步、offload、异步等待等。
- **runtime API time**：host 上 CUDA API 调用本身的耗时。

多个 stream 上 kernel 可能重叠，因此 **GPU work 可以大于 host range**，不能直接把 kernel 时间之和当作该阶段 wall time，也不能把所有 host API 时间再相加当作总时间。

另有 19 个未标记的小 kernel，总计约 0.078 ms，未强行归入某个模型。CPU SaveVideo/FFmpeg 编码不属于 GPU 模型 kernel，不计入模型 NVTX 区间。

## 2. 各阶段总览

下表中的 memcpy 以十进制 GB/MB 表示。GPU work 为 kernel duration 之和，不是 wall time。

| 阶段 | host range | kernel 数 | GPU work | memcpy |
|---|---:|---:|---:|---:|
| `H3.text.qwen` | 113.043 s | 2,366 | 444.075 ms | 15.237 GB |
| `H3.text.refiner` | 14.275 s | 32 | 0.831 ms | 1.596 GB |
| `H3.dit.step.01` | 178.617 s | 2,990 | 1,708.844 ms | 19.626 GB |
| `H3.dit.step.02..20` 每一步 | 约 1.16 s | 各 2,986 | 约 1,700.5 ms | 各 175.701 MB |
| `H3.sampler.res_multistep` 整个区间 | 211.329 s | 1,067（DiT 外调度） | 2.742 ms | 116.145 MB |
| `H3.audio_vae.decode` | 3.050 s | 2,679 | 34.288 ms | 259.412 MB |
| `H3.video_vae.decode` | 25.427 s | 1,979（ViT 之外） | 10.228 ms | 617.103 MB |
| `H3.video_vae.vit.01` | 22.379 s | 792 | 80.697 ms | 4.869 GB |
| `H3.video_vae.vit.02..42` 每一次 | 约 35–63 ms | 各 648 | 约 22–78 ms | 常规各 16 KiB |

注意：`H3.sampler.res_multistep` 是整个采样父区间，其中的 1,067 个 kernel 是扣除各 DiT 子区间后的 scheduler/其他操作，不包含 59,724 个 DiT kernel。

`ViT.01` 比常规多 144 个 kernel（792 vs 648），说明首次 decoder 有额外初始化；不能把每次 ViT 调用都固定为 792 个 kernel。

## 3. Qwen：NVFP4 反量化后实际是 FP32 计算

### 实测计算链

```text
NVFP4 packed weight
  → dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, float, __nv_fp8_e4m3,...>
  → FP32 GEMM（cutlass3x_sm100_simt_sgemm_f32...）
  → FP32 RMSNorm / RoPE / residual / SwiGLU
  → fmha_cutlassF_f32_aligned_64x128_rf_sm80
```

这里输出的 `float` 同时出现在 NVFP4 反量化 kernel 类型和普通 FP32 GEMM 名称中，因此不再只是“初始化 dtype 不等于计算 dtype”的谨慎推测，而是 trace 直接证明的计算路径。

### 350 次 NVFP4 反量化

| 项 | 实测值 |
|---|---:|
| kernel | `dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, float, __nv_fp8_e4m3,...>` |
| 次数 | 350 = 50 layers × 7 Linear |
| 总时间 | 26.807 ms |
| 平均时间 | 76.593 µs |

按 grid 统计：

| gridX | 次数 | 对应权重类别 |
|---:|---:|---|
| 10,240 | 100 | k_proj / v_proj |
| 81,920 | 100 | q_proj / o_proj |
| 256,000 | 150 | gate_proj / up_proj / down_proj |

q/o 的权重元素总数相同，gate/up/down 的大 FFN 权重元素总数相同，所以 grid 只能按类别归并；不能用它再把每一个 q 与 o 的唯一实例区分开。

### 350 次 FP32 GEMM 与七类 Linear

| Linear | 调用次数 | GEMM 模板与 grid（x,y） | 源码中的输入→输出 |
|---|---:|---|---|
| q_proj | 50 | 64×32，(128,13) | 5120→8192 |
| k_proj / v_proj | 100 | 64×32，(16,13) | 5120→1024（各 50） |
| o_proj | 50 | 128×64，(40,7) | 8192→5120 |
| gate_proj / up_proj | 100 | 128×64，(200,7) | 5120→25600（各 50） |
| down_proj | 50 | 128×64，(40,7) | 25600→5120 |

o_proj 和 down_proj 的 output shape 一样，单个 GEMM instance 不能只凭 grid 区分，但调用顺序、每类 50 次、总 350 次与 50 层×7 Linear 完全一致。

`simt_sgemm_f32` 是普通 FP32 GEMM 路径，不是 FP4 Tensor Core 的 `scaled_mm_nvfp4`。全部 155 种 kernel 中没有 `scaled_mm_nvfp4` 或 `cublas_gemm_blockwise_fp4`。

### Qwen attention 与其他 op

- 50 个 `fmha_cutlassF_f32_aligned_64x128_rf_sm80`：总 10.115 ms，平均 202.295 µs。
- 200 个 FP32 `vectorized_layer_norm_kernel`：每层 input/post-attention/Q/K norm 共 4 个。
- 50 个 FP32 `silu_kernel`，与 50 层 MLP 对应。
- Qwen 区间总 GPU work 为 444.075 ms；host range 为 113.043 s，memcpy 为 15.237 GB。冷执行中大量 host/权重准备开销不能归因于 attention 或 GEMM 本身。

`fmha_cutlassF_f32` 是 PyTorch memory-efficient attention 路径；主 DiT 则走另一条 BF16 FlashAttention 路径，不能混称。

## 4. 主 DiT：BF16 FlashAttention + INT8 ConvRot 主投影

### 本次实际 packed sequence

activation quant kernel 的 grid M 为 **15,787**。已知：

```text
video rows = 14,985
audio rows = 414
text rows  = 15,787 - 14,985 - 414 = 388
```

所以本次 prompt 的实际 text_len=388，packed sequence：

```text
[388 text | 414 audio | 14,985 video]，S=15,787
```

这比静态公式 `S=L+15,399` 更进了一步：L 已从真实 kernel grid 反推出来。

### 主 attention：实测 BF16 FlashAttention

```text
pytorch_flash::flash_fwd_kernel<
  Flash_fwd_kernel_traits<128,128,64,4,...,
  cutlass::bfloat16_t,...>>
```

| 项 | 实测值 |
|---|---:|
| 主 DiT 调用次数 | 1,000 = 20 steps × 50 blocks |
| 总时间 | 17.660879 s |
| 平均每次 | 17.660879 ms |
| 文本 refiner 同 kernel | 另 2 次，合计约 0.061 ms |
| 全局同 kernel 次数 | 1,002 |

因此前文“PyTorch SDPA 最底层后端未实测”已被本次结果更新：**主 DiT 是 BF16 FlashAttention，不是 `ck.int8_attention`，也不是 FP32 mem-efficient attention。**

### 主 INT8 GEMM：4000 次

```text
20 steps × 50 blocks × (qkv_proj + out_proj + fc1 + fc2) = 4000
```

| 层类型（按调用顺序、grid 和模板区分） | 次数 | 实际 CUTLASS 路径 | 总时间 | 平均每次 |
|---|---:|---|---:|---:|
| qkv_proj + fc1 | 2,000 | identity swizzle，BF16 output，INT8 input | 8.776803 s | 4.388402 ms |
| out_proj + fc2 | 2,000 | LeanStreamKT，BF16 output，INT8 input | 3.630500 s | 1.815250 ms |

qkv_proj 的 gridY=84，fc1 的 gridY=112，与输出维度 21,504 和 28,672 对应。模板名称含 `arch::Sm80`，这是实际加载的 CUTLASS 兼容 kernel 模板，不是把 B200 误认成 SM80 硬件。

同一 DiT 阶段另外还有 20 个 `cutlass_80_simt_sgemm_128x32_8x5_tn_align1`，即每一步一个非 INT8 的单精度 GEMM。因此全部 `cutlass::Kernel2` 为 4020 个，但其中只有 4000 个是 `signed char` INT8 GEMM；额外的 20 个不计入上述 INT8 数量，也不应因未显式标出 BF16 类型而误称为 FP16。

### 在线 ConvRot：4000 次

| 层/输入宽度 | 次数 | 实际 kernel 配置 | 总时间 | 平均每次 |
|---|---:|---|---:|---:|
| qkv/fc1，K=5,376 | 2,000 | block_threads=256，code 0 | 498.528 ms | 249.264 µs |
| out，K=7,168 | 1,000 | block_threads=512，code 0 | 337.920 ms | 337.920 µs |
| fc2，K=14,336，SwiGLU | 1,000 | block_threads=1024，code 2 | 911.173 ms | 911.173 µs |

实际 kernel 名为：

```text
comfy::<unnamed>::quantize_int8_rowwise_convrot64_kernel<__nv_bfloat16,...>
```

这证明 ConvRot 在线激活量化不是静态“可用功能”，而是每个主 INT8 Linear 都实际执行。fc2 的 SwiGLU 也进入了对应融合量化路径，不是全部拆成独立 eager activation。

### RoPE、residual 与 compute work 分布

- 1,000 个 BF16 `rope_kernel`，对应 fused RMSNorm+RoPE，合计 720.691 ms，平均 720.691 µs。
- 6,000 个分段 `addcmul`，对应 `20×50×2×3` 个 modality 段 residual，合计 452.751 ms。
- 20 步 DiT 共 59,724 个 kernel：首步 2,990，其余每步 2,986。

按全部 20 步 DiT kernel work=34.030 s 计算：

| 类别 | work 占比 |
|---|---:|
| BF16 FlashAttention | 约 51.9% |
| INT8 CUTLASS GEMM | 约 36.5% |
| ConvRot 在线量化 | 约 5.1% |
| fused RMSNorm+RoPE | 约 2.1% |
| 其他 op | 约 4.4% |

这些比例是 kernel work 占比，**不是阶段 wall-time 占比**。

## 5. Video VAE：42 次 ViT、6048 次 INT8 GEMM、INT8 attention

### 42 次 decoder 的来源

```text
7 temporal chunks
× 3 个高度 tile rows
× 每组宽度 tile 按 (4+1) 合 batch
= 42 次 ViT3DDecoder.forward
```

每次常规 ViT 有 648 个 kernel，第一次有 792 个，含额外初始化。每次 36 层，因此每类层调用为：

```text
42 × 36 = 1512
```

### INT8 Linear 与 residual epilogue

| 层类型 | 调用次数 | 计算路径 |
|---|---:|---|
| to_qkv | 1,512 | RMSNorm → ConvRot → INT8 GEMM，无 residual |
| to_out | 1,512 | ConvRot → INT8 GEMM → residual scale |
| ff.w1 | 1,512 | RMSNorm → ConvRot → INT8 GEMM，无 residual |
| ff.w2 | 1,512 | SwiGLU → ConvRot → INT8 GEMM → residual scale |
| 合计 | 6,048 | 42×36×4 |

实际 CUTLASS 统计：

```text
带 VisitorAuxLoad：   3024 个，对应 to_out/ff.w2 的 residual epilogue
不带 VisitorAuxLoad： 3024 个，对应 to_qkv/ff.w1
```

所以 VAE residual 融合已由 trace 证实，不再只是源码候选。

ConvRot 量化也有 6048 次：RMSNorm 输入 3024 次、无输入激活 1512 次、SwiGLU 1512 次。

### INT8 attention

```text
qk_int_sv_i8_attn_kernel<...>
```

- 1,512 次，合计 284.758 ms，平均 188.332 µs；
- `quant_qk_fused`：1,512 次；
- `quant_v_int8_kernel`：1,512 次；
- FP16 `rope_kernel`：1,512 次，合计 188.218 ms；
- `detect_k_anchor`：1,512 次。

这条路径与主 DiT 的 BF16 FlashAttention 完全不同：video VAE 实测使用 Kitchen/Sage INT8 attention，并含 Q/K/V 量化和 FP32 softmax 相关计算。

首次 ViT 的 host range 为 22.379 s、memcpy 为 4.869 GB，而后续每次约 35–63 ms，说明冷执行与常规 decoder 必须分开分析。

## 6. Audio VAE：实测 FP32，没有量化 kernel

音频阶段共 2,679 个 kernel，GPU work 为 34.288 ms，host range 为 3.050 s，memcpy 为 259.412 MB。

主要 kernel：

- FP32 Conv1d / ConvTranspose1d 对应的 cuDNN 和 implicit GEMM；
- `dgrad2d_grouped_direct_kernel`：127 次，合计 11.757 ms；
- `conv_depthwise2d_forward_kernel_generic`：127 次；
- `replication_pad_forward_kernel1d`：254 次；
- `sin_kernel`、exp、mul/add、residual、copy 等逐点 op。

`dgrad` 名称容易误导：它在这里服务于 BigVGAN 的 ConvTranspose1d 前向，不表示发生训练 backward。音频没有 INT8、NVFP4 或 INT8 attention kernel，与 checkpoint 的 FP32 配置一致。

## 7. 采样器和保存阶段

`res_multistep` 中扣除 20 个 DiT 子区间后，还有 1,067 个小 kernel，合计约 2.742 ms，主要是：

- `expm1`、`exp`、`log`、`nan_to_num`；
- `add`、`mul`、`div`、`neg`；
- `cat`、`copy`、fill 和其他 TensorIterator op。

这直接支持“res_multistep 不是一个单独 CUDA kernel，而是一组普通 tensor op”的结论。`eta=0`，没有额外 ancestral 噪声 kernel 注入路径。

CreateVideo/SaveVideo 使用 CPU tensor cast、NumPy、PyAV/FFmpeg 完成 MP4 编码，不是 DiT/VAE GPU kernel。PyAV 已验证输出与原 workflow 一致。

## 8. 冷启动与 runtime API：能解释什么、不能解释什么

全部 runtime API 中，host 时间靠前的项目为：

| API | 次数 | host total |
|---|---:|---:|
| `cudaMemsetAsync` | 4,293 | 13.764 s |
| `cudaStreamSynchronize` | 1,038 | 10.573 s |
| `cudaHostRegister` | 852 | 8.194 s |
| `cudaLaunchKernel` | 93,597 | 6.718 s |
| `cuMemCreate` | 1,153 | 3.140 s |
| `cuMemSetAccess` | 1,153 | 1.453 s |
| `cuMemGetInfo_v2` | 86 | 1.234 s |
| `cudaMemcpyAsync` | 4,275 | 1.077 s |
| `cudaMallocAsync` | 99,972 | 0.692 s |
| `cudaFreeAsync` | 98,443 | 0.073 s |

Qwen 的 `cudaHostRegister` 为 2.979 s，DiT 第一步为 4.034 s。首次执行还有 library/module load、host buffer 注册、VBAR 映射和权重 cast 等。

但这些 API 的 host total 仍远小于某些冷阶段 host range，例如 Qwen 的 113 s。由此必须保留一个边界：

- CUPTI 能证明实际 GPU kernel、memcpy 和 CUDA API；
- 磁盘/mmap page fault、Python、模型结构处理、部分权重准备、host 等待并不都包含在 kernel trace 中；
- 不能把 Qwen 的 113 s 或首步 178.6 s 全部解释成 GPU kernel 慢或 memcpy 慢；
- 也不能把不同运行环境、host cache 状态下的总时间直接当作稳定性能对比。

## 9. 对原静态分析的修正与确认

| 原静态分析 | CUPTI 实测结果 |
|---|---|
| 主 DiT 的 SDPA 后端未知 | **BF16 `pytorch_flash::flash_fwd_kernel`** |
| Qwen 的 full-precision GEMM dtype 不写死 | **FP32 GEMM，FP32 mem-efficient attention** |
| NVFP4 不做 FP4 GEMM（由 metadata/源码推断） | **350 次反量化 + 350 次 FP32 GEMM，未出现 FP4 scaled_mm** |
| INT8 ConvRot/CUTLASS 是优先候选 | **主 DiT 4000 次、video VAE 6048 次实际命中** |
| VAE residual 可以融合 | **3024 个 `VisitorAuxLoad` 实际命中** |
| video VAE INT8 attention 条件满足 | **1512 个 `qk_int_sv_i8_attn_kernel` 实际执行** |
| allocation graph 不等于 CUDA Graph replay | **全部 kernel 的 graphNodeId 为空，未观察到 graph node** |
| video/audio VAE 是两条不同解码路径 | **INT8 ViT video decode 与 FP32 BigVGAN audio decode 均已分开采集** |

原静态分析中的“尚无 CUPTI”只能视为当时证据边界。后续判断应以本报告的真实 trace 为准。

## 10. 可复核产物

### 原始证据

- `logs/cupti_h3_t2v/h3_t2v_cold.nsys-rep`
- `logs/cupti_h3_t2v/h3_t2v_cold.sqlite`
- `logs/cupti_h3_t2v/history.json`
- `logs/cupti_h3_t2v/run_result.json`
- `logs/cupti_h3_t2v/server_nsys.log`
- `logs/cupti_h3_t2v/client.log`

### 解析结果

目录：`logs/cupti_h3_t2v/analysis/`

- `manifest.json`：总量和映射规则；
- `global_kernel_summary.csv`：155 种 kernel 的全局统计；
- `phase_kernel_summary.csv`：各模型/阶段统计；
- `range_kernel_summary.csv`：每个 DiT step、每个 ViT call 的统计；
- `range_summary.csv`：NVTX host range、kernel、memcpy、memset；
- `runtime_api_summary.csv`：所有 runtime API，不只是 launch；
- `kernel_events.csv.gz`：逐 kernel 原始记录和 launch 参数；
- `nvtx_kern_sum.txt`：Nsight 自带的 NVTX kernel 汇总，用于交叉核验。

### 复现脚本

- `profile_h3_cupti.sh`：独立 nsys ComfyUI launcher；
- `run_h3_cupti.py`：启动/停止 profiler、提交原 prompt、轮询结果；
- `analyze_h3_cupti.py`：SQLite 关联、聚合和导出；
- `ComfyUI/custom_nodes/h3_cupti_profiler.py`：NVTX range 和 profiler 控制接口。

详细命令、问题与解决过程见 `run_book.md` 的“CUPTI 逐 Kernel 实测”章节。
