# H3 dense Linear MXFP8 / MXFP4 接入 ComfyUI / Comfy-Kitchen：设计 Proposal

> 状态：**供审阅的设计稿，尚未实现，尚未运行 MXFP GEMM**。
>
> 目标 workflow：`video_minimax_h3_t2v.json`。
>
> 首要硬件：本机 B200（SM100）。优先支持当前 workflow 的 dense Linear，不展开 MoE、专家路由、训练量化或完整跨平台后端矩阵。
>
> 本文将已核实的现状、推荐设计和待实测事项分开陈述。文中的接口代码是设计示意，不是已经存在的实现。

## 1. 结论与推荐路线

**可以做成，而且不必改写 H3 模型网络，也不必让 ComfyUI 依赖 vLLM。** 推荐路线如下：

1. 在 Comfy-Kitchen 内补齐 MXFP8 的真实 CUDA GEMM 和 MXFP4 的完整量化布局。
2. 首选 **FlashInfer 的 CUTLASS / CuTe-DSL block-scaled GEMM**，由 Comfy-Kitchen 负责格式适配、能力检查和 kernel 选择；不复制 FlashInfer 的底层 CUDA/CUTLASS 源码。
3. 继续走 ComfyUI 已有的 `QuantizedTensor → mixed_precision_ops.Linear → .comfy_quant` 路径，不增加 H3 专用模型节点。
4. 原生模型仍是单个 `.safetensors` 文件。compressed-tensors 的 MXFP8/MXFP4 权重通过转换器无损映射为 Comfy 原生格式，随后与 BF16 转出的 MX 权重共用同一套运行时代码。
5. 借鉴 vLLM 的“候选列表 → `is_supported` → `can_implement` → 失败原因”策略，但保留 Comfy-Kitchen 已有 registry。形成两层选择：外层选择 CUDA/eager，内层选择 FlashInfer 具体 kernel。
6. **W8A8、W4A4 和 weight-only W4A16/W8A16 必须明确区分。** 不把 Marlin 或 dequantize 后普通 GEMM 静默冒充成 MXFP8/MXFP4 真 GEMM。
7. 先实现 dense Linear 的正确性，再考虑 RMSNorm/SwiGLU/residual 融合。视频 VAE 继续用 Sage INT8 attention，主 DiT 继续用 BF16 FlashAttention，音频 VAE 保持 FP32。

对现有判断的一个重要修正是：

- 当前 INT8、NVFP4 dequant/GEMM、Sage INT8 attention 的 Python 调度确实在 Comfy-Kitchen；C++ 实现由其 `_C` 扩展承载。
- **新增 MX 路径的实际 GEMM 可以来自 FlashInfer，而不是必须全部来自 comfy-kitchen `_C`。** Comfy-Kitchen 是统一接入点，不必是所有 kernel 的源代码所有者。

## 2. 范围、证据和术语

### 2.1 已核实的基线

- Workflow：[video_minimax_h3_t2v.json](./video_minimax_h3_t2v.json)。
- 已执行 API prompt：[api_prompt.json](./logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json)。
- 现有推理分析：[comfyui_h3_t2v_kernel_analysis.md](./comfyui_h3_t2v_kernel_analysis.md)。
- 既有 CUPTI trace：`logs/cupti_h3_t2v/h3_t2v_cold.sqlite`。
- ComfyUI 调研版本：`989d93d0d968e704c589ce0146bb18c77f6b50cc`。
- Comfy-Kitchen：0.2.41；PyTorch：2.13.0+cu130。
- vLLM 参考环境：0.29.0，位于 `/home/kaokaolv/.venv-vllmomni029`。
- **当前 ComfyUI 虚拟环境没有 FlashInfer**；本机另一个环境有 FlashInfer 源码，不代表当前服务可以直接调用。

原 workflow 的 turbo 配置和已有 CUPTI 复测配置不是同一组采样参数。后续性能/画质对比必须固定实际 API prompt，不能把原模板 turbo 结果与 20-step trace 混为一组。

### 2.2 “dense”在这里的准确含义

当前需要改造的都是稠密 Linear，没有 MoE 路由专家；但整个模型并不只有 Linear。

| 模块 | 稠密 Linear | 保留的其他计算 |
|---|---|---|
| 主 DiT | 50 个 block，每块 `qkv_proj / out_proj / fc1 / fc2` | BF16 FlashAttention、RMSNorm、RoPE、调制、patch/输出层 |
| Qwen 文本编码器 | 50 层，每层 `q/k/v/o_proj` 和 `gate/up/down_proj` | attention、归一化、embedding、文本 connector |
| 视频 VAE ViT | 36 层，每层 `to_qkv / to_out / ff.w1 / ff.w2` | Sage INT8 attention、RMS/RoPE、卷积和解码 |
| 音频 VAE | 不在范围内 | FP32 Conv1d、ConvTranspose1d、Snake 等 |

主 DiT 中可选的 `to_gate_compress` 服务于稀疏 attention 扩展，不是 MoE expert。当前 dense forward 不使用它，本 proposal 也不因此扩展为 MoE 方案。

### 2.3 三种容易混淆的计算

- **W8A8 MXFP8**：权重和 activation 都是 E4M3 FP8，按 32 元素共享 E8M0 scale，执行 block-scaled GEMM。
- **W4A4 MXFP4**：权重和 activation 都是 E2M1 FP4，按 32 元素共享 E8M0 scale，执行 block-scaled GEMM。
- **W4A16/W8A16**：只压缩权重，activation 保持 FP16/BF16，例如 Marlin 路线。可以另做对照，但不属于本方案的 W4A4/W8A8 验收。

“加载了 MXFP 权重”也不等于“执行了 MXFP GEMM”。dequantize 后 FP32/BF16 普通 GEMM 只能作为参考实现或显式 fallback。

## 3. 当前 H3 的实际目标层

以下 K、N 来自现有模型和 CUPTI 分析；静态脚本检查确认，**这些目标层的 K、N 都能被 128 整除**。因此不需要为了当前 H3 对 contraction K 补零，也不需要支持任意异形 Linear 才能开始。

### 3.1 主 DiT

| 层 | K | N | 层数 | 总 Linear 数 |
|---|---:|---:|---:|---:|
| `qkv_proj` | 5376 | 21504 | 50 | 50 |
| `out_proj` | 7168 | 5376 | 50 | 50 |
| `fc1` | 5376 | 28672 | 50 | 50 |
| `fc2` | 14336 | 5376 | 50 | 50 |

当前为 INT8 tensor-wise + ConvRot（group 256）。已有 trace 中 20 步共 4000 次 INT8 GEMM；本例 packed M=15787。主 attention 是 BF16 FlashAttention，不是 INT8 attention。

### 3.2 Qwen 文本编码器

| 层 | K | N | 层数 | 总 Linear 数 |
|---|---:|---:|---:|---:|
| `q_proj` | 5120 | 8192 | 50 | 50 |
| `k_proj / v_proj` | 5120 | 1024 | 各 50 | 100 |
| `o_proj` | 8192 | 5120 | 50 | 50 |
| `gate_proj / up_proj` | 5120 | 25600 | 各 50 | 100 |
| `down_proj` | 25600 | 5120 | 50 | 50 |

当前 350 个 Linear 是 NVFP4 存储，但 `full_precision_matrix_mult=true` 和文本编码器 loader 策略导致 **dequantize 后 FP32 GEMM**。已有 trace 是 350 次 NVFP4 dequant + 350 次 FP32 SGEMM，不是 FP4 Tensor Core GEMM。本例 text length=388。

### 3.3 视频 VAE ViT

| 层 | K | N | 层数 | 总 Linear 数 |
|---|---:|---:|---:|---:|
| `to_qkv` | 2048 | 6144 | 36 | 36 |
| `to_out` | 2048 | 2048 | 36 | 36 |
| `ff.w1` | 2048 | 16384 | 36 | 36 |
| `ff.w2` | 8192 | 2048 | 36 | 36 |

当前为 INT8 ConvRot。已有 trace 中 42 次 ViT forward 共 6048 次 INT8 GEMM；M=1797 或 7188。`to_out/ff.w2` 有 residual，`ff.w2` 前有 SwiGLU，`to_qkv/ff.w1` 前有 RMSNorm。

### 3.4 只用于估算的权重 payload

下表只计算上述 Linear 的 qdata 和理论线性 scale，分别按 BF16 16 bit、MXFP8 8.25 bit、MXFP4 4.25 bit/元素估算。不是完整 checkpoint 大小，也不是实测 VRAM。

| 模块 | Linear 权重元素 | BF16 GiB | MXFP8 GiB | MXFP4 GiB |
|---|---:|---:|---:|---:|
| DiT | 19,267,584,000 | 35.889 | 18.505 | 9.533 |
| Qwen | 24,379,392,000 | 45.410 | 23.415 | 12.062 |
| Video VAE ViT | 2,415,919,104 | 4.500 | 2.320 | 1.195 |

目标 K、N 已对齐，当前形状的 scale padding 不额外增加理论比例；其他模型权重、bias、embedding、LoRA、临时 activation、工作区和 allocator 仍需另算。

## 4. 现有实现的真实缺口

### 4.1 MXFP8 不是从零开始，但缺少当前 CUDA 真 GEMM

已有组件：

- `comfy_kitchen/tensor/mxfp8.py`：`TensorCoreMXFP8Layout`，FP8 E4M3、32 元素一组、E8M0 scale、128x4 blocked scale、逻辑 transpose。
- `comfy_kitchen/backends/cuda/__init__.py:1841`：`quantize_mxfp8`，量化输出 `[M,K]` FP8 和 blocked scale。
- `comfy_kitchen/__init__.py`：公开 `scaled_mm_mxfp8(...)`。
- `backends/eager/quantization.py:371`：eager `scaled_mm_mxfp8`，进入 `scaled_mm_v2`。
- `ComfyUI/comfy/quant_ops.py`：已有 MXFP8 layout 子类和 `QUANT_ALGOS["mxfp8"]`。

缺口：当前 CUDA backend capability 列表有 `quantize_mxfp8`，**没有 `scaled_mm_mxfp8`**。所以不能把已有 Python API 解释为 B200 上已有真实 MXFP8 CUDA GEMM。

另一个需要区分的事实是：`comfy_kitchen/scaled_mm_v2.py:101-126` 在 PyTorch 提供该能力时会调用 `torch.nn.functional.scaled_mm`，并有其他兼容分支。因此不能反过来把 eager 一律称为 fake。它是否在当前 B200/PyTorch 上形成真实 MXFP8 Tensor Core GEMM，必须在 P0 单独用 CUPTI 验证；在验证前，不把它冒充为已经证明可用的 CUDA backend。

### 4.2 MXFP4 不能通过 NVFP4 改几个参数获得

需要补齐：

- packed E2M1 quantize/dequantize 和 eager reference；
- `TensorCoreMXFP4Layout` 及序列化；
- `quantize_mxfp4` / `scaled_mm_mxfp4` 公开接口及 fake/meta；
- CUDA backend adapter 和能力检查；
- ComfyUI `QUANT_ALGOS`、注册、加载和保存映射。

`scaled_mm_nvfp4` 当前虽然有一段按 scale dtype 区分 MXFP4 的代码，但 `block_length` 固定为 16，E8M0 分支直接报“不支持”，因此不能把它当成已有 MXFP4 kernel。

### 4.3 当前三层选择机制各自负责什么

1. ComfyUI `QuantizedTensor.__torch_dispatch__`：选择 layout-specific `linear/mm/addmm` handler。
2. Comfy-Kitchen `BackendRegistry`：按 `ascend → cuda → triton → eager` 和 `FunctionConstraints` 选择实现。
3. vLLM：在某一格式的多个 kernel 之间做 `is_supported/can_implement` 选择。

推荐复用前两层，在 CUDA backend 内增加第三层的小型 `MXLinearKernel` 选择器。**不重写整个 registry，也不在 H3 模型里硬编码 FlashInfer。**

### 4.4 现有 silent fallback 不适合作为性能验收

`mxfp8.py` 和 `nvfp4.py` 的 linear handler 会捕获部分异常后 dequantize 再普通 GEMM。这样能提高兼容性，却会让“模型能生成”掩盖“没有走目标 kernel”。

本方案改为：

- 非 Linear 的通用 op 仍可 dequantize fallback；
- `full_precision_matrix_mult=true` 是明确的参考路径；
- `full_precision_matrix_mult=false` 时，MX Linear 找不到真 backend 要给出完整失败原因；
- 只有显式指定 emulation/eager 才允许参考实现；
- 选中的 kernel 在执行时抛出的错误继续抛出，不能再伪装成“硬件不支持”而静默换路。

## 5. 数学格式和量化契约

### 5.1 统一的逻辑表示

对逻辑矩阵 `X[R,K]`，沿最后一维每 32 个元素形成一个 block：

```text
A[r,g] = max(abs(X[r, 32g : 32g+32]))
X ≈ X_hat[r, 32g+j] = Q[r, 32g+j] × S[r,g],  j=0..31
```

其中：

| 格式 | Q 的存储 | S 的存储 | 全局 scale |
|---|---|---|---|
| MXFP8 | E4M3，每元素 1 byte | E8M0，`S=2^(e-127)` | 无 |
| MXFP4 | E2M1，每元素 4 bit，两个/byte | E8M0，`S=2^(e-127)` | 无 |
| NVFP4，仅作对照 | E2M1，两个/byte | E4M3，group 16 | 有 tensor scale |

MXFP4 packed byte 约定：**低 nibble 是偶数元素，高 nibble 是奇数元素**。必须固定，不允许转换器和 kernel 各自猜测。

### 5.2 MXFP8 的量化

Comfy-Kitchen eager 和 vLLM 参考实现的主要规则为：

```text
e = clamp(ceil(log2(max(A, tiny) / 448)) + 127, 0, 254)
S = 2^(e - 127)
Q = round_to_E4M3(clamp(X / S, -448, 448))
```

全零 block、极小值、饱和和 NaN/Inf 的处理必须与选定实现逐项测试。Comfy-Kitchen 的 eager `e8m0_to_f32` 对 exponent 0 有特殊处理，因此不能只拿另一个库的指数编码表推断它完全一致。

### 5.3 MXFP4 的量化和舍入不是附带细节

vLLM `mxfp4_utils.py` 的参考 Triton 权重量化先对 block amax 做 round-to-nearest-even，再用：

```text
exponent = floor(log2(rounded_amax)) - 2
S = 2^clamp(exponent, -127, 127)
Q = round_to_nearest_even_E2M1(X / S)
packed = Q[2i] | (Q[2i+1] << 4)
```

FlashInfer 另有 CUDA 和实验性 CuTe-DSL quantizer。**不能事先宣称 vLLM、FlashInfer 和自写参考实现的每个输出 byte 都相同。** 原生文件必须记录 quantizer/rounding 版本；格式验收要分别检查：

1. 32 元素分组是否正确；
2. scale 的选择和零 block 规则是否一致；
3. E2M1 的 RNE 和饱和是否正确；
4. packed nibble 顺序是否正确；
5. activation 量化是否与权重使用兼容规则。

### 5.4 真 GEMM 的定义

MX Linear 的目标计算是：

```text
Y[m,n] = bias[n] + Σg ( SA[m,g] × SB[n,g] ×
                         Σj(QA[m,32g+j] × QB[n,32g+j]) )
```

实际 Tensor Core 按 block-scaled 指令及其 tile 顺序执行，accumulation 以 FP32 语义理解，输出 FP16/BF16。不能为了简化实现先物化完整 `dequantize(A)` 和 `dequantize(B)`，再声称走了真 MX GEMM。

参考实现可以做上述 dequantize + FP32 矩阵乘法，用于校验；它不代表现实 kernel 的精确累加顺序，也不保证与硬件 bitwise 相等。

## 6. 原生文件和 runtime scale 布局

### 6.1 逻辑格式与物理布局分开命名

- compressed-tensors 的交换语义：`weight/weight_scale` 或 `weight_packed/weight_scale`，scale 是逻辑 `[N,K/32]`。
- Comfy-Kitchen 当前 MXFP8 runtime：qdata 保持 `[N,K]`，scale 已按 128x4 重排。
- FlashInfer：接受同语义的 1D 128x4 scale；qdata B 可以是 `[N,K]` 的 `.t()` 视图。

本方案第一版原生文件沿用 Comfy-Kitchen 现有的 **128x4 scale 存储**，避免改变旧 MXFP8 文件语义。compressed-tensors 转换器负责线性 scale → 128x4 scale 的无损重排。

这不是宣称所有未来硬件都接受 128x4；它是当前 B200 两个首选 backend 共用的物理格式。未来新增其他 layout 时必须在 loader 做一次显式转换，不能每次 forward 重排。

### 6.2 128x4 scale 的形状

对逻辑 `[R,K]` 数据：

```text
R_scale = ceil(R / 128) × 128
K_scale = ceil((K / 32) / 4) × 4
scale.numel = R_scale × K_scale
```

可将它保留为 `[R_scale,K_scale]` 的 contiguous tensor；调用 FlashInfer 时 `.reshape(-1)`，只生成 view，不复制数据。

本轮已做一项纯 CPU 索引验证：对 `[128,8]`、按顺序编码的 scale，Comfy-Kitchen `to_blocked` 与 vLLM `sf.view(M/128,4,32,K/128,4).transpose(1,3).contiguous()` 的输出 **1024 个 byte 全部相同**，`from_blocked` 也恢复原矩阵。这证明了两套 Python 重排逻辑在该样例上兼容；它不代替后续对真实 CUDA quantizer 输出和 FlashInfer GEMM 的验证。

当前 K、N 全部对齐 128。M 可以不是 128 的倍数，scale 的 M tile 仍要补齐；这不等于 GEMM 输出要永久保留 padding 行。

例如 DiT 的 M=15787，如果复用当前 MXFP8 的 32 行 padding，则实际 activation 为 15808 行，输出再裁回 15787 行。M padding 只影响边界，不能参与有效输出。

### 6.3 `.comfy_quant` 建议格式

每层保存：

```json
{
  "format": "mxfp4",
  "group_size": 32,
  "scale_dtype": "e8m0",
  "scale_layout": "128x4",
  "packing": "e2m1_low_even",
  "activation_scheme": "dynamic",
  "quantizer": "<converter-and-rounding-version>",
  "full_precision_matrix_mult": false
}
```

MXFP8 对应 `format="mxfp8"`，无 FP4 packing 字段。要求：

- `weight`：MXFP8 为 `[N,K] float8_e4m3fn`；MXFP4 为 `[N,K/2] uint8`。
- `weight_scale`：uint8 或其 `float8_e8m0fnu` view，物理 128x4 layout。
- 不保存无意义的 `weight_scale_2/input_scale/pre_quant_scale`；MX 不需要 NVFP4 global scale。
- 未量化权重和其他 tensor 原样保留。
- 文件的顶层 `_quantization_metadata` 仍按 ComfyUI 已有结构描述各层，由 `convert_old_quants` 生成 `.comfy_quant`。

缺少 group size、scale layout 或字段互相矛盾时明确报错；不能先当 BF16 加载再继续推理。

## 7. Comfy-Kitchen 接口设计

### 7.1 保持现有 MXFP8 API

```python
quantize_mxfp8(x, pad_32x=False) -> (qdata_fp8, scale_e8m0)
dequantize_mxfp8(qx, block_scales, output_type=torch.bfloat16)

scaled_mm_mxfp8(
    a,                         # [M,K], FP8 E4M3
    b,                         # [N,K], FP8 E4M3，不是预先转置的权重
    block_scale_a,             # 逻辑 A 的 128x4 scale
    block_scale_b,             # 逻辑 B 的 128x4 scale
    bias=None,
    out_dtype=None,
)                            # [M,N]
```

`scaled_mm_mxfp8` 保持 `a @ b.T + bias` 语义。FlashInfer 适配器内部调用 `b.t()`，**不转置 `block_scale_b`**，因为 scale 仍按权重原始 `[N,K/32]` 分组。

### 7.2 新增对应的 MXFP4 API

```python
quantize_mxfp4(x, pad_32x=False) -> (qdata_uint8, scale_e8m0)
dequantize_mxfp4(qx, block_scales, output_type=torch.bfloat16)

scaled_mm_mxfp4(
    a,                         # [M,K/2], packed E2M1
    b,                         # [N,K/2], packed E2M1
    block_scale_a,
    block_scale_b,
    bias=None,
    out_dtype=None,
)                            # [M,N]
```

约束：

- K 是逻辑 contraction 维度，不是 packed byte 数；
- K 必须被 32 整除，当前 H3 实际全部被 128 整除；
- `out_dtype` 真 kernel 第一版只接受 FP16/BF16；
- bias 如果 backend 不支持原生 epilogue，则在 GEMM 后单独加，保持明确的输出 dtype；
- `alpha`、NVFP4 tensor scale 不属于 MXFP4 接口。

FlashInfer MXFP4 调用固定为：

```python
mm_fp4(
    a, b.t(), scale_a_1d, scale_b_1d,
    alpha=None,
    out_dtype=out_dtype,
    block_size=32,
    use_nvfp4=False,
    backend=selected_backend,
)
```

### 7.3 量化和 fake/meta 契约

- 输入必须是二维 contiguous tensor；shape 转换由 ComfyUI 外层负责。
- MXFP8 qdata 为 `[R_pad32,K_pad32]`，MXFP4 为 `[R_pad32,K_pad32/2]`。
- scale fake shape 与真实 128x4 padding 完全一致。
- `scaled_mm_mxfp*_fake` 返回 `[M,N]`，dtype/device 与真实输出契约一致。
- quantizer 的版本信息不能仅靠 Python 函数名推断；文件 manifest 记录实际选择。
- torch.compile 下 backend Python 代码隐藏在 custom op 后；首次 JIT/autotune 不允许发生在 CUDA Graph capture 内。

## 8. Tensor layout 设计

### 8.1 复用并修正 MXFP8 layout

`TensorCoreMXFP8Layout` 继续拥有：

- `Params.scale`：E8M0 blocked scale；
- `orig_dtype`、`orig_shape`；
- 逻辑 `transposed`；
- `get_storage_shape/get_logical_shape_from_storage`；
- `linear/mm/addmm` 和 `state_dict_tensors`。

需要调整的不是数学格式，而是 fast-path 的错误处理和 backend dispatch。

### 8.2 新增 TensorCoreMXFP4Layout

建议与 MXFP8 对称实现：

```text
Params.scale = E8M0 blocked scale
Params.orig_shape = (N,K)                  # 逻辑形状
qdata.shape = (ceil32(N), ceil32(K)/2)     # packed 物理形状
get_padded_shape = ceil32(N), ceil32(K)
get_storage_shape = padded_N, padded_K/2
```

`.t()` 只改变逻辑 flag，不重新 pack，也不数值转置 scale。`linear` 要求两个 operand 都是同 layout 的 QuantizedTensor；`mm` 要求第二个 operand 已逻辑转置。

### 8.3 非 Linear op 和序列化

- `.to(device)`、`.clone()`、offload/pin 必须同时处理 qdata 和 scale。
- `.dequantize()` 先 unswizzle scale，再按 32 元素恢复，最后裁掉 logical padding。
- `state_dict_tensors()` 输出 `weight` 和 `weight_scale`，与 loader 使用相同布局。
- 任意 view/reshape 若没有明确 packed 语义，走 dequantize，不保留一个看似有效但物理含义错误的 QuantizedTensor。

### 8.4 不复制 INT8 的融合接口

初版 MX layout 对带 `input_act/residual` 的 `linear_input_act` 返回 `NotImplemented`。现有 ComfyUI 调用方随后执行：

```text
RMSNorm 或 SwiGLU（模块层原有实现）
  → MXFP quantize activation
  → MXFP GEMM
  → bias / residual（现有后处理）
```

这样不会把 INT8 ConvRot 的旋转、fused epilogue 或 quantization scale 错误带入 MX。

## 9. vLLM 风格的 kernel 选择

### 9.1 两层 dispatch

```text
ComfyUI F.linear
  → QuantizedTensor layout handler
  → torch.ops.comfy_kitchen.scaled_mm_mxfp8 / mxfp4
  → BackendRegistry 选择 cuda（外层）
  → MXLinearSelector 选择具体 FlashInfer backend（内层）
  → quantize activation + 真 GEMM
```

外层 `FunctionConstraints` 负责通用 dtype、device、维度和最低 compute capability。内层负责真实 device、第三方模块、CUDA/CuTe 版本、K/N、输出 dtype、scale layout 和 activation scheme。

fast path 必须锁定经过验证的 CUDA 实现，不能在它失败后由外层 registry 自动落到名称相同但语义未验证的 eager 实现。显式 `eager_reference` 由 layout 直接 dequantize 后计算，不借普通 registry fallback 伪装成 fast path。现有 `torch_scaled_mm_v2` 若要成为 fast-path 候选，必须独立通过真实 kernel 验证。

硬件检查使用输入 tensor 的实际 device，不能只读取创建进程时的 current device。

### 9.2 候选接口

设计示意：

```python
@dataclass(frozen=True)
class MXLinearConfig:
    fmt: Literal["mxfp8", "mxfp4"]
    n: int
    k: int                         # 逻辑 K
    activation_scheme: Literal["dynamic"]
    out_dtype: torch.dtype
    scale_layout: Literal["128x4"]

class MXLinearKernel(ABC):
    name: str

    @classmethod
    def is_supported(cls, device: torch.device) -> tuple[bool, str | None]: ...

    @classmethod
    def can_implement(cls, config: MXLinearConfig) -> tuple[bool, str | None]: ...

    def apply(self, a, b, scale_a, scale_b, bias, out_dtype) -> Tensor: ...
```

当前两个 FlashInfer 候选都可直接消费 128x4 scale，所以 `apply` 只需 contiguous/1D view。未来若加入不同物理格式，应在权重加载/准备阶段一次性 repack，由模型自己的量化参数拥有新 buffer；不增加进程级 tensor cache，也不在每次 forward 重排。

### 9.3 第一版候选矩阵

以下是**源码支持的设计候选，不是本机已经跑通的矩阵**：

| 格式 | 候选 | 计算 | 硬件/依赖检查 | 第一版定位 |
|---|---|---|---|---|
| MXFP8 | `flashinfer_cutedsl` | W8A8 | SM100/SM103 family、FlashInfer、CuTe-DSL、版本和 shape | 优先验证，参考 vLLM 顺序 |
| MXFP8 | `flashinfer_cutlass` | W8A8 | SM100+、FlashInfer CUTLASS、128x4 scale | 可比较的备选 |
| MXFP8 | `torch_scaled_mm_v2` | 待 CUPTI 确认 | PyTorch 实际 API、scale recipe 和 B200 kernel 约束 | P0 再决定是否启用 |
| MXFP4 | `flashinfer_cutedsl` | W4A4 | 本机接口列出的 SM 家族、FlashInfer、CuTe-DSL、N%8 | 首选 B200 路线 |
| MXFP4 | `flashinfer_cutlass` | W4A4 | 实际安装的 FlashInfer requirement 必须通过 | 验证后启用的备选 |
| MXFP8/4 | `eager_reference` | 本文定义的 dequant + FP32 reference | 无真 kernel 要求 | 仅显式 reference/emulation |
| MXFP8/4 | Marlin | W8A16/W4A16 | 各自的 Marlin 约束 | 不在 W8A8/W4A4 默认候选中 |

不直接把 vLLM 的全部 ROCm/XPU/MoE/B12x 选择表复制过来。SM120/121 的 B12x、cuDNN、TRT-LLM 和 Marlin 可以作为后续独立候选，但不阻塞当前 B200 H3。

### 9.4 选择、覆盖和错误报告

建议一个统一配置：

```text
COMFY_KITCHEN_MX_LINEAR_BACKEND=auto
# auto | flashinfer_cutedsl | flashinfer_cutlass | eager_reference
```

选择顺序：

1. 过滤显式禁用/覆盖项；
2. `is_supported(actual_device)`；
3. `can_implement(format, activation_scheme, N,K,out_dtype,scale_layout)`；
4. 返回第一个可用候选；
5. 没有候选时报告每一个拒绝原因。

FlashInfer 内部 tuning 只能在已选 backend 内选择 tactic，不能悄悄换成另一个计算格式。实际 backend 名称、版本、N/K/out_dtype 在首次选择时记录一次，便于和 CUPTI 对照。

**缺失依赖与运行错误分开处理**：可选 import/probe 失败可排除候选；已经选中的 GEMM 失败则原样抛出并带 backend 上下文，不能借 fallback 隐藏真正的 kernel bug。

## 10. FlashInfer adapter 的实现方式

### 10.1 推荐的薄适配，而不是 fork kernel

新增 `comfy_kitchen/backends/cuda/mx_linear.py`，负责：

1. 可选依赖探测和版本能力检查；
2. `MXLinearKernel` 两个 FlashInfer 实现；
3. activation quantizer 的选择；
4. `mm_mxfp8/mm_fp4` 参数转换；
5. 错误原因和选择日志。

`backends/cuda/__init__.py` 只增加公共 wrapper 和 constraints。没有安装 FlashInfer 时，INT8/NVFP4 等其他 CUDA backend 必须照常工作。

### 10.2 量化与 GEMM 不必来自同一个子后端

可将“稳定 CUDA activation quantizer”与“CuTe-DSL GEMM”组合，只要 scale layout 和舍入契约通过测试。权重转换器、activation quantizer、GEMM backend 分别记录，不假定它们必须同名。

优先顺序：

1. 小形状验证 FlashInfer CUDA quantizer + FlashInfer GEMM；
2. 验证 CuTe-DSL quantizer 的数值和启动开销；
3. 将确定可用的组合放入对应 kernel candidate；
4. 对 H3 实际 M/N/K 做 warmup 和正确性复核。

### 10.3 kernel 来源可追踪

预期调用链：

```text
ck.scaled_mm_mxfp4
  → cuda.mx_linear.FlashInferCuteDslMXFP4.apply
  → flashinfer.mxfp4_quantize / 选定的 activation quantizer
  → flashinfer.mm_fp4(block_size=32,use_nvfp4=False,backend="cute-dsl")
  → FlashInfer CuTe/CUTLASS SM100 block-scaled kernel
```

MXFP8 类似，但调用 `mm_mxfp8`。最终 kernel 名称必须以实际 CUPTI trace 为准，设计阶段不虚构 kernel 名称或 SM 指令 trace。

### 10.4 B200 CuTe-DSL kernel 的实际计算结构

本机 `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py` 已核实：它不是将两个矩阵先反量化再普通 GEMM，而是 `Sm100BlockScaledPersistentDenseGemmKernel` 直接消费 A、B、SFA、SFB。文件注明该实现由 TensorRT-LLM 的 persistent dense block-scaled GEMM 移植并适配 FlashInfer。

关键数据流为：

```text
A/B qdata 和独立 SFA/SFB
  → 构造 block-scaled tiled MMA 与 TMA descriptor
  → TMA warp 搬运 A/B/SFA/SFB 到 shared-memory stages
  → SFA/SFB 写入 tcgen05.MMA 对应字段
  → cute.gemm 对量化 operand 做 block-scaled MMA
  → FP32 accumulator（tensor memory）
  → epilogue 转为输出 dtype、alpha 和输出搬运
```

源码对应关系：

- `dense_blockscaled_gemm_sm100_common.py:84-205`：为 A/B/SFA/SFB 创建 TMA 描述符；`tile_atom_to_shape_SF` 将 scale byte buffer 解释为硬件所需 atom layout。
- `dense_blockscaled_gemm_sm100.py:109-140`：accumulator 是 FP32；epilogue warp 为 0–3，MMA warp 为 4，TMA warp 为 5，采用 warp specialization。
- `dense_blockscaled_gemm_sm100.py:156-180`：`make_blockscaled_trivial_tiled_mma` 构造 tcgen05 block-scaled MMA。
- `dense_blockscaled_gemm_sm100.py:193-215`：MMA instruction 每步 K=256 bit，即 FP8 为 32 元素、FP4 为 64 元素；tile K 再包含 4 个 instruction K step。它与“每 32 个元素一个 E8M0 scale”是不同层次，不能把 instruction K 和 scale group 混为一谈。
- `dense_blockscaled_gemm_sm100.py:262-307`：分别建立 A/B/SFA/SFB 的 shared-memory layout 和 accumulator/scale 的 tensor-memory 分配。
- `dense_blockscaled_gemm_sm100.py:550-573`：`PipelineTmaUmma` 管理 A/B 输入流水，`PipelineUmmaAsync` 管理 accumulator。
- `dense_blockscaled_gemm_sm100.py:1127-1144`：设置 SFA/SFB 后调用 `cute.gemm`，之后持续启用 accumulator。
- 底层类支持 FP32 accumulator；FlashInfer 公开 `mm_fp4/mm_mxfp8` wrapper 的本机接口仍只接受 FP16/BF16 输出，不能把底层能力扩大解释为当前公开 API 已支持 FP32 输出。

该实现还支持 persistent tile scheduler、tile/cluster 配置及有限候选的 autotuning。MMA tiler M 为 128/256，N 为 64/128/192/256；这描述的是 tile，不要求整个 H3 Linear 的 M、N 只能是这些数。

因此，Comfy-Kitchen 需要保证的是 **qdata、scale、逻辑形状和 backend 约束正确**，而不是自己再实现一套 tcgen05 mainloop。最终实际 launch 参数、selected tactic 和 SASS 指令仍由 P0 和 CUPTI 验证。

## 11. ComfyUI 模型加载与执行映射

### 11.1 注册新格式

`comfy/quant_ops.py`：

- 引入并注册 `TensorCoreMXFP4Layout`；
- 添加 `QUANT_ALGOS["mxfp4"]`；
- `storage_t=torch.uint8`、`group_size=32`、动态 activation；
- MXFP8 保持现有注册。

`comfy/ops.py` 的 `_load_quantized_module`：

- MXFP8/MXFP4 都读取 `weight_scale` 并以 `view(torch.float8_e8m0fnu)` 解释，不做浮点数值转换；
- 校验 packed shape、group 32、scale layout 和逻辑 `orig_shape`；
- 构造对应 `Params(scale=..., orig_dtype=..., orig_shape=...)`；
- 不把未知 quant format 静默退化为 BF16。

### 11.2 解决文本编码器 full-precision 默认值

当前 `sd1_clip.py` 给量化文本编码器传入 `full_precision_mm=True`。建议不全局修改它，而是在 `_load_quantized_module` 只对 MXFP8/MXFP4 使用如下优先级：

```text
disabled_formats（强制禁止 fast path）
  > 新 MX checkpoint 显式 full_precision_matrix_mult
  > mixed_precision_ops 的父级默认
```

新 MX 文件显式保存 `full_precision_matrix_mult=false`，因此能覆盖父级默认。未写该字段时继续继承父级默认；其他量化格式保留原有分支，现有 NVFP4 文本编码器仍保持原来的 FP32 路径。

保存时也要对新 MX 格式显式写 false，否则 save/load 一次后可能丢失 fast-path 策略。`disabled_formats` 仍然优先禁止 fast path，并且禁止先量化 activation 再走普通 GEMM。

### 11.3 FP32 文本编码器的输出策略

FlashInfer 这里只输出 FP16/BF16，不能按现有 FP32 Qwen 的 `x.dtype` 直接请求 FP32 输出。

第一版建议：

- MX Linear 的 activation 量化输入规范化为 FP16/BF16；对 FP32 Qwen activation 明确转换为 BF16 再量化；
- Qwen MX GEMM 输出 BF16；
- FP32 bias/residual 按现有 torch 类型提升规则处理，非量化 attention 的代码不改；
- 实际 attention dtype 和后续 connector dtype 必须在 Qwen 阶段单独 trace；
- 该 dtype 变化属于量化实验的一部分，必须记录，不能宣称与旧 FP32 GEMM 完全等价。

DiT 的 BF16、视频 VAE 的 FP16/BF16 按模型原 dtype 输出。训练/带梯度路径不纳入第一版 fast path，继续使用明确的 dequantize 参考路径。

### 11.4 H3 各模块接入

**主 DiT**：普通 `operations.Linear` 走 MX handler；`fc2` 前的 SwiGLU 由现有模块逻辑先做。attention、RoPE 不变。

**视频 VAE**：`linear_input_act` 的可选融合先返回 `NotImplemented`，由模块层执行 RMSNorm/SwiGLU/residual。保留 Sage INT8 attention，不将“MX Linear”误等同于整个 VAE 全部 MX。

**Qwen**：350 个目标 Linear 真 GEMM，embedding 和 attention 不量化。文本编码器其他未在本次 T2V 使用的模块不自动纳入。

**音频 VAE**：完全不改。

**LoRA/patch**：第一版如果存在 `weight_function/bias_function`，沿用 dequantize 后打 patch 的路径，不在 packed MX 数据上直接做 LoRA 数学。质量/性能基线优先固定已有 API prompt，不混入另一组 LoRA 配置。

## 12. BF16 与 compressed-tensors 的统一转换器

### 12.1 两条入口，一个运行时

```text
BF16/FP16 safetensors
  → 逐层量化（明确的 quantizer 版本）
  ┐
  ├→ Comfy 原生 .safetensors + .comfy_quant → 同一个 QuantizedTensor / GEMM
  │
compressed-tensors MXFP8/MXFP4
  → 键名和 scale layout 无损转换（不重新量化）
  ┘
```

建议转换工具与模型代码分离，输入输出都是单文件；使用 `safetensors.safe_open` 逐层处理，避免同时物化完整 BF16 和量化模型。

### 12.2 BF16 → MX

推荐命令接口（未来实现，不是本轮已执行命令）：

```bash
python tools/convert_mx_checkpoint.py \
  --input model_bf16.safetensors \
  --output model_mxfp8.safetensors \
  --format mxfp8 \
  --layer-manifest h3_t2v_dense_layers.json \
  --quantizer <versioned-quantizer> \
  --device cuda
```

`--format mxfp4` 走同样的 loader/runtime，只更换量化器和存储格式。

要求：

- 优先从 BF16/FP16 源权重转换，避免 INT8/NVFP4 → float → MX 的二次量化；
- layer manifest 来自当前 H3 目标层，不盲目量化文件里的每一个二维 tensor；
- 每层只临时保留源权重、qdata 和 scale，控制 CPU/GPU 峰值；
- 非目标 key 保留原名、dtype、数据；
- 文件 metadata 记录源文件、量化器、group size、scale layout、目标层和输出 hash；
- 不从 INT8 ConvRot 权重反推“原始 BF16”。如果确需从已量化权重做实验，必须是显式选项并标注 double-quantized，不能作为首选质量基线。

### 12.3 compressed-tensors MXFP8 映射

```text
weight [N,K] float8_e4m3fn        → weight，byte 不变
weight_scale [N,K/32] uint8 E8M0 → 128x4 重排，值不变
其他 weight/bias                 → 原样保留
```

vLLM 的 `CompressedTensorsW8A8Mxfp8` 证实 activation 是运行时动态量化，因此不需要伪造静态 `input_scale`。

### 12.4 compressed-tensors MXFP4 映射

```text
weight_packed [N,K/2] uint8      → weight，只改名，nibble/byte 不变
weight_scale [N,K/32] uint8 E8M0 → 128x4 重排，值不变
```

vLLM 的 `CompressedTensorsW4A4Mxfp4` 明确没有 global scale；不得把 NVFP4 的 `weight_scale_2` 强行塞入。

### 12.5 recipe、名称和边界

- 标准 compressed-tensors 的 `config.json` 是外部文件，当前 ComfyUI 单文件 loader 不会自动发现完整目录 recipe。转换器必须显式读取 recipe 或读取 safetensors 中已嵌入的量化 metadata。
- 不能只凭 uint8/float8 dtype 猜 MXFP4/MXFP8。必须验证 quantization config、scheme、group=32、scale dtype、activation scheme 和 shape。
- 对实际支持的 compressed-tensors 版本，使用其配置解析能力或严格匹配已核实的 scheme；未知 format 报错，而不是猜测 `pack-quantized` 的含义。
- H3 没有 tensor-parallel 分片，首版不实现通用 TP shard 拼接。若源模型 `q/k/v`、`gate/up` 分开，需要显式 H3 key mapping 和分块校验，不能默认拼接。
- ComfyUI 运行时不需要 vLLM，也不需要 compressed-tensors。这些依赖最多属于离线转换环境。

第一版通过转换后的原生单文件实现“compressed-tensors 权重在 ComfyUI 加载并 dispatch”。直接加载一个含外部 `config.json` 的 Hugging Face 目录不是当前 H3 的必要条件，可后续在 `convert_old_quants` 同一入口增加内嵌 recipe 的直接适配，而不是增加另一套模型 loader。

## 13. 文件级实施清单

以下是推荐的最小责任划分，不是要求一次提交全部修改。

### Comfy-Kitchen

| 文件 | 修改 |
|---|---|
| `tensor/mxfp8.py` | 保留格式，收紧 fast-path、接入真实 CUDA backend |
| `tensor/mxfp4.py` | 新增 layout、quant/dequant、linear/mm、序列化 |
| `tensor/__init__.py` | 导出/注册 MXFP4 |
| `backends/cuda/mx_linear.py` | 新增候选接口、硬件选择、FlashInfer adapter |
| `backends/cuda/__init__.py` | 公共 wrapper、constraints、可选 backend 注册 |
| `backends/eager/quantization.py` | MXFP4 reference，MXFP8/4 custom-op fake/meta |
| `__init__.py` | 导出 `quantize_mxfp4/dequantize_mxfp4/scaled_mm_mxfp4` |

### ComfyUI

| 文件 | 修改 |
|---|---|
| `comfy/quant_ops.py` | MXFP4 import/layout 注册和 `QUANT_ALGOS` |
| `comfy/ops.py` | MX scale/形状加载、显式 full-precision 优先级、保存、FP32 activation 策略 |

### 独立工具

| 文件 | 修改 |
|---|---|
| `tools/convert_mx_checkpoint.py` | BF16 和 compressed-tensors 转 Comfy 原生单文件 |
| `h3_t2v_dense_layers.json` | 由当前已核实模型生成的目标层 manifest |

不为这个功能新增 H3 专用 workflow 节点，不让模型类 import FlashInfer，不让 kernel 层理解 ComfyUI 的模型名、workflow 或节点。

## 14. 验证设计

### 14.1 P0：依赖与真实能力预检

在任何模型转换之前，在独立验证环境锁定 FlashInfer、CuTe-DSL、CUDA 和 PyTorch 版本，直接用小矩阵验证：

- `mxfp8_quantize` 和 `mm_mxfp8`；
- `mxfp4_quantize` 和 `mm_fp4(block_size=32,use_nvfp4=False)`；
- 现有 `torch_scaled_mm_v2` 在 B200 上是否产生真实 MXFP8 kernel（只核实，不预先承诺）；
- B200 上 CUTLASS/CuTe-DSL 各自可用性；
- FP16/BF16 输出、非 128 倍数 M、H3 的所有 N/K；
- 当前 device 的 stream 是否正确。

不把另一个 vLLM venv 的 `.so` 用 PYTHONPATH 强行混入 ComfyUI venv。版本未锁定前不写死安装命令，也不宣称本机 FlashInfer 已经可用。

### 14.2 格式单元验证

1. 全部 16 个 E2M1 code 的解码，含负数、0.5、1、6。
2. packed byte 的低/高 nibble 顺序。
3. E8M0 exponent、全零 block、极小值、极大值和饱和。
4. `[N,K/32] ↔ 128x4 ↔ [N,K/32]` 索引可逆，padding 不参与有效值。
5. compressed-tensors 转换前后 qdata byte 完全相同，逻辑 scale 完全相同。
6. `.comfy_quant` save → load 往返后 `orig_shape/storage_shape/layout` 完全一致。
7. 非 32 对齐的一般小矩阵验证 padding/crop；H3 已对齐形状验证无多余 K padding。

### 14.3 Linear 正确性

对每种格式、backend 和每类 N/K：

- 用同一组 qdata/scale，与 FP32 dequant-reference 比较；
- 比较 M=1、小 M、H3 实际 M、带/不带 bias；
- 检查 N/K 形状、输出 dtype 和非 contiguous 输入边界；
- 检查 FP32 Qwen activation → BF16 MX GEMM 的明确策略；
- 对同一 FlashInfer backend/tactic 做 vLLM wrapper 与 Comfy adapter 的同输入对照；
- 测试失败候选的拒绝原因，以及显式 eager reference 是否可见。

量化格式转换可以要求 bitwise 一致；硬件 GEMM 只要求在预先冻结的数值容差内，不把累加顺序差异误判成格式错误。容差应在 P0 用参考实现测量后固定，不能看完结果再放宽。

### 14.4 真实调用栈和 CUPTI

沿用现有 `run_h3_cupti.py`、NVTX profiler 和 SQLite 分析方式，新增 MX 采集而不覆盖原 trace。至少验证：

- Python：`MixedPrecisionOps.Linear → QuantizedTensor → scaled_mm_mxfp* → MXLinearSelector → FlashInfer`；
- MXFP8 的 activation quantization 和 block-scaled GEMM；
- MXFP4 的 activation quantization 和 block-scaled GEMM；
- GEMM 前没有逐层 `dequantize_mxfp*`；
- 没有意外落到普通 BF16/FP32 SGEMM；
- 主 DiT FlashAttention、视频 VAE Sage INT8 attention、音频 FP32 conv 保持原有类型；
- 实际 kernel 名称、grid、register、shared memory、耗时来自 trace，不预先编造。

对当前 prompt 的目标计数基线：

| 模块 | 目标真 MX Linear GEMM 数 | 不应被误算的其他 kernel |
|---|---:|---|
| 主 DiT，20 步 | 4000 | 每步 BF16 FlashAttention 和其他 float SGEMM |
| Qwen | 350 | FP32 attention、embedding 和其他模块 |
| 视频 VAE ViT | 6048 | 1512 次 Sage INT8 attention、RoPE、卷积等 |

这些数字用于检查覆盖完整性；MX 替换后其他 elementwise kernel 数可以改变，不能强行要求总 kernel 数仍是 95,226。

### 14.5 分模块质量验证

避免只看最终视频而无法定位误差来源：

1. **Qwen**：固定 token/tags，比较旧 FP32/NVFP4 路径与 MX 路径的 conditioning；检查 shape、dtype、NaN/Inf。
2. **DiT 单步**：固定同一 conditioning、latent、sigma 和随机状态，比较每个目标 block 和最终 velocity。
3. **视频 VAE**：固定同一 latent，比较 ViT 中间输出、解码帧和 residual 行为。
4. **端到端**：相同 prompt、seed、分辨率、frames、采样参数和 LoRA 配置，分别跑 BF16（若源权重可用）、当前 INT8/NVFP4、MXFP8、MXFP4。
5. 视频比较解码后的帧和 PCM 音频，不直接比较 MP4 container byte；记录尺寸、帧数、fps、音频采样率和可解码性。
6. 当前 Qwen 是“NVFP4 存储 + FP32 GEMM”，与 MXFP4 W4A4 的比较必须明确标注。若增加真实 NVFP4 GEMM 对照，应作为单独实验配置。

### 14.6 性能、显存和冷启动

分别报告：

- 权重转换时间和峰值 CPU/GPU 内存，不计入 steady-state 推理；
- 首次 import/JIT/autotune 的冷启动；
- 每个代表 M/N/K 的 warmup 后 quantize、GEMM、bias/residual 时间；
- Qwen、DiT 20 steps、视频 VAE、完整 workflow wall time；
- quantized resident weights、scale、临时 activation、backend workspace、峰值 allocated/reserved VRAM；
- MXFP4/MXFP8 与当前 INT8/NVFP4 的逐层 payload 和真实显存差异。

不能只凭 FP4 bit 数宣称端到端加速。量化 activation、shape autotune、Python 调度、额外 padding 和未融合 RMSNorm/SwiGLU 都可能抵消 GEMM 收益。

## 15. 分阶段实施顺序

### P0：独立 FlashInfer smoke

锁定版本，跑通 B200 小矩阵和 H3 所有形状，确定 quantizer、scale layout、backend 和容差。若这里失败，先修依赖/接口，不开始转大模型。

### P1：MXFP8 DiT 垂直切片

补齐 CUDA `scaled_mm_mxfp8`、native converter 和真 kernel 验证；只转换主 DiT 的 200 个目标 Linear。先验证单步，再验证 20 步。

### P2：MXFP4 DiT 垂直切片

新增 MXFP4 layout/API/loading，复用同一 backend 框架；比较 MXFP8/MXFP4 与当前 INT8 的单步和完整采样结果。

### P3：视频 VAE

覆盖 144 个 Linear，保留 Sage INT8 attention；先走显式 RMSNorm/SwiGLU/residual，验证 6048 次 GEMM 和输出视频。

### P4：Qwen

处理 full-precision 覆盖和 BF16 输出策略，覆盖 350 个 Linear；检查 conditioning、后续 attention dtype 和显存改善。

### P5：compressed-tensors 无损导入

验证 MXFP8/MXFP4 qdata byte 不变、scale 逻辑值不变，并与同权重直接调 vLLM/FlashInfer 的结果对照。

### P6：端到端对照和可选优化

完成四组质量、CUPTI、显存、冷启动报告。只有在 profile 证明值得时，再评估 fused RMSNorm+quant、SwiGLU+quant 和 bias/residual epilogue。

Marlin、B12x、其他 GPU、训练、MoE、CUDA Graph 全图化和通用 compressed-tensors 目录 loader 都不阻塞当前目标。

## 16. 主要风险与防止误判的措施

| 风险 | 后果 | 设计措施 |
|---|---|---|
| 有 Python API 就以为有真 kernel | 实际一直 dequant + 普通 GEMM | fast-path 失败可见，强制 CUPTI 验证 |
| MXFP4 与 NVFP4 混淆 | group/global scale 错误，结果错误 | 独立 API、独立 metadata、严格校验 |
| scale 被普通 `.t()` 转置 | 分组与权重不再对应 | 保留 `[N,K/32]` 语义，128x4 显式转换 |
| 不同 quantizer 舍入不同 | 同格式权重却产生不同质量 | quantizer 版本化，逐 byte 验证 |
| 文本编码器父级 full precision | 新 MXFP4 仍走 FP32 | 文件显式 false 覆盖父级默认，旧默认不变 |
| FP32 输出不受支持 | Qwen kernel 调用失败或偷偷 fallback | 明确 BF16 输出策略和独立验证 |
| 从 INT8/NVFP4 再量化 | 二次量化损失，无法公平比较 | 首选 BF16 源，特殊转换明确标记 |
| venv 混用二进制依赖 | ABI、加载器或 CUDA 错误 | 独立锁定环境，不跨 venv PYTHONPATH |
| runtime 异常被捕获后换路 | kernel bug 被掩盖 | 候选探测与执行错误分离 |
| 每次 forward 重排权重 | 性能差、显存峰值大 | 文件/加载阶段一次转换，无全局 tensor cache |
| 直接量化整个文件 | 误量化 embedding/未使用模块 | H3 layer manifest 和形状校验 |
| autotune 混入性能结果 | 冷启动和稳态数据失真 | 冷/热分别测量，trace 分开保存 |
| packed 权重做 LoRA | 数学错误或重复量化 | 初版保留 dequant patch 路径 |

## 17. 审阅时需要确认的决策

本 proposal 推荐的默认值是：

1. **先 FlashInfer adapter，后评估是否需要 comfy-kitchen `_C` 原生实现。**
2. **第一版只保证 B200 的 dense W8A8/W4A4；其他后端采用同一接口逐步扩展。**
3. **原生单文件优先，compressed-tensors 离线无损转换，不引入运行时 vLLM 依赖。**
4. **正确性优先，INT8 的融合不在 MX 第一版强行复刻。**
5. **新 MX 文件显式请求 fast path，旧 NVFP4/INT8 默认行为不变。**
6. **Qwen 的 FP32 → BF16 MX Linear 输出策略是显式实验选择，需要实测，而非暗中改变。**
7. **任何“跑通”结论必须同时有输出、调用栈和 CUPTI 证据。**

## 18. 参考来源

### 官方格式与 API

- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：E8M0、32 元素 block、FP8/FP4 转换和 RNE。
- [FlashInfer `mm_mxfp8`](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_mxfp8.html)：输入、1D 128x4 scale、输出和 backend 约束。
- [FlashInfer `mm_fp4`](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)：MXFP4 的 `block_size=32,use_nvfp4=False` 和 scale 要求。
- [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)。

### vLLM 源码

- [dense kernel registry/selection](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py)
- [MXFP8 FlashInfer/Marlin kernels](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py)
- [MXFP4 FlashInfer kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mxfp4/flashinfer.py)
- [compressed-tensors W8A8 MXFP8](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py)
- [compressed-tensors W4A4 MXFP4](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py)
- [MXFP8 utilities](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py)
- [MXFP4 utilities](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py)

本地阅读以 `/home/kaokaolv/.venv-vllmomni029/lib/python3.12/site-packages/vllm` 和 `.../flashinfer` 的实际文件为准。官方页面可能对应不同版本，实施时以锁定的依赖版本重新核对，不把网页参数列表无条件套到本机包。

### ComfyUI / H3 证据

- `ComfyUI/comfy/quant_ops.py`、`comfy/ops.py`、`comfy/utils.py`、`comfy/sd1_clip.py`。
- `ComfyUI/comfy/ldm/minimax/model.py`、`vae.py`。
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)。
- 本机 `comfy_kitchen/tensor/mxfp8.py`、`nvfp4.py`、`backends/cuda/__init__.py`、`backends/eager/quantization.py`、`float_utils.py`。
- 本机 `logs/cupti_h3_t2v/h3_t2v_cold.sqlite` 及已有分析脚本。

---

**本 proposal 的完成不代表 MXFP8/MXFP4 已在 H3 上运行。** 下一步应先审阅格式、dtype 和 backend 策略，再从 P0 的小矩阵实测开始。
