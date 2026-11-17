# run_book_quant.md — 量化任务记录

> 本文件是量化相关任务的独立记录（自 2026-11-17 起），此前的模型结构/精度分析记录在 `run_book.md`。
> 每次量化任务记录：目标、所用代码版本、目标模块、脚本、运行命令、结果与偏差。

---

## 1. 任务：用 auto-round 把 MiniMax-H3 pruned DiT 量化到 MXFP8

### 1.1 目标

- Base：`Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`
  （本地默认路径 `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`，40.23 GB / 20.111 B / 532 tensor）。
- 参考量化文件：`ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
  （20.970 GB；200 个 `comfy_quant` 模块）。
- 工具：`/home/kaokaolv/comfyui_h3/auto-round`（repo 版本 **0.17.0**）。
- 参考脚本：`/home/kaokaolv/my_commit/t_mxfp4_llmc.py`
  （`AutoRound(model, scheme="MXFP4", iters=0).quantize_and_save(format="llm_compressor")`）。
- 目标格式：MXFP8（E4M3 权重 + 每 32 权重一个 E8M0 block scale）。

### 1.2 目标模块（与 int8 文件完全一致）

`minimax_h3_fl2va_pruned_int8_convrot.safetensors` 的 200 个量化模块：

```
blocks.{0..49}.attn.qkv_proj
blocks.{0..49}.attn.out_proj
blocks.{0..49}.mlp.fc1
blocks.{0..49}.mlp.fc2
```

其余模块保持原精度（token_refiner BF16、AdaLN F16、patch/final head F32、norm BF16 等）。
脚本也可从 int8 文件的 `comfy_quant` 键自动读取目标集合，避免手写列表漂移。

### 1.3 auto-round 代码阅读结论（0.17.0）

- MXFP8 属于 auto-round 的 model-free 方案：`scheme="MXFP8"`，`group_size=32`，输出 compressed-tensors 的 `mxfp8-quantized`。
- model-free 的逐层入口：`auto_round/utils/model_free_utils.py::_quantize_weight_mxfp`
  - 内部调用 `auto_round.data_type.utils.get_quant_func(data_type, bits=8, sym=True, iters=0)`；
  - 默认 MXFP 走 `opt_rtn_mx_fp8`（优化 RTN，`disable_opt_rtn=True` 可退回普通 RTN）；
  - 通过 `auto_round/export/formats/autoround/qlinear_fp.py::QuantLinear.pack` 生成
    `weight`（`float8_e4m3fn`）和 `weight_scale`（`uint8`，形状 `[N, K/32]`，row-major）。
- E8M0 scale 编码：`QuantLinear.pack` 中 `final_scale = (shared_exp + 127).clamp(0, 254).to(torch.uint8)`。
- auto-round 的 model-free 输出是 compressed-tensors 目录格式（`weight` + `weight_scale` + `quantization_config`），
  与 ComfyUI 的单文件 `comfy_quant` 格式不同，需要二次重排。

### 1.4 ComfyUI mxfp8 格式要求

ComfyUI `comfy/quant_ops.py` 的 `mxfp8`：

- `weight`：`float8_e4m3fn`，`[N, K]`；
- `weight_scale`：E8M0（文件里可存 `uint8`，`pop_scale(..., torch.float8_e8m0fnu)` 会 view）；
- block size 32；
- scale 必须是 **cuBLAS blocked/swizzled 布局**（`comfy/float.py` 的 `to_blocked(..., flatten=False)`，
  与 `comfy_kitchen.float_utils.to_blocked` 相同），不是 auto-round 的 row-major `[N, K/32]`；
- 每模块 `comfy_quant`：`{"format": "mxfp8"}`；若要走反量化 + 普通 F.linear，可加 `"full_precision_matrix_mult": true`。
- `TensorCoreMXFP8Layout.MIN_SM_VERSION = (10, 0)`：真正 MXFP8 tensor-core 路径需要 Blackwell（SM≥10）。

### 1.5 脚本

新增：[`quantize_h3_mxfp8_autoround.py`](quantize_h3_mxfp8_autoround.py)

处理流程：

1. 读 base 的 safetensors header；从 int8 文件读 200 个 `comfy_quant` 模块（或默认 pattern）；
2. 逐 tensor 读取，只对目标 `.weight` 调用 auto-round 的 `_quantize_weight_mxfp`；
3. 把 auto-round 的 row-major `uint8` scale 过一遍 `to_blocked(..., flatten=False)`；
4. 写 `weight`（F8_E4M3）、`weight_scale`（U8）、`comfy_quant`（U8 JSON）；
5. 非目标 tensor 原样拷贝；
6. 用 `safetensors.torch.save_file` 写单个文件。

脚本内嵌了 `to_blocked`（与 `comfy_kitchen.float_utils.to_blocked` 等价），因此不依赖 ComfyUI 的 import。

### 1.6 环境

- ComfyUI `.venv`：有 `torch`、`safetensors`、`comfy_kitchen`，但**没有** `auto_round`。
- `~/.venv-aqa-gpu`：已安装 `auto_round 0.15.0`（dist-info），但 repo 是 **0.17.0**；建议用 `PYTHONPATH` 指向 repo，或按 auto-round AGENTS 的方式 editable 安装：
  ```bash
  PYTHONPATH=/home/kaokaolv/comfyui_h3/auto-round \
  /home/kaokaolv/.venv-aqa-gpu/bin/python quantize_h3_mxfp8_autoround.py ...
  ```
  或
  ```bash
  /home/kaokaolv/.venv-aqa-gpu/bin/pip install --no-build-isolation -e /home/kaokaolv/comfyui_h3/auto-round
  ```
- 注意：auto-round 的 import 较慢（torch/transformers 初始化），建议用 `-u` 或后台运行。

### 1.7 运行命令

```bash
cd /home/kaokaolv/comfyui_h3

# 1) 先 dry-run，确认 200 个目标模块（只需 torch，不需要 auto_round）
.venv/bin/python quantize_h3_mxfp8_autoround.py \
  --src ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
  --reference-int8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --dry-run

# 2) 正式量化（CUDA + auto-round 默认 opt_rtn）
AUTO_ROUND_LIGHT=1 AUTO_ROUND_REPO=/home/kaokaolv/comfyui_h3/auto-round \
PYTHONPATH=/home/kaokaolv/comfyui_h3/auto-round \
/home/kaokaolv/.venv-aqa-gpu/bin/python -u quantize_h3_mxfp8_autoround.py \
  --src ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
  --dst ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors \
  --reference-int8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --device cuda

# 3) 如果目标 GPU 不是 Blackwell，加载时不想走 MXFP8 tensor-core：
#    加 --full-precision-mm，comfy_quant 会写 "full_precision_matrix_mult": true
```

### 1.8 状态

- [x] 读完 auto-round 0.17.0 的 MXFP8 路径（`model_free.py`、`model_free_utils.py`、`data_type/mxfp.py`、`qlinear_fp.py`）。
- [x] 读完 ComfyUI 的 mxfp8 格式（`quant_ops.py`、`float.py`、`kitchen/tensor/mxfp8.py`）。
- [x] 脚本 `quantize_h3_mxfp8_autoround.py` 完成，`py_compile` 通过。
- [x] dry-run（用 int8 header 代替 base）：从 `comfy_quant` 读到 **200** 个目标模块，block 0–49。
- [x] 已下载 `minimax_h3_fl2va_pruned_bf16.safetensors`（40.23 GB，sha256 匹配）。
- [x] auto-round 0.17.0 已用 `AUTO_ROUND_LIGHT=1` 路径跑通（import ~40 s，单层 smoke test 通过）。
- [x] 已用 `TensorCoreMXFP8Layout` 反量化 3 层验证，rel_rmse 2.66%。

### 1.9 原计划的延后原因（已执行）

最初因 base 未下载、机器上有其他 auto_round 任务而延后；用户确认后已按 §2 执行完成（下载 + 200/200 量化 + 校验）。

### 1.10 待确认

1. 是否要同时产出 auto-round 原生的 compressed-tensors 目录（`format="llm_compressor"`），还是只产出 ComfyUI 单文件？
2. 目标 GPU 已确认是 B200（SM100），本次使用默认 MXFP8 tensor-core 路径，未加 `--full-precision-mm`。
3. 本次使用 auto-round 默认 `opt_rtn`；200 层量化 281 s。若只做格式验证可加 `--disable-opt-rtn`。

---

## 2. 执行记录：下载 + MXFP8 量化

### 2.1 下载 pruned BF16

```bash
cd /home/kaokaolv/comfyui_h3
HF_HOME=/home/kaokaolv/comfyui_h3/.cache/huggingface \
.venv/bin/hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
  --local-dir ComfyUI/models
```

结果：

- 文件：`ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`
- 大小：`40,225,724,176` B（与 HF API 一致）
- hf 缓存 metadata 的 sha256：`a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4`
  （与 HF LFS oid 一致，revision `4b53a35191437de05370da548e59aeabe2a5a09a`）
- 日志：`logs/download_pruned_bf16.log`

### 2.2 量化环境

- ComfyUI `.venv` 没有 auto_round；`~/.venv-aqa-gpu` 的 auto_round 是 0.15.0（没有 model-free / mxfp8 路径）。
- 使用 repo 0.17.0 + `PYTHONPATH`。但 `import auto_round` 会执行 `auto_round/__init__.py`（AutoRound/transformers 等），在本机共享 FS 上超过 240 s 仍未完成。
- 解决办法：脚本支持 `AUTO_ROUND_LIGHT=1` + `AUTO_ROUND_REPO=...`，用 fake package 绕过 `__init__.py`，只导入
  `auto_round.data_type.mxfp`（`quant_mx` / `quant_mx_opt_rtn`）。实测 import 约 40 s，CUDA 可用。
- 量化在 `cuda:3`（B200，当时 GPU 7 被其他用户占满，GPU 3 空闲）。

### 2.3 量化命令

```bash
cd /home/kaokaolv/comfyui_h3
AUTO_ROUND_LIGHT=1 AUTO_ROUND_REPO=/home/kaokaolv/comfyui_h3/auto-round \
AUTO_ROUND_LIGHT=1 AUTO_ROUND_REPO=/home/kaokaolv/comfyui_h3/auto-round \
PYTHONPATH=/home/kaokaolv/comfyui_h3/auto-round \
/home/kaokaolv/.venv-aqa-gpu/bin/python -u quantize_h3_mxfp8_autoround.py \
  --src ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
  --dst ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors \
  --reference-int8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --device cuda:3 --overwrite
```

日志：`logs/quant_h3_mxfp8.log`。

### 2.4 进度

- 200 个目标模块已从 int8 header 读出；block 0–49。
- 单层耗时（GPU）：qkv 约 2.2 s，fc1 约 2.9 s，out/fc2 约 0.7–1.5 s；第 1 层含 import 约 29 s。
- 预计 200 层 ~7–8 min，最后 `save_file` 写 ~21 GB。

### 2.5 第一次运行中止：safetensors 共享内存

第一次跑到 57/200 时主动中止。原因：脚本把所有模块的 `comfy_quant` 指向同一个 `torch.uint8` 张量，
`safetensors.torch.save_file` 会拒绝共享内存（实测报错 `Some tensors share memory`）。
修正：每个模块 `.clone()` 一份；清理未完成的 dst 后重跑。
（非目标 tensor 由 `safe_open.get_tensor` 各自返回，不共享。）

### 2.6 第二次运行

- 重跑 80/200 用时 62 s；单层 0.2–2.9 s。
- 等待 200 层完成 + `save_file` 写 ~21 GB。
- 目标文件：`ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors`。

### 2.7 输出预期（用于校验）

- tensor 总数：532（base）+ 200（weight_scale）+ 200（comfy_quant）= **932**。
- dtype 预期：F8_E4M3 ×200、U8 ×400（200 scale + 200 blob）、BF16 ×220、F16 ×102、F32 ×10。
- 200 个 `comfy_quant` 模块，内容 `{"format":"mxfp8"}`。
- 校验脚本：`validate_h3_mxfp8.py`（用 `comfy_kitchen.TensorCoreMXFP8Layout` 反量化 3 个代表层，和 BF16 原权重比 RMSE）。

### 2.8 第二次运行结果（截至保存阶段）

- 200/200 完成，累计 281 s（不含 import 29 s）；单层 0.2–2.9 s。
- 已进入 `save_file`：932 tensors，目标 ~21.5 GB。
- 保存完成后运行：
  ```bash
  .venv/bin/python validate_h3_mxfp8.py \
    --mxfp8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors \
    --base ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
  ```

### 2.9 完成结果与校验

输出文件：

- `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors`
- 大小：**21,560,297,704 B = 21.560 GB**
- 200 个 MXFP8 模块；保存阶段耗时约 220 s（200/200 量化 281 s，总任务 501 s）。

header 统计（`validate_h3_mxfp8.py`）：

```
tensors: 932
dtypes: {'F32': 10, 'BF16': 220, 'F16': 102, 'F8_E4M3': 200, 'U8': 400}
comfy_quant modules: 200
```

与预期完全一致：200 个 F8_E4M3 权重、200 个 U8 `weight_scale`、200 个 U8 `comfy_quant`；
非目标 tensor 保持 BF16 220 / F16 102 / F32 10。

用 ComfyUI 的 `comfy_kitchen.TensorCoreMXFP8Layout` 反量化 3 个代表层，并与 BF16 base 对比：

| 模块 | shape | scale shape | rel_rmse |
|---|---|---|---|
| `blocks.0.attn.qkv_proj` | 21504×5376 | 21504×168 | **2.66%** |
| `blocks.0.mlp.fc1` | 28672×5376 | 28672×168 | **2.66%** |
| `blocks.49.mlp.fc2` | 5376×14336 | 5376×448 | **2.66%** |

这说明 blocked E8M0 scale 布局、F8_E4M3 权重、`comfy_quant` 元数据都被 ComfyUI 的布局正确识别，
反量化误差在 MXFP8 的预期范围内（权重 ~2.7% RMSE）。

未做的验证：尚未用该文件在 ComfyUI 里跑完整 T2V 生成（需要同时加载 TE/两个 VAE 和 workflow）；
本次只验证了 checkpoint 格式与单层反量化。MXFP8 tensor-core 执行需要 SM≥10（B200 可用）。

---

## 3. int8_convrot 反量化误差（对照 BF16 base）

### 3.1 方法

新增 [`validate_h3_int8_convrot.py`](validate_h3_int8_convrot.py)：

- 读 `minimax_h3_fl2va_pruned_int8_convrot.safetensors` 的 `weight`（I8）和 `weight_scale`（F32 `[N,1]`）；
- 用 ComfyUI 自己的 `comfy_kitchen.TensorWiseINT8Layout`：
  ```python
  params = TensorWiseINT8Layout.Params(
      scale=s.cuda(), orig_dtype=torch.bfloat16, orig_shape=tuple(ref.shape),
      is_weight=True, convrot=True, convrot_groupsize=256)
  deq = QuantizedTensor(w.cuda(), "TensorWiseINT8Layout", params).dequantize()
  ```
  该路径调用 `dequantize_int8_convrot_weight_dtype`，会做 256 组 Hadamard 反旋转 + per-output-channel scale 反量化；
- 与 `minimax_h3_fl2va_pruned_bf16.safetensors` 的同名权重逐元素对比，计算 RMSE / rel_RMSE / cosine。

注意：cosine 必须用 float64 计算。115M 元素的 float32 dot/norm 会因累加误差给出 >1 的“cosine”（实测 qkv 的 float32 比值 1.0415，float64 为 0.99996）。

### 3.2 int8 结果

header：932 tensor，`F32 210 / BF16 220 / F16 102 / I8 200 / U8 200`；200 个 `comfy_quant` 模块
（`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`）。

| 模块 | shape | rel_RMSE | cosine (float64) |
|---|---|---|---|
| `blocks.0.attn.qkv_proj` | 21504×5376 | **0.8985%** | 0.999960 |
| `blocks.0.attn.out_proj` | 5376×7168 | **1.2113%** | 0.999927 |
| `blocks.0.mlp.fc1` | 28672×5376 | **0.8973%** | 0.999960 |
| `blocks.49.mlp.fc2` | 5376×14336 | **0.9640%** | 0.999954 |

### 3.3 与 MXFP8 的对比（同一批模块、同一 BF16 base）

| 模块 | int8_convrot rel_RMSE | mxfp8 rel_RMSE（auto-round opt_rtn） |
|---|---|---|
| `blocks.0.attn.qkv_proj` | 0.8985% | 2.6606% |
| `blocks.0.attn.out_proj` | 1.2113% | 2.6611% |
| `blocks.0.mlp.fc1` | 0.8973% | 2.6608% |
| `blocks.49.mlp.fc2` | 0.9640% | 2.6620% |

结论：在这几个主投影上，int8_convrot 的权重 RMSE 约为 MXFP8 的 **1/2.2–1/3**。
原因：int8 每输出通道一个 F32 scale（加 256 组旋转抑制 outlier），而 MXFP8 是每 32 个权重一个
E8M0（2 的幂）scale，E4M3 只有 3 位 mantissa；MXFP8 的 block 更细，但单值相对精度更低。

脚本：
- `validate_h3_int8_convrot.py`（本次新增）
- `validate_h3_mxfp8.py`（上一节）
