# Comfy-Org/MiniMax-H3 五个单文件模型：结构、精度与参数量分析

> 分析对象：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) 中的 5 个文件
> 分析日期：2026-11-17
> 数据来源：HF model card、上游 `MiniMaxAI/MiniMax-H3` 的 `config.json`、本地 safetensors 文件头（逐 tensor dtype/shape/data_offsets）、ComfyUI 源码 `comfy/ldm/minimax/*`、`comfy/quant_ops.py`、`comfy_kitchen` 源码。
> 所有"参数量"均为**逻辑参数量**（4 bit 权重按每字节 2 个参数还原），不是文件字节数；量化 side-car（scale、comfy_quant、AWQ pre_quant_scale、LoRA alpha）单独统计，不计入参数量。
>
> 修订记录：2026-11-17 初版；同日依据 `cupti_h3_t2v_kernel_report.md` 与 `comfyui_h3_t2v_kernel_analysis.md` 的 CUPTI 实测，修正 TE 计算 dtype 为 FP32、补充 Video VAE INT8 attention、新增 §11（重新量化的 base 来源）与 §12（交叉校验）。

---

## 0. 速览

| # | 文件 | 角色 | 文件大小 | 逻辑参数量 | 存储精度构成（按字节） | 有效 bit/参数 |
|---|---|---|---|---|---|---|
| 1 | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 视频+音频 DiT | 20.970 GB | **20.111 B** | int8 91.88%，bf16 7.62%，fp16 0.42%，fp32 0.08% | 8.34 |
| 2 | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Qwen3-VL-32B 文本/视觉编码器 | 15.687 GB | **25.753 B** | nvfp4 77.71%，fp8 9.71%，bf16 7.62%，int8 4.96% | 4.87 |
| 3 | `vae/minimax_h3_video_vae_int8_convrot.safetensors` | 视频 VAE | 2.811 GB | **2.604 B** | int8 85.94%，fp16 12.83%，fp32 1.22% | 8.64 |
| 4 | `vae/minimax_h3_audio_vae_fp32.safetensors` | 音频 VAE | 0.605 GB | **0.151 B** | fp32 100% | 32.0 |
| 5 | `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8 步蒸馏 LoRA | 1.956 GB | **0.978 B** | bf16 100% | 16.0 |

关键结论（详细证据见后文）：

1. **DiT 的 "pruned" 不是删层**。50 层 block 一层不少，删掉的是每层 AdaLN 的全宽调制矩阵：`[96768, 2688]`（BF16，2.60 亿参数/层）被替换成 `[96768, 8]`（FP16，0.87 M/层）+ 共享的时间嵌入曲线表 `adaln_t_table [1025, 8]`，同时删除 `time_embedder`。这一项共减少 **13.01 B 参数（39.3%）**，与 66.28 GB → 40.23 GB（bf16 版本）的文件差完全对得上。
2. **TE 的 nvfp4 是"存储量化"**：`comfy_quant` 里带 `"full_precision_matrix_mult": true`，权重按 NVFP4 存盘，反量化成计算 dtype 再做普通 F.linear，因此不依赖 Blackwell FP4 tensor core。计算 dtype 不是固定 BF16：本次真实 T2V 的 CUPTI 实测是 **FP32**（350 次 `dequantize_nvfp4_kernel<...,float>` + 350 次 `cutlass3x_sm100_simt_sgemm_f32`），见 §12。
3. **int8_convrot 是 W8A8**：权重 int8 + 每输出通道 F32 scale + 按 256 分组的 Hadamard 旋转（convrot）；运行时激活值在线旋转并动态 per-row 量化后走 INT8 GEMM。
4. 三个 DiT/VAE 文件的量化覆盖面不同：DiT 只量化 block 内的 4 个线性层（token_refiner、patch proj、final head 保持 BF16/FP32）；视频 VAE 只量化 ViT decoder 的 36×4 个线性层（encoder 保持 FP16）；音频 VAE 全 FP32。

---

## 1. 证据链与核验方法

### 1.1 Model card 中的相关事实

[Comfy-Org/MiniMax-H3 README](https://huggingface.co/Comfy-Org/MiniMax-H3)：

- 这是 ComfyUI 的单文件重打包仓库，原始模型为 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)、[alibaba-pai/MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)、[Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental)。
- 文本编码器 nvfp4 版本转换自 [cybermotaz/Qwen3-VL-32B-Instruct-NVFP4](https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4)（NVFP4 + AWQ_FULL 校准）。
- "This `nvfp4` text encoder does not require Blackwell GPU to use."
- "For diffusion models prefer `int8_convrot` if you are able to use pytorch with cu130. `fp8_scaled` should only be used if you cannot use `int8_convrot`."
- 目录约定：`diffusion_models/`、`text_encoders/`、`vae/`、`loras/`、`model_patches/`、`embeddings/`。

### 1.2 上游 config（FL2VA）

`MiniMaxAI/MiniMax-H3/FL2VA/transformer/config.json`：

```json
{
  "_class_name": "MiniMaxH3DiTModel",
  "hidden_size": 5376, "num_layers": 50, "token_refiner_num_layers": 2,
  "num_attention_heads": 56, "attention_head_dim": 128, "ffn_hidden_size": 14336,
  "latents_dim": 24, "audio_latents_dim": 32, "patch_size": [1, 2, 2],
  "text_dim": 5120, "timestep_input_dim": 256,
  "time_embed_hidden_size": 5376, "time_embed_dim": 2688,
  "adaln_out_features": 96768, "final_adaln_out_features": 10752,
  "rope_inv_freq_len": 16, "norm_eps": 1e-05, "qk_norm_eps": 1e-05
}
```

`FL2VA/text_encoder/config.json`：`Qwen3VLForConditionalGeneration`，text_config 为 hidden 5120 / 64 层 / 64 Q heads / 8 KV heads / head_dim 128 / intermediate 25600 / vocab 151936 / RoPE θ=5e6、mRoPE sections `[24,20,20]`；vision_config 为 27 层、hidden 1152、16 heads、intermediate 4304、patch 16、spatial_merge 2。

`FL2VA/video_vae/config.json` + `video_vae/source/config.json`：`AutoencoderKLLegacy`，ch=128、ch_mult `[1,2,2,4,4,8]`、z_channels 24、space_down `[2,2,2,2,1,1]`（16×）、time_down `[1,2,2,1,1,1]`（4×）、`use_vit_decoder: true`，ViT decoder 36 层 / 32 heads / dim_head 64 / rope_theta 100。

`FL2VA/audio_vae/config.json` + `FL2VA/audio_vae/config.yaml`：32 kHz、32 latent channels、输出 2 声道、encoder_rates `[2,4,4,5,5]`、decoder_rates `[5,5,2,2,2,2,2]`、encoder_dim 64、latent_dim 2048、decoder_dim 1024、BigVGAN。

### 1.3 直接从 safetensors 读取

脚本 [`analyze_h3_safetensors.py`](analyze_h3_safetensors.py) 只读 safetensors 头（8 字节长度 + JSON header）和 `comfy_quant` 小 blob，不加载权重张量。得到的 tensor 数：

| 文件 | tensor 数 | header 字节 |
|---|---|---|
| DiT int8 | 932 | 95,416 |
| TE nvfp4 | 2,054 | 231,400 |
| video VAE int8 | 850 | 99,512 |
| audio VAE fp32 | 917 | 105,520 |
| LoRA bf16 | 624 | 73,632 |

同时抽样读取了若干张量做数值合理性检查（INT8 值域 −127..127、per-channel scale 量级、NVFP4 block scale 为 FP8 且与 F32 全局 scale 相乘、LoRA alpha 实际存储值等），见 §8。

### 1.4 本地文件与 HF 文件的一致性

本地 5 个文件字节数与 HF API 返回的 `size` **完全一致**；sha256 在 `setup_comfyui_h3.sh` 下载阶段已按 HF LFS 清单校验通过（见 §8.2）。

---

## 2. Diffusion model：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`

### 2.1 基本参数

| 项目 | 值 |
|---|---|
| 文件大小 | 20,970,379,616 B = 20.970 GB |
| tensor 数 | 932 |
| 逻辑参数量 | 20,111,438,744（20.111 B） |
| `__metadata__` | **无**（该文件不带任何元数据） |
| 量化格式 | 200 × `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` |

### 2.2 模型结构

这是一个**视频+音频联合 DiT**（FL2VA：首尾帧 → 视频+音频），把 text / 视觉条件 / audio / video 拼成一条 packed 序列做全注意力：

```
[ text (Qwen3-VL hidden 5120 → cond_proj 5376 → token_refiner 2 blocks)
| cond rows (首尾帧 latent, patchify 1×2×2)
| audio rows (32ch × stereo 2, 40 Hz)
| video rows (24ch, patch 1×2×2) ]
```

| 组件 | 结构 | 参数量 |
|---|---|---|
| `video_patch_proj` | Linear 96 → 5376（96 = 24×1×2×2），FP32 | 521,472 |
| `audio_patch_proj` | Linear 32 → 5376，FP32 | 177,408 |
| `condition_proj` | Linear 5120 → 5376，BF16 | 27,530,496 |
| `token_refiner` | 2 × RefinerBlock（无 AdaLN），BF16 | 770,730,752 |
| `blocks` | 50 × DiTBlock | 19,311,680,000 |
| `final_layer` | AdaLN + video head 5376→96 + audio head 5376→32 | 790,400 |
| `adaln_t_table` | `[1025, 8]` FP32 时间嵌入曲线基 | 8,200 |
| `rope.inv_freq` | `[16]` FP32（3 轴 × 16，旋转 96/128 维） | 16 |

单个 DiTBlock（hidden 5376，56 heads × 128，SwiGLU ffn 14336）：

| 子模块 | shape | 存储精度 | 每 block 参数 |
|---|---|---|---|
| `attn.qkv_proj` | 21504 × 5376（3×7168） | **int8** | 115,605,504 |
| `attn.out_proj` | 5376 × 7168 | **int8** | 38,535,168 |
| `mlp.fc1` | 28672 × 5376（2×14336，SwiGLU） | **int8** | 154,140,672 |
| `mlp.fc2` | 5376 × 14336 | **int8** | 77,070,336 |
| `adaln_proj.linear` | 96768 × 8 + 96768 bias（6 组调制 × 3 模态） | **fp16** | 870,912 |
| `norm1`, `norm2` | 5376 | bf16 | 10,752 |
| `attn.q_norm`, `attn.k_norm` | 128 | bf16 | 256 |

每 block 合计 386,233,600，×50 = 19.312 B（96.02%）。

> 运行时校验（CUPTI）：本次 T2V 无首尾帧条件，实际 packed 序列为 **[388 text | 414 audio | 14,985 video]，S=15,787**；20 步 × 50 block × 4 个 INT8 层 = **4000 次 INT8 GEMM**，另有 4000 次在线 ConvRot；主 attention 是 **1000 次 BF16 FlashAttention**（`pytorch_flash::flash_fwd_kernel`），不是 INT8 attention。见 §12。

### 2.3 "pruned" 到底删了什么（核心发现）

对比上游完整 bf16 checkpoint（13 个 shard，535 个 tensor，33.123 B 参数，66.28 GB）与本文件：

| 键 | 完整版 | pruned 版 | 每 block 参数差 |
|---|---|---|---|
| `blocks.N.adaln_proj.linear.weight` | `[96768, 2688]` BF16 | `[96768, 8]` FP16 | −259,272,704 |
| `blocks.N.adaln_proj.linear.bias` | `[96768]` | `[96768]` | 0 |
| `time_embedder.proj_in/proj_out` | 存在（256→5376→2688） | **不存在** | — |
| `adaln_t_table` | 不存在 | `[1025, 8]` FP32 | +8,200 |
| `final_layer.adaln_proj.linear.weight` | `[10752, 2688]` | `[10752, 8]` | −28.8 M |

- 50 层 × 259,272,704 + final 28,815,360 + time_embedder 15,835,584 − table 8,200 ≈ **13.008 B**；
- 完整版 33.123 B − pruned 版 20.111 B = **13.012 B**，误差 0.004 B（0.03%，来自 bias/对齐）；
- bf16 文件差 66.28 GB − 40.23 GB = 26.05 GB ≈ 13.01 B × 2 B，完全吻合。

ComfyUI 源码对此有明确注释（`comfy/ldm/minimax/model.py:520-535`）：

```python
# curve-form checkpoints replace the time embedder and full-width adaln weights
# with a small shared basis of the time-embedding curve
self.use_adaln_curves = adaln_curve_grid is not None
```

推理时 `_time_embedding()` 用 `adaln_t_table` 在 t∈[0,1] 上做线性插值得到 8 维坐标，再乘以每层 `[96768, 8]` 的 AdaLN 矩阵（`model.py:641-649`）。**即 pruned = 把 AdaLN 的时间嵌入分支做了 rank-8 的低秩/曲线蒸馏，不是删 transformer 层，也不是剪 MLP/attention 宽度。**

> 交叉验证：直接读取 Comfy-Org `minimax_h3_fl2va_pruned_bf16.safetensors` 的 header——532 个 tensor，参数总量 **20,111,438,744**，与本地 int8 文件逐位一致；dtype 为 BF16 420 个、F16 102 个（AdaLN）、F32 10 个（patch/final head、table），`adaln_proj` 仍是 `[96768, 8]` F16，且同样没有 `time_embedder`。说明 "pruned bf16" 就是这份 20.11 B 权重，未再裁剪。

### 2.4 各模块精度与参数量占比

| 模块（50 层聚合） | 存储精度 | 参数量 | 占模型 |
|---|---|---|---|
| `blocks.*.mlp.fc1.weight` | int8（convrot W8A8） | 7.7070 B | **38.32%** |
| `blocks.*.attn.qkv_proj.weight` | int8（convrot W8A8） | 5.7803 B | **28.74%** |
| `blocks.*.mlp.fc2.weight` | int8（convrot W8A8） | 3.8535 B | **19.16%** |
| `blocks.*.attn.out_proj.weight` | int8（convrot W8A8） | 1.9268 B | **9.58%** |
| `token_refiner.blocks.*` 四个线性层 | bf16 | 0.7707 B | 3.83% |
| `blocks.*.adaln_proj.linear`（weight+bias） | fp16 | 0.0435 B | 0.22% |
| `condition_proj` | bf16 | 0.0275 B | 0.14% |
| `blocks.*.norm1/norm2/q_norm/k_norm` | bf16 | 0.00055 B | 0.003% |
| `final_layer.video_out`（96×5376） | **fp32** | 0.00052 B | 0.003% |
| `video_patch_proj`（96→5376） | **fp32** | 0.00052 B | 0.003% |
| `audio_patch_proj`（32→5376） | **fp32** | 0.00018 B | 0.001% |
| `final_layer.audio_out`（32×5376） | **fp32** | 0.00017 B | 0.001% |
| `final_layer.adaln_proj`（10752×8） | fp16 | 0.000097 B | <0.001% |
| `adaln_t_table` / `rope.inv_freq` | fp32 | 0.000008 B | <0.001% |

按功能归类：

| 功能块 | 参数 | 占比 |
|---|---|---|
| 主 DiT blocks（50 层，含 AdaLN 0.0435 B） | 19.312 B | 96.02% |
| ├ attention（qkv+out） | 7.707 B | 38.32% |
| └ MLP（fc1+fc2） | 11.561 B | 57.48% |
| token_refiner（2 层） | 0.771 B | 3.83% |
| 条件/输出投影（condition_proj、patch proj、final head、table） | 0.029 B | 0.14% |

注：AdaLN 的 0.0435 B 已计入主 blocks，不在最后一行重复统计；三行合计 20.111 B。

### 2.5 int8_convrot 格式细节

`comfy_quant` blob（每模块 72 字节）：

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

磁盘布局（以 `blocks.0.mlp.fc1` 为例）：

| tensor | dtype | shape | 说明 |
|---|---|---|---|
| `weight` | I8 | `[28672, 5376]` | int8 权重，值域实测 −127..127 |
| `weight_scale` | F32 | `[28672, 1]` | **每输出通道**一个 scale（本层实测 2.37e-4..7.89e-3） |
| `comfy_quant` | U8 | `[72]` | JSON 元数据 |

计算路径（`comfy/quant_ops.py` + `comfy_kitchen/tensor/int8.py`）：

1. `TensorWiseINT8Layout`，`is_weight=True`、`per_channel=True`、`convrot=True`；
2. 权重按 256 一组做正交 Hadamard 旋转后再量化（抑制 per-channel outlier）；
3. 推理时 `ck.int8_linear` 对激活做**在线旋转 + 动态 per-row 量化**，走 INT8 tensor-core GEMM，输出 BF16。

也就是说文件名中的 `int8_convrot` 实际是 **W8A8**（权重 8 bit、激活动态 8 bit），不是 W8A16。

### 2.6 磁盘构成

| dtype | 字节 | 占比 |
|---|---|---|
| I8（200 个量化线性层） | 19,267,584,000 | 91.88% |
| BF16（token_refiner、norm、condition_proj） | 1,597,634,048 | 7.62% |
| F16（AdaLN 调制矩阵） | 87,284,736 | 0.42% |
| F32（weight_scale、patch/final head、table） | 17,767,008 | 0.08% |
| U8（comfy_quant 元数据） | 14,400 | ~0 |

---

## 3. Text encoder：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

### 3.1 基本参数

| 项目 | 值 |
|---|---|
| 文件大小 | 15,687,142,551 B = 15.687 GB |
| tensor 数 | 2,054 |
| 逻辑参数量 | 25,753,095,920（25.753 B） |
| `__metadata__` | 无 |
| 量化格式 | 350 × `{"format":"nvfp4","full_precision_matrix_mult":true}` + 1 × `{"format":"int8_tensorwise"}`（embedding） |

### 3.2 结构：Qwen3-VL-32B 截断到 50 层 + 完整 ViT

| 组件 | 结构 | 精度 | 参数量 | 占比 |
|---|---|---|---|---|
| `model.layers.0..49` | 50 × Qwen3-VL text layer | **nvfp4**（线性层） | 24.380 B | **94.67%** |
| `model.embed_tokens` | 151936 × 5120 | **int8** per-row | 0.7779 B | 3.02% |
| `visual.*` | 27 层 ViT + deepstack merger | **bf16** | 0.5953 B | 2.31% |

单层 text layer（hidden 5120，64 Q heads / 8 KV heads，head_dim 128，GQA 8:1，SwiGLU 25600）：

| 子模块 | shape | 精度 | 每层参数 |
|---|---|---|---|
| `self_attn.q_proj` | 8192 × 5120 | nvfp4 | 41.94 M |
| `self_attn.k_proj` | 1024 × 5120 | nvfp4 | 5.24 M |
| `self_attn.v_proj` | 1024 × 5120 | nvfp4 | 5.24 M |
| `self_attn.o_proj` | 5120 × 8192 | nvfp4 | 41.94 M |
| `mlp.gate_proj` | 25600 × 5120 | nvfp4 | 131.07 M |
| `mlp.up_proj` | 25600 × 5120 | nvfp4 | 131.07 M |
| `mlp.down_proj` | 5120 × 25600 | nvfp4 | 131.07 M |
| 4 × RMSNorm（input/post_attn/q_norm/k_norm） | 5120/5120/128/128 | bf16 | 10,496 |
| `o_proj.pre_quant_scale`、`down_proj.pre_quant_scale`（side-car，不计入参数） | 8192 / 25600 | bf16（AWQ） | 33,792 |

每层 ≈ 487.60 M（权重 + norms，不含 33,792 个 pre_quant side-car 元素）；50 层 = 24.380 B。层内占比：**MLP 80.65%（393.22 M）**，attention 19.35%（94.37 M）。

`pre_quant_scale` 的分布是**不对称**的：每层只有 `o_proj` 和 `down_proj` 各一个，`q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj` 都没有；表里的 33,792 是**每层**的元素数（8192 + 25600），全文件 50 层共 100 个张量、1,689,600 个元素。§3.3 会给出源码与上游仓库的对照。

注意：

- 文件只有 50 层，且**没有 `model.norm` / `lm_head`**——与 `comfy/text_encoders/minimax.py` 的说明一致："truncated to 50 layers … the conditioning is the unnormalized hidden state after LM layer 50 (no final norm)"。
- `visual.*` 是完整 Qwen3-VL ViT：27 blocks（hidden 1152、16 heads、MLP 4304、patch 16、merge 2）、`visual.merger`、3 个 `deepstack_merger_list`（对应 index 8/16/24）、`pos_embed [2304,1152]`、`patch_embed [1152,3,2,16,16]`。全部 BF16，共 0.5953 B（2.31%）。

> 运行时校验（CUPTI）：50 层 × 7 = **350 次 NVFP4 反量化 + 350 次 FP32 GEMM**；50 次 FP32 attention；200 个 FP32 layer_norm。全部 155 种 kernel 中没有 `scaled_mm_nvfp4`。见 §12。

### 3.3 NVFP4 + AWQ 的精确布局

以同时具备全部 side-car 的 `model.layers.0.self_attn.o_proj` 为例（`q_proj` 没有 `pre_quant_scale`，见下）：

| tensor | dtype | shape | 说明 |
|---|---|---|---|
| `weight` | U8 | `[5120, 4096]` | 每字节 2 个 4-bit 值（K=8192 → 4096 字节） |
| `weight_scale` | F8_E4M3 | `[5120, 512]` | 每 16 个权重一个 block scale（8192/16=512） |
| `weight_scale_2` | F32 | `[]` | 每 tensor 一个全局 scale（本层实测 7.847e-4） |
| `pre_quant_scale` | BF16 | `[8192]` | AWQ 输入侧平滑系数（本层实测 0.293..3.406，mean 2.045） |
| `comfy_quant` | U8 | `[55]` | `{"format":"nvfp4","full_precision_matrix_mult":true}` |

反量化关系：`w = code(4bit) × block_scale(fp8) × scale_2(fp32)`。本层 row 0 实测 block scale 为 1.125..7.5（512 个），与 `scale_2 = 7.847e-4` 相乘后，每个 16-权重块的步长约为 **8.8e-4 .. 5.9e-3**；4-bit 码值范围 0..±6，因此单块可表示的权重量级为 0..±0.035。这里不做“与原始 BF16 权重逐值等价”的断言——文件内只有量化后的权重和三层 scale，是否与某个 BF16 版本逐值一致需要另取 BF16 权重做逐元素对照，本报告只描述文件自身的 scale 结构。

**为什么只有 `o_proj` / `down_proj` 有 `pre_quant_scale`：**

- 文件事实：全文件 100 个 `pre_quant_scale`（50 层 × 2），模块名只有 `self_attn.o_proj` 和 `mlp.down_proj` 两种；`q/k/v/gate/up` 均无。
- 上游对照：源仓库 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` 的第 1 个 shard 里，`pre_quant_scale` 同样只出现在 `self_attn.o_proj` 和 `mlp.down_proj`（该 shard 覆盖 layer 0..8，共 9 个 o_proj + 8 个 down_proj = 17 个），说明这是 modelopt NVFP4_AWQ 的导出方式，不是 Comfy-Org 转换时新增的层级。
- 作用：AWQ 的 per-input-channel 平滑系数，在量化线性层之前乘到输入上（`comfy/ops.py:1618-1620`）。`o_proj` 和 `down_proj` 是 attention / MLP 两个残差分支的输出投影，`pre_quant_scale` 作用在它们的输入端；`q/k/v/gate/up` 的输入没有这个系数。

`full_precision_matrix_mult: true` 的效果（`comfy/ops.py:1160`）：

```python
if self._full_precision_mm:
    out = torch.nn.functional.linear(x, self._v_weight.to(dtype=x.dtype).dequantize())
```

即**权重反量化成 `x.dtype` 后走普通 F.linear**，不需要 FP4 tensor core，所以 model card 说"不需要 Blackwell"。本次真实运行里 `x.dtype` 是 FP32（CUPTI 实测 350 次 `dequantize_nvfp4_kernel<...,float>` + 350 次 FP32 `cutlass sgemm`），不是 BF16；BF16 只是文件里非量化权重的存储 dtype。

`embed_tokens` 是另一套格式：`I8 [151936, 5120]` + `F32 [151936, 1]` per-row scale + `{"format":"int8_tensorwise"}`，即 int8 per-row embedding 表（3.02% 参数）。

### 3.4 磁盘构成与有效位宽

| dtype | 字节 | 占比 |
|---|---|---|
| U8（350 个 NVFP4 权重，24.379 B 参数） | 12,189,715,279 | 77.71% |
| F8_E4M3（350 个 block scale，共 1.524 B 个 scale 值） | 1,523,712,000 | 9.71% |
| BF16（ViT 0.595 B + 每层 norm/pre_quant） | 1,194,962,400 | 7.62% |
| I8（embedding 0.778 B） | 777,912,320 | 4.96% |
| F32（350 个 tensor scale） | 609,144 | 0.004% |

NVFP4 部分的有效存储 = 4 bit 权重 + 0.5 bit/参数的 FP8 block scale = **4.5 bit/参数**；整体 4.87 bit/参数。这就是 32 B 级模型压到 15.7 GB 的原因。

### 3.5 来源与重新量化的 base

model card 明确写了这份 nvfp4 文本编码器的来源：

> "The Qwen3-VL-32B `nvfp4_awq` quant is converted from: https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4"

实测该源仓库（`model.safetensors.index.json` + shard header）：

| 项目 | cybermotaz 源仓库 | Comfy-Org 转换后 |
|---|---|---|
| tensor 数 | 2,530 | 2,054 |
| 总大小 | 21.861 GB | 15.687 GB |
| text layer | 64 层（0..63） | 50 层（0..49） |
| `lm_head` / `model.norm` | 有 | **无** |
| visual tower | 有（`model.visual.*`，BF16） | 有（`visual.*`，BF16） |
| `input_scale` | 448 个（64 层 × 7 线性层） | **无** |
| `pre_quant_scale` | 128 个（64 层 × 2） | 100 个（50 层 × 2） |
| 量化配置 | modelopt 0.40.0，NVFP4_AWQ，group 16，`pre_quant_scale: true`，exclude `lm_head` / `model.visual*` | 保留 NVFP4 + block scale + scale_2 + pre_quant_scale，key 改为 ComfyUI 形式 |

Comfy-Org 做的转换是：64→50 层截断；删除 `lm_head` 和 `model.norm`；`model.language_model.*` → `model.layers.*`、`model.visual.*` → `visual.*`；去掉 `input_scale`；visual 保持 BF16。

**AWQ scale 与 block scale 的实际处理**（实测）：

- `pre_quant_scale`（AWQ 输入侧平滑，BF16，仅 o_proj/down_proj）：**保留且逐位相同**。抽查 `layers.0.self_attn.o_proj.pre_quant_scale`，源仓库与 Comfy-Org 文件 `torch.equal` 为 True，范围 0.293..3.406、mean 2.045 完全一致。
- `input_scale`（ModelOpt 静态激活 scale，F32 标量，源仓库 448 个）：**Comfy-Org 文件没有**。ComfyUI 的 `full_precision_matrix_mult` 路径不做激活量化，所以不需要它；如果目标运行时是 ModelOpt/vLLM，则要保留。
- `weight_scale` / `weight_scale_2`（NVFP4 的 block / tensor scale，不是 AWQ scale）：`weight_scale_2` 抽查与源相同（5.958e-4）；但 **block scale 与 U8 code 的排列并不逐字节相同**。实测 q_proj 第 0 行：4-bit code 的直方图完全一致（同一批 code 的排列），但 U8 字节顺序不同；block scale 的数值多重集也不同（源 14/15，目标 13/15）。这说明转换还做了 block 分组/布局重排，并重新计算了 block scale（`to_blocked`/`from_blocked` 不能直接复现这一差异）。

因此：**cybermotaz 是量化权重的来源，但不是可以直接拷贝的 ComfyUI 布局**。要复现现有 nvfp4 文件，最稳的是以 Comfy-Org 的 nvfp4 文件本身为 base；要从 BF16 重做，则用下面的 50 层 bf16 重新走量化流程。

| 目的 | base | 大小 | 说明 |
|---|---|---|---|
| 复现/继续修改现有 nvfp4 权重 | `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` | 21.861 GB，5 shards，64 层 | 量化权重的上游来源；需要截断到 50 层、改名、去 lm_head/model.norm，并做 block 分组/scale 重排（不能直接拷贝 U8/block scale） |
| 从 BF16 重新量化（推荐） | `Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51.506 GB，902 tensor，50 层 | 已截断、已改名、保留 visual；参数 25.753 B，与现有 nvfp4 文件一一对应 |
| 从原始 Qwen 重新开始 | `Qwen/Qwen3-VL-32B-Instruct` 或 `MiniMaxAI/MiniMax-H3/FL2VA/text_encoder` | 66.715 GB，64 层 + lm_head | 需自己截断到 50 层、去 lm_head/model.norm、保留 visual |

注意：cybermotaz 的 `input_scale` 在 Comfy-Org 文件中被去掉了；ComfyUI 的 nvfp4 配置虽然声明了 `input_scale` 参数，但这份文件没有该字段。如果复现原量化流程，需要确认目标框架是否使用 `input_scale`。

---

## 4. Video VAE：`minimax_h3_video_vae_int8_convrot.safetensors`

### 4.1 基本参数

| 项目 | 值 |
|---|---|
| 文件大小 | 2,811,065,184 B = 2.811 GB |
| tensor 数 | 850 |
| 逻辑参数量 | 2,603,871,080（2.604 B） |
| 量化格式 | 144 × `int8_tensorwise` + convrot 256 |
| `__metadata__` | `{"minimax_h3_video_vae": ...}`，内含 `latents_mean/std`（24 维）、`vae_clip_length=17`、`vae_token_drop=3`、完整 `source_config` |

### 4.2 结构：Conv3D encoder + 36 层 ViT decoder

- **encoder**：`EncoderFCN3D`，3D causal conv，通道 128→256→256→512→512→1024，每级 2 个 ResnetBlock3D，空间下采样 `[2,2,2,2,1,1]`（16×），时间下采样 `[1,2,2,1,1,1]`（4×）；`conv_out` 输出 48 = 24 mean + 24 logvar；`quant_conv` 48→48。
- **decoder**：`ViT3DDecoder`，patch `4×16×16`，dim 2048（32 heads × 64），36 层 TransformerBlock（RMSNorm + partial RoPE + SwiGLU FFN inner 8192，`w1` 输出 16384 是 gate×value 拼接），4 个 register token + 1 个 mask token（推理不用），`proj_out` 2048→3072（3×4×16×16）；`post_quant_conv` 24→24。
- latent：24 通道，空间 16×、时间 4×，`scale_factor=1.0`，用 24 维 `latents_mean/std` 归一化。

> 运行时校验（CUPTI）：一次 T2V decode 实际执行 **42 次 ViT forward**（7 时间 chunk × 3 高度 tile × (4+1) 宽度 batch），每次 36 层；共 **6048 次 INT8 GEMM** 和 **1512 次 INT8 attention**（`qk_int_sv_i8_attn_kernel`）。

### 4.3 精度与参数量占比

| 模块 | 存储精度 | 参数量 | 占比 | 字节 |
|---|---|---|---|---|
| `decoder.transformer_blocks.*.ff.w1`（36 层） | int8 | 1.2080 B | 46.39% | 1,207.96 MB |
| `decoder.transformer_blocks.*.ff.w2` | int8 | 0.6040 B | 23.20% | 603.98 MB |
| `decoder.transformer_blocks.*.attn.to_qkv` | int8 | 0.4530 B | 17.40% | 452.98 MB |
| `decoder.transformer_blocks.*.attn.to_out` | int8 | 0.1510 B | 5.80% | 150.99 MB |
| `encoder.down.*.conv1/conv2`（24 个 conv） | **fp16** | 0.1672 B | 6.42% | 334.43 MB |
| `encoder.conv_in/conv_out/downsample/nin_shortcut`（9 个 conv） | **fp16** | 0.0131 B | 0.50% | 26.18 MB |
| `decoder.proj_out`（3072×2048） | fp32 | 0.0063 B | 0.24% | 25.17 MB |
| decoder 线性层 bias / weight_scale + norm1/2/scale1/2 | fp32 | 0.0022 B | 0.08% | 8.85 MB |
| `decoder.norm_out` / `register_tokens` / `x_embedder` / `mask_token` | fp32 | 0.00007 B | ~0 | 0.3 MB |
| `quant_conv` / `post_quant_conv` | fp16 | 0.000003 B | ~0 | 6.9 KB |

按大块：

| 大块 | 参数 | 占比 |
|---|---|---|
| ViT decoder（36 层，int8 权重 + FP32 bias/norm/scale） | 2.4172 B | 92.84% |
| Conv3D encoder（fp16） | 0.1803 B | 6.92% |
| decoder 前后处理（proj_out/x_embedder/norm） | 0.0064 B | 0.24% |
| quant_conv 对 | 0.000003 B | ~0 |

### 4.4 int8 布局

与 DiT 相同：`I8 [N,K]` + `F32 [N,1]` per-output-channel scale + `U8 [72]` comfy_quant，`convrot_groupsize=256`。抽样 `decoder.transformer_blocks.0.attn.to_qkv`：weight 值域 −127..127，scale 0.0002..0.0011。

值得注意的是 **encoder 没有被量化**（FP16 conv），只有 decoder 的 144 个线性层是 int8；这解释了为什么 decoder 2.42 B 参数（int8）只占约 2.43 GB，而 encoder 0.18 B 参数（fp16）占 0.36 GB。

---

## 5. Audio VAE：`minimax_h3_audio_vae_fp32.safetensors`

### 5.1 基本参数

| 项目 | 值 |
|---|---|
| 文件大小 | 605,254,808 B = 0.605 GB |
| tensor 数 | 917 |
| 逻辑参数量 | 151,287,320（0.1513 B） |
| 精度 | **100% FP32**（917 个 tensor 全部 F32） |
| `__metadata__` | `{"minimax_h3_audio_vae": ...}`：32 kHz、latent 32 ch、2 声道、`latents_mean/std` 32 维、encoder_rates `[2,4,4,5,5]`、decoder_rates `[5,5,2,2,2,2,2]`、encoder_dim 64、latent_dim 2048、decoder_dim 1024、BigVGAN |

### 5.2 结构

- **encoder**：DAC 风格 1D conv，1→64→128→256→512→1024→2048，strides `2,4,4,5,5`（hop = 800 samples @32k = **40 latent frames/s**），每级 `EncoderBlock` 内含 3 个 dilated ResidualUnit（dilation 1/3/9）+ Snake1d。
- **pre_block**：`AttnProjection`，8 heads 的 causal attention + GeGLU MLP，把 2048 维压到 32 维 latent（12.67 M 参数）。
- **decoder**：BigVGAN，`dec_in_proj` 32→2048，`conv_pre` 2048→1024（k=7），7 级 upsample `5,5,2,2,2,2,2`（×800），每级 3 个 AMPBlock1（kernel 3/7/11，dilation (1,3,5)），SnakeBeta + 低通/上采样滤波器，最后 `conv_post` 1024→1；左右声道独立处理。

### 5.3 参数量占比

| 模块 | 参数 | 占比 |
|---|---|---|
| `encoder`（DAC encoder） | 73.70 M | **48.72%** |
| `decoder`（BigVGAN） | 64.85 M | **42.86%** |
| `pre_block`（attention 投影） | 12.67 M | 8.37% |
| `dec_in_proj`（32→2048） | 0.066 M | 0.04% |
| `mean_proj` / `logs_proj`（32→32，logs 推理不用） | 0.002 M | ~0 |

其中最重的单个张量：`encoder.block.*` 的深层 conv（30 个，33.5 M，22.17%）、`decoder.resblocks.*.convs1/convs2`（各 63 个，各 22.0 M，各 14.56%）、`decoder.conv_pre [1024,2048,7]`（14.7 M，9.71%）、`pre_block.attn.qkv [6144,2048]`（12.6 M，8.32%）。

全 FP32 是这份文件 0.605 GB 的直接原因（151.3 M × 4 B ≈ 605 MB）。model card 中没有提供该 VAE 的 int8/bf16 版本。

> 运行时校验（CUPTI）：音频阶段 2,679 个 kernel，全部 FP32 Conv1d / ConvTranspose1d / Snake / 逐点 op，没有 INT8/NVFP4 kernel。

---

## 6. LoRA：`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

### 6.1 基本参数

| 项目 | 值 |
|---|---|
| 文件大小 | 1,956,193,000 B = 1.956 GB |
| tensor 数 | 624 |
| 逻辑参数量 | 978,059,264（0.978 B），100% BF16 |
| `__metadata__` | `training_rank=128`、`training_alpha=8.0`、`training_scale=0.0625`、`base_model=Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`、`source_format=Diffusers PEFT LoRA`、`target_format=ComfyUI generic LoRA`、`qkv_fusion=block diagonal B; concat A; alpha multiplied by 3`、`swi_glu_mapping=Diffusers [value;gate] -> ComfyUI [gate;value]` |

### 6.2 目标模块与秩

LoRA 覆盖 **50 个 DiT block + 2 个 token_refiner block**，每 block 4 个目标模块，共 52 × 4 = 208 个模块、416 个 A/B 张量、208 个 alpha 标量：

| 目标模块 | `lora_A` shape | `lora_B` shape | rank | alpha | 每 block 参数 |
|---|---|---|---|---|---|
| `attn.qkv_proj` | `[384, 5376]` | `[21504, 384]` | **384**（=3×128，q/k/v 拼接） | **24.0** | 10,321,920 |
| `attn.out_proj` | `[128, 7168]` | `[5376, 128]` | 128 | 8.0 | 1,605,632 |
| `mlp.fc1` | `[128, 5376]` | `[28672, 128]` | 128 | 8.0 | 4,358,144 |
| `mlp.fc2` | `[128, 14336]` | `[5376, 128]` | 128 | 8.0 | 2,523,136 |
| 合计 | | | | | **18,808,832** |

- 208 个 alpha 的实际存储值只有两种：**8.0（156 个）** 和 **24.0（52 个，全部是 qkv_proj）**。
- 有效缩放统一为 `alpha/rank`：qkv 为 24/384 = 1/16；其余 8/128 = 1/16，与 `training_scale=0.0625` 一致。
- qkv 的融合方式：B 是 3 个 rank-128 块组成的块对角矩阵，A 是 3 段拼接，所以每个 q/k/v 子块仍是 rank 128。
- SwiGLU 的 `fc1` 在 Diffusers 中是 `[value; gate]` 顺序，转换到 ComfyUI 时重排为 `[gate; value]`（metadata 明示）。

### 6.3 参数量占比

| 目标（52 个 block，含 2 个 token_refiner block） | 参数 | 占 LoRA |
|---|---|---|
| `attn.qkv_proj` A+B | 536.74 M | **54.88%** |
| `mlp.fc1` A+B | 226.62 M | 23.17% |
| `mlp.fc2` A+B | 131.20 M | 13.41% |
| `attn.out_proj` A+B | 83.49 M | 8.54% |
| 合计 | 978.06 M | 100% |

**LoRA / DiT 参数比 = 978.06 M / 20.111 B = 4.86%**（文件大小比 1.956 GB / 20.970 GB = 9.3%，因为 LoRA 是 BF16 而 DiT 主体是 int8）。

> 运行时边界：参考 T2V 运行 `turbo_mode=false`、主模型 0 patches，**没有加载该 LoRA**；本章是对文件本身的静态分析，运行时证据只覆盖 DiT / TE / 两个 VAE。

---

## 7. 汇总：精度 × 参数量 × 磁盘

### 7.1 五个文件的精度分布（按参数量）

| 文件 | int8 | nvfp4 | bf16 | fp16 | fp32 |
|---|---|---|---|---|---|
| DiT pruned int8 | **95.80%** | — | 3.97% | 0.22% | 0.01% |
| TE nvfp4 | 3.02%（embed） | **94.67%** | 2.31%（ViT） | — | ~0 |
| Video VAE int8 | **92.78%** | — | — | 6.92%（encoder） | 0.29% |
| Audio VAE fp32 | — | — | — | — | **100%** |
| LoRA bf16 | — | — | **100%** | — | — |

### 7.2 磁盘构成（GB = 10^9 B）

| 文件 | 总大小 | 权重主体 | 量化 side-car | 其他 |
|---|---|---|---|---|
| DiT int8 | 20.970 GB | int8 19.268 GB | 17.8 MB（F32 scale + U8 blob） | bf16 1.598 GB + fp16 0.087 GB |
| TE nvfp4 | 15.687 GB | nvfp4 12.190 GB | 1.524 GB（FP8 block scale）+ 0.6 MB（F32） | embed 0.778 GB + ViT 1.195 GB |
| Video VAE | 2.811 GB | int8 2.416 GB | 3.8 MB（F32 scale + blob） | fp16 0.361 GB + fp32 0.034 GB |
| Audio VAE | 0.605 GB | — | — | fp32 0.605 GB |
| LoRA | 1.956 GB | bf16 1.956 GB | 832 B（alpha） | — |

### 7.3 精度语义速查

| 标记 | 含义 | 在本仓库中的实际用法 |
|---|---|---|
| `int8_convrot` | int8 权重 + 每输出通道 F32 scale + 256 组 Hadamard 旋转；**W8A8** GEMM | DiT 200 层、Video VAE 144 层 |
| `nvfp4` | 4-bit E2M1 + 每 16 权重一个 FP8 E4M3 block scale + 每 tensor 一个 F32 scale；本文件 `full_precision_matrix_mult=true` → **存储 4-bit，反量化后普通 F.linear；实测计算 FP32** | TE 350 层 |
| `int8_tensorwise`（embedding） | int8 per-row embedding 表 | TE `embed_tokens` |
| `fp16` | IEEE half | DiT AdaLN 矩阵、Video VAE encoder conv |
| `bf16` | bfloat16 | DiT token_refiner / norm、TE ViT / norm、LoRA |
| `fp32` | float32 | Audio VAE 全部、DiT patch/final head、各量化 scale、视频 VAE 的 bias/norm |

### 7.4 与上游完整模型的对照

| 版本 | 参数量 | bf16 大小 | 说明 |
|---|---|---|---|
| 上游 `FL2VA/transformer`（完整） | 33.123 B | 66.28 GB | AdaLN `[96768,2688]` + `time_embedder` |
| Comfy-Org `pruned_bf16` | 20.111 B | 40.23 GB | AdaLN `[96768,8]` + `adaln_t_table[1025,8]`，无 time_embedder |
| Comfy-Org `pruned_int8_convrot`（本文） | 20.111 B | 20.97 GB | 同上，block 线性层 int8 |

---

## 8. 复现与核验

### 8.1 结构与精度统计

```bash
cd /home/kaokaolv/comfyui_h3
.venv/bin/python analyze_h3_safetensors.py            # 五个文件的总览
.venv/bin/python analyze_h3_safetensors.py diT        # 单文件
```

脚本只读 safetensors header 和 `comfy_quant` blob；`comfy_quant` 的 JSON 内容、每 tensor 的 dtype/shape/字节偏移均原样输出。

### 8.2 文件完整性

本地文件由 `setup_comfyui_h3.sh` 在固定 revision `4b53a35191437de05370da548e59aeabe2a5a09a` 下用 `hf download` 获取，并在同一脚本里用内嵌清单做了 `sha256sum --check`；`logs/setup_comfyui_h3.log` 记录 5 个文件全部 `OK`。该清单与 HF API 返回的 LFS oid 完全一致：

| 文件 | HF LFS oid / 本地清单 | 2026-11-17 复核 |
|---|---|---|
| DiT pruned int8 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` | 大小一致，setup 时 sha256 OK |
| TE nvfp4 | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` | 大小一致，setup 时 sha256 OK |
| Video VAE int8 | `52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e` | 大小一致，setup 时 sha256 OK |
| Audio VAE fp32 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` | 大小一致，setup 时 sha256 OK |
| LoRA 8-step | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` | 大小一致，setup 时 sha256 OK |

证据链：`setup_comfyui_h3.sh:126-139`（下载 + `sha256sum --check` 清单）、`logs/setup_comfyui_h3.log:293-297`（5 个 `OK`）、HF `/api/models/Comfy-Org/MiniMax-H3`（文件大小与 oid）。

### 8.3 关键源码位置

| 内容 | 位置 |
|---|---|
| DiT 模型实现（packed 序列、AdaLN 曲线、音频/视频双流） | `ComfyUI/comfy/ldm/minimax/model.py` |
| 模型检测（从 state_dict 反推 50 层、56 heads、latent 24、audio 32） | `ComfyUI/comfy/model_detection.py:404-437` |
| 支持的模型/采样 shift（video 12.0 / audio 3.0） | `ComfyUI/comfy/supported_models.py:984-1014` |
| int8/convrot/nvfp4 量化格式注册 | `ComfyUI/comfy/quant_ops.py:211-250` |
| 量化 Linear 的加载与 `full_precision_matrix_mult` | `ComfyUI/comfy/ops.py:1375-1425, 1155-1175` |
| INT8 layout / convrot 反量化 | `.venv/lib/python3.12/site-packages/comfy_kitchen/tensor/int8.py:60-180` |
| Video VAE（Conv3D encoder + ViT decoder） | `ComfyUI/comfy/ldm/minimax/vae.py:191-430, 565-660` |
| Audio VAE（DAC + AttnProjection + BigVGAN） | `ComfyUI/comfy/ldm/minimax/audio_vae.py:195-443` |
| TE 截断与 token tag 规则 | `ComfyUI/comfy/text_encoders/minimax.py:1-40` |

### 8.4 抽样数值校验（摘录）

| 张量 | 结果 |
|---|---|
| `blocks.0.attn.qkv_proj.weight`（I8） | min −127 / max 127，mean ≈ 0 |
| `blocks.0.attn.qkv_proj.weight_scale`（F32） | 21,504 个，0.0006..0.0071 |
| `blocks.0.mlp.fc1` 反量化前 4 行 | std 0.127..0.149，absmax 0.43..0.59 |
| `model.layers.0.self_attn.q_proj.weight_scale`（F8_E4M3） | 8192×320，值 1.25..448 |
| `model.layers.0.self_attn.q_proj.weight_scale_2`（F32） | 5.96e-4 |
| `model.layers.0.self_attn.o_proj.pre_quant_scale`（BF16） | 0.293..3.406，mean 2.04 |
| `decoder.transformer_blocks.0.attn.to_qkv.weight_scale` | 6,144 个，0.0002..0.0011 |
| LoRA alpha | 8.0 ×156，24.0 ×52 |

---

## 9. 结论

1. 这 5 个文件覆盖了 ComfyUI 推理链路的四类组件：**DiT（int8 W8A8）→ TE（NVFP4 存储量化，反量化后 FP32 计算）→ 视频 VAE（decoder int8 / encoder FP16）→ 音频 VAE（全 FP32）**，再叠加一个 **BF16 的 8 步蒸馏 LoRA**。
2. 参数量重心：DiT 的 96% 在 50 层 block 内（MLP 57.5% / attention 38.3%）；TE 的 94.7% 在 50 层 text layer（MLP 80.7% / attention 19.3%）；视频 VAE 的 92.8% 在 36 层 ViT decoder；音频 VAE 的 91.6% 在 encoder+decoder。
3. 磁盘效率：TE 4.87 bit/参数（NVFP4）、DiT 8.34 bit/参数（int8 + BF16 非量化岛）、视频 VAE 8.64 bit/参数、LoRA 16 bit/参数、音频 VAE 32 bit/参数。
4. `pruned` 的真实含义是 **AdaLN 时间分支的 rank-8 曲线蒸馏**（−13.01 B 参数，−39.3%），不是层剪枝；这一点由上游 shard header 的 `[96768,2688]` vs 本地 `[96768,8]`、参数量差与文件大小差三方交叉验证。
5. `nvfp4_awq` 的 "AWQ" 体现在 `o_proj` / `down_proj` 的 `pre_quant_scale`（BF16，输入侧平滑），"nvfp4" 体现在 U8 packed 权重 + FP8 block scale + F32 tensor scale 三层 scale 结构。

---

## 10. 关键结论的证据来源与推断边界

本节把前文每条结论拆到"原始证据"层面，并明确区分三类：

- **实测**：直接从 safetensors header、张量值或 HF API 读出，可复跑；
- **文档**：HF model card、上游 `config.json` 的明文；
- **推断**：由实测数据 + 源码语义交叉确认，但文档没有直接这样写。

| # | 结论 | 主要证据 | 类型 |
|---|---|---|---|
| 1 | `pruned` = AdaLN 曲线低秩化，不是删层 | 本地/上游 header 形状差 + 参数量差 + `model.py:525` 注释 | 实测 + 推断 |
| 2 | `int8_convrot` 是 W8A8 | 文件内 `comfy_quant` blob + `int8_linear` 源码对激活做 int8 量化 | 实测 + 源码 |
| 3 | TE 的 nvfp4 是存储量化、反量化后 FP32 计算 | 文件内 `full_precision_matrix_mult=true` + `ops.py:1160` + CUPTI 实测 | 实测 + 文档 |
| 4 | 三个文件量化覆盖面不同 | 全量 tensor 的 dtype 普查 | 实测 |
| 5 | LoRA rank 128 / alpha 8 / 52 模块 / 4.86% | 文件 `__metadata__` + alpha 张量值 + 键名统计 + 除法 | 实测 |
| 6 | 本地文件与 HF 一致 | `setup_comfyui_h3.sh` 的 sha256 清单 + setup 日志 + HF API size/oid | 实测 |

### 10.1 `pruned` 的证据链

这一条 model card 没有解释，属于**推断**，但有三组独立数据指向同一结论。

**(a) 本地 int8 文件头（实测）**

- `blocks.0..49` 共 50 个 block，一个不少；
- `blocks.N.adaln_proj.linear.weight` 形状 `[96768, 8]`、dtype F16；
- `adaln_t_table` 形状 `[1025, 8]`、F32；
- 没有任何 `time_embedder.*` 键。

**(b) 上游完整 checkpoint（实测）**

用 HTTP Range 只读 13 个 shard 的 header（每个约 6 KB）：

- 535 个 tensor，参数总量 **33.123 B**，66.28 GB；
- `blocks.N.adaln_proj.linear.weight` 形状 `[96768, 2688]`、BF16；
- 存在 `time_embedder.proj_in [5376,256]`、`time_embedder.proj_out [2688,5376]`；
- `final_layer.adaln_proj.linear.weight` 为 `[10752, 2688]`。

**(c) 参数量与文件大小对账（实测 + 算术）**

```
50 × (96768×2688 − 96768×8) = 12,963,635,200
final AdaLN 差                =     28,815,360
time_embedder                =     15,835,584
adaln_t_table                =         −8,200
─────────────────────────────────────────────
合计                          ≈ 13,008,277,944 = 13.008 B

完整版 33.123 B − 本地 20.111 B = 13.012 B（误差 0.03%）
bf16 文件差 66.28 GB − 40.23 GB = 26.05 GB ≈ 13.01 B × 2 B
```

另外直接读取 Comfy-Org `minimax_h3_fl2va_pruned_bf16.safetensors` 的 header：532 个 tensor，参数总量 **20,111,438,744**，与本地 int8 文件完全相同；`adaln_proj` 仍是 `[96768,8]` F16，同样没有 `time_embedder`。说明 "pruned bf16" 就是这份 20.11 B 权重，未再裁剪。

**(d) 源码语义（推断依据）**

- `comfy/ldm/minimax/model.py:525`：
  `# curve-form checkpoints replace the time embedder and full-width adaln weights with a small shared basis of the time-embedding curve`
- `model.py:524`：`self.use_adaln_curves = adaln_curve_grid is not None`
- `model.py:639-649`：`_time_embedding()` 用 `adaln_t_table` 在 t∈[0,1] 上线性插值出 8 维坐标，再乘以每层 `[96768,8]` 矩阵
- `comfy/model_detection.py:421-425`：存在 `adaln_t_table` 时，`adaln_curve_grid = table[0]`、`time_embed_dim = table[1]`，即 1025 和 8

**边界**：数据本身（形状、参数量、文件大小）都是实测；"pruned 这个词具体指 AdaLN 曲线蒸馏"是由这些数据 + ComfyUI 注释推出的解释，model card 未给出定义。

### 10.2 `int8_convrot` = W8A8 的证据

**(a) 文件内元数据（实测）**

每个量化模块的 `comfy_quant` blob（72 字节）内容：

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

权重布局实测：`weight` I8 `[28672, 5376]`（−127..127），`weight_scale` F32 `[28672, 1]`（每输出通道一个，0.0006..0.0071）。

**(b) 注册表（源码）**

`comfy/quant_ops.py:238-243`：

```python
QUANT_ALGOS["int8_tensorwise"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "TensorWiseINT8Layout",
    "quantize_input": False,
}
```

`comfy_kitchen/tensor/int8.py:117`：`convrot: If True, apply orthogonal group-wise Hadamard rotation to weight.`

**(c) 运行时激活量化（源码）**

“A8”来自 kernel 实现，不在 checkpoint 里。`comfy_kitchen/backends/cuda/__init__.py` 的 `int8_linear`（2128 行起）在 convrot 分支中先记了一句注释，再分配 int8 激活缓冲：

```python
# cuBLAS INT8 GEMM requires row-wise quantized activations and tensor-wise quantized weights.
if convrot:
    x_qdata = torch.empty((m, k), dtype=torch.int8, device=x.device)
    x_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    _C.quantize_int8_rowwise_convrot64(x_2d, x_qdata, x_scale, convrot_groupsize, ...)
# 另外两条等价路径（同样产出 int8 激活 + per-row F32 scale）：
#   x_qdata, x_scale = quantize_int8_rowwise_convrot(x_2d, convrot_groupsize)
#   x_qdata, x_scale = quantize_and_rotate_rowwise(x_2d, hadamard, convrot_groupsize)
```

即激活先做在线 Hadamard 旋转 + per-row int8 量化，再走 INT8 GEMM。`quantize_input: False` 只表示框架不把输入预打包成 QuantizedTensor（checkpoint 也不存 `input_scale`），激活量化由 `ck.int8_linear` 在 kernel 内完成，不代表运行时激活不量化。

### 10.3 TE 的 nvfp4 是存储量化的证据

**(a) 文件内元数据（实测）**

350 个线性层的 blob：

```json
{"format": "nvfp4", "full_precision_matrix_mult": true}
```

**(b) 执行路径（源码）**

`comfy/ops.py:1160`：

```python
if self._full_precision_mm:
    out = torch.nn.functional.linear(x, self._v_weight.to(dtype=x.dtype).dequantize())
```

权重先 `dequantize()` 成计算 dtype（本次运行为 FP32）再做普通 F.linear，因此不需要 FP4 tensor core。

**(c) 布局（实测 + 源码）**

文件头实测 + `comfy/quant_ops.py:222-227`（`group_size: 16`）：

| tensor | dtype | shape（以 `o_proj` 为例） |
|---|---|---|
| `weight` | U8 | `[5120, 4096]`（K=8192，每字节 2 个 4-bit 值） |
| `weight_scale` | F8_E4M3 | `[5120, 512]`（8192/16） |
| `weight_scale_2` | F32 | `[]`（标量，本层实测 7.847e-4） |
| `pre_quant_scale` | BF16 | `[8192]`（仅 `o_proj` / `down_proj`） |

**(d) 文档（model card）**

> "This `nvfp4` text encoder does not require Blackwell GPU to use."

AWQ 的体现：全文件 100 个 `pre_quant_scale`（50 层 × 2），模块名只有 `o_proj` 和 `down_proj`；源仓库 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` 的第 1 个 shard 也是同样分布（9 个 o_proj + 8 个 down_proj = 17 个），说明这不是 Comfy-Org 转换时新增的。`comfy/ops.py:1618-1620` 把它乘到量化线性层的输入上（输入侧平滑）。

**(e) 关于"weight std"的说明**

初版 §3.3 写过"乘积量级 ~0.01，与权重 std 吻合"。这句话指的是反量化后权重的标准差，但它是未经验证的表述：文件内没有 BF16 原权重，无法只凭 scale 证明反量化值与某个 BF16 版本逐值一致。现已改为只陈述可测量的 scale 算术（见 §3.3），不对"与原始权重 std 吻合"下结论。

### 10.4 量化覆盖面的证据

纯 `.weight` 张量的 dtype 普查（`analyze_h3_safetensors.py` 可复跑）：

| 文件 | 结论 | 实测计数 |
|---|---|---|
| DiT | 只量化 block 内 4 个线性层 | I8 恰好 200 = 50 × 4；`token_refiner.*` 全 BF16；patch/final head F32 |
| Video VAE | 只量化 decoder 线性层 | I8 恰好 144 = 36 × 4；encoder 33 个 conv 权重全 F16；bias/norm 全 F32 |
| Audio VAE | 未量化 | 917 个 tensor 全 F32 |

### 10.5 LoRA 参数的证据

- `__metadata__`（实测，文件自带）：`training_rank=128`、`training_alpha=8.0`、`training_scale=0.0625`、`base_model=Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`、`qkv_fusion=block diagonal B; concat A; alpha multiplied by 3`、`swi_glu_mapping=Diffusers [value;gate] -> ComfyUI [gate;value]`。
- alpha 张量值（实测）：8.0 × 156 个，24.0 × 52 个（24 = 8×3，全部在 `qkv_proj`）。
- A/B 形状（实测 header）：qkv `[384,5376]` / `[21504,384]`，out `[128,7168]` / `[5376,128]`，fc1 `[128,5376]` / `[28672,128]`，fc2 `[128,14336]` / `[5376,128]`。
- 模块数（键名统计）：`blocks.*` 50 组 + `token_refiner.blocks.*` 2 组，每组 4 个目标 = 208 个模块、416 个 A/B 张量。
- 占比（算术）：978,059,264 / 20,111,438,744 = **4.86%**。

### 10.6 文件完整性证据

- `setup_comfyui_h3.sh:117-139`：在 revision `4b53a35191437de05370da548e59aeabe2a5a09a` 下 `hf download` 这 5 个文件，并内嵌 sha256 清单执行 `sha256sum --check`。
- `logs/setup_comfyui_h3.log:293-297`：5 个文件全部 `OK`。
- 清单中的 5 个 sha256 与 HF API 返回的 LFS oid 完全一致；2026-11-17 复核文件大小仍与 HF API 一致。
- 说明：本次没有重算 41 GB 的全量 sha256（磁盘 I/O 过慢，已中止），完整性依据是"下载时校验 + 当前大小一致"。

### 10.7 一句话总结证据边界

- **文件头/张量值能直接回答的**：参数量、dtype、shape、量化格式、scale 布局、LoRA 秩与 alpha、模块覆盖面——全部是实测。
- **需要源码才能回答的**：W8A8 还是 W8A16、nvfp4 是存储量化还是计算量化、convrot 的旋转粒度、AdaLN 曲线如何工作——由 ComfyUI / comfy_kitchen 源码确认。
- **唯一属于推断的**：`pruned` 一词的语义（AdaLN 曲线蒸馏）。数据与源码高度一致，但 model card 没有明文定义。

---

## 11. 重新量化的 base 来源（五个文件）

如果要重新量化，先要区分“现有量化文件的直接权重来源”和“从 BF16/FP16 从零量化的干净起点”。下表都经过 header 核对。

| 目标文件 | 直接 base（同参数、同布局） | 上游原始 | 规模 / 备注 |
|---|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | `MiniMaxAI/MiniMax-H3/FL2VA/transformer/*`（13 shards） | pruned_bf16：40.23 GB / 20.111 B / 532 tensor；上游：66.28 GB / 33.123 B，含 `time_embedder` 与 `[96768,2688]` AdaLN |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 从零量化用 `Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`；复现现有 nvfp4 权重用 `cybermotaz/Qwen3-VL-32B-Instruct-NVFP4` | `Qwen/Qwen3-VL-32B-Instruct` 或 `MiniMaxAI/MiniMax-H3/FL2VA/text_encoder` | Comfy-Org bf16：51.506 GB / 25.753 B / 902 tensor / 50 层；cybermotaz：21.861 GB / 64 层 / 2,530 tensor；原始 Qwen：66.715 GB / 64 层 + lm_head |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors` | `MiniMaxAI/MiniMax-H3/FL2VA/video_vae/source/model.safetensors` | fp16 base：5.208 GB / 2.604 B / 562 tensor / 全 F16；上游 source：10.416 GB / FP32 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 本身未量化，可直接作为 base | `MiniMaxAI/MiniMax-H3/FL2VA/audio_vae/model.safetensors` | 605 MB / 151.3 M / 全 FP32 |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 不量化；合并到 `minimax_h3_fl2va_bf16.safetensors` | `lightx2v/Minimax-h3-Turbo`（训练源） | 1.956 GB / 978.06 M / BF16；LoRA 的 metadata 指定 base 为 `minimax_h3_fl2va_bf16` |

选择建议与注意事项：

1. **要复现现有 int8 DiT**：用 `pruned_bf16`（20.111 B），不要用完整 `bf16`（33.123 B）；两者差 13.01 B，全部在 AdaLN/时间嵌入分支（§2.3）。
2. **要从 BF16 重做 TE**：用 Comfy-Org 的 50 层 `qwen3vl_32b_minimax_h3_bf16` 最省事——它已经截断到 50 层、删掉 `lm_head` / `model.norm`、保留 `visual.*`，且 key 与现有 nvfp4 文件一一对应。若从原始 Qwen3-VL-32B 开始，需要自己截断 64→50、删 `lm_head`/`model.norm`、把 `model.language_model.*` 映射为 `model.layers.*`、`model.visual.*` 映射为 `visual.*`。
3. **要复现现有 nvfp4 权重本身**：`cybermotaz` 是量化权重的上游来源（model card 原文，21.861 GB / 64 层），但它带 448 个 `input_scale`（Comfy-Org 已去掉）、`lm_head`、64 层，而且 U8 code 与 block scale 的排列/数值与 Comfy-Org 文件不同（实测 4-bit code 直方图相同、block scale 多重集不同）。因此不能直接拷贝张量；如果只是要现有 Comfy-Org nvfp4 的权重，直接以该 nvfp4 文件为 base 更稳。AWQ 的 `pre_quant_scale` 已逐位保留（100 个，o_proj/down_proj）。
4. **Video VAE**：fp16 base 的 562 个 tensor 与 int8 文件的编码器/解码器结构一致，只是 decoder 的 144 个线性层在 int8 文件里被量化；encoder 在两个文件里都是 F16。
5. **Audio VAE**：没有量化版本，直接以 fp32 文件（或上游 `FL2VA/audio_vae/model.safetensors`）为 base。
6. **Ref2VA**：同一仓库还有 `minimax_h3_ref2va_bf16` / `_pruned_bf16`，上游在 `Ref2VA/transformer`（同样 50 层、`time_embed_dim 2688` 的 full AdaLN）。
7. **许可**：这些 base 均在 MiniMax-H3 Community License 下，重分发/商用前需核对 `LICENSE`。

---

## 12. 与真实运行记录的交叉校验

参考文档：

- `comfyui_h3_t2v_kernel_analysis.md`（静态调用栈 + 第 17 章 CUPTI 实测）
- `cupti_h3_t2v_kernel_report.md`（CUPTI/Nsight 逐 kernel 报告）

两者使用的基线与本文档分析的**是同一批文件**：`kernel_analysis` §1.1 列出的 4 个模型文件字节数与本文 §0 / §2.1 / §3.1 / §4.1 / §5.1 完全一致（TE 15,687,142,551；DiT 20,970,379,616；video VAE 2,811,065,184；audio VAE 605,254,808）。这为下面的交叉校验提供了同一对象前提。

### 12.1 实测确认的结论

| 本文档结论 | 运行记录中的实测证据 | 结论 |
|---|---|---|
| DiT 有 50 个主 block，仅其中 200 个线性层是 int8 | CUPTI：`20 steps × 50 blocks × 4 = 4000` 次 INT8 GEMM；`kernel_analysis` §3.2 的 200 个 `.comfy_quant` 表 | 一致 |
| DiT 的 int8 是 convrot W8A8，激活在线旋转+per-row 量化 | CUPTI：4000 次 `quantize_int8_rowwise_convrot64_kernel<__nv_bfloat16,...>`；`kernel_analysis` §8.1 | 一致 |
| DiT 主 attention 不是 INT8 | CUPTI：1000 次 `pytorch_flash::flash_fwd_kernel`（BF16），无 DiT INT8 attention kernel | 一致 |
| DiT 其他模块（token_refiner、patch proj、AdaLN、final head）不是 int8 | `kernel_analysis` §3.2 明确“不能根据文件名归入 200 个 INT8 Linear”；CUPTI 只有 4000 次 INT8 GEMM | 一致 |
| TE 是 Qwen3-VL-32B 的前 50 层，每层 7 个线性层 | CUPTI：`350 = 50 × 7` 次 NVFP4 反量化；200 个 FP32 layer_norm（4/层）；50 次 attention | 一致 |
| TE 的 nvfp4 是存储量化，不是 FP4 GEMM | CUPTI：350 次 `dequantize_nvfp4_kernel<...,float>` + 350 次 FP32 GEMM，无 `scaled_mm_nvfp4`；`kernel_analysis` §4.4 | 一致（但计算 dtype 需修正，见 §12.2） |
| Video VAE decoder 是 36 层 ViT，144 个 int8 线性层 | CUPTI：42 次 ViT × 36 层 × 4 = 6048 次 INT8 GEMM；`kernel_analysis` §9.2 的 144 = 36×4 | 一致 |
| Video VAE 的 attention 也是 INT8 | CUPTI：1512 次 `qk_int_sv_i8_attn_kernel` + `quant_qk_fused`/`quant_v_int8` | 一致，且本文已补注 |
| Video VAE 的 decoder residual 走融合 epilogue | CUPTI：3024 个带 `VisitorAuxLoad` 的 CUTLASS kernel，对应 `to_out`/`ff.w2` | 一致 |
| Audio VAE 全 FP32，没有量化 kernel | CUPTI：2,679 个 FP32 Conv1d/ConvTranspose1d/Snake/逐点 kernel，无 INT8/NVFP4 | 一致 |
| 文件大小/参数量 | `kernel_analysis` §1.1 的 4 个文件字节数与本文完全一致 | 一致 |

### 12.2 需要修正的地方

1. **TE 的计算 dtype 不是 BF16，而是 FP32。**
   - 本文初版根据 `ops.py:1160` 的 `F.linear(x, weight.dequantize())` 写成“反量化成 BF16”。这只说明了“反量化后普通 F.linear”，没有确认 `x.dtype`。
   - CUPTI 实测：`dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, float, ...>` 后接 `cutlass3x_sm100_simt_sgemm_f32`，Qwen attention 是 `fmha_cutlassF_f32_aligned`。因此实际计算是 FP32。
   - 本次修订已把 §0、§3.3、§7.3、§9、§10.3 的“BF16 计算”改为“反量化后普通 F.linear；本次实测 FP32”。

2. **Video VAE 的 attention 精度需要补注。**
   - 本文只写了 144 个 int8 线性层，没有提 attention。CUPTI 实测 decoder 的 attention 走 `qk_int_sv_i8_attn_kernel`（1512 次），并带 `quant_qk_fused`/`quant_v_int8`。
   - 已在 §4.2 补注。

3. **参考文档 `kernel_analysis` §3.2/§7.4 的 `TimeEmbedder` 描述与本 checkpoint 不符。**
   - `kernel_analysis` §7.4 写的是 `TimeEmbedder: 256→5376→2688`，但本文分析的同一个 DiT 文件（字节数相同）**没有 `time_embedder.*` 键**，而是 `adaln_t_table [1025, 8]` + `blocks.N.adaln_proj.linear.weight [96768, 8]`。
   - `comfy/model_detection.py:421-425` 在存在 `adaln_t_table` 时设置 `adaln_curve_grid = table[0]`、`time_embed_dim = table[1]`；`model.py:524-527` 据此走 `use_adaln_curves=True`，不创建 `TimeEmbedder`。
   - 因此这两段应视为沿用了 full 模型的静态描述；本文 §2.3 以 header + 检测代码为准。该差异不影响 CUPTI 的 kernel 计数（AdaLN 本来就不在 200 个 INT8 Linear 里），但会影响对时间嵌入路径的描述。

4. **“weight std 吻合”已删除。**
   - 见 §3.3 与 §10.3(e)：文件内没有 BF16 原权重，不能只凭三层 scale 证明反量化值与某个 BF16 版本逐值一致；改为陈述可测量的 scale 算术。

### 12.3 运行记录没有覆盖的部分

- **LoRA 未加载**：参考 T2V 运行 `turbo_mode=false`、主模型 0 patches，因此 LoRA 章节是纯静态文件分析；没有“LoRA 合并后 kernel”的运行时证据。
- **Video VAE encoder 未执行**：T2V 只跑 decoder；encoder 的 F16 精度结论来自 header，不是运行时 kernel。
- **Audio VAE encoder 未执行**：T2V 只跑 BigVGAN decoder；`pre_block`/encoder 的 FP32 结论来自 header。
- **Ref2VA / image 变体未运行**：本文只分析 FL2VA 的 5 个文件；Ref2VA 的结构同源但未在本次 trace 中出现。
- **GPU 差异**：参考 trace 在 NVIDIA B200（sm100）上采集；本文的格式结论（int8 W8A8、nvfp4 存储量化）是格式/代码层面的，不依赖具体 GPU，但具体 kernel 名（`cutlass3x_sm100_*`、`cutlass::arch::Sm80` 模板）来自那台机器。

### 12.4 本次复核修正的内部数字

除了 §12.2 的跨文档修正，本次逐表复算还修正了以下内部不一致（均为算术/归属问题，不改变模型结论）：

| 位置 | 初版 | 修正后 | 原因 |
|---|---|---|---|
| §2.4 功能归类表 | 条件/输出投影 + AdaLN = 0.072 B / 0.36% | 条件/输出投影 = 0.029 B / 0.14%，AdaLN 0.0435 B 计入主 blocks | 初版把 AdaLN 同时计入 blocks 和最后一行，重复统计 |
| §3.2 层参数 | 每层 487.64 M，50 层 24.382 B | 每层 487.60 M，50 层 24.380 B | 24.382 B 包含 pre_quant side-car，与总参数 25.753 B 不一致 |
| §3.2 pre_quant 数量 | “全文件 100 个……也就是表里的 33,792 个元素” | 33,792 是每层元素数；全文件 100 个张量 / 1,689,600 个元素 | 表格按层统计，文件按全模型统计 |
| §2.5 fc1 scale 范围 | 0.0006..0.0071 | 2.37e-4..7.89e-3 | 初版误用了 qkv_proj 的实测范围 |
| §4.3 decoder bias/scale 行 | 0.0014 B / 2.9 MB | 0.0022 B / 8.85 MB（含 norm1/2/scale1/2） | 初版少算了 norm 和 per-channel scale |
| §6.3 LoRA 表 | qkv 516.10 M + 单独 token_refiner 37.62 M；fc1/fc2/out 为 52 模块合计 | 四类均按 52 个 block 合计：qkv 536.74 M、fc1 226.62 M、fc2 131.20 M、out 83.49 M | 初版混用了 50-block 与 52-block 统计，且 token_refiner 行与前四行重复 |
| §9 结论 | 视频 VAE 92.9% | 92.8% | 2.4172 / 2.6039 = 92.84% |

复核后各表合计：DiT 20,111,438,744；TE 25,753,095,920（不含 pre_quant side-car）；video VAE 2,603,871,080；audio VAE 151,287,320；LoRA 978,059,264。均与 `analyze_h3_safetensors.py` 的输出一致。

---

## 13. 量化误差实测对比：int8_convrot vs MXFP8

为了给 §2/§3 的精度描述一个可量化的参照，使用 ComfyUI 自己的 layout op 把量化权重反量化回 BF16，再与 `minimax_h3_fl2va_pruned_bf16.safetensors` 的同名权重逐元素比较。

### 13.1 方法

- int8_convrot：`comfy_kitchen.TensorWiseINT8Layout.Params(scale, convrot=True, convrot_groupsize=256)` +
  `QuantizedTensor.dequantize()`，走 `dequantize_int8_convrot_weight_dtype`（256 组 Hadamard 反旋转 + per-channel scale）。
- MXFP8：`comfy_kitchen.TensorCoreMXFP8Layout.Params(scale, ...)` + `QuantizedTensor.dequantize()`，
  走 `dequantize_mxfp8`（blocked E8M0 scale + E4M3 权重）。
- 指标：`rel_RMSE = RMSE(deq, bf16) / std(bf16)`；cosine 用 float64 计算
  （115M 元素的 float32 dot/norm 会给出 >1 的伪 cosine）。
- 脚本：`validate_h3_int8_convrot.py`、`validate_h3_mxfp8.py`。

### 13.2 结果（4 个代表层，同一 BF16 base）

| 模块 | shape | int8_convrot rel_RMSE | MXFP8 rel_RMSE |
|---|---|---|---|
| `blocks.0.attn.qkv_proj` | 21504×5376 | **0.8985%** | 2.6606% |
| `blocks.0.attn.out_proj` | 5376×7168 | **1.2113%** | 2.6611% |
| `blocks.0.mlp.fc1` | 28672×5376 | **0.8973%** | 2.6608% |
| `blocks.49.mlp.fc2` | 5376×14336 | **0.9640%** | 2.6620% |

int8_convrot 的权重 RMSE 约为 MXFP8 的 1/2.2–1/3。差异来源：

- int8 每个输出通道一个 F32 scale，且 256 组 Hadamard 旋转把 outlier 分散；
- MXFP8 每 32 个权重一个 E8M0（2 的幂）scale，E4M3 只有 3 位 mantissa，单值相对精度更低（block 更细但精度更低）。

### 13.3 边界

- 这是**权重级**误差，不是端到端视频质量；不能直接由 rel_RMSE 推断生成质量。
- 只测了 4 个代表层；200 个模块的分布可能略有差异。
- MXFP8 数值来自 auto-round 0.17.0 `quant_mx_opt_rtn` 的导出（`minimax_h3_fl2va_pruned_mxfp8.safetensors`），
  int8 数值来自 Comfy-Org 的 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`。
