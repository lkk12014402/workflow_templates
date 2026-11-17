
### comfyui workflow: video_minimax_h3_t2v.json

1. 整个flow的执行逻辑

2. flow里涉及到的模型，以及源模型或bf16模型，以及模型简介


### 函数调用逻辑细节

分别介绍每个模型的函数调用栈，用到的kernel，dense linear的计算类型或kernel，attention以及计算类型或kernel


### 支持video_minimax_h3_t2v.json用源模型对应的mxfp4/mxfp8量化模型

#### 重新量化策略
1. 量化规则
int8相关的模型，用源模型量化为mxfp8
nvfp4相关的模型，用源模型量化为mxfp4
bf16/fp2相关的模型保持不变

2. 量化的方式
用auto-round量化，初版，采用auto-round的model-free路径，对单独的safetensor文件做对应的mxfp8/mxfp4量化。后期再用auto-round内置的优化方法（tuning/awq/scale search)

#### comfyui推理策略

1. 采用proposal comfyui_h3_mxfp8_mxfp4_proposal.md: 先走flashinfer相关的kernel，支持dense W8A8/W4A4。先在B200上跑通验证。后期也可以接入xpu对应的kernel

2. 走vllm-omni组件，你调研过


对比二者的差异



