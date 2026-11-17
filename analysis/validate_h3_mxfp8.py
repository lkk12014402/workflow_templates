#!/usr/bin/env python3
"""Validate a ComfyUI MXFP8 checkpoint produced by quantize_h3_mxfp8_autoround.py.

Checks:
  * tensor count / dtype census of the output header;
  * for a few target layers, rebuild a ComfyUI TensorCoreMXFP8Layout QuantizedTensor
    and dequantize it on GPU;
  * compare the dequantized weight with the BF16 base weight (RMSE, relative RMSE).

Run with the ComfyUI .venv (needs comfy_kitchen + torch):
    .venv/bin/python validate_h3_mxfp8.py \
        --mxfp8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors \
        --base  ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter

import torch


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mxfp8", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument(
        "--modules",
        nargs="*",
        default=["blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1", "blocks.49.mlp.fc2"],
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    from safetensors import safe_open
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.mxfp8 import TensorCoreMXFP8Layout

    hdr = read_header(args.mxfp8)
    dtypes = Counter(v["dtype"] for v in hdr.values())
    print(f"tensors: {len(hdr)}")
    print("dtypes:", dict(dtypes))
    qmods = {k[: -len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant")}
    print(f"comfy_quant modules: {len(qmods)}")
    if len(qmods) != 200:
        print(f"WARNING: expected 200 MXFP8 modules, got {len(qmods)}")

    with safe_open(args.mxfp8, framework="pt", device="cpu") as fq, safe_open(
        args.base, framework="pt", device="cpu"
    ) as fb:
        for mod in args.modules:
            w = fq.get_tensor(f"{mod}.weight")
            s = fq.get_tensor(f"{mod}.weight_scale")
            ref = fb.get_tensor(f"{mod}.weight")
            qconf = fq.get_tensor(f"{mod}.comfy_quant").numpy().tobytes().decode()
            params = TensorCoreMXFP8Layout.Params(
                scale=s.view(torch.float8_e8m0fnu).to(args.device),
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(ref.shape),
            )
            qt = QuantizedTensor(w.to(args.device), "TensorCoreMXFP8Layout", params)
            deq = qt.dequantize().float().cpu()
            r = ref.float()
            rmse = (deq - r).pow(2).mean().sqrt().item()
            rel = rmse / r.std().item()
            print(
                f"{mod}: weight={tuple(w.shape)} {w.dtype} scale={tuple(s.shape)} {s.dtype} "
                f"conf={qconf} rmse={rmse:.6f} rel_rmse={rel:.4%}"
            )
            del w, s, ref, qt, deq, r
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
