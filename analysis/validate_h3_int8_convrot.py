#!/usr/bin/env python3
"""Validate the int8_convrot DiT checkpoint against the BF16 base.

Uses ComfyUI's own layout op:
    comfy_kitchen.TensorWiseINT8Layout.dequantize()
which calls dequantize_int8_convrot_weight_dtype (un-rotates the 256-group
Hadamard rotation) and per-output-channel scale.

Run with the ComfyUI .venv:
    .venv/bin/python validate_h3_int8_convrot.py \
        --int8 ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --base ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
        --device cuda:3
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
    ap.add_argument("--int8", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument(
        "--modules",
        nargs="*",
        default=[
            "blocks.0.attn.qkv_proj",
            "blocks.0.attn.out_proj",
            "blocks.0.mlp.fc1",
            "blocks.49.mlp.fc2",
        ],
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    from safetensors import safe_open
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    hdr = read_header(args.int8)
    dtypes = Counter(v["dtype"] for v in hdr.values())
    print(f"tensors: {len(hdr)}")
    print("dtypes:", dict(dtypes))
    qmods = {k[: -len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant")}
    print(f"comfy_quant modules: {len(qmods)}")

    with safe_open(args.int8, framework="pt", device="cpu") as fq, safe_open(
        args.base, framework="pt", device="cpu"
    ) as fb:
        for mod in args.modules:
            w = fq.get_tensor(f"{mod}.weight")
            s = fq.get_tensor(f"{mod}.weight_scale")
            ref = fb.get_tensor(f"{mod}.weight")
            qconf = fq.get_tensor(f"{mod}.comfy_quant").numpy().tobytes().decode()
            params = TensorWiseINT8Layout.Params(
                scale=s.to(args.device),
                orig_dtype=torch.bfloat16,
                orig_shape=tuple(ref.shape),
                is_weight=True,
                convrot=True,
                convrot_groupsize=256,
            )
            qt = QuantizedTensor(w.to(args.device), "TensorWiseINT8Layout", params)
            deq = qt.dequantize().float().cpu()
            r = ref.float()
            rmse = (deq - r).pow(2).mean().sqrt().item()
            rel = rmse / r.std().item()
            # FP32 dot/norm over 100M+ elements loses precision (ratio can exceed 1);
            # use float64 for the reported cosine.
            a, b = deq.flatten().double(), r.flatten().double()
            cos = (a @ b / (a.norm() * b.norm())).item()
            print(
                f"{mod}: weight={tuple(w.shape)} {w.dtype} scale={tuple(s.shape)} {s.dtype} "
                f"conf={qconf} rmse={rmse:.6f} rel_rmse={rel:.4%} cos={cos:.6f}"
            )
            del w, s, ref, qt, deq, r
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
