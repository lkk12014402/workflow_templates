#!/usr/bin/env python3
"""Quantize the Comfy-Org MiniMax-H3 pruned DiT (BF16 single file) to MXFP8
with the auto-round repository at /home/kaokaolv/comfyui_h3/auto-round.

Target modules are exactly the 200 modules that
`minimax_h3_fl2va_pruned_int8_convrot.safetensors` quantizes:

    blocks.{0..49}.attn.qkv_proj
    blocks.{0..49}.attn.out_proj
    blocks.{0..49}.mlp.fc1
    blocks.{0..49}.mlp.fc2

Everything else (token_refiner, condition_proj, AdaLN, patch/final heads,
norms, table, rope) is copied unchanged.

Output is a ComfyUI-loadable single-file checkpoint:
    <module>.weight        float8_e4m3fn  [N, K]
    <module>.weight_scale  uint8 E8M0 exponents [N, K/32], cuBLAS blocked layout
    <module>.comfy_quant   uint8 JSON {"format": "mxfp8"}

Environment:
    AUTO_ROUND_LIGHT=1 AUTO_ROUND_REPO=/home/kaokaolv/comfyui_h3/auto-round \
    PYTHONPATH=/home/kaokaolv/comfyui_h3/auto-round \
    /home/kaokaolv/.venv-aqa-gpu/bin/python quantize_h3_mxfp8_autoround.py ...

    AUTO_ROUND_LIGHT=1 bypasses auto_round/__init__.py (which pulls in
    transformers/datasets and took >240 s on this shared FS); the light path
    imports only auto_round.data_type.mxfp (~40 s).

The auto-round repo is 0.17.0. It is not installed in the ComfyUI .venv, so
either add it to PYTHONPATH (as above) or `pip install --no-build-isolation -e`
it into the environment you use. The script only needs torch + auto-round;
it embeds ComfyUI's `to_blocked` so it does not import ComfyUI.

Notes:
- auto-round's model-free MXFP8 writing convention stores a row-major uint8
  scale [N, K/32]. ComfyUI's TensorCoreMXFP8Layout expects the cuBLAS blocked
  (swizzled) layout, so the scale is passed through `to_blocked(..., flatten=False)`.
- All 200 target shapes are multiples of 32 (and N of 128, K/32 of 4), so no
  padding is needed. The script asserts this.
- The MXFP8 fast path in ComfyUI/comfy_kitchen requires SM >= 10 (Blackwell).
  On older GPUs use `--full-precision-mm` to write
  `{"format":"mxfp8","full_precision_matrix_mult":true}` instead, which makes
  ComfyUI dequantize to the compute dtype and use a plain F.linear.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# auto-round import (repo at /home/kaokaolv/comfyui_h3/auto-round)
# ---------------------------------------------------------------------------
_BACKEND = None


def _maybe_light_package():
    """Optionally bypass auto_round/__init__.py (heavy transformers imports).

    Set AUTO_ROUND_LIGHT=1 and AUTO_ROUND_REPO=/path/to/auto-round to import
    auto_round.data_type.mxfp directly. This is what the slow shared-FS
    environment needs; a normal pip install works without it.
    """
    if os.environ.get("AUTO_ROUND_LIGHT") != "1":
        return
    import types

    repo = os.environ.get("AUTO_ROUND_REPO", "/home/kaokaolv/comfyui_h3/auto-round")
    pkg_path = os.path.join(repo, "auto_round")
    if "auto_round" not in sys.modules and os.path.isdir(pkg_path):
        pkg = types.ModuleType("auto_round")
        pkg.__path__ = [pkg_path]
        pkg.__package__ = "auto_round"
        sys.modules["auto_round"] = pkg


def _load_backend():
    """Load auto-round's MXFP8 quantizer.

    Preferred: the light path (auto_round.data_type.mxfp), which only needs
    torch and avoids transformers/accelerate/datasets. Fallback: the model-free
    layer helper, which returns already-packed weight/scale tensors.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    _maybe_light_package()
    try:
        from auto_round.data_type.mxfp import quant_mx, quant_mx_opt_rtn

        _BACKEND = ("light", (quant_mx, quant_mx_opt_rtn))
    except ImportError:
        try:
            from auto_round.utils.model_free_utils import _quantize_weight_mxfp
        except ImportError as exc:  # pragma: no cover - environment hint
            raise SystemExit(
                "Cannot import auto_round. Run with\n"
                "  PYTHONPATH=/home/kaokaolv/comfyui_h3/auto-round \\\n"
                "  /home/kaokaolv/.venv-aqa-gpu/bin/python quantize_h3_mxfp8_autoround.py ...\n"
                "or install the repo editable into your environment:\n"
                "  pip install --no-build-isolation -e /home/kaokaolv/comfyui_h3/auto-round\n"
                f"Original error: {exc}"
            )
        _BACKEND = ("model_free", _quantize_weight_mxfp)
    return _BACKEND


def _quantize_light(weight: torch.Tensor, device: str, disable_opt_rtn: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 QDQ via auto-round's data_type.mxfp, packed like QuantLinear.pack.

    quant_mx / quant_mx_opt_rtn return (dequantized_weight, shared_exp, None).
    QuantLinear.pack stores weight = dequant / 2**shared_exp (float8_e4m3fn)
    and scale = (shared_exp + 127).clamp(0, 254) (uint8 E8M0).
    """
    kind, funcs = _load_backend()
    if kind != "light":
        raise RuntimeError("light backend not loaded")
    quant_mx, quant_mx_opt_rtn = funcs
    func = quant_mx if disable_opt_rtn else quant_mx_opt_rtn
    qdq, shared_exp, _ = func(weight.to(device=device, dtype=torch.float32), bits=8, group_size=32, data_type="mx_fp8")
    n, k = weight.shape
    shared_exp = shared_exp.reshape(n, k // 32)
    scale = torch.pow(2.0, shared_exp)
    fp8 = (qdq.reshape(n, k // 32, 32) / scale.unsqueeze(-1)).reshape(n, k).to(torch.float8_e4m3fn).cpu()
    e8m0 = (shared_exp + 127).clamp(0, 254).to(torch.uint8).cpu()
    return fp8, e8m0


def _quantize_model_free(
    weight: torch.Tensor, module: str, device: str, disable_opt_rtn: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    kind, fn = _load_backend()
    if kind != "model_free":
        raise RuntimeError("model_free backend not loaded")
    out = fn(
        weight,
        module,
        bits=8,
        group_size=32,
        data_type="mx_fp8",
        device=device,
        disable_opt_rtn=disable_opt_rtn,
    )
    return out[f"{module}.weight"], out[f"{module}.weight_scale"]


# ---------------------------------------------------------------------------
# cuBLAS blocked-scale layout, identical to comfy_kitchen.float_utils.to_blocked
# (copied so the script does not need ComfyUI on PYTHONPATH)
# ---------------------------------------------------------------------------
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """Rearrange [H, W] scales into the cuBLAS 128x4 tiled layout."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


# ---------------------------------------------------------------------------
# safetensors header helpers
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def target_modules_from_int8(path: str) -> set[str]:
    """Return the module names flagged by comfy_quant in the int8 checkpoint."""
    hdr = read_header(path)
    mods = {k[: -len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant")}
    if not mods:
        raise ValueError(f"No comfy_quant modules found in {path}")
    return mods


def default_target_modules(blocks: int = 50) -> set[str]:
    mods = set()
    for i in range(blocks):
        mods.add(f"blocks.{i}.attn.qkv_proj")
        mods.add(f"blocks.{i}.attn.out_proj")
        mods.add(f"blocks.{i}.mlp.fc1")
        mods.add(f"blocks.{i}.mlp.fc2")
    return mods


def quantize_one(
    weight: torch.Tensor,
    module: str,
    device: str,
    disable_opt_rtn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (fp8_weight, blocked_e8m0_scale) for one 2-D weight."""
    if weight.ndim != 2:
        raise ValueError(f"{module}: expected 2-D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"{module}: expected BF16/FP16/FP32 source weight, got {weight.dtype}")
    n, k = weight.shape
    if n % 32 or k % 32:
        raise ValueError(f"{module}: shape {tuple(weight.shape)} is not a multiple of 32")
    if n % 128 or (k // 32) % 4:
        raise ValueError(
            f"{module}: shape {tuple(weight.shape)} needs padding for the blocked scale layout"
        )

    kind, _ = _load_backend()
    if kind == "light":
        w, scale = _quantize_light(weight, device, disable_opt_rtn)
    else:
        w, scale = _quantize_model_free(weight, module, device, disable_opt_rtn)
    # scale is uint8 [N, K/32], row-major
    if w.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{module}: auto-round returned {w.dtype}, expected float8_e4m3fn")
    if scale.dtype != torch.uint8:
        raise TypeError(f"{module}: auto-round returned scale {scale.dtype}, expected uint8")
    if tuple(scale.shape) != (n, k // 32):
        raise ValueError(f"{module}: scale shape {tuple(scale.shape)} != {(n, k // 32)}")

    # E8M0 0xFF is NaN; ComfyUI clamps to 0xFE. auto-round clamps to 255, so
    # clamp once more before writing the ComfyUI layout.
    scale = scale.clamp_(0, 254)
    scale_blocked = to_blocked(scale, flatten=False)
    return w, scale_blocked


def parse_blocks(spec: str | None, default: int = 50) -> range:
    if not spec:
        return range(default)
    if "-" in spec:
        a, b = spec.split("-", 1)
        return range(int(a), int(b) + 1)
    return range(int(spec), int(spec) + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src",
        default="ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        help="BF16 pruned DiT (Comfy-Org single file)",
    )
    ap.add_argument(
        "--dst",
        default="ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_mxfp8.safetensors",
        help="output MXFP8 single file",
    )
    ap.add_argument(
        "--reference-int8",
        default="ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        help="read the target module list from this checkpoint's comfy_quant; pass '' to use the default pattern",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--disable-opt-rtn", action="store_true", help="plain RTN instead of auto-round optimized RTN")
    ap.add_argument(
        "--full-precision-mm",
        action="store_true",
        help='write {"format":"mxfp8","full_precision_matrix_mult":true} (no MXFP8 tensor-core path)',
    )
    ap.add_argument("--blocks", default=None, help="e.g. 0-9 for a quick subset (default: all 50)")
    ap.add_argument("--dry-run", action="store_true", help="only print the target list and exit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst) and not args.overwrite:
        raise SystemExit(f"{args.dst} exists; pass --overwrite to replace it")

    hdr = read_header(args.src)
    if args.reference_int8:
        targets = target_modules_from_int8(args.reference_int8)
        print(f"target modules from {args.reference_int8}: {len(targets)}")
    else:
        targets = default_target_modules()
        print(f"target modules from default pattern: {len(targets)}")

    blocks = parse_blocks(args.blocks)
    allowed = {m for m in targets if m.startswith(tuple(f"blocks.{i}." for i in blocks))}
    if not allowed:
        raise SystemExit("no target modules in the selected block range")

    # Sanity: every target weight exists in the source header.
    missing = [f"{m}.weight" for m in allowed if f"{m}.weight" not in hdr]
    if missing:
        raise SystemExit(f"{len(missing)} target weights missing from source, e.g. {missing[:3]}")
    print(f"blocks {blocks.start}-{blocks.stop - 1}: {len(allowed)} modules to quantize")
    if args.dry_run:
        for m in sorted(allowed):
            print(m)
        return 0

    from safetensors import safe_open
    from safetensors.torch import save_file

    qconf = {"format": "mxfp8"}
    if args.full_precision_mm:
        qconf["full_precision_matrix_mult"] = True
    qconf_bytes = torch.tensor(list(json.dumps(qconf).encode("utf-8")), dtype=torch.uint8)

    out: dict[str, torch.Tensor] = {}
    t0 = time.time()
    done = 0
    total_targets = len(allowed)

    with safe_open(args.src, framework="pt", device="cpu") as f:
        keys = list(hdr)
        for idx, key in enumerate(keys, 1):
            module = key[: -len(".weight")] if key.endswith(".weight") else None
            if module in allowed:
                w = f.get_tensor(key)
                t1 = time.time()
                qw, qs = quantize_one(w, module, args.device, args.disable_opt_rtn)
                out[f"{module}.weight"] = qw
                out[f"{module}.weight_scale"] = qs
                out[f"{module}.comfy_quant"] = qconf_bytes.clone()
                done += 1
                print(
                    f"[{done}/{total_targets}] {module} {tuple(w.shape)} "
                    f"-> fp8 + e8m0 ({time.time() - t1:.1f}s, total {time.time() - t0:.0f}s)"
                )
                del w, qw, qs
            else:
                out[key] = f.get_tensor(key)
            if idx % 200 == 0:
                print(f"  ... {idx}/{len(keys)} tensors, {time.time() - t0:.0f}s")

    print(f"saving {len(out)} tensors to {args.dst}")
    save_file(out, args.dst)
    size = os.path.getsize(args.dst)
    print(f"done: {args.dst} ({size / 1e9:.3f} GB), {done} MXFP8 modules")
    if not args.full_precision_mm:
        print(
            "note: MXFP8 tensor-core execution requires SM >= 10 (Blackwell); "
            "on older GPUs re-run with --full-precision-mm or load with a dequant fallback."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
