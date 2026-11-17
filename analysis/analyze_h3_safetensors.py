"""Structural / precision analysis of the Comfy-Org MiniMax-H3 single-file checkpoints.

Reads only the safetensors headers (plus the small comfy_quant metadata blobs),
so it never loads tensor payloads into memory.

Logical parameter accounting:
  * int8 weights (I8)              : 1 value per stored element
  * nvfp4 weights  (U8 packed)     : 2 values per stored byte
  * everything else                : 1 value per stored element
Quantization side-cars (weight_scale, weight_scale_2, pre_quant_scale,
comfy_quant, lora alpha) are reported separately, never as parameters.
"""

import json
import os
import struct
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "diT": "ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "TE": "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors",
    "lora": "ComfyUI/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}
SIDECAR_SUFFIXES = (".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".comfy_quant", ".alpha")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_start = 8 + n
    meta = hdr.pop("__metadata__", None)
    return hdr, meta, data_start


def read_blob(path, data_start, info):
    with open(path, "rb") as f:
        f.seek(data_start + info["data_offsets"][0])
        return f.read(info["data_offsets"][1] - info["data_offsets"][0])


def quants(path, hdr, data_start):
    out = {}
    for name, info in hdr.items():
        if name.endswith(".comfy_quant"):
            mod = name[: -len(".comfy_quant")]
            try:
                out[mod] = json.loads(read_blob(path, data_start, info))
            except Exception:
                out[mod] = {}
    return out


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def weight_params(mod, info, qinfo):
    n = numel(info["shape"])
    fmt = qinfo.get(mod, {}).get("format")
    if fmt in ("nvfp4", "mxfp4"):
        return n * 2
    if fmt in ("asym_w4a8_int8", "asym_w4a8_int8_lloyd"):
        return n * 2
    if fmt == "w6a8_int8":
        return n * 4 // 3
    return n


def load(key):
    path = os.path.join(ROOT, FILES[key])
    hdr, meta, data_start = read_header(path)
    qinfo = quants(path, hdr, data_start)
    return path, hdr, meta, qinfo


def classify(key, name):
    if name.endswith(".weight"):
        return "weight"
    for suf in SIDECAR_SUFFIXES:
        if name.endswith(suf):
            if suf == ".alpha" and key != "lora":
                return "aux"  # Snake alpha/beta are real parameters, not LoRA side-cars
            return suf[1:]
    return "aux"


def module_of(name):
    for suf in (".weight_scale_2", ".weight_scale", ".pre_quant_scale", ".comfy_quant", ".weight", ".bias", ".alpha"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def summary(key):
    path, hdr, meta, qinfo = load(key)
    size = os.path.getsize(path)
    groups = defaultdict(lambda: {"params": 0, "bytes": 0, "tensors": 0, "dtypes": Counter()})
    dtype_params = Counter()
    dtype_bytes = Counter()
    sidecar_params = Counter()
    total_params = 0

    for name, info in hdr.items():
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        mod = module_of(name)
        kind = classify(key, name)
        if kind == "weight":
            p = weight_params(mod, info, qinfo)
            total_params += p
            dtype_params[info["dtype"]] += p
            groups[mod]["dtypes"][info["dtype"]] += p
        elif kind in ("bias", "aux"):
            p = numel(info["shape"])
            total_params += p
            dtype_params[info["dtype"]] += p
            groups[mod]["dtypes"][info["dtype"]] += p
        else:
            sidecar_params[kind] += numel(info["shape"])
        dtype_bytes[info["dtype"]] += nbytes
        groups[mod]["params"] += (weight_params(mod, info, qinfo) if kind == "weight"
                                  else numel(info["shape"]) if kind in ("bias", "aux") else 0)
        groups[mod]["bytes"] += nbytes
        groups[mod]["tensors"] += 1

    print("=" * 110)
    print(f"[{key}] {os.path.basename(path)}")
    print(f"  file size           : {size:,} B  ({size/1e9:.3f} GB / {size/2**30:.2f} GiB)")
    print(f"  stored tensors      : {len(hdr)}")
    print(f"  logical parameters  : {total_params:,}  ({total_params/1e9:.3f} B)")
    print(f"  scale/metadata cats : {dict(sidecar_params)}")
    print("  params by dtype     : " + ", ".join(
        f"{d} {n/1e9:.3f}B ({100*n/total_params:.2f}%)" for d, n in dtype_params.most_common()))
    print("  bytes by dtype      : " + ", ".join(
        f"{d} {n/1e9:.3f}GB ({100*n/size:.2f}%)" for d, n in dtype_bytes.most_common()))
    if qinfo:
        fmts = Counter(json.dumps(v, sort_keys=True) for v in qinfo.values())
        for blob, cnt in fmts.most_common():
            print(f"  quant {cnt:4}x {blob}")
    if meta:
        print(f"  metadata keys       : {list(meta)}")
    print()
    return dict(path=path, size=size, hdr=hdr, meta=meta, qinfo=qinfo,
                total_params=total_params, groups=groups, dtype_params=dtype_params,
                dtype_bytes=dtype_bytes, sidecar=sidecar_params)


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(FILES)):
        summary(k)
