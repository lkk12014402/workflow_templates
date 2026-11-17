#!/usr/bin/env python3
import csv
import gzip
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


root = Path(__file__).resolve().parent
trace = root / "logs/cupti_h3_t2v"
analysis = trace / "analysis"
manifest = json.loads((analysis / "manifest.json").read_text())
conn = sqlite3.connect(trace / "h3_t2v_cold.sqlite")
assert manifest["kernel_count"] == 95226 == conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
assert manifest["runtime_api_count"] == 421188 == conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
assert manifest["memcpy_count"] == 5596 == conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
assert manifest["memcpy_bytes"] == 46506971601 == conn.execute("SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
assert manifest["distinct_kernels"] == 155

ranges = list(csv.DictReader((analysis / "range_summary.csv").open()))
assert sum(row["label"].startswith("H3.dit.step.") for row in ranges) == 20
assert sum(row["label"].startswith("H3.video_vae.vit.") for row in ranges) == 42
assert len(ranges) == 67

phase = list(csv.DictReader((analysis / "phase_kernel_summary.csv").open()))


def count_if(group, predicate):
    return sum(int(row["count"]) for row in phase if row["range_group"] == group and predicate(row["kernel"]))


assert count_if("H3.dit.all_steps", lambda name: "pytorch_flash::flash_fwd_kernel" in name) == 1000
assert count_if("H3.dit.all_steps", lambda name: "cutlass::Kernel2" in name) == 4020
assert count_if("H3.dit.all_steps", lambda name: "cutlass::Kernel2" in name and "signed char" in name) == 4000
assert count_if("H3.dit.all_steps", lambda name: "cutlass::Kernel2" in name and "signed char" not in name) == 20
assert count_if("H3.dit.all_steps", lambda name: "quantize_int8_rowwise_convrot64" in name) == 4000
assert count_if("H3.text.qwen", lambda name: "dequantize_nvfp4_kernel" in name) == 350
assert count_if("H3.text.qwen", lambda name: "cutlass3x_sm100_simt_sgemm" in name) == 350
assert count_if("H3.text.qwen", lambda name: "fmha_cutlassF_f32" in name) == 50
assert count_if("H3.video_vae.all_vit", lambda name: "cutlass::Kernel2" in name) == 6048
assert count_if("H3.video_vae.all_vit", lambda name: "qk_int_sv_i8_attn_kernel" in name) == 1512
assert count_if("H3.video_vae.all_vit", lambda name: "quantize_int8_rowwise_convrot64" in name) == 6048
assert count_if("H3.audio_vae.decode", lambda name: "nvfp4" in name) == 0
assert count_if("H3.audio_vae.decode", lambda name: "int8" in name) == 0

aux = Counter()
graph = Counter()
with gzip.open(analysis / "kernel_events.csv.gz", "rt") as source:
    for row in csv.DictReader(source):
        if row["range_group"] == "H3.video_vae.all_vit" and "cutlass::Kernel2" in row["name"]:
            aux["auxload" if "VisitorAuxLoad" in row["name"] else "no_auxload"] += 1
        graph[bool(row["graphNodeId"])] += 1
assert aux == {"no_auxload": 3024, "auxload": 3024}, aux
assert graph == {False: 95226}, graph

global_rows = list(csv.DictReader((analysis / "global_kernel_summary.csv").open()))
assert sum(int(row["count"]) for row in global_rows) == 95226
assert not any("scaled_mm_nvfp4" in row["kernel"] or "cublas_gemm_blockwise_fp4" in row["kernel"] for row in global_rows)

main_text = (root / "comfyui_h3_t2v_kernel_analysis.md").read_text()
report_text = (root / "cupti_h3_t2v_kernel_report.md").read_text()
assert main_text.count("```") % 2 == 0
assert report_text.count("```") % 2 == 0
assert len(re.findall(r"^## \d+\.", main_text, re.M)) == 18
assert "cupti_h3_t2v_kernel_report.md" in main_text
assert report_text.startswith("# MiniMax H3 T2V：CUPTI 逐 Kernel 实测报告")
for reference in [
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
    "grid=(124,1,56)",
    "grid=(7,64,1)",
    "grid=(15,32,4)",
    "M_batch4 = 4×1797 = 7188",
    "cutlass_80_simt_sgemm",
    "cupti_h3_t2v_kernel_report.md",
    "comfy_kitchen/backends/cuda/_C.abi3.so",
]:
    assert reference in main_text, reference
assert "host API  further" not in main_text
for relative in [
    "logs/cupti_h3_t2v/h3_t2v_cold.nsys-rep",
    "logs/cupti_h3_t2v/h3_t2v_cold.sqlite",
    "logs/cupti_h3_t2v/analysis/nvtx_kern_sum.txt",
    "logs/cupti_h3_t2v/output/video/MiniMax_H3_00001_.mp4",
]:
    assert (root / relative).is_file(), relative

print("all_cupti_evidence_checks_passed")
print(json.dumps({
    "kernels": 95226,
    "distinct_kernels": 155,
    "nvtx_ranges": 67,
    "dit_steps": 20,
    "video_vit_calls": 42,
    "dit_cutlass_all": 4020,
    "dit_int8_gemm": 4000,
    "other_float_sgemm": 20,
    "vae_int8_gemm": 6048,
    "qwen_nvfp4_dequant": 350,
    "vae_auxload": dict(aux),
    "graph_nodes": 0,
}, indent=2))
