#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / "logs" / "model_card_runs"
GROUPS = ["video_minimax_h3_t2v", "i2v", "pika", "ref2va", "controlnet", "p-h3"]


def gib(value):
    return value / 2**30


def summarize_run(run_dir):
    result = json.loads((run_dir / "result.json").read_text())
    prompt_path = run_dir / "api_prompt.json"
    graph = json.loads(prompt_path.read_text())["prompt"] if prompt_path.exists() else {}
    samples_path = run_dir / "memory_samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.exists() else []
    events_path = run_dir / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    step_values = [
        event["event"]["data"].get("max")
        for event in events
        if event.get("event", {}).get("type") == "progress"
        and graph.get(event["event"]["data"].get("node"), {}).get("class_type") in ("SamplerCustomAdvanced", "KSampler")
    ]
    sampler_steps = max(step_values) if step_values else None
    valid = [sample for sample in samples if "memory" in sample]
    models = {}
    nodes = {}
    loaded_total_peak = 0
    weights_at_nvml_peak = 0
    other_at_nvml_peak_mib = 0
    if valid:
        nvml_peak_sample = max(valid, key=lambda sample: sample["nvml_used_mib"])
        weights_at_nvml_peak = sum(model["loaded_size"] for model in nvml_peak_sample["memory"]["models"])
        other_at_nvml_peak_mib = nvml_peak_sample["nvml_used_mib"] - weights_at_nvml_peak / 2**20
    for sample in valid:
        loaded_total_peak = max(loaded_total_peak, sum(model["loaded_size"] for model in sample["memory"]["models"]))
        elapsed = sample["elapsed"]
        node = sample.get("node") or "unassigned"
        node_data = nodes.setdefault(node, {
            "first_elapsed": elapsed,
            "last_elapsed": elapsed,
            "nvml_used_mib_max": 0,
            "allocated_max": 0,
            "reserved_max": 0,
            "samples": 0,
        })
        node_data["first_elapsed"] = min(node_data["first_elapsed"], elapsed)
        node_data["last_elapsed"] = max(node_data["last_elapsed"], elapsed)
        node_data["nvml_used_mib_max"] = max(node_data["nvml_used_mib_max"], sample["nvml_used_mib"])
        node_data["samples"] += 1
        for device in sample["memory"]["torch"].values():
            node_data["allocated_max"] = max(node_data["allocated_max"], device["allocated"])
            node_data["reserved_max"] = max(node_data["reserved_max"], device["reserved"])
        for model in sample["memory"]["models"]:
            key = f"{model['model_class']} patches={model['patch_count']}"
            current = models.setdefault(key, {
                "model_class": model["model_class"],
                "patcher_class": model["patcher_class"],
                "patch_count": model["patch_count"],
                "model_size": model["model_size"],
                "loaded_size_max": 0,
                "offloaded_size_min": model["offloaded_size"],
                "dynamic": model["dynamic"],
            })
            current["model_size"] = max(current["model_size"], model["model_size"])
            current["loaded_size_max"] = max(current["loaded_size_max"], model["loaded_size"])
            current["offloaded_size_min"] = min(current["offloaded_size_min"], model["offloaded_size"])
    labeled_nodes = {}
    for node, data in nodes.items():
        if node == "unassigned":
            label = node
        else:
            label = f"{node} ({graph.get(node, {}).get('class_type', 'unknown')})"
        labeled_nodes[label] = data
    return {
        "workflow": result["workflow"],
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak": result.get("peak", {}),
        "models": list(models.values()),
        "model_loaded_total_peak": loaded_total_peak,
        "weights_at_nvml_peak": weights_at_nvml_peak,
        "other_at_nvml_peak_mib": other_at_nvml_peak_mib,
        "nodes": labeled_nodes,
        "peak_node": max(labeled_nodes, key=lambda node: labeled_nodes[node]["nvml_used_mib_max"]) if labeled_nodes else None,
        "sampler_steps": sampler_steps,
        "media": result.get("media", []),
        "error": result.get("error"),
    }


def markdown(summary):
    lines = [
        "# MiniMax H3 显存与执行结果",
        "",
        "单位：GiB（除 NVML MiB 和时间外）。NVML 是整卡进程口径；loaded_size 是 ComfyUI 模型权重口径；节点峰值为 0.5 秒采样近似值。",
        "",
        "## 总览",
        "",
        "| Workflow | 状态 | 步数 | 峰值节点 | 用时(s) | NVML 峰值(MiB) | 权重 loaded 峰值 | 峰值时其他占用(MiB) | Torch allocated 峰值 | Torch reserved 峰值 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary:
        torch = run["peak"].get("torch", {}).get("cuda:0", {})
        lines.append(
            f"| {run['workflow']} | {run['status']} | {run['sampler_steps'] or '-'} | {run['peak_node'] or '-'} | {run['elapsed_seconds']:.1f} | "
            f"{run['peak'].get('nvml_used_mib_max', 0)} | {gib(run['model_loaded_total_peak']):.2f} | "
            f"{run['other_at_nvml_peak_mib']:.0f} | {gib(torch.get('max_allocated', 0)):.2f} | {gib(torch.get('max_reserved', 0)):.2f} |"
        )
    lines.extend(["", "## 输出与媒体验证", "", "| Workflow | 输出 | 验证 |", "|---|---|---|"])
    for run in summary:
        for media in run["media"]:
            lines.append(f"| {run['workflow']} | `{media['path']}` | {media['validation']} |")
        if not run["media"]:
            lines.append(f"| {run['workflow']} | 无 | {run['status']} |")
    lines.extend(["", "## 各模型模块", ""])
    for run in summary:
        lines.extend([
            f"### {run['workflow']}",
            "",
            "| 模型 | patch 数 | model_size | loaded_size 最大值 | offloaded 最小值 | dynamic |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for model in run["models"]:
            lines.append(
                f"| {model['model_class']} | {model['patch_count']} | {gib(model['model_size']):.2f} | "
                f"{gib(model['loaded_size_max']):.2f} | {gib(model['offloaded_size_min']):.2f} | {model['dynamic']} |"
            )
        if not run["models"]:
            lines.append("| 无采样数据 |  |  |  |  |  |")
        lines.extend([
            "",
            "节点阶段（当前节点标签来自 WebSocket executing 事件）：",
            "",
            "| 节点 | 起止采样时间(s) | NVML 峰值(MiB) | allocated 采样峰值 | reserved 采样峰值 | 样本数 |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for node, data in run["nodes"].items():
            lines.append(
                f"| {node} | {data['first_elapsed']:.1f}–{data['last_elapsed']:.1f} | {data['nvml_used_mib_max']} | "
                f"{gib(data['allocated_max']):.2f} | {gib(data['reserved_max']):.2f} | {data['samples']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    summary = []
    for group in GROUPS:
        if group == "video_minimax_h3_t2v":
            names = [group]
        else:
            summary_path = RUNS / f"summary_{group}.json"
            if summary_path.exists():
                names = [item["workflow"] for item in json.loads(summary_path.read_text())]
            else:
                names = []
        for name in names:
            run_dir = RUNS / name
            if (run_dir / "result.json").exists():
                summary.append(summarize_run(run_dir))
    for run_dir in sorted(RUNS.iterdir()):
        if run_dir.is_dir() and (run_dir / "result.json").exists() and run_dir.name not in {run["workflow"] for run in summary}:
            summary.append(summarize_run(run_dir))
    (RUNS / "vram_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (RUNS / "vram_report.md").write_text(markdown(summary))
    print(f"wrote {RUNS / 'vram_report.md'} for {len(summary)} workflows")


if __name__ == "__main__":
    main()
