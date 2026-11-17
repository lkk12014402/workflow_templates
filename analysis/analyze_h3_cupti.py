#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


KERNEL_FIELDS = [
    "start", "end", "deviceId", "streamId", "correlationId", "gridX", "gridY", "gridZ",
    "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory",
    "dynamicSharedMemory", "localMemoryPerThread", "localMemoryTotal", "graphNodeId",
    "shortName", "demangledName",
]


def table_exists(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def read_ranges(conn):
    ranges = []
    if not table_exists(conn, "NVTX_EVENTS"):
        return ranges
    for start, end, text in conn.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'H3.%' AND end IS NOT NULL"):
        ranges.append({"label": text.lstrip(":"), "start": start, "end": end, "duration_ns": end - start})
    return sorted(ranges, key=lambda item: (item["start"], -item["end"]))


def range_group(label):
    if label.startswith("H3.dit.step."):
        return "H3.dit.all_steps"
    if label.startswith("H3.video_vae.vit."):
        return "H3.video_vae.all_vit"
    return label


def new_stat():
    return {"count": 0, "total_ns": 0, "min_ns": None, "max_ns": 0}


def update(stat, duration):
    stat["count"] += 1
    stat["total_ns"] += duration
    stat["min_ns"] = duration if stat["min_ns"] is None else min(stat["min_ns"], duration)
    stat["max_ns"] = max(stat["max_ns"], duration)


def finish_stat(stat):
    return {**stat, "mean_ns": stat["total_ns"] / stat["count"] if stat["count"] else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.trace}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ranges = read_ranges(conn)

    def phase(timestamp):
        candidates = [item for item in ranges if item["start"] <= timestamp < item["end"]]
        if not candidates:
            return "unlabeled"
        return min(candidates, key=lambda item: item["duration_ns"])["label"]

    runtime = {}
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for row in conn.execute("SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            runtime[row[2]] = {"start": row[0], "end": row[1], "name_id": row[3]}

    string_values = {}
    for row in conn.execute("SELECT id, value FROM StringIds"):
        string_values[row[0]] = row[1]
    for value in runtime.values():
        value["name"] = string_values.get(value["name_id"], f"nameId:{value['name_id']}")

    api_stats = defaultdict(new_stat)
    for api in runtime.values():
        label = phase(api["start"])
        update(api_stats[(label, api["name"])], api["end"] - api["start"])

    global_stats = defaultdict(new_stat)
    phase_stats = defaultdict(new_stat)
    range_stats = defaultdict(new_stat)
    range_totals = {item["label"]: {"kernel_count": 0, "kernel_time_ns": 0, "memcpy_count": 0, "memcpy_bytes": 0, "memcpy_time_ns": 0, "memset_count": 0, "memset_bytes": 0} for item in ranges}
    kernel_count = 0
    kernel_time = 0
    events_path = args.out / "kernel_events.csv.gz"
    event_columns = [
        "range", "range_group", "start", "end", "duration_ns", "name", "correlationId",
        "deviceId", "streamId", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
        "registersPerThread", "staticSharedMemory", "dynamicSharedMemory", "localMemoryPerThread",
        "localMemoryTotal", "graphNodeId", "api_name", "api_start",
    ]

    with gzip.open(events_path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=event_columns)
        writer.writeheader()
        query = f"SELECT {', '.join(KERNEL_FIELDS)} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        for row in conn.execute(query):
            item = dict(row)
            duration = item["end"] - item["start"]
            api = runtime.get(item["correlationId"])
            timestamp = api["start"] if api is not None else item["start"]
            label = phase(timestamp)
            group = range_group(label)
            name = string_values.get(item["demangledName"]) or string_values.get(item["shortName"]) or f"nameId:{item['demangledName']}"
            kernel_count += 1
            kernel_time += duration
            update(global_stats[name], duration)
            update(phase_stats[(group, name)], duration)
            update(range_stats[(label, name)], duration)
            if label in range_totals:
                range_totals[label]["kernel_count"] += 1
                range_totals[label]["kernel_time_ns"] += duration
            event = {key: item.get(key) for key in KERNEL_FIELDS if key not in ("shortName", "demangledName")}
            event.update({
                "range": label, "range_group": group, "duration_ns": duration, "name": name,
                "api_name": api["name"] if api is not None else None,
                "api_start": api["start"] if api is not None else None,
            })
            writer.writerow(event)

    memcpy_count = 0
    memcpy_bytes = 0
    memcpy_time_ns = 0
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for row in conn.execute("SELECT start, end, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            api = runtime.get(row[2])
            timestamp = api["start"] if api is not None else row[0]
            label = phase(timestamp)
            memcpy_count += 1
            memcpy_bytes += row[3]
            memcpy_time_ns += row[1] - row[0]
            if label in range_totals:
                range_totals[label]["memcpy_count"] += 1
                range_totals[label]["memcpy_bytes"] += row[3]
                range_totals[label]["memcpy_time_ns"] += row[1] - row[0]

    memset_count = 0
    memset_bytes = 0
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for row in conn.execute("SELECT start, end, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            api = runtime.get(row[2])
            timestamp = api["start"] if api is not None else row[0]
            label = phase(timestamp)
            memset_count += 1
            memset_bytes += row[3]
            if label in range_totals:
                range_totals[label]["memset_count"] += 1
                range_totals[label]["memset_bytes"] += row[3]

    def write_summary(path, rows, fieldnames):
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    global_rows = [{"kernel": key, **finish_stat(value)} for key, value in global_stats.items()]
    global_rows.sort(key=lambda item: item["total_ns"], reverse=True)
    write_summary(args.out / "global_kernel_summary.csv", global_rows,
                  ["kernel", "count", "total_ns", "mean_ns", "min_ns", "max_ns"])

    phase_rows = [{"range_group": key[0], "kernel": key[1], **finish_stat(value)} for key, value in phase_stats.items()]
    phase_rows.sort(key=lambda item: (item["range_group"], -item["total_ns"]))
    write_summary(args.out / "phase_kernel_summary.csv", phase_rows,
                  ["range_group", "kernel", "count", "total_ns", "mean_ns", "min_ns", "max_ns"])

    range_rows = [{"range": key[0], "kernel": key[1], **finish_stat(value)} for key, value in range_stats.items()]
    range_rows.sort(key=lambda item: (item["range"], -item["total_ns"]))
    write_summary(args.out / "range_kernel_summary.csv", range_rows,
                  ["range", "kernel", "count", "total_ns", "mean_ns", "min_ns", "max_ns"])

    api_rows = [{"range": key[0], "api": key[1], **finish_stat(value)} for key, value in api_stats.items()]
    api_rows.sort(key=lambda item: (item["range"], -item["total_ns"]))
    write_summary(args.out / "runtime_api_summary.csv", api_rows,
                  ["range", "api", "count", "total_ns", "mean_ns", "min_ns", "max_ns"])

    ranges_out = []
    for item in ranges:
        label = item["label"]
        ranges_out.append({**item, **range_totals[label]})
    write_summary(args.out / "range_summary.csv", ranges_out,
                  ["label", "start", "end", "duration_ns", "kernel_count", "kernel_time_ns", "memcpy_count", "memcpy_bytes", "memcpy_time_ns", "memset_count", "memset_bytes"])

    run_result_path = args.trace.parent / "run_result.json"
    run_result = json.loads(run_result_path.read_text()) if run_result_path.exists() else None
    manifest = {
        "trace": str(args.trace),
        "trace_bytes": args.trace.stat().st_size,
        "analysis_time_unix": time.time(),
        "kernel_count": kernel_count,
        "kernel_time_ns": kernel_time,
        "distinct_kernels": len(global_stats),
        "nvtx_range_count": len(ranges),
        "memcpy_count": memcpy_count,
        "memcpy_bytes": memcpy_bytes,
        "memcpy_time_ns": memcpy_time_ns,
        "memset_count": memset_count,
        "memset_bytes": memset_bytes,
        "runtime_api_count": len(runtime),
        "run_result": run_result,
        "mapping": "CUDA runtime correlationId start timestamp, falling back to GPU kernel start; assign the smallest enclosing H3 NVTX range",
        "time_warning": "Kernel durations on different streams may overlap. Their sum is work, not phase wall time. NVTX duration is host range wall time and can include initialization, transfers, sync and CPU work.",
        "range_groups": "H3.dit.step.NN -> H3.dit.all_steps; H3.video_vae.vit.NN -> H3.video_vae.all_vit; other labels unchanged",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = ["# CUPTI analysis summary", "", f"- Trace: `{args.trace}`", f"- Kernel events: {kernel_count}", f"- Distinct kernels: {len(global_stats)}", f"- NVTX ranges: {len(ranges)}", "", "## Top kernels by summed GPU time", ""]
    for item in global_rows[:30]:
        lines.append(f"- `{item['kernel']}`: count={item['count']}, total_ms={item['total_ns']/1e6:.3f}, mean_us={item['mean_ns']/1e3:.3f}")
    lines += ["", "## NVTX ranges", ""]
    for item in ranges_out:
        lines.append(f"- `{item['label']}`: wall_ms={item['duration_ns']/1e6:.3f}, kernels={item['kernel_count']}, kernel_work_ms={item['kernel_time_ns']/1e6:.3f}, memcpy_MiB={item['memcpy_bytes']/2**20:.3f}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: manifest[key] for key in ("kernel_count", "distinct_kernels", "nvtx_range_count", "memcpy_count", "memcpy_bytes")}, indent=2))
    print("analysis_output", args.out)


if __name__ == "__main__":
    main()
