#!/usr/bin/env python3
import argparse
import http.client
import json
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent


def request(port, method, path, body=None, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"{method} {path}: HTTP {response.status}: {data.decode(errors='replace')}")
    return json.loads(data) if data else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--trace-base", default=str(BASE / "logs/cupti_h3_t2v/h3_t2v_cold"))
    args = parser.parse_args()
    out_dir = BASE / "logs/cupti_h3_t2v"
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + 180
    while True:
        try:
            status = request(args.port, "GET", "/h3/cupti/status", timeout=5)
            break
        except (OSError, RuntimeError):
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    print("profiler_status_before_start", status, flush=True)

    prompt = json.loads((BASE / "logs/model_card_runs/video_minimax_h3_t2v/api_prompt.json").read_text())["prompt"]
    started = time.monotonic()
    result = {"prompt_id": None, "success": False, "started_monotonic": started}
    prompt_id = None
    profiler_started = False
    try:
        result["profiler_start"] = request(args.port, "POST", "/h3/cupti/start")
        profiler_started = True
        response = request(args.port, "POST", "/prompt", {"prompt": prompt, "client_id": "h3-cupti"})
        prompt_id = response["prompt_id"]
        result["prompt_id"] = prompt_id
        result["post_elapsed"] = time.monotonic() - started
        print("prompt_submitted", prompt_id, "elapsed", result["post_elapsed"], flush=True)

        last_progress = -60
        while True:
            elapsed = time.monotonic() - started
            if elapsed > args.timeout:
                raise TimeoutError(f"prompt {prompt_id} exceeded {args.timeout}s")
            history = request(args.port, "GET", f"/history/{prompt_id}")
            if prompt_id in history:
                record = history[prompt_id]
                (out_dir / "history.json").write_text(json.dumps(record, indent=2) + "\n")
                status = record.get("status", {})
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    result["success"] = status.get("status_str") == "success"
                    result["history_status"] = status
                    if not result["success"]:
                        raise RuntimeError(f"prompt failed: {json.dumps(status)}")
                    break
            if elapsed - last_progress >= 30:
                print("profile_running", f"{elapsed:.1f}s", flush=True)
                last_progress = elapsed
            time.sleep(5)
    finally:
        if profiler_started:
            try:
                result["profiler_stop"] = request(args.port, "POST", "/h3/cupti/stop")
            except Exception as exc:
                result["profiler_stop_error"] = str(exc)
        result["elapsed_seconds"] = time.monotonic() - started
        (out_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("run_result", json.dumps(result), flush=True)

    for suffix in (".nsys-rep", ".sqlite"):
        path = Path(args.trace_base + suffix)
        wait_deadline = time.monotonic() + 180
        while not path.exists() and time.monotonic() < wait_deadline:
            time.sleep(2)
        if not path.exists():
            raise FileNotFoundError(f"trace file not generated: {path}")
        print("trace_file", str(path), path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
