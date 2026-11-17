#!/usr/bin/env python3
import argparse
import asyncio
import concurrent.futures
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import aiohttp
import av
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

BASE = Path(__file__).resolve().parent
COMFY = BASE / "ComfyUI"
MODELS = COMFY / "models"
INPUT = COMFY / "input"
WORKFLOWS = BASE / "model_card_workflows"
RUNS = BASE / "logs" / "model_card_runs"
MANIFEST_PATH = RUNS / "download_manifest.json"
HOST = "127.0.0.1"
PORT = 8188
GPU_ID = int(os.environ.get("GPU_ID", "2"))

os.environ.setdefault("HF_HOME", str(BASE / ".cache" / "huggingface"))
os.environ.setdefault("TMPDIR", str(BASE / ".tmp"))

GROUPS = {
    "baseline": ["video_minimax_h3_t2v"],
    "i2v": [
        "video_minimax_h3_i2v",
        "video_minimax_h3_i2v_continuation",
        "image_minimax_h3_t2i",
    ],
    "pika": [
        "video_pika_h3_flex_t2v",
        "video_pika_h3_flex_fl2va",
        "video_pika_h3_flex_ref2va",
    ],
    "ref2va": [
        "video_minimax_h3_r2v",
        "video_minimax_h3_multiframe_reference",
    ],
    "controlnet": ["video_minimax_h3_fun_controlnet_union"],
    "p-h3": ["image_p_h3_t2i", "image_p_h3_image_edit"],
}

GROUP_MODELS = {
    "baseline": [],
    "i2v": ["vae/minimax_h3_video_vae_fp16.safetensors"],
    "pika": [
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "loras/pika_h3_flex_fl2va_v1_bf16.safetensors",
        "loras/pika_h3_flex_ref2va_v1_bf16.safetensors",
    ],
    "ref2va": [
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    ],
    "controlnet": [
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",
        "checkpoints/sdpose_wholebody_fp16.safetensors",
        "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors",
    ],
    "p-h3": [
        "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
        "text_encoders/p-h3-image_prompt_upsampler_w4a8.safetensors",
        "vae/p-h3-image_vae_int8_convrot.safetensors",
        "loras/p-h3-image_lora_bf16.safetensors",
    ],
}

H3_REVISION = "4b53a35191437de05370da548e59aeabe2a5a09a"
MODEL_SPECS = {
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"),
    "loras/pika_h3_flex_fl2va_v1_bf16.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "cc5453aeecd6189468409003fe4cf1b57e52ff3a830a0a154b87decbe07358b7"),
    "loras/pika_h3_flex_ref2va_v1_bf16.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "de3745ceb7d8d5c20898dd8fa3a05f1c0b00089ccd55297b2a8bab95c766b0e0"),
    "vae/minimax_h3_video_vae_fp16.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
    "model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors": (
        "Comfy-Org/MiniMax-H3", H3_REVISION, "9c645c0a308c8af361efd43b409710f6f8fec0db297c29503e141a84991fed0c"),
    "checkpoints/sdpose_wholebody_fp16.safetensors": (
        "Comfy-Org/SDPose", "acb43fbe8142b54d9ebd52b720b81dd4e14c8d81", "63d01f9a7494560693b24767f4469d59c9d3266b31ff0a253e74d1e611442721"),
    "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors": (
        "Comfy-Org/SDPose", "acb43fbe8142b54d9ebd52b720b81dd4e14c8d81", "581f9af9bbabb664d1891cbccd823308b176ecd409146f954dfa39af3bec2476"),
    "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors": (
        "PrunaAI/p-h3-image-comfy", "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543", "fb42c9ff0597e61729475d0b48b69b591ffe06d0b442a79dde22b14fadbe1c0f"),
    "text_encoders/p-h3-image_prompt_upsampler_w4a8.safetensors": (
        "PrunaAI/p-h3-image-comfy", "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543", "74b206c0e3133aa0d1983fd8658c1526a53863cdcca5184538388e927f4e989b"),
    "vae/p-h3-image_vae_int8_convrot.safetensors": (
        "PrunaAI/p-h3-image-comfy", "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543", "583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe"),
    "loras/p-h3-image_lora_bf16.safetensors": (
        "PrunaAI/p-h3-image-comfy", "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543", "0e9013ae2873e3415dcb7bb59a9677293bdbdb6f663e138a79bb9bdced00ea61"),
}

UNCONNECTED = object()


def request(method, path, body=None, timeout=120):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    payload = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"{method} {path}: HTTP {response.status}: {data.decode(errors='replace')}")
    return json.loads(data) if data else {}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest):
    RUNS.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "h3-batch-runner"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def template_commit():
    source = WORKFLOWS / "source.json"
    if source.exists():
        return json.loads(source.read_text())["commit"]
    data = json.loads(fetch_bytes("https://api.github.com/repos/Comfy-Org/workflow_templates/commits/main"))
    commit = data["sha"]
    WORKFLOWS.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"commit": commit}, indent=2) + "\n")
    return commit


def ensure_workflow(name):
    WORKFLOWS.mkdir(parents=True, exist_ok=True)
    path = WORKFLOWS / f"{name}.json"
    commit = template_commit()
    url = f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{commit}/templates/{name}.json"
    content = fetch_bytes(url)
    if path.exists() and path.read_bytes() == content:
        return path
    path.write_bytes(content)
    return path


def all_nodes(workflow):
    return list(workflow["nodes"]) + [
        node
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        for node in subgraph.get("nodes", [])
    ]


def ensure_assets(name, workflow):
    commit = template_commit()
    manifest = load_manifest()
    manifest.setdefault("assets", {})
    INPUT.mkdir(parents=True, exist_ok=True)
    assets = set()
    for node in all_nodes(workflow):
        if node["type"] not in ("LoadImage", "LoadVideo"):
            continue
        values = node.get("widgets_values_named", {})
        asset = values.get("image") or values.get("file")
        if asset:
            assets.add(asset)
    for asset in sorted(assets):
        content = fetch_bytes(f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{commit}/input/{asset}")
        path = INPUT / asset
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite different input asset: {path}")
        path.write_bytes(content)
        manifest["assets"][asset] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        print(f"asset {asset}: {len(content)} bytes", flush=True)
    save_manifest(manifest)


def download_models(group):
    manifest = load_manifest()
    manifest.setdefault("models", {})
    for filename in GROUP_MODELS[group]:
        repo, revision, expected = MODEL_SPECS[filename]
        path = MODELS / filename
        if path.exists() and manifest["models"].get(filename, {}).get("sha256") == expected:
            print(f"model {filename}: already verified", flush=True)
            continue
        if not path.exists():
            print(f"downloading {repo}/{filename}", flush=True)
            path = Path(hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                local_dir=str(MODELS),
            ))
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
        manifest["models"][filename] = {
            "repo": repo,
            "revision": revision,
            "sha256": actual,
            "size": path.stat().st_size,
        }
        save_manifest(manifest)
        print(f"model {filename}: SHA256 OK", flush=True)


def download_group(group):
    for name in GROUPS[group]:
        workflow_path = ensure_workflow(name)
        workflow = json.loads(workflow_path.read_text())
        ensure_assets(name, workflow)
    download_models(group)


class WorkflowConverter:
    def __init__(self, workflow, object_info):
        self.workflow = workflow
        self.object_info = object_info
        self.subgraphs = {
            subgraph["id"]: subgraph
            for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        }
        self.api = {}

    def scope(self, nodes, links):
        node_map = {node["id"]: node for node in nodes}
        link_map = {}
        for link in links:
            if isinstance(link, dict):
                link_map[link["id"]] = (link["origin_id"], link["origin_slot"])
            else:
                link_map[link[0]] = (link[1], link[2])
        return {"nodes": node_map, "links": link_map}

    def context(self, scope, prefix="", instance=None, parent=None, subgraph=None):
        return {
            "scope": scope,
            "prefix": prefix,
            "instance": instance,
            "parent": parent,
            "subgraph": subgraph,
        }

    def nested_context(self, ctx, instance, subgraph):
        return self.context(
            self.scope(subgraph["nodes"], subgraph.get("links", [])),
            prefix=f"{ctx['prefix']}{instance['id']}:",
            instance=instance,
            parent=ctx,
            subgraph=subgraph,
        )

    def external_value(self, ctx, slot):
        instance = ctx["instance"]
        subgraph = ctx["subgraph"]
        input_name = subgraph["inputs"][slot]["name"]
        instance_input = next((item for item in instance.get("inputs", []) if item["name"] == input_name), None)
        if instance_input is not None and instance_input.get("link") is not None:
            return self.resolve_link(ctx["parent"], instance_input["link"])
        values = instance.get("widgets_values_named", {})
        return values.get(input_name, UNCONNECTED)

    def resolve_link(self, ctx, link_id):
        origin_id, origin_slot = ctx["scope"]["links"][link_id]
        return self.resolve_output(ctx, origin_id, origin_slot)

    def resolve_output(self, ctx, node_id, output_slot):
        if node_id == -10:
            return self.external_value(ctx, output_slot)
        node = ctx["scope"]["nodes"][node_id]
        if node.get("mode", 0) == 4:
            output_type = node.get("outputs", [])[output_slot]["type"]
            candidates = [item for item in node.get("inputs", []) if item.get("link") is not None]
            matching = next((item for item in candidates if item["type"] == output_type), candidates[0])
            return self.resolve_link(ctx, matching["link"])
        if node["type"] in self.subgraphs:
            subgraph = self.subgraphs[node["type"]]
            nested = self.nested_context(ctx, node, subgraph)
            for link in subgraph.get("links", []):
                if link["target_id"] == -20 and link["target_slot"] == output_slot:
                    return self.resolve_output(nested, link["origin_id"], link["origin_slot"])
            raise RuntimeError(f"Unmapped subgraph output {node_id}:{output_slot}")
        return [f"{ctx['prefix']}{node_id}", output_slot]

    def convert_node(self, ctx, node):
        class_type = node["type"]
        info = self.object_info[class_type]
        inputs_schema = info.get("input", {})
        allowed = set(inputs_schema.get("required", {})) | set(inputs_schema.get("optional", {}))
        values = node.get("widgets_values_named", {})
        inputs = {
            key: value
            for key, value in values.items()
            if key in allowed or "." in key
        }
        if class_type == "SaveVideo":
            inputs.pop("codec", None)
            if "format" in inputs and "format.codec" not in inputs:
                inputs["format.codec"] = values.get("codec", "auto")
        for item in node.get("inputs", []):
            if item.get("link") is None:
                continue
            value = self.resolve_link(ctx, item["link"])
            if value is not UNCONNECTED:
                inputs[item["name"]] = value
        self.api[f"{ctx['prefix']}{node['id']}"] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    def convert_scope(self, ctx):
        for node in ctx["scope"]["nodes"].values():
            if node["type"] == "MarkdownNote" or node.get("mode", 0) == 4:
                continue
            if node["type"] in self.subgraphs:
                subgraph = self.subgraphs[node["type"]]
                self.convert_scope(self.nested_context(ctx, node, subgraph))
            else:
                self.convert_node(ctx, node)

    def prune(self):
        output_nodes = [
            node_id
            for node_id, node in self.api.items()
            if self.object_info[node["class_type"]].get("output_node")
        ]
        if not output_nodes:
            raise RuntimeError("Workflow has no output nodes")
        needed = set()

        def visit(node_id):
            if node_id in needed:
                return
            needed.add(node_id)
            for value in self.api[node_id]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in self.api:
                    visit(value[0])

        for node_id in output_nodes:
            visit(node_id)
        self.api = {node_id: self.api[node_id] for node_id in needed}

    def convert(self):
        top = self.context(self.scope(self.workflow["nodes"], self.workflow.get("links", [])))
        self.convert_scope(top)
        self.prune()
        return self.api


def nvml_used():
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    return int(output.strip())


class Monitor:
    def __init__(self, run_dir, client_id):
        self.run_dir = run_dir
        self.client_id = client_id
        self.started = time.time()
        self.samples = []
        self.events = []
        self.state = {"prompt_id": None, "current_node": None}
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.poll_thread = threading.Thread(target=self.poll_loop, daemon=True)
        self.ws_thread = threading.Thread(target=self.ws_loop, daemon=True)

    def start(self):
        self.poll_thread.start()
        self.ws_thread.start()
        self.ready.wait(15)

    def sample(self, label):
        try:
            memory = request("GET", "/h3/memory", timeout=30)
            used = nvml_used()
            sample = {
                "elapsed": time.time() - self.started,
                "label": label,
                "node": self.state["current_node"],
                "nvml_used_mib": used,
                "memory": memory,
            }
            with self.lock:
                self.samples.append(sample)
        except Exception as exc:
            with self.lock:
                self.samples.append({"elapsed": time.time() - self.started, "label": label, "error": str(exc)})

    def poll_loop(self):
        self.sample("baseline")
        while not self.stop_event.is_set():
            self.sample("poll")
            self.stop_event.wait(0.5)

    def ws_loop(self):
        asyncio.run(self.ws_main())

    async def ws_main(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"ws://{HOST}:{PORT}/ws?clientId={self.client_id}", heartbeat=30) as ws:
                    self.ready.set()
                    while not self.stop_event.is_set():
                        try:
                            message = await ws.receive(timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        data = event.get("data", {})
                        prompt_id = data.get("prompt_id")
                        if self.state["prompt_id"] and prompt_id != self.state["prompt_id"]:
                            continue
                        with self.lock:
                            self.events.append({"elapsed": time.time() - self.started, "event": event})
                        event_type = event.get("type")
                        if event_type == "executing":
                            self.state["current_node"] = data.get("node")
                            self.sample(f"executing:{data.get('node')}")
                        elif event_type == "executed":
                            self.sample(f"executed:{data.get('node')}")
                        elif event_type == "progress":
                            self.sample("progress")
                        elif event_type in ("execution_start", "execution_cached", "execution_success", "execution_error", "status"):
                            self.sample(event_type)
        except Exception as exc:
            with self.lock:
                self.events.append({"elapsed": time.time() - self.started, "error": str(exc)})
            self.ready.set()

    def stop(self):
        self.sample("final")
        self.stop_event.set()
        self.poll_thread.join(10)
        self.ws_thread.join(10)
        (self.run_dir / "memory_samples.json").write_text(json.dumps(self.samples, indent=2) + "\n")
        (self.run_dir / "events.json").write_text(json.dumps(self.events, indent=2) + "\n")


def output_files(item):
    files = []

    def collect(obj):
        if isinstance(obj, dict):
            if "filename" in obj and obj.get("type") == "output":
                files.append({key: obj.get(key) for key in ("filename", "subfolder", "type")})
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)

    collect(item.get("outputs", {}))
    unique = []
    seen = set()
    for file in files:
        key = (file["type"], file["subfolder"], file["filename"])
        if key not in seen:
            seen.add(key)
            unique.append(file)
    return unique


def validate_media(file):
    path = COMFY / "output" / (file["subfolder"] or "") / file["filename"]
    if not path.is_file():
        raise RuntimeError(f"Missing output file: {path}")
    report = {"path": str(path), "size": path.stat().st_size}
    suffix = path.suffix.lower()
    if suffix in (".mp4", ".mov", ".webm", ".mkv"):
        with av.open(str(path)) as container:
            report["duration"] = container.duration / 1e6 if container.duration else None
            report["streams"] = []
            for stream in container.streams:
                entry = {"type": stream.type, "codec": stream.codec_context.name}
                if stream.type == "video":
                    entry.update(width=stream.codec_context.width, height=stream.codec_context.height, fps=str(stream.average_rate))
                if stream.type == "audio":
                    entry.update(sample_rate=stream.codec_context.sample_rate, channels=stream.codec_context.channels)
                report["streams"].append(entry)
            video_frames = 0
            video_std = []
            for frame in container.decode(video=0):
                array = frame.to_ndarray(format="rgb24")
                video_frames += 1
                video_std.append(float(array.std()))
            report["decoded_video_frames"] = video_frames
            report["video_std_range"] = [min(video_std), max(video_std)]
            if not video_frames or max(video_std) <= 1:
                raise RuntimeError(f"Blank or invalid video: {path}")
        if any(stream["type"] == "audio" for stream in report["streams"]):
            with av.open(str(path)) as container:
                audio_max = 0.0
                audio_samples = 0
                for frame in container.decode(audio=0):
                    array = frame.to_ndarray().astype(np.float64)
                    audio_samples += frame.samples
                    audio_max = max(audio_max, float(np.abs(array).max()))
                report["decoded_audio_samples"] = audio_samples
                report["audio_max_abs"] = audio_max
                if not audio_samples or audio_max <= 1e-6:
                    raise RuntimeError(f"Silent or invalid audio: {path}")
    elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image.convert("RGB"))
            report.update(width=image.width, height=image.height, mode=image.mode, std=float(array.std()))
            if array.std() <= 1:
                raise RuntimeError(f"Blank image: {path}")
    else:
        raise RuntimeError(f"Unsupported media output: {path}")
    report["validation"] = "PASS"
    return report


def publish_workflow(name):
    source = WORKFLOWS / f"{name}.json"
    target_dir = COMFY / "user" / "default" / "workflows"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.json"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError(f"Refusing to overwrite different user workflow: {target}")
    shutil.copyfile(source, target)


def run_one(name):
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = {"workflow": name, "started": started, "status": "running"}
    monitor = None
    try:
        workflow_path = ensure_workflow(name)
        workflow = json.loads(workflow_path.read_text())
        object_info = request("GET", "/object_info", timeout=180)
        graph = WorkflowConverter(workflow, object_info).convert()
        (run_dir / "api_prompt.json").write_text(json.dumps({"prompt": graph}, indent=2, ensure_ascii=False) + "\n")
        request("POST", "/free", {"unload_models": True, "free_memory": True})
        baseline = request("POST", "/h3/memory/reset", {})
        (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        client_id = f"h3-batch-{uuid.uuid4()}"
        monitor = Monitor(run_dir, client_id)
        monitor.start()
        response = request("POST", "/prompt", {"prompt": graph, "client_id": client_id})
        prompt_id = response["prompt_id"]
        monitor.state["prompt_id"] = prompt_id
        result["prompt_id"] = prompt_id
        print(f"{name}: submitted {prompt_id}, {len(graph)} API nodes", flush=True)
        while True:
            history = request("GET", f"/history/{prompt_id}")
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    break
            time.sleep(2)
        (run_dir / "history.json").write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
        if status.get("status_str") != "success" or not status.get("completed"):
            raise RuntimeError(f"Execution failed: {json.dumps(status)}")
        files = output_files(item)
        if not files:
            raise RuntimeError("No output files")
        result["files"] = files
        result["media"] = [validate_media(file) for file in files]
        result["status"] = "success"
        publish_workflow(name)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"{name}: ERROR {exc}", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()
            if monitor.samples:
                valid = [sample for sample in monitor.samples if "memory" in sample]
                if valid:
                    final = valid[-1]["memory"]
                    result["peak"] = {
                        "nvml_used_mib_max": max(sample["nvml_used_mib"] for sample in valid),
                        "torch": final["torch"],
                        "models_final": final["models"],
                    }
        result["elapsed_seconds"] = time.time() - started
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        try:
            request("POST", "/free", {"unload_models": True, "free_memory": True})
        except Exception as exc:
            print(f"{name}: cleanup warning {exc}", flush=True)
    print(f"{name}: {result['status']}, {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def run_group(group):
    download_group(group)
    results = []
    for name in GROUPS[group]:
        results.append(run_one(name))
    summary = RUNS / f"summary_{group}.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if any(result["status"] != "success" for result in results):
        raise SystemExit(1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "run", "convert"))
    parser.add_argument("group", choices=tuple(GROUPS))
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.action == "download":
        download_group(args.group)
    elif args.action == "run":
        run_group(args.group)
    else:
        for name in GROUPS[args.group]:
            workflow = json.loads(ensure_workflow(name).read_text())
            object_info = request("GET", "/object_info", timeout=180)
            graph = WorkflowConverter(workflow, object_info).convert()
            run_dir = RUNS / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "api_prompt.json").write_text(json.dumps({"prompt": graph}, indent=2, ensure_ascii=False) + "\n")
            print(f"{name}: {len(graph)} API nodes")


if __name__ == "__main__":
    main()
