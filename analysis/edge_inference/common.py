"""Shared paths, hashing and environment capture for the FT-0010 / EXP-0010 harness.

Nothing here touches torch/onnxruntime so it imports on the Windows host.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "edge_inference"
PROTOCOL_PATH = PKG / "protocol.json"

HF_REPO = "nvidia/PhysicalAI-SmartSpaces"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/MTMC_Tracking_2026"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}"


def cache_dir() -> Path:
    """Raw downloads (never committed). Override with FLEXTERN_EDGE_CACHE."""
    p = Path(os.environ.get("FLEXTERN_EDGE_CACHE", Path.home() / "flextern_edge"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def load_protocol() -> dict:
    with open(PROTOCOL_PATH, encoding="utf8") as f:
        return json.load(f)


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "analysis/edge_inference"))}


def environment() -> dict:
    """Software/hardware identity recorded with every run."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "git": git_info(),
    }
    # importlib.metadata, NOT import: importing torch would inflate the RSS of torch-free (ORT) runs
    from importlib import metadata
    for dist in ("numpy", "torch", "torchvision", "onnx", "onnxruntime", "opencv-python-headless", "av", "psutil", "ijson"):
        try:
            env[dist] = metadata.version(dist)
        except Exception:
            env[dist] = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                env["cpu_model"] = line.split(":", 1)[1].strip()
                break
        env["logical_cpus"] = os.cpu_count()
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                env["mem_total_kb"] = int(line.split()[1])
                break
        env["wsl2"] = "microsoft" in platform.release().lower()
    except OSError:
        env["cpu_model"] = platform.processor()
        env["logical_cpus"] = os.cpu_count()
    return env
