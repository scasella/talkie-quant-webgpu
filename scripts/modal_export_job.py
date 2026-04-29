#!/usr/bin/env python3
"""Detached Modal jobs for finishing the Talkie ONNX export pipeline."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "onnx-webgpu-export-job"
CACHE_VOLUME = "talkie-web-cache"
WORK_DIR = "/cache/talkie-onnx"
ENV_KEYS_TO_FORWARD = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def forwarded_env(repo_root: Path) -> dict[str, str]:
    dotenv_values = load_dotenv(repo_root / ".env")
    env: dict[str, str] = {}
    for key in ENV_KEYS_TO_FORWARD:
        value = os.environ.get(key) or dotenv_values.get(key)
        if value:
            env[key] = value
    if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" in env:
        env["HF_TOKEN"] = env["HUGGING_FACE_HUB_TOKEN"]
    if "HUGGING_FACE_HUB_TOKEN" not in env and "HF_TOKEN" in env:
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    env.setdefault("HF_HOME", "/cache/huggingface")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    env.setdefault("TORCH_HOME", "/cache/torch")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ENV = forwarded_env(REPO_ROOT)
SECRET = modal.Secret.from_dict(REMOTE_ENV)
VOLUME = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "accelerate>=1.0.0",
        "hf_transfer>=0.1.8",
        "hf_xet>=1.1.0",
        "huggingface_hub>=0.36.0",
        "numpy>=1.26.0",
        "onnx>=1.17.0",
        "onnxconverter-common>=1.14.0",
        "onnxruntime-gpu>=1.23.0",
        "onnxscript>=0.5.0",
        "safetensors>=0.4.5",
        "tokenizers>=0.20.0",
        "tqdm>=4.66.0",
        "transformers>=5.6.2",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/app",
        ignore=[
            ".env",
            ".env.*",
            ".git",
            ".playwright-mcp",
            "node_modules",
            "dist",
            ".vite",
            ".cache",
            ".modal",
            "__pycache__",
            "*.pyc",
            "models",
            "artifacts",
            "*.onnx",
            "*.onnx_data",
            "*.onnx_data_*",
            "*.safetensors",
            "*.pt",
            "*.bin",
            "*.log",
        ],
    )
)

app = modal.App(APP_NAME)


def command_for_mode(mode: str) -> list[str]:
    base = [
        "python",
        "scripts/export_talkie_onnx.py",
        "--skip-export",
        "--skip-fp-validation",
        "--skip-quant-validation",
        "--work-dir",
        WORK_DIR,
    ]
    if mode == "q4":
        return [*base, "--skip-q8", "--skip-upload"]
    if mode == "q8":
        return [*base, "--skip-q4", "--skip-upload"]
    if mode == "upload":
        return [*base, "--skip-quantize", "--split-external-data"]
    if mode == "split-upload":
        return [*base, "--skip-quantize", "--split-external-data"]
    if mode == "all":
        return base
    if mode == "q8-fp16":
        return [*base, "--skip-quantize", "--skip-upload", "--prepare-q8-fp16"]
    if mode == "q8-fold":
        return [*base, "--skip-quantize", "--skip-upload", "--prepare-q8-folded"]
    if mode == "q8-from-folded":
        return [
            *base,
            "--skip-q4",
            "--skip-upload",
            "--q8-source-path",
            f"{WORK_DIR}/hub/onnx/model_fp16_folded.onnx",
        ]
    raise ValueError(f"Unknown mode {mode!r}; expected q4, q8, q8-fp16, q8-fold, q8-from-folded, split-upload, upload, or all")


@app.function(
    image=IMAGE,
    gpu="H100",
    timeout=24 * 60 * 60,
    memory=524288,
    volumes={"/cache": VOLUME},
    secrets=[SECRET],
)
def run_export_mode(mode: str) -> None:
    command = command_for_mode(mode)
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd="/app", check=False)
    VOLUME.commit()
    if completed.returncode:
        sys.exit(completed.returncode)


@app.function(
    image=IMAGE,
    cpu=32,
    timeout=24 * 60 * 60,
    memory=344064,
    volumes={"/cache": VOLUME},
    secrets=[SECRET],
)
def run_export_mode_cpu(mode: str) -> None:
    command = command_for_mode(mode)
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd="/app", check=False)
    VOLUME.commit()
    if completed.returncode:
        sys.exit(completed.returncode)


@app.local_entrypoint()
def main(mode: str = "q4") -> None:
    function = run_export_mode_cpu if mode in {"q8", "upload", "split-upload"} else run_export_mode
    call = function.spawn(mode)
    print(f"Spawned Talkie export mode={mode} call_id={call.object_id}", flush=True)
    print(call.get_dashboard_url(), flush=True)
