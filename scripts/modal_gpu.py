#!/usr/bin/env python3
"""Run commands on a remote Modal GPU sandbox."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path


DEFAULT_APP_NAME = "talkie-web-onnx"
DEFAULT_GPU = "T4"
DEFAULT_TIMEOUT_MINS = 30
DEFAULT_CACHE_VOLUME = "talkie-web-cache"
QUARANTINED_COMMAND_MARKERS = (
    "export_talkie_onnx.py",
    "talkie-onnx",
)

ENV_KEYS_TO_FORWARD = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)


def stream_lines(stream, dest) -> None:
    for line in stream:
        dest.write(line)
        dest.flush()


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


def build_forwarded_env(repo_root: Path) -> dict[str, str]:
    dotenv_values = load_dotenv(repo_root / ".env")
    forwarded: dict[str, str] = {}
    for key in ENV_KEYS_TO_FORWARD:
        value = os.environ.get(key) or dotenv_values.get(key)
        if value:
            forwarded[key] = value
    if "HF_TOKEN" not in forwarded and "HUGGING_FACE_HUB_TOKEN" in forwarded:
        forwarded["HF_TOKEN"] = forwarded["HUGGING_FACE_HUB_TOKEN"]
    if "HUGGING_FACE_HUB_TOKEN" not in forwarded and "HF_TOKEN" in forwarded:
        forwarded["HUGGING_FACE_HUB_TOKEN"] = forwarded["HF_TOKEN"]
    return forwarded


def main() -> None:
    try:
        import modal
    except ImportError:
        print("ERROR: install Modal locally first: pip install modal && modal setup", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU type, e.g. T4, A10G, A100, H100")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINS, help="Timeout in minutes")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME, help="Modal app name")
    parser.add_argument("--memory", type=int, default=None, help="Sandbox memory in MiB")
    parser.add_argument("--cache-volume", default=DEFAULT_CACHE_VOLUME, help="Modal Volume name for HF/Torch caches")
    parser.add_argument(
        "--allow-talkie-export",
        action="store_true",
        help="Explicitly allow the quarantined Talkie ONNX export path.",
    )
    parser.add_argument(
        "--keep-alive-on-client-exit",
        action="store_true",
        help="Do not terminate the sandbox from the local cleanup block if the client exits unexpectedly.",
    )

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        parser.print_help()
        print("\nCOMMAND must be provided after --, for example:")
        print("  python scripts/modal_gpu.py --gpu=H100 -- python -c \"import torch; print(torch.cuda.is_available())\"")
        return

    try:
        sep = sys.argv.index("--")
    except ValueError:
        print("Usage: scripts/modal_gpu.py [--gpu GPU] [--timeout MINS] -- COMMAND...", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args(sys.argv[1:sep])
    command = sys.argv[sep + 1 :]
    if not command:
        print("ERROR: missing COMMAND after --", file=sys.stderr)
        sys.exit(1)
    joined = " ".join(command)
    export_quarantined = any(marker in joined for marker in QUARANTINED_COMMAND_MARKERS)
    export_allowed = args.allow_talkie_export and os.environ.get("TALKIE_ALLOW_EXPENSIVE_EXPORT") == "1"
    if export_quarantined and not export_allowed:
        print(
            "ERROR: refusing to launch the quarantined Talkie ONNX export path through scripts/modal_gpu.py. "
            "This path can spend real GPU time and upload large Hub artifacts. Pass "
            "--allow-talkie-export with TALKIE_ALLOW_EXPENSIVE_EXPORT=1 only for an intentional supervised export.",
            file=sys.stderr,
        )
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[1]
    forwarded_env = build_forwarded_env(repo_root)
    forwarded_env.setdefault("HF_HOME", "/cache/huggingface")
    forwarded_env.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    forwarded_env.setdefault("TORCH_HOME", "/cache/torch")
    forwarded_env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    secret = modal.Secret.from_dict(forwarded_env) if forwarded_env else None
    cache_volume = modal.Volume.from_name(args.cache_volume, create_if_missing=True)

    image = (
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
            repo_root,
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

    app = modal.App.lookup(args.app_name, create_if_missing=True)
    sandbox = None
    try:
        sandbox = modal.Sandbox.create(
            *command,
            image=image,
            gpu=args.gpu,
            timeout=args.timeout * 60,
            workdir="/app",
            app=app,
            secrets=[secret] if secret is not None else None,
            memory=args.memory,
            volumes={"/cache": cache_volume},
        )

        stderr_thread = threading.Thread(target=stream_lines, args=(sandbox.stderr, sys.stderr), daemon=True)
        stderr_thread.start()
        stream_lines(sandbox.stdout, sys.stdout)
        stderr_thread.join(timeout=5)

        sandbox.wait(raise_on_termination=False)
        sys.exit(sandbox.returncode or 0)
    finally:
        if sandbox is not None:
            try:
                if sandbox.returncode is None and not args.keep_alive_on_client_exit:
                    sandbox.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
