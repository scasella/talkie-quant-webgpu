#!/usr/bin/env python3
"""Check that the published Talkie ONNX repo has the expected browser files."""

from __future__ import annotations

import argparse
import json
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


EXPECTED = {
    "README.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "onnx/model_q4f16.onnx",
    "onnx/model_quantized.onnx",
}

EXPECTED_EXTERNAL_MODELS = ("model_q4f16.onnx", "model_quantized.onnx")
EXPECTED_KV = {
    "onnx/model_kv_q4f16.onnx",
    "onnx/model_kv_quantized.onnx",
}
EXPECTED_KV_EXTERNAL_MODELS = ("model_kv_q4f16.onnx", "model_kv_quantized.onnx")
EXPECTED_EXTERNAL_CHUNKS = {
    "model_q4f16.onnx": 22,
    "model_quantized.onnx": 31,
    "model_kv_q4f16.onnx": 32,
    "model_kv_quantized.onnx": 42,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", nargs="?", default="scasella91/talkie-1930-13b-it-ONNX")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--require-kv", action="store_true", help="Require the cached model_kv q4/q8 artifacts")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    files = set(api.list_repo_files(args.repo_id, repo_type="model", revision=args.revision))
    expected = set(EXPECTED)
    expected_external_models = list(EXPECTED_EXTERNAL_MODELS)
    if args.require_kv:
        expected |= EXPECTED_KV
        expected_external_models.extend(EXPECTED_KV_EXTERNAL_MODELS)
    missing = sorted(expected - files)
    external_data = sorted(path for path in files if path.startswith("onnx/") and ".onnx_data" in path)

    if missing:
        print("Missing expected files:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    config_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="config.json",
        repo_type="model",
        revision=args.revision,
        token=token,
    )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    transformers_js_config = config.get("transformers.js_config") or {}
    external_format = transformers_js_config.get("use_external_data_format") or {}
    bad_external_entries = []
    for name in expected_external_models:
        actual = int(external_format.get(name, 0))
        expected_count = EXPECTED_EXTERNAL_CHUNKS[name]
        if actual != expected_count:
            bad_external_entries.append({"file": name, "expected": expected_count, "actual": actual})
    if bad_external_entries:
        print("config.json has incorrect Transformers.js external-data entries:")
        print(json.dumps(bad_external_entries, indent=2))
        sys.exit(1)
    missing_chunks: list[str] = []
    expected_chunks: set[str] = set()
    for model_name in expected_external_models:
        chunk_count = int(external_format[model_name])
        data_base = f"onnx/{model_name}_data"
        for index in range(chunk_count):
            chunk_path = data_base if index == 0 else f"{data_base}_{index}"
            expected_chunks.add(chunk_path)
            if chunk_path not in files:
                missing_chunks.append(chunk_path)
    if missing_chunks:
        print("Missing external data chunks:")
        for path in missing_chunks:
            print(f"  - {path}")
        sys.exit(1)
    stale_chunks = sorted(
        path
        for path in external_data
        if any(path == f"onnx/{name}_data" or path.startswith(f"onnx/{name}_data_") for name in expected_external_models)
        and path not in expected_chunks
    )
    if stale_chunks:
        print("Stale external data chunks:")
        for path in stale_chunks:
            print(f"  - {path}")
        sys.exit(1)

    print(f"Found {len(files)} files in {args.repo_id}@{args.revision}.")
    if external_data:
        print("External data files:")
        for path in external_data:
            print(f"  - {path}")
    print("Hub artifact check passed.")


if __name__ == "__main__":
    main()
