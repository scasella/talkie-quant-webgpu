#!/usr/bin/env python3
"""Create gzip-compressed ONNX external-data chunks from a Hugging Face model repo."""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path

import requests
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("model_name", help="Bare ONNX filename, e.g. model_kv_fast_q4f16.onnx")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", default="output/compressed-external-data")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    if "/" in args.model_name or not args.model_name.endswith(".onnx"):
        raise SystemExit("model_name must be a bare .onnx filename")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    files = set(api.list_repo_files(args.repo_id, repo_type="model", revision=args.revision))
    sizes = repo_file_sizes(api, args.repo_id, args.revision)
    chunk_paths = external_chunk_paths(files, args.model_name)
    if not chunk_paths:
        raise SystemExit(f"No external-data chunks found for {args.model_name}")

    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    total_raw = 0
    total_gzip = 0
    for path in chunk_paths:
        output_path = out_dir / f"{Path(path).name}.gz"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"SKIP {path} -> {output_path}")
        else:
            print(f"COMPRESS {path} -> {output_path}", flush=True)
            stream_gzip_from_hub(args.repo_id, args.revision, path, output_path, args.compression_level, token)
        raw_size = sizes.get(path, 0)
        gzip_size = output_path.stat().st_size
        total_raw += raw_size
        total_gzip += gzip_size
        ratio = gzip_size / raw_size if raw_size else 0
        print(f"  raw={raw_size} gzip={gzip_size} ratio={ratio:.3f}", flush=True)

    print(f"TOTAL raw={total_raw} gzip={total_gzip} ratio={total_gzip / total_raw:.3f}", flush=True)

    if args.upload:
        if not token:
            raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for upload")
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo="onnx",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message or f"Add gzip external-data chunks for {args.model_name}",
        )


def external_chunk_paths(files: set[str], model_name: str) -> list[str]:
    paths: list[str] = []
    index = 0
    while True:
        path = f"onnx/{model_name}_data" if index == 0 else f"onnx/{model_name}_data_{index}"
        if path not in files:
            break
        paths.append(path)
        index += 1
    return paths


def stream_gzip_from_hub(
    repo_id: str,
    revision: str,
    path: str,
    output_path: Path,
    compression_level: int,
    token: str | None,
) -> None:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with gzip.open(tmp_path, "wb", compresslevel=compression_level) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(output_path)


def repo_file_sizes(api: HfApi, repo_id: str, revision: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(repo_id, path_in_repo="onnx", recursive=False, repo_type="model", revision=revision, expand=True):
        if item.size is not None:
            sizes[item.path] = int(item.size)
    return sizes


if __name__ == "__main__":
    main()
