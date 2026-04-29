#!/usr/bin/env python3
"""Inspect quantized ONNX logits from the cached Talkie export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from export_talkie_onnx import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, prepare_prompt


def inspect_model(path: Path, input_ids: np.ndarray, tokenizer, providers: list[str]) -> dict:
    session = ort.InferenceSession(str(path), providers=providers)
    output_name = session.get_outputs()[0].name
    got = session.run([output_name], {"input_ids": input_ids})[0]
    scores = got[0, -1].astype(np.float32)
    finite = np.isfinite(scores)
    finite_ids = np.flatnonzero(finite)
    if finite_ids.size:
        top = np.argsort(scores[finite])[-10:][::-1]
        top_ids = finite_ids[top].astype(int).tolist()
        min_score = float(scores[finite].min())
        max_score = float(scores[finite].max())
    else:
        top_ids = []
        min_score = None
        max_score = None
    return {
        "path": str(path),
        "providers": session.get_providers(),
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(scores).sum()),
        "posinf": int(np.isposinf(scores).sum()),
        "neginf": int(np.isneginf(scores).sum()),
        "min": min_score,
        "max": max_score,
        "top_ids": top_ids,
        "top_text": [tokenizer.decode([token_id]) for token_id in top_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/cache/talkie-onnx")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="Write a short notice about wireless telegraphy.")
    parser.add_argument(
        "--names",
        nargs="+",
        default=["model_q4f16.onnx", "model_quantized.onnx"],
        help="ONNX filenames under the hub/onnx directory to inspect.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Use only ONNX Runtime CPUExecutionProvider.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    input_ids = prepare_prompt(tokenizer, args.prompt).cpu().numpy().astype(np.int64)
    onnx_dir = Path(args.work_dir) / "hub" / "onnx"
    providers = ["CPUExecutionProvider"] if args.cpu_only else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    results = []
    for name in args.names:
        path = onnx_dir / name
        if path.exists():
            try:
                results.append(inspect_model(path, input_ids, tokenizer, providers))
            except Exception as error:
                results.append(
                    {
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        else:
            results.append({"path": str(path), "error": "missing"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
