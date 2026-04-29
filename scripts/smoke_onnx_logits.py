#!/usr/bin/env python3
"""Run a tiny ONNX logits smoke test for cached Talkie artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def smoke_model(path: Path, input_ids: np.ndarray) -> dict:
    print(f"LOAD_START {path.name}", flush=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"RUN_START {path.name}", flush=True)
    output = session.run([session.get_outputs()[0].name], {"input_ids": input_ids})[0]
    scores = output[0, -1].astype(np.float32)
    finite = np.isfinite(scores)
    finite_ids = np.flatnonzero(finite)
    if finite_ids.size:
        top = np.argsort(scores[finite])[-5:][::-1]
        top_ids = finite_ids[top].astype(int).tolist()
        min_score = float(scores[finite].min())
        max_score = float(scores[finite].max())
    else:
        top_ids = []
        min_score = None
        max_score = None
    return {
        "name": path.name,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(scores).sum()),
        "min": min_score,
        "max": max_score,
        "top_ids": top_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/cache/talkie-onnx")
    parser.add_argument("--names", nargs="+", default=["model_q4f16.onnx", "model_quantized.onnx"])
    parser.add_argument("--input-ids", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    input_ids = np.asarray([args.input_ids], dtype=np.int64)
    onnx_dir = Path(args.work_dir) / "hub" / "onnx"
    results = [smoke_model(onnx_dir / name, input_ids) for name in args.names]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
