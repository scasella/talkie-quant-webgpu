#!/usr/bin/env python3
"""Print ONNX node input/output types around a named node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def tensor_type_name(elem_type: int) -> str:
    return onnx.TensorProto.DataType.Name(elem_type)


def value_type_map(model) -> dict[str, str]:
    types: dict[str, str] = {}
    for initializer in model.graph.initializer:
        types[initializer.name] = tensor_type_name(initializer.data_type)
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.type.HasField("tensor_type"):
            elem_type = value.type.tensor_type.elem_type
            if elem_type:
                types[value.name] = tensor_type_name(elem_type)
    return types


def producer_map(model) -> dict[str, object]:
    producers: dict[str, object] = {}
    for node in model.graph.node:
        for output in node.output:
            if output:
                producers[output] = node
    return producers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--node-name", default="/Add")
    parser.add_argument("--op-type", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    model = onnx.load_model(str(args.model_path), load_external_data=False)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as error:
        print(json.dumps({"shape_inference_error": f"{type(error).__name__}: {error}"}, indent=2))
        inferred = model

    types = value_type_map(inferred)
    producers = producer_map(inferred)
    records = []
    for node in inferred.graph.node:
        if args.op_type and node.op_type != args.op_type:
            continue
        if args.node_name and node.name != args.node_name and args.node_name not in node.name:
            continue
        record = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": [
                {
                    "name": value,
                    "type": types.get(value),
                    "producer": getattr(producers.get(value), "op_type", None),
                    "producer_name": getattr(producers.get(value), "name", None),
                }
                for value in node.input
            ],
            "outputs": [{"name": value, "type": types.get(value)} for value in node.output],
        }
        records.append(record)
        if len(records) >= args.limit:
            break
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
