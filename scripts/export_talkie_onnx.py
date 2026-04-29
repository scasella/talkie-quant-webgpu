#!/usr/bin/env python3
"""Export Talkie to ONNX, quantize for Transformers.js, and optionally publish."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "lewtun/talkie-1930-13b-it-hf"
DEFAULT_MODEL_REVISION = "6311dedf518470856a8503f2080bb4b54fcb3323"
DEFAULT_OUTPUT_REPO = "scasella91/talkie-1930-13b-it-ONNX"
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_OPSET = 18
STOP_TOKEN_IDS = [65535, 65536]
DEFAULT_EXTERNAL_DATA_CHUNK_MIB = 1024
METADATA_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
]


class TalkieFullSequenceOnnxWrapper(nn.Module):
    """Export wrapper that avoids Talkie's lazy Python-side RoPE cache."""

    def __init__(self, model: nn.Module, max_seq_len: int, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len
        self.compute_dtype = compute_dtype
        self.head_dim = model.config.head_dim
        self.attn_scale = self.head_dim ** -0.5
        cos, sin = self._precompute_rotary_embeddings(
            max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
            device=next(model.parameters()).device,
            dtype=compute_dtype,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        causal_mask = torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min, device=cos.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer("causal_mask", causal_mask[None, None, :, :], persistent=False)

    @staticmethod
    def _precompute_rotary_embeddings(
        seq_len: int,
        head_dim: int,
        base: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype), sin.to(dtype)
        return cos[None, :, None, :], sin[None, :, None, :]

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5
        xf = x.float()
        normed = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
        return normed.to(dtype=x.dtype)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.compute_dtype) if x.is_floating_point() else x

    def _linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(self._compute(x), self._compute(weight))

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self._compute(self.model.model.embed.weight))

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=3).to(dtype=x.dtype)

    def _attention(self, attn: nn.Module, x: torch.Tensor, seq_len: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        q = self._linear(x, attn.attn_query.weight).view(batch, seq_len, attn.n_head, attn.head_dim)
        k = self._linear(x, attn.attn_key.weight).view(batch, seq_len, attn.n_head, attn.head_dim)
        v = self._linear(x, attn.attn_value.weight).view(batch, seq_len, attn.n_head, attn.head_dim)

        q = self._apply_rotary_emb(q, cos, sin)
        k = self._apply_rotary_emb(k, cos, sin)
        q = self._rms_norm(q)
        k = self._rms_norm(k)
        q = q * attn.head_gain.head_g.to(dtype=q.dtype).view(1, 1, -1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.attn_scale
        scores = scores + self.causal_mask[:, :, :seq_len, :seq_len]
        probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        y = torch.matmul(probs, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self._linear(y, attn.attn_resid.weight)

    def _mlp(self, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate = self._linear(x, mlp.mlp_gate.weight)
        linear = self._linear(x, mlp.mlp_linear.weight)
        return self._linear(self._silu(gate) * linear, mlp.mlp_resid.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        cos = self.rope_cos[:, :seq_len]
        sin = self.rope_sin[:, :seq_len]
        x = self._embed(input_ids)
        x = self._rms_norm(x)
        e_x = x
        for block in self.model.model.blocks:
            attn_in = self._rms_norm(x)
            attn_out = self._attention(block.attn, attn_in, seq_len, cos, sin)
            x = x + attn_out * block.attn_gain.a_g.to(dtype=attn_out.dtype)
            mlp_in = self._rms_norm(x)
            mlp_out = self._mlp(block.mlp, mlp_in)
            x = x + mlp_out * block.mlp_gain.a_g.to(dtype=mlp_out.dtype)
            x = x + e_x * block.embed_skip.a_g.to(dtype=e_x.dtype)
        hidden_states = self._rms_norm(x)
        lm_head = self._compute(self.model.lm_head_gain(self.model.lm_head))
        logits = F.linear(hidden_states, lm_head).float()
        return logits


@dataclass
class ValidationResult:
    name: str
    top1: int
    reference_top5: list[int]
    passed: bool


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@contextlib.contextmanager
def heartbeat(label: str, interval_sec: int = 20):
    done = threading.Event()
    start = time.monotonic()

    def beat() -> None:
        while not done.wait(interval_sec):
            print(
                json.dumps(
                    {
                        "heartbeat": label,
                        "elapsed_sec": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1)


def ensure_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required")
    return token


def prepare_prompt(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids


def export_onnx(
    wrapper: nn.Module,
    input_ids: torch.Tensor,
    onnx_path: Path,
    opset: int,
    dynamo: bool,
    max_seq_len: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx_path.exists():
        onnx_path.unlink()
    data_path = onnx_path.with_name(f"{onnx_path.name}_data")
    if data_path.exists():
        data_path.unlink()

    kwargs = dict(
        args=(input_ids,),
        f=str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=opset,
        external_data=True,
    )
    if dynamo:
        from torch.export import Dim

        kwargs["dynamo"] = True
        kwargs["dynamic_shapes"] = {
            "input_ids": {
                0: Dim("batch", min=1),
                1: Dim("sequence", min=2, max=max_seq_len),
            }
        }
    else:
        try:
            from torch.onnx._internal.torchscript_exporter import _globals as torchscript_exporter_globals

            torchscript_exporter_globals.GLOBALS.onnx_shape_inference = False
        except Exception:
            pass
        kwargs["dynamo"] = False
        kwargs["dynamic_axes"] = {
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        }
        kwargs["do_constant_folding"] = False

    torch.onnx.export(wrapper, **kwargs)


def validate_onnx(
    onnx_path: Path,
    input_ids: torch.Tensor,
    name: str,
    reference_top5: list[int],
    require_top1_match: bool,
) -> ValidationResult:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
    output_name = session.get_outputs()[0].name
    got = session.run([output_name], {"input_ids": input_ids.cpu().numpy().astype(np.int64)})[0]
    got_top1 = int(np.argmax(got[0, -1]))
    passed = got_top1 == reference_top5[0] if require_top1_match else got_top1 in reference_top5
    print(
        json.dumps(
            {
                "validation": name,
                "top1": got_top1,
                "reference_top5": reference_top5,
                "passed": passed,
            },
            indent=2,
        ),
        flush=True,
    )
    del got
    del session
    gc.collect()
    return ValidationResult(name=name, top1=got_top1, reference_top5=reference_top5, passed=passed)


def reference_top5_from_wrapper(wrapper: nn.Module, input_ids: torch.Tensor) -> list[int]:
    with torch.inference_mode():
        reference = wrapper(input_ids.to(next(wrapper.parameters()).device)).detach().float().cpu().numpy()
    return np.argsort(reference[0, -1])[-5:][::-1].astype(int).tolist()


def validate_wrapper_against_model(
    model: nn.Module,
    wrapper: nn.Module,
    input_ids: torch.Tensor,
) -> None:
    with torch.inference_mode():
        reference = model(input_ids.to(next(model.parameters()).device)).logits.detach().float().cpu().numpy()
        got = wrapper(input_ids.to(next(wrapper.parameters()).device)).detach().float().cpu().numpy()
    reference_top5 = np.argsort(reference[0, -1])[-5:][::-1].astype(int).tolist()
    got_top1 = int(np.argmax(got[0, -1]))
    passed = got_top1 in reference_top5
    print(
        json.dumps(
            {
                "validation": "wrapper_vs_model",
                "wrapper_top1": got_top1,
                "model_top5": reference_top5,
                "passed": passed,
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("Export wrapper does not match source model top-5")


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def remove_onnx_sidecars(model_path: Path) -> None:
    data_path = model_path.with_name(f"{model_path.name}_data")
    for path in (model_path, data_path):
        if path.exists():
            path.unlink()


def save_onnx_external(model, model_path: Path) -> None:
    import onnx

    remove_onnx_sidecars(model_path)
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_path.name}_data",
        size_threshold=1024,
        convert_attribute=False,
    )


def external_value(tensor, key: str) -> str | None:
    for entry in tensor.external_data:
        if entry.key == key:
            return entry.value
    return None


def set_external_values(tensor, values: dict[str, str | int]) -> None:
    del tensor.external_data[:]
    for key, value in values.items():
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)


def data_chunk_name(model_path: Path, chunk_index: int) -> str:
    base = f"{model_path.name}_data"
    return base if chunk_index == 0 else f"{base}_{chunk_index}"


def split_external_data_file(model_path: Path, chunk_size_bytes: int) -> int:
    import onnx

    source_path = model_path.with_name(data_chunk_name(model_path, 0))
    existing_chunks = sorted(model_path.parent.glob(f"{model_path.name}_data_*"))
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    model = onnx.load_model(str(model_path), load_external_data=False)
    external_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
    ]
    if not external_initializers:
        return 0

    source_name = source_path.name
    if existing_chunks:
        locations = {external_value(initializer, "location") for initializer in external_initializers}
        chunk_count = len(existing_chunks) + 1
        expected_locations = {data_chunk_name(model_path, index) for index in range(chunk_count)}
        if locations and locations != {source_name} and locations <= expected_locations:
            missing_chunks = [
                model_path.with_name(data_chunk_name(model_path, index))
                for index in range(chunk_count)
                if not model_path.with_name(data_chunk_name(model_path, index)).exists()
            ]
            if missing_chunks:
                raise FileNotFoundError(", ".join(str(path) for path in missing_chunks))
            return chunk_count
        if source_path.stat().st_size <= chunk_size_bytes:
            return chunk_count

    for initializer in external_initializers:
        if external_value(initializer, "location") != source_name:
            raise RuntimeError(f"{model_path.name} is already split or references an unexpected external data file")

    tmp_dir = model_path.with_name(f".{model_path.name}_chunks_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    tmp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")

    chunk_index = 0
    chunk_offset = 0
    chunk_handle = None
    chunk_count = 0

    try:
        with source_path.open("rb") as source:
            for initializer in sorted(external_initializers, key=lambda item: int(external_value(item, "offset") or 0)):
                offset = external_value(initializer, "offset")
                length = external_value(initializer, "length")
                if offset is None or length is None:
                    raise RuntimeError(f"{initializer.name} is missing external data offset/length")
                tensor_offset = int(offset)
                tensor_length = int(length)
                if chunk_handle is None or (chunk_offset > 0 and chunk_offset + tensor_length > chunk_size_bytes):
                    if chunk_handle is not None:
                        chunk_handle.close()
                    chunk_name = data_chunk_name(model_path, chunk_index)
                    chunk_handle = (tmp_dir / chunk_name).open("wb")
                    chunk_count += 1
                    chunk_offset = 0
                    chunk_index += 1
                chunk_name = data_chunk_name(model_path, chunk_index - 1)
                source.seek(tensor_offset)
                remaining = tensor_length
                while remaining:
                    data = source.read(min(16 * 1024 * 1024, remaining))
                    if not data:
                        raise RuntimeError(f"Unexpected EOF while reading {initializer.name}")
                    chunk_handle.write(data)
                    remaining -= len(data)
                set_external_values(
                    initializer,
                    {
                        "location": chunk_name,
                        "offset": chunk_offset,
                        "length": tensor_length,
                    },
                )
                chunk_offset += tensor_length
        if chunk_handle is not None:
            chunk_handle.close()
            chunk_handle = None

        onnx.save_model(model, str(tmp_model_path))
        for old_chunk in model_path.parent.glob(f"{model_path.name}_data*"):
            old_chunk.unlink()
        for new_chunk in sorted(tmp_dir.iterdir()):
            shutil.move(str(new_chunk), model_path.parent / new_chunk.name)
        shutil.move(str(tmp_model_path), model_path)
    finally:
        if chunk_handle is not None:
            chunk_handle.close()
        if tmp_model_path.exists():
            tmp_model_path.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return chunk_count


def split_browser_external_data(onnx_dir: Path, chunk_mib: int) -> dict[str, int]:
    chunk_size_bytes = chunk_mib * 1024 * 1024
    if chunk_size_bytes <= 0:
        raise ValueError("External data chunk size must be positive")

    chunk_counts: dict[str, int] = {}
    for name in ("model_q4f16.onnx", "model_quantized.onnx"):
        model_path = onnx_dir / name
        if not model_path.exists():
            continue
        chunks = split_external_data_file(model_path, chunk_size_bytes)
        if chunks:
            chunk_counts[name] = chunks
            print(f"Split {name} external data into {chunks} chunk(s)", flush=True)
    return chunk_counts


def cast_array_for_onnx_type(array: np.ndarray, tensor_type: int) -> np.ndarray | None:
    import onnx

    if tensor_type == onnx.TensorProto.FLOAT:
        return np.asarray(array, dtype=np.float32)
    if tensor_type == onnx.TensorProto.FLOAT16:
        return np.asarray(array, dtype=np.float16)
    if tensor_type == onnx.TensorProto.DOUBLE:
        return np.asarray(array, dtype=np.float64)
    return None


def get_int_attribute(node, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def get_ints_attribute(node, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return [int(value) for value in attr.ints]
    return None


def build_consumers(graph) -> dict[str, list[tuple[object, int]]]:
    consumers: dict[str, list[tuple[object, int]]] = {}
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name:
                consumers.setdefault(input_name, []).append((node, index))
    return consumers


def replace_input(node, old_name: str, new_name: str) -> None:
    for index, input_name in enumerate(node.input):
        if input_name == old_name:
            node.input[index] = new_name


def fold_initializer_casts(model) -> int:
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers = build_consumers(graph)
    folded = 0

    for node in list(graph.node):
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None:
            continue
        initializer_consumers = consumers.get(initializer.name, [])
        if len(initializer_consumers) != 1 or initializer_consumers[0][0] is not node or initializer_consumers[0][1] != 0:
            continue

        to_type = get_int_attribute(node, "to")
        if to_type is None:
            continue
        array = numpy_helper.to_array(initializer)
        casted = cast_array_for_onnx_type(array, to_type)
        if casted is None:
            continue

        initializer.CopyFrom(numpy_helper.from_array(casted, name=initializer.name))
        for consumer, _index in consumers.get(node.output[0], []):
            replace_input(consumer, node.output[0], initializer.name)
        graph.node.remove(node)
        folded += 1

    return folded


def fold_initializer_transposes_for_matmul(model) -> int:
    from onnx import numpy_helper

    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers = build_consumers(graph)
    folded = 0

    for node in list(graph.node):
        if node.op_type != "Transpose" or len(node.input) != 1 or len(node.output) != 1:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None:
            continue
        initializer_consumers = consumers.get(initializer.name, [])
        if len(initializer_consumers) != 1 or initializer_consumers[0][0] is not node or initializer_consumers[0][1] != 0:
            continue

        output_consumers = consumers.get(node.output[0], [])
        if not output_consumers:
            continue
        if any(consumer.op_type != "MatMul" or index != 1 for consumer, index in output_consumers):
            continue

        array = numpy_helper.to_array(initializer)
        if array.ndim != 2:
            continue
        perm = get_ints_attribute(node, "perm") or [1, 0]
        if perm != [1, 0]:
            continue

        transposed = np.asarray(array).transpose(perm).copy()
        initializer.CopyFrom(numpy_helper.from_array(transposed, name=initializer.name))
        for consumer, _index in output_consumers:
            replace_input(consumer, node.output[0], initializer.name)
        graph.node.remove(node)
        folded += 1

    return folded


def build_producers(graph) -> dict[str, object]:
    producers: dict[str, object] = {}
    for node in graph.node:
        for output_name in node.output:
            if output_name:
                producers[output_name] = node
    return producers


def trace_initializer_chain_to_value(graph, value_name: str, allowed_ops: set[str]) -> tuple[object, list[object]] | None:
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers = build_consumers(graph)
    producers = build_producers(graph)
    chain: list[object] = []
    current = value_name

    while current not in initializers:
        producer = producers.get(current)
        if producer is None or producer.op_type not in allowed_ops or len(producer.input) != 1:
            return None
        output_consumers = consumers.get(current, [])
        if len(output_consumers) != 1:
            return None
        chain.append(producer)
        current = producer.input[0]

    if not chain:
        return None
    initializer_consumers = consumers.get(current, [])
    if len(initializer_consumers) != 1 or initializer_consumers[0][0] is not chain[-1]:
        return None
    return initializers[current], chain


def apply_initializer_chain_array(array: np.ndarray, chain: list[object], force_float16: bool) -> np.ndarray | None:
    result = np.asarray(array)
    for node in reversed(chain):
        if node.op_type == "Transpose":
            perm = get_ints_attribute(node, "perm") or list(reversed(range(result.ndim)))
            if sorted(perm) != list(range(result.ndim)):
                return None
            result = result.transpose(perm)
        elif node.op_type == "Cast":
            to_type = get_int_attribute(node, "to")
            if to_type is None:
                return None
            if force_float16:
                result = np.asarray(result, dtype=np.float16)
            else:
                casted = cast_array_for_onnx_type(result, to_type)
                if casted is None:
                    return None
                result = casted
        else:
            return None
    if force_float16:
        result = np.asarray(result, dtype=np.float16)
    return np.asarray(result).copy()


def fold_matmul_initializer_chains(model, force_float16: bool = False) -> int:
    from onnx import numpy_helper

    graph = model.graph
    folded = 0
    nodes_to_remove: list[object] = []

    for matmul in list(graph.node):
        if matmul.op_type != "MatMul" or len(matmul.input) < 2:
            continue
        traced = trace_initializer_chain_to_value(graph, matmul.input[1], allowed_ops={"Cast", "Transpose"})
        if traced is None:
            continue
        initializer, chain = traced
        array = numpy_helper.to_array(initializer)
        if array.ndim != 2:
            continue
        folded_array = apply_initializer_chain_array(array, chain, force_float16=force_float16)
        if folded_array is None or folded_array.ndim != 2:
            continue

        initializer.CopyFrom(numpy_helper.from_array(folded_array, name=initializer.name))
        matmul.input[1] = initializer.name
        nodes_to_remove.extend(chain)
        folded += 1

    for node in nodes_to_remove:
        try:
            graph.node.remove(node)
        except ValueError:
            pass

    return folded


def count_foldable_initializer_transposes_for_matmul(model) -> int:
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers = build_consumers(graph)
    foldable = 0

    for node in graph.node:
        if node.op_type != "Transpose" or len(node.input) != 1 or len(node.output) != 1:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None or len(initializer.dims) != 2:
            continue
        initializer_consumers = consumers.get(initializer.name, [])
        if len(initializer_consumers) != 1 or initializer_consumers[0][0] is not node or initializer_consumers[0][1] != 0:
            continue
        output_consumers = consumers.get(node.output[0], [])
        if not output_consumers:
            continue
        if any(consumer.op_type != "MatMul" or index != 1 for consumer, index in output_consumers):
            continue
        perm = get_ints_attribute(node, "perm") or [1, 0]
        if perm == [1, 0]:
            foldable += 1

    return foldable


def preprocess_onnx_for_quantization(model_path: Path, fold_casts: bool) -> None:
    import onnx

    print(f"Preprocessing ONNX graph for weight-only quantization -> {model_path}", flush=True)
    if not fold_casts:
        light_model = onnx.load_model(str(model_path), load_external_data=False)
        foldable_transposes = count_foldable_initializer_transposes_for_matmul(light_model)
        del light_model
        if foldable_transposes == 0:
            print(
                json.dumps(
                    {
                        "preprocess": "fold_initializer_transforms",
                        "folded_casts": 0,
                        "folded_transposes": 0,
                        "skipped_external_rewrite": True,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return

    model = onnx.load_model(str(model_path), load_external_data=True)
    folded_casts = fold_initializer_casts(model) if fold_casts else 0
    folded_transposes = fold_initializer_transposes_for_matmul(model)
    save_onnx_external(model, model_path)
    del model
    gc.collect()
    print(
        json.dumps(
            {
                "preprocess": "fold_initializer_transforms",
                "folded_casts": folded_casts,
                "folded_transposes": folded_transposes,
            },
            indent=2,
        ),
        flush=True,
    )


def summarize_onnx_graph(model_path: Path) -> None:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    initializers = {initializer.name for initializer in model.graph.initializer}
    transpose_sources = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Transpose" and len(node.input) == 1 and len(node.output) == 1
    }
    op_counts: dict[str, int] = {}
    matmul_total = 0
    matmul_direct_initializer_b = 0
    matmul_transpose_initializer_b = 0
    gemm_total = 0

    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type == "MatMul":
            matmul_total += 1
            if len(node.input) > 1 and node.input[1] in initializers:
                matmul_direct_initializer_b += 1
            if len(node.input) > 1 and transpose_sources.get(node.input[1]) in initializers:
                matmul_transpose_initializer_b += 1
        elif node.op_type == "Gemm":
            gemm_total += 1

    print(
        json.dumps(
            {
                "onnx_graph": "summary",
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
                "matmul_total": matmul_total,
                "matmul_direct_initializer_b": matmul_direct_initializer_b,
                "matmul_transpose_initializer_b": matmul_transpose_initializer_b,
                "gemm_total": gemm_total,
                "top_ops": sorted(op_counts.items(), key=lambda item: item[1], reverse=True)[:12],
            },
            indent=2,
        ),
        flush=True,
    )
    del model
    gc.collect()


def rewrite_tensor_type(type_proto, old_type: int, new_type: int) -> int:
    if not type_proto.HasField("tensor_type"):
        return 0
    tensor_type = type_proto.tensor_type
    if tensor_type.elem_type != old_type:
        return 0
    tensor_type.elem_type = new_type
    return 1


def convert_bfloat16_graph(model, target_onnx_type: int, target_numpy_dtype: np.dtype) -> dict[str, int]:
    import onnx
    from onnx import numpy_helper

    converted_initializers = 0
    converted_constants = 0
    converted_casts = 0
    converted_type_info = 0

    for initializer in model.graph.initializer:
        if initializer.data_type != onnx.TensorProto.BFLOAT16:
            continue
        array = numpy_helper.to_array(initializer)
        initializer.CopyFrom(numpy_helper.from_array(np.asarray(array, dtype=target_numpy_dtype), name=initializer.name))
        converted_initializers += 1

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.BFLOAT16:
                    attr.i = target_onnx_type
                    converted_casts += 1
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.BFLOAT16:
                    array = numpy_helper.to_array(attr.t)
                    attr.t.CopyFrom(numpy_helper.from_array(np.asarray(array, dtype=target_numpy_dtype), name=attr.t.name))
                    converted_constants += 1

    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        converted_type_info += rewrite_tensor_type(value_info.type, onnx.TensorProto.BFLOAT16, target_onnx_type)

    return {
        "initializers": converted_initializers,
        "constants": converted_constants,
        "casts": converted_casts,
        "type_info": converted_type_info,
    }


def convert_bfloat16_graph_to_float16(model) -> dict[str, int]:
    import onnx

    return convert_bfloat16_graph(model, onnx.TensorProto.FLOAT16, np.dtype(np.float16))


def convert_bfloat16_graph_to_float32(model) -> dict[str, int]:
    import onnx

    return convert_bfloat16_graph(model, onnx.TensorProto.FLOAT, np.dtype(np.float32))


def build_value_type_map(model) -> dict[str, int]:
    types: dict[str, int] = {}
    for initializer in model.graph.initializer:
        types[initializer.name] = int(initializer.data_type)
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.elem_type:
            types[value_info.name] = int(value_info.type.tensor_type.elem_type)
    return types


def update_value_info_type(model, value_name: str, elem_type: int) -> None:
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value_info.name == value_name and value_info.type.HasField("tensor_type"):
            value_info.type.tensor_type.elem_type = elem_type


def cast_initializer_to_onnx_type(initializer, elem_type: int) -> bool:
    from onnx import numpy_helper

    casted = cast_array_for_onnx_type(numpy_helper.to_array(initializer), elem_type)
    if casted is None:
        return False
    initializer.CopyFrom(numpy_helper.from_array(casted, name=initializer.name))
    return True


def cast_constant_to_onnx_type(node, elem_type: int) -> bool:
    from onnx import numpy_helper

    for attr in node.attribute:
        if attr.name != "value" or not attr.HasField("t"):
            continue
        casted = cast_array_for_onnx_type(numpy_helper.to_array(attr.t), elem_type)
        if casted is None:
            return False
        attr.t.CopyFrom(numpy_helper.from_array(casted, name=attr.t.name))
        return True
    return False


def align_constant_types_for_elementwise_ops(
    model,
    types: dict[str, int] | None = None,
    align_initializers: bool = True,
) -> int:
    import onnx

    if types is None:
        try:
            typed_model = onnx.shape_inference.infer_shapes(model)
            types = build_value_type_map(typed_model)
        except Exception:
            types = build_value_type_map(model)

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producers = build_producers(model.graph)
    elementwise_ops = {"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max"}
    aligned = 0

    for node in model.graph.node:
        if node.op_type not in elementwise_ops or len(node.input) < 2:
            continue
        known_inputs = [(index, value, types.get(value)) for index, value in enumerate(node.input[:2])]
        if any(elem_type is None for _index, _value, elem_type in known_inputs):
            continue
        if known_inputs[0][2] == known_inputs[1][2]:
            continue

        for index, value_name, elem_type in known_inputs:
            other_type = known_inputs[1 - index][2]
            if elem_type == other_type or other_type is None:
                continue

            casted = False
            initializer = initializers.get(value_name)
            if initializer is not None and align_initializers:
                casted = cast_initializer_to_onnx_type(initializer, other_type)
            else:
                producer = producers.get(value_name)
                if producer is not None and producer.op_type == "Constant":
                    casted = cast_constant_to_onnx_type(producer, other_type)

            if casted:
                update_value_info_type(model, value_name, other_type)
                types[value_name] = other_type
                aligned += 1
                break

    return aligned


def infer_saved_value_types(model_path: Path) -> dict[str, int]:
    import onnx

    light_model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        typed_model = onnx.shape_inference.infer_shapes(light_model)
        return build_value_type_map(typed_model)
    finally:
        del light_model
        gc.collect()


def align_saved_constant_types_for_elementwise_ops(model_path: Path) -> int:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        try:
            typed_model = onnx.shape_inference.infer_shapes(model)
            types = build_value_type_map(typed_model)
        except Exception:
            types = build_value_type_map(model)
        aligned = align_constant_types_for_elementwise_ops(model, types=types, align_initializers=False)
        if aligned:
            onnx.save_model(model, str(model_path))
        return aligned
    finally:
        del model
        gc.collect()


def insert_quantized_matmul_input_casts(model, types: dict[str, int]) -> int:
    import onnx
    from onnx import helper

    supported_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    inserted = 0
    new_nodes = []

    for node in model.graph.node:
        if node.op_type == "MatMulNBits" and len(node.input) >= 3:
            activation_type = types.get(node.input[0])
            scale_type = types.get(node.input[2])
            if (
                activation_type in supported_types
                and scale_type in supported_types
                and activation_type != scale_type
            ):
                clean_name = (node.name or f"MatMulNBits_{inserted}").replace("/", "_").strip("_")
                cast_output = f"{node.input[0]}__{clean_name}_cast_to_{scale_type}"
                cast_name = f"{node.name or clean_name}_CastInputToScaleType"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[node.input[0]],
                        outputs=[cast_output],
                        name=cast_name,
                        to=scale_type,
                    )
                )
                node.input[0] = cast_output
                inserted += 1
        new_nodes.append(node)

    if inserted:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return inserted


def insert_regular_matmul_input_casts(model, types: dict[str, int]) -> int:
    import onnx
    from onnx import helper

    supported_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    initializers = {initializer.name for initializer in model.graph.initializer}
    inserted = 0
    new_nodes = []

    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 2 and node.output:
            output_type = types.get(node.output[0])
            input_types = [types.get(node.input[0]), types.get(node.input[1])]
            target_type = output_type if output_type in supported_types else None
            if target_type is None:
                known = [item for item in input_types if item in supported_types]
                target_type = known[0] if known else None
            if target_type in supported_types:
                for input_index, input_type in enumerate(input_types):
                    if input_type == target_type or node.input[input_index] in initializers:
                        continue
                    clean_name = (node.name or f"MatMul_{inserted}").replace("/", "_").strip("_")
                    cast_output = f"{node.input[input_index]}__{clean_name}_input{input_index}_cast_to_{target_type}"
                    cast_name = f"{node.name or clean_name}_CastInput{input_index}ToMatMulType"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            inputs=[node.input[input_index]],
                            outputs=[cast_output],
                            name=cast_name,
                            to=target_type,
                        )
                    )
                    node.input[input_index] = cast_output
                    inserted += 1
        new_nodes.append(node)

    if inserted:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return inserted


def insert_elementwise_input_casts(model, types: dict[str, int]) -> int:
    import onnx
    from onnx import helper

    supported_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    elementwise_ops = {"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max"}
    inserted = 0
    new_nodes = []

    for node in model.graph.node:
        if node.op_type in elementwise_ops and len(node.input) >= 2 and node.output:
            output_type = types.get(node.output[0])
            input_types = [types.get(node.input[0]), types.get(node.input[1])]
            target_type = output_type if output_type in supported_types else None
            if target_type is None:
                known_types = [item for item in input_types if item in supported_types]
                target_type = known_types[0] if known_types else None
            if target_type in supported_types:
                for input_index, input_type in enumerate(input_types):
                    if input_type == target_type:
                        continue
                    clean_name = (node.name or f"{node.op_type}_{inserted}").replace("/", "_").strip("_")
                    cast_output = f"{node.input[input_index]}__{clean_name}_input{input_index}_cast_to_{target_type}"
                    cast_name = f"{node.name or clean_name}_CastInput{input_index}ToOutputType"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            inputs=[node.input[input_index]],
                            outputs=[cast_output],
                            name=cast_name,
                            to=target_type,
                        )
                    )
                    node.input[input_index] = cast_output
                    inserted += 1
        new_nodes.append(node)

    if inserted:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return inserted


def align_saved_quantized_matmul_types(model_path: Path) -> int:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        try:
            typed_model = onnx.shape_inference.infer_shapes(model)
            types = build_value_type_map(typed_model)
        except Exception:
            types = build_value_type_map(model)
        inserted = insert_quantized_matmul_input_casts(model, types)
        if inserted:
            onnx.save_model(model, str(model_path))
        return inserted
    finally:
        del model
        gc.collect()


def align_saved_regular_matmul_types(model_path: Path) -> int:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        try:
            typed_model = onnx.shape_inference.infer_shapes(model)
            types = build_value_type_map(typed_model)
        except Exception:
            types = build_value_type_map(model)
        inserted = insert_regular_matmul_input_casts(model, types)
        if inserted:
            onnx.save_model(model, str(model_path))
        return inserted
    finally:
        del model
        gc.collect()


def align_saved_elementwise_runtime_types(model_path: Path) -> int:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        try:
            typed_model = onnx.shape_inference.infer_shapes(model)
            types = build_value_type_map(typed_model)
        except Exception:
            types = build_value_type_map(model)
        inserted = insert_elementwise_input_casts(model, types)
        if inserted:
            onnx.save_model(model, str(model_path))
        return inserted
    finally:
        del model
        gc.collect()


def strip_saved_value_info(model_path: Path) -> int:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    try:
        count = len(model.graph.value_info)
        if count:
            del model.graph.value_info[:]
            onnx.save_model(model, str(model_path))
        return count
    finally:
        del model
        gc.collect()


def update_transformers_js_config(out_dir: Path, external_data_chunks: dict[str, int] | None = None) -> None:
    config_path = out_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    transformers_js_config = config.setdefault("transformers.js_config", {})
    transformers_js_config.setdefault("dtype", "q4f16")
    transformers_js_config["use_external_data_format"] = external_data_chunks or {
        "model_q4f16.onnx": 1,
        "model_quantized.onnx": 1,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def copy_metadata(model_id: str, revision: str, out_dir: Path, token: str) -> None:
    snapshot_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=token,
        allow_patterns=METADATA_FILES,
    )
    for rel in METADATA_FILES:
        src = Path(snapshot_dir) / rel
        if src.exists():
            shutil.copy2(src, out_dir / rel)

    update_transformers_js_config(out_dir)

    generation_config = out_dir / "generation_config.json"
    if generation_config.exists():
        data = json.loads(generation_config.read_text(encoding="utf-8"))
    else:
        data = {}
    data.setdefault("eos_token_id", STOP_TOKEN_IDS)
    data.setdefault("pad_token_id", STOP_TOKEN_IDS[0])
    generation_config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_model_card(out_dir: Path, source_model: str, source_revision: str, default_dtype: str) -> None:
    readme = f"""---
license: apache-2.0
library_name: transformers.js
language:
- en
tags:
- transformers.js
- onnx
- webgpu
- text-generation
- conversational
- talkie
- vintage
- historical
- quantized
- browser-ai
- q4f16
- q8
pipeline_tag: text-generation
base_model: {source_model}
---

# Talkie 1930 13B IT ONNX WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU quantization of
[`{source_model}`](https://huggingface.co/{source_model}) for browser inference
with Transformers.js.

- Live browser demo: <https://scasella.github.io/talkie-quant-webgpu/>
- GitHub runner and export scripts: <https://github.com/scasella/talkie-quant-webgpu>
- Source model revision: `{source_revision}`
- Default browser dtype: `{default_dtype}`
- Fallback dtype: `q8`
- Stop token IDs: `{STOP_TOKEN_IDS}`

This model keeps the source tokenizer, chat template, generation config, and
Apache-2.0 license metadata. It is not an official Talkie release.

## Files

| File | Runtime dtype | Use |
| --- | --- | --- |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Default browser path |
| `onnx/model_quantized.onnx` | q8 | Fallback path |

The `config.json` includes the Transformers.js external-data chunk map used by
the browser loader.

## Browser Use

The easiest path is the hosted demo:

```text
https://scasella.github.io/talkie-quant-webgpu/
```

For local development:

```bash
git clone https://github.com/scasella/talkie-quant-webgpu.git
cd talkie-quant-webgpu
npm install
npm run dev
```

The app loads this repo with `device: "webgpu"` and `dtype: "{default_dtype}"`
first, then falls back to `q8` if needed.

## Transformers.js Notes

Talkie has a custom `model_type`, so stock `pipeline("text-generation")` is not
used here. The browser runner formats messages with the shipped chat template,
tokenizes them, runs the full accumulated `input_ids` for each new token, samples
on the CPU, suppresses token `0`, and stops on token IDs `65535` or `65536`.

Minimal loading sketch:

```ts
import {{ AutoModel, AutoTokenizer }} from "@huggingface/transformers";

const repo = "scasella91/talkie-1930-13b-it-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModel.from_pretrained(repo, {{
  device: "webgpu",
  dtype: "{default_dtype}"
}});
```

Use the GitHub runner for the complete manual generation loop.

## Known Limitations

- First load is large and slow because the model is split across external-data
  chunks.
- Browser cache writes may hit quota; that is noisy but not necessarily fatal if
  the model reaches `Ready`.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Generation is full-sequence and intentionally slower than a KV-cache decoder.

## Attribution

Talkie was developed by Alec Radford, Nick Levine, and David Duvenaud. This ONNX
repo builds on the Hugging Face Transformers-format conversion by
[`{source_model}`](https://huggingface.co/{source_model}) and the original
Talkie project at <https://github.com/talkie-lm/talkie>.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def quantize_q4_model(model, save_path: Path, block_size: int, accuracy_level: int | None):
    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

        quantizer = MatMul4BitsQuantizer(
            model=model,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=accuracy_level,
        )
        quantizer.process()
        return quantizer.model.model
    except ModuleNotFoundError:
        import onnxruntime.quantization.matmul_nbits_quantizer as nbits_quantizer

        quant_config = nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=accuracy_level,
            quant_format=nbits_quantizer.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul", "Gather"),
            quant_axes=(("MatMul", 0), ("Gather", 1)),
            bits=4,
        )
        quantizer = nbits_quantizer.MatMulNBitsQuantizer(
            model,
            nodes_to_exclude=None,
            nodes_to_include=None,
            algo_config=quant_config,
        )
        quantizer.process()
        quantized_model = getattr(quantizer.model, "model", quantizer.model)
        return quantized_model


def quantize_with_onnxruntime(
    onnx_dir: Path,
    q4_block_size: int,
    q4_accuracy_level: int | None,
    skip_q4: bool,
    skip_q8: bool,
    q4_final_fp16: bool,
    q8_source_path: Path | None = None,
) -> None:
    import onnx
    from onnxconverter_common.float16 import convert_float_to_float16
    from onnxruntime.quantization import QuantType, QuantizationMode
    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
    from onnxruntime.quantization.registry import IntegerOpsRegistry

    model_path = onnx_dir / "model.onnx"
    q8_path = onnx_dir / "model_quantized.onnx"
    q4_path = onnx_dir / "model_q4f16.onnx"

    logging.getLogger("onnxruntime.quantization.matmul_4bits_quantizer").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.INFO)

    if skip_q4:
        print("Skipping q4f16 quantization.", flush=True)
    else:
        print(f"Quantizing q4f16 -> {q4_path}", flush=True)
        with heartbeat("q4_load_convert_fold"):
            q4_model = onnx.load_model(str(model_path), load_external_data=True)
            q4_bf16_to_f32 = convert_bfloat16_graph_to_float32(q4_model)
            print(json.dumps({"preprocess": "q4_bfloat16_to_float32", **q4_bf16_to_f32}, indent=2), flush=True)
            q4_folded_matmuls = fold_matmul_initializer_chains(q4_model, force_float16=False)
            print(json.dumps({"preprocess": "q4_fold_matmul_initializer_chains", "folded_matmuls": q4_folded_matmuls}, indent=2), flush=True)
        with heartbeat("q4_quantize"):
            q4f16_model = quantize_q4_model(q4_model, q4_path, q4_block_size, q4_accuracy_level)
        if q4_final_fp16:
            try:
                q4f16_model = convert_float_to_float16(q4f16_model, keep_io_types=True, disable_shape_infer=True)
            except ValueError as error:
                if "already converted to float16" not in str(error):
                    raise
                print("q4f16 graph is already float16-ready; skipping final fp16 conversion.", flush=True)
        else:
            print("Keeping q4 graph runtime tensors in float32 for WebGPU numerical stability.", flush=True)
        with heartbeat("q4_save"):
            save_onnx_external(q4f16_model, q4_path)
        with heartbeat("q4_align_saved_elementwise_types"):
            q4_aligned_elementwise = align_saved_constant_types_for_elementwise_ops(q4_path)
            q4_inserted_matmul_casts = align_saved_quantized_matmul_types(q4_path)
            q4_inserted_elementwise_casts = align_saved_elementwise_runtime_types(q4_path)
            q4_inserted_regular_matmul_casts = align_saved_regular_matmul_types(q4_path)
            q4_stripped_value_info = strip_saved_value_info(q4_path)
        print(
            json.dumps(
                {
                    "preprocess": "q4_align_saved_runtime_types",
                    "aligned_inputs": q4_aligned_elementwise,
                    "inserted_matmul_casts": q4_inserted_matmul_casts,
                    "inserted_elementwise_casts": q4_inserted_elementwise_casts,
                    "inserted_regular_matmul_casts": q4_inserted_regular_matmul_casts,
                    "stripped_value_info": q4_stripped_value_info,
                },
                indent=2,
            ),
            flush=True,
        )
        del q4_model
        del q4f16_model
        gc.collect()

    if skip_q8:
        print("Skipping q8 quantization.", flush=True)
        return

    print(f"Quantizing q8 -> {q8_path}", flush=True)
    with heartbeat("q8_load_convert_fold"):
        q8_model_path = q8_source_path or model_path
        q8_model = onnx.load_model(str(q8_model_path), load_external_data=True)
        if q8_source_path is None:
            q8_bf16_to_f32 = convert_bfloat16_graph_to_float32(q8_model)
            print(json.dumps({"preprocess": "q8_bfloat16_to_float32", **q8_bf16_to_f32}, indent=2), flush=True)
            q8_folded_matmuls = fold_matmul_initializer_chains(q8_model, force_float16=False)
            print(json.dumps({"preprocess": "q8_fold_matmul_initializer_chains", "folded_matmuls": q8_folded_matmuls}, indent=2), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "preprocess": "q8_use_preprocessed_source",
                        "source": str(q8_source_path),
                    },
                    indent=2,
                ),
                flush=True,
            )
    q8_quantizer = ONNXQuantizer(
        q8_model,
        per_channel=False,
        reduce_range=False,
        mode=QuantizationMode.IntegerOps,
        static=False,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QUInt8,
        tensors_range=None,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=set(IntegerOpsRegistry.keys()),
        extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True},
    )
    with heartbeat("q8_quantize"):
        q8_quantizer.quantize_model()
    with heartbeat("q8_save"):
        save_onnx_external(q8_quantizer.model.model, q8_path)
    del q8_model
    del q8_quantizer
    gc.collect()


def prepare_q8_fp16_model(onnx_dir: Path) -> None:
    import onnx

    source_path = onnx_dir / "model.onnx"
    fp16_path = onnx_dir / "model_fp16.onnx"
    print(f"Preparing q8 fp16 checkpoint -> {fp16_path}", flush=True)
    with heartbeat("q8_prepare_fp16"):
        model = onnx.load_model(str(source_path), load_external_data=True)
        stats = convert_bfloat16_graph_to_float16(model)
        print(json.dumps({"preprocess": "q8_bfloat16_to_float16", **stats}, indent=2), flush=True)
        save_onnx_external(model, fp16_path)
    del model
    gc.collect()


def prepare_q8_folded_model(onnx_dir: Path) -> None:
    import onnx

    fp16_path = onnx_dir / "model_fp16.onnx"
    folded_path = onnx_dir / "model_fp16_folded.onnx"
    if not fp16_path.exists():
        raise FileNotFoundError(fp16_path)
    print(f"Preparing q8 folded checkpoint -> {folded_path}", flush=True)
    with heartbeat("q8_prepare_folded"):
        model = onnx.load_model(str(fp16_path), load_external_data=True)
        folded_matmuls = fold_matmul_initializer_chains(model, force_float16=True)
        print(json.dumps({"preprocess": "q8_fold_matmul_initializer_chains", "folded_matmuls": folded_matmuls}, indent=2), flush=True)
        save_onnx_external(model, folded_path)
    del model
    gc.collect()


def upload_to_hub(out_dir: Path, repo_id: str, private: bool, token: str, upload_fp_model: bool) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    allow_patterns = None
    delete_patterns = None
    if not upload_fp_model:
        allow_patterns = [
            "README.md",
            *METADATA_FILES,
            "onnx/model_q4f16.onnx",
            "onnx/model_q4f16.onnx_data*",
            "onnx/model_quantized.onnx",
            "onnx/model_quantized.onnx_data*",
        ]
        delete_patterns = ["onnx/model.*", "onnx/model_fp16*"]
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(out_dir),
        commit_message="Add Talkie ONNX WebGPU export",
        allow_patterns=allow_patterns,
        delete_patterns=delete_patterns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--output-repo", default=DEFAULT_OUTPUT_REPO)
    parser.add_argument("--work-dir", default="/tmp/talkie-onnx")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--prompt", default="Write a short notice about wireless telegraphy.")
    parser.add_argument("--private", action="store_true", help="Create/upload to a private Hub repo")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--export-compute-dtype", choices=["source", "float32", "float16"], default="source")
    parser.add_argument("--q4-block-size", type=int, default=32)
    parser.add_argument("--q4-accuracy-level", type=int, default=4)
    parser.add_argument("--q4-final-fp16", action="store_true", help="Convert the q4 graph runtime tensors to fp16 after quantization")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-q4", action="store_true", help="Skip q4f16 generation; useful for q8 recovery runs")
    parser.add_argument("--skip-q8", action="store_true", help="Only produce q4f16; useful for recovery runs")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-fp-validation", action="store_true")
    parser.add_argument("--skip-quant-validation", action="store_true")
    parser.add_argument("--upload-fp-model", action="store_true", help="Upload the large fp model.onnx artifact too")
    parser.add_argument("--prepare-q8-fp16", action="store_true", help="Write onnx/model_fp16.onnx and exit")
    parser.add_argument("--prepare-q8-folded", action="store_true", help="Write onnx/model_fp16_folded.onnx and exit")
    parser.add_argument("--q8-source-path", default=None, help="Use a preprocessed ONNX source for q8 quantization")
    parser.add_argument(
        "--split-external-data",
        dest="split_external_data",
        action="store_true",
        default=True,
        help="Split browser ONNX sidecar files into Web-safe chunks",
    )
    parser.add_argument(
        "--no-split-external-data",
        dest="split_external_data",
        action="store_false",
        help="Keep one external data file per ONNX graph",
    )
    parser.add_argument("--external-data-chunk-mib", type=int, default=DEFAULT_EXTERNAL_DATA_CHUNK_MIB)
    parser.add_argument("--legacy-export", action="store_true", help="Use legacy torch.onnx exporter")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def resolve_export_compute_dtype(name: str, source_dtype: torch.dtype) -> torch.dtype:
    if name == "source":
        return source_dtype
    if name == "float16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    token = ensure_token()
    work_dir = Path(args.work_dir)
    out_dir = work_dir / "hub"
    onnx_dir = out_dir / "onnx"
    model_path = onnx_dir / "model.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "revision": args.revision,
                "output_repo": args.output_repo,
                "max_seq_len": args.max_seq_len,
                "torch_dtype": args.torch_dtype,
                "export_compute_dtype": args.export_compute_dtype,
                "q4_block_size": args.q4_block_size,
                "q4_accuracy_level": args.q4_accuracy_level,
                "work_dir": str(work_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, token=token, trust_remote_code=True)
    input_ids = prepare_prompt(tokenizer, args.prompt)
    reference_top5: list[int] | None = None

    if not args.skip_export:
        torch_dtype = resolve_torch_dtype(args.torch_dtype)
        export_compute_dtype = resolve_export_compute_dtype(args.export_compute_dtype, torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            token=token,
            trust_remote_code=True,
            dtype=torch_dtype,
        ).to("cuda")
        model.eval()
        wrapper = TalkieFullSequenceOnnxWrapper(model, args.max_seq_len, compute_dtype=export_compute_dtype).eval().to("cuda")
        input_ids = input_ids.to("cuda")
        validate_wrapper_against_model(model, wrapper, input_ids)

        try:
            export_onnx(
                wrapper,
                input_ids,
                model_path,
                args.opset,
                dynamo=not args.legacy_export,
                max_seq_len=args.max_seq_len,
            )
        except Exception:
            if args.legacy_export:
                raise
            print("Dynamo export failed; retrying with legacy exporter.", file=sys.stderr, flush=True)
            export_onnx(wrapper, input_ids, model_path, args.opset, dynamo=False, max_seq_len=args.max_seq_len)

        reference_top5 = reference_top5_from_wrapper(wrapper, input_ids)
        del model
        del wrapper
        release_cuda_memory()
        preprocess_onnx_for_quantization(model_path, fold_casts=args.export_compute_dtype == "float16")
        summarize_onnx_graph(model_path)
        if not args.skip_fp_validation:
            result = validate_onnx(
                model_path,
                input_ids.cpu(),
                "fp_export",
                reference_top5=reference_top5,
                require_top1_match=True,
            )
            if not result.passed:
                raise RuntimeError("ONNX export validation failed")
    else:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not (args.skip_fp_validation and args.skip_quant_validation):
            torch_dtype = resolve_torch_dtype(args.torch_dtype)
            export_compute_dtype = resolve_export_compute_dtype(args.export_compute_dtype, torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                revision=args.revision,
                token=token,
                trust_remote_code=True,
                dtype=torch_dtype,
            ).to("cuda")
            wrapper = TalkieFullSequenceOnnxWrapper(model.eval(), args.max_seq_len, compute_dtype=export_compute_dtype).eval().to("cuda")
            reference_top5 = reference_top5_from_wrapper(wrapper, input_ids)
            del model
            del wrapper
            release_cuda_memory()
        preprocess_onnx_for_quantization(model_path, fold_casts=args.export_compute_dtype == "float16")
        summarize_onnx_graph(model_path)
        if not args.skip_fp_validation:
            if reference_top5 is None:
                raise RuntimeError("reference_top5 is required for fp validation")
            result = validate_onnx(
                model_path,
                input_ids.cpu(),
                "fp_export",
                reference_top5=reference_top5,
                require_top1_match=True,
            )
            if not result.passed:
                raise RuntimeError("ONNX export validation failed")

    copy_metadata(args.model_id, args.revision, out_dir, token)
    write_model_card(out_dir, args.model_id, args.revision, default_dtype="q4f16")

    if args.prepare_q8_fp16:
        prepare_q8_fp16_model(onnx_dir)
        print("DONE", flush=True)
        return

    if args.prepare_q8_folded:
        prepare_q8_folded_model(onnx_dir)
        print("DONE", flush=True)
        return

    if not args.skip_quantize:
        quantize_with_onnxruntime(
            onnx_dir,
            q4_block_size=args.q4_block_size,
            q4_accuracy_level=args.q4_accuracy_level,
            skip_q4=args.skip_q4,
            skip_q8=args.skip_q8,
            q4_final_fp16=args.q4_final_fp16,
            q8_source_path=Path(args.q8_source_path) if args.q8_source_path else None,
        )
        q4_path = onnx_dir / "model_q4f16.onnx"
        q8_path = onnx_dir / "model_quantized.onnx"
        if args.skip_quant_validation:
            print("Skipping quantized ONNX validation.", flush=True)
        elif args.skip_q4:
            print("q4f16 validation skipped because q4f16 quantization was skipped.", flush=True)
        elif q4_path.exists():
            if reference_top5 is None:
                raise RuntimeError("reference_top5 is required for quantized validation")
            q4_result = validate_onnx(
                q4_path,
                input_ids.cpu(),
                "q4f16",
                reference_top5=reference_top5,
                require_top1_match=False,
            )
            if not q4_result.passed:
                raise RuntimeError("q4f16 validation failed")
        if args.skip_quant_validation:
            pass
        elif q8_path.exists():
            q8_result = validate_onnx(
                q8_path,
                input_ids.cpu(),
                "q8",
                reference_top5=reference_top5,
                require_top1_match=False,
            )
            if not q8_result.passed:
                raise RuntimeError("q8 validation failed")
        elif args.skip_q8:
            print("q8 validation skipped because q8 quantization was skipped.", flush=True)
        else:
            raise FileNotFoundError(q8_path)

    if args.split_external_data:
        external_data_chunks = split_browser_external_data(onnx_dir, args.external_data_chunk_mib)
        if external_data_chunks:
            update_transformers_js_config(out_dir, external_data_chunks)

    if not args.skip_upload:
        upload_to_hub(out_dir, args.output_repo, args.private, token, upload_fp_model=args.upload_fp_model)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
