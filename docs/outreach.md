# Outreach Templates

These are drafts only. Do not post automatically.

These are ready for maintainer review. The browser demo now defaults to the
fast cached q4f16 artifact, which passed a Chrome/WebGPU smoke on a 24 GB M4
Pro at about 3.17 tok/s rolling token latency after cold load. Cold load and
first-token latency are still large, so keep that caveat visible.

## Hugging Face Community

Title: q4f16/q8 ONNX WebGPU artifacts for Talkie 1930 13B

I published an unofficial community ONNX/WebGPU quantization of Talkie 1930 13B:

- Model: https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX
- Live browser demo: https://scasella.github.io/talkie-quant-webgpu/
- GitHub runner/scripts: https://github.com/scasella/talkie-quant-webgpu

It includes a fast cached q4f16 browser artifact, cached/full q4f16 and q8
fallbacks, tokenizer/config/chat template, and a Transformers.js browser runner
with a direct ONNX Runtime WebGPU generation loop. The fast cached q4f16
artifact is about 13 GB across 55 chunks and passed a 16-word Chrome/WebGPU
smoke on a 24 GB M4 Pro at about 3.17 tok/s rolling token latency.
This is not an official Talkie release, but it should make the model much
easier to try from a local browser with WebGPU.

## GitHub Release

Talkie Quant WebGPU v0.1.0 publishes a community ONNX/WebGPU artifact path
for Talkie 1930 13B:

- fast cached q4f16 ONNX default for browser decoding, with 55 external-data
  chunks and about 3.17 tok/s rolling token latency on the M4 Pro smoke
- cached q4f16/q8 ONNX fallbacks, with 32 and 42 external-data chunks
- full-sequence q4f16/q8 fallbacks with 22 and 31 external-data chunks
- Static React/Vite WebGPU chat demo using Transformers.js
- Manual Talkie generation loop, token `0` suppression, and stop IDs
  `65535`/`65536`
- Advanced Modal scripts for reproducing the export

Try the demo: https://scasella.github.io/talkie-quant-webgpu/

## Discord Or Social

Talkie 1930 13B now has an unofficial ONNX/WebGPU quant:
https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX

I also published a static demo and runner:
https://scasella.github.io/talkie-quant-webgpu/

It runs fully client-side with Transformers.js plus direct ONNX Runtime WebGPU
for the KV-cache graph. The default fast cached q4 path passed a 16-word
Chrome/WebGPU smoke at about 3.17 tok/s rolling token latency. First load is
big and TTFT is still slow, but it finally makes Talkie tryable without setting
up a CUDA box.

## Note To Original Talkie Community

I made an unofficial community ONNX/WebGPU quantization of Talkie 1930 13B for
browser use. The repo preserves the source tokenizer, chat template, generation
config, and Apache-2.0 metadata, and links clearly back to the original Talkie
project and HF Transformers-format conversion.

Model: https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX
Demo: https://scasella.github.io/talkie-quant-webgpu/
Code: https://github.com/scasella/talkie-quant-webgpu
