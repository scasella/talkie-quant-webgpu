# Outreach Templates

These are drafts only. Do not post automatically.

These are ready for maintainer review. The current browser demo defaults to the
smaller full-sequence q4f16 artifact, which passed a Chrome/WebGPU smoke on a
24 GB M4 Pro. Cached KV-cache artifacts are published but still experimental in
the browser.

## Hugging Face Community

Title: q4f16/q8 ONNX WebGPU artifacts for Talkie 1930 13B

I published an unofficial community ONNX/WebGPU quantization of Talkie 1930 13B:

- Model: https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX
- Live browser demo: https://scasella.github.io/talkie-quant-webgpu/
- GitHub runner/scripts: https://github.com/scasella/talkie-quant-webgpu

It includes full-sequence q4f16/q8 browser fallbacks, cached q4f16/q8 artifacts,
tokenizer/config/chat template, and a Transformers.js browser runner with a
manual generation loop. The cached q4f16 artifact is 16.53 GB across 32 chunks;
the browser demo currently defaults to the smaller 10.58 GB full-sequence q4,
which is slower but has passed a Chrome/WebGPU smoke.
This is not an official Talkie release, but it should make the model much
easier to try from a local browser with WebGPU.

## GitHub Release

Talkie Quant WebGPU v0.1.0 publishes a community ONNX/WebGPU artifact path
for Talkie 1930 13B:

- cached q4f16/q8 ONNX artifacts for faster browser decoding, with 32 and 42
  external-data chunks
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

It runs fully client-side with Transformers.js and defaults to the 10.58 GB
full-sequence q4 path that passed a Chrome/WebGPU smoke. Cached q4f16/q8
artifacts are also published for the faster KV-cache path, but that browser path
is still experimental. First load is big, but it finally makes Talkie tryable
without setting up a CUDA box.

## Note To Original Talkie Community

I made an unofficial community ONNX/WebGPU quantization of Talkie 1930 13B for
browser use. The repo preserves the source tokenizer, chat template, generation
config, and Apache-2.0 metadata, and links clearly back to the original Talkie
project and HF Transformers-format conversion.

Model: https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX
Demo: https://scasella.github.io/talkie-quant-webgpu/
Code: https://github.com/scasella/talkie-quant-webgpu
