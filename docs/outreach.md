# Outreach Templates

These are drafts only. Do not post automatically.

These are ready for maintainer review. The browser demo now defaults to the
fast cached q4f16 artifact, which passed a Chrome/WebGPU smoke on a 24 GB M4
Pro at about 3.60 tok/s rolling token latency after the default warmup. Cold
load is still large, but measured TTFT after Ready fell to about 1.1s. A gzip
external-data opt-in reduces the fast q4 transfer to about 8.85 GB and reached
about 316.6s Ready / 17.3s TTFT, but it still did not meet the 2x cold-start
target. A smaller cold64 q4 candidate is also available as an opt-in artifact.

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
smoke on a 24 GB M4 Pro at about 3.60 tok/s rolling token latency after the
default warmup.
For cold-start testing, the gzip external-data opt-in reduced transfer to about
8.85 GB and reached about 316.6s Ready / 17.3s TTFT / 3.61 tok/s with
`compressed=1`, `warmup=0`, and `fetches=8`. It is opt-in because it still
misses the 265s Ready target.
This is not an official Talkie release, but it should make the model much
easier to try from a local browser with WebGPU.

## GitHub Release

Talkie Quant WebGPU v0.2.0 updates the first browser-ready community
ONNX/WebGPU path for Talkie 1930 13B with the fast cached q4f16 runtime.

What changed:

- moved the default browser path from full-sequence q4f16 to direct ONNX
  Runtime WebGPU KV-cache decoding
- improved steady decode from about 0.61 tok/s to about 3.60 tok/s reported
  rolling token latency on the same 24 GB M4 Pro Chrome/WebGPU smoke after the
  default warmup
- reduced measured TTFT after Ready to about 1.1s by warming a tiny cached
  decode path after session creation
- published `onnx/model_kv_fast_q4f16.onnx`, about 13 GB across 55 chunks, with
  q/k attention projections quantized and value projections left unquantized
- kept cached q4/q8 and full-sequence q4/q8 artifacts as fallbacks
- published an opt-in `model_kv_cold64_q4f16.onnx` cold-start candidate, about
  12.25 GB across 54 chunks, while keeping the default unchanged because the
  cold-start win was modest
- published gzip external-data companions for `model_kv_fast_q4f16.onnx`, about
  8.85 GB transferred, which cut the best measured cold-start run to about
  316.6s Ready / 17.3s TTFT / 3.61 tok/s while remaining opt-in because it
  missed the 265s Ready target
- documented the trial-and-error path: full-sequence baseline, browser
  allocation failures, graph-optimization failures, fetch-throttling fix, and
  the failed all-attention-projection q4 candidate

Try the demo: https://scasella.github.io/talkie-quant-webgpu/
Read the journey: https://github.com/scasella/talkie-quant-webgpu#performance-journey

## Discord Or Social

Talkie 1930 13B now has an unofficial ONNX/WebGPU quant:
https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX

I also published a static demo and runner:
https://scasella.github.io/talkie-quant-webgpu/

It runs fully client-side with Transformers.js plus direct ONNX Runtime WebGPU
for the KV-cache graph. The default fast cached q4 path passed a 16-word
Chrome/WebGPU smoke at about 3.60 tok/s rolling token latency after warmup.
First load is still big. The compressed opt-in got the best cold-start smoke
down to about 316.6s Ready / 17.3s TTFT, but the default stays conservative
until the 2x Ready target is actually met.

## Note To Original Talkie Community

I made an unofficial community ONNX/WebGPU quantization of Talkie 1930 13B for
browser use. The repo preserves the source tokenizer, chat template, generation
config, and Apache-2.0 metadata, and links clearly back to the original Talkie
project and HF Transformers-format conversion.

Model: https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX
Demo: https://scasella.github.io/talkie-quant-webgpu/
Code: https://github.com/scasella/talkie-quant-webgpu
