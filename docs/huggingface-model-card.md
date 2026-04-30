---
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
base_model: lewtun/talkie-1930-13b-it-hf
---

# Talkie 1930 13B IT ONNX WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU quantization of
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
for browser inference with Transformers.js.

- Live browser demo: <https://scasella.github.io/talkie-quant-webgpu/>
- GitHub runner and export scripts: <https://github.com/scasella/talkie-quant-webgpu>
- Source model revision: `6311dedf518470856a8503f2080bb4b54fcb3323`
- Validated ONNX artifact commit: `8353531db9d507d96b8a92f5aceb12ff71b6b753`
- Default browser dtype: `q4f16`
- Fallback dtype: `q8`
- Stop token IDs: `[65535, 65536]`

This model keeps the source tokenizer, chat template, generation config, and
Apache-2.0 license metadata. It is not an official Talkie release. The ONNX
artifacts validate outside the browser, and the smaller full-sequence q4f16 path
has passed a Chrome/WebGPU smoke on a 24 GB M4 Pro. Cached KV-cache artifacts
are published but still experimental in the browser.

Later model-card-only commits may move this repo's HEAD without changing the
validated ONNX artifacts.

## Quantization At A Glance

The preferred performance direction is the **cached q4f16 ONNX** file:
**16.53 GB** across **32** external-data chunks. It is larger than the smaller
full-sequence q4 artifact because the cached export intentionally leaves the
key/value projections that build the KV cache unquantized, which keeps cached
logits aligned with the source model while avoiding a full prompt re-run on
every generated token.

| Artifact | Size | Reduction vs source | Notes |
| --- | ---: | ---: | --- |
| BF16 source safetensors | 26.56&nbsp;GB | baseline | `lewtun/talkie-1930-13b-it-hf` |
| Cached q4f16 ONNX | 16.53&nbsp;GB | 38% smaller | Experimental KV-cache browser path; most MatMuls q4, K/V projections unquantized |
| Cached q8 ONNX fallback | 21.60&nbsp;GB | 19% smaller | KV-cache fallback; K/V projections unquantized |
| Full-sequence q4f16 fallback | 10.58&nbsp;GB | 60% smaller | Smaller download, slower generation |
| Full-sequence q8 fallback | 15.31&nbsp;GB | 42% smaller | Full-prompt fallback path |

The q4f16 files are larger than a theoretical pure 4-bit checkpoint because
ONNX stores scales, metadata, and unquantized tensors. The cached files trade a
larger first download for faster per-token decoding.

## Files

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_kv_q4f16.onnx` | hybrid q4f16 | Preferred cached browser path, 16.53&nbsp;GB | 32 |
| `onnx/model_kv_quantized.onnx` | hybrid q8 | Cached fallback path, 21.60&nbsp;GB | 42 |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Full-sequence fallback, 10.58&nbsp;GB | 22 |
| `onnx/model_quantized.onnx` | q8 | Full-sequence fallback, 15.31&nbsp;GB | 31 |

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

The app defaults to the smaller full-sequence q4f16 artifact and exposes cached
q4f16 as an explicit option. Cached q4f16 is the preferred performance direction
because it avoids rerunning the full prompt every token, but it is still an
experimental Chrome/WebGPU load path.

## Transformers.js Notes

Talkie has a custom `model_type`, so stock `pipeline("text-generation")` is not
used here. The browser runner formats messages with the shipped chat template,
tokenizes them, runs an explicit manual generation loop, samples on the CPU,
suppresses token `0`, and stops on token IDs `65535` or `65536`.

Minimal loading sketch:

```ts
import { AutoModel, AutoTokenizer } from "@huggingface/transformers";

const repo = "scasella91/talkie-1930-13b-it-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModel.from_pretrained(repo, {
  device: "webgpu",
  dtype: "q4f16"
});
```

Use the GitHub runner for the complete manual generation loop, including manual
`past_key_values.*` cache management for the cached files.

## Validation

- Hub artifact validation confirms tokenizer/config files, ONNX files, and all
  q4/q8 external-data chunks.
- The cached q4f16 artifact validated against the PyTorch full-sequence wrapper
  on CUDA and CPU providers.
- The cached q8 fallback validated against the same reference on the CPU
  provider and is kept as a browser/WebGPU fallback.
- Chromium on a 24 GB M4 Pro loaded full-sequence q4f16 with `cache=0` and ONNX
  Runtime graph optimization disabled, then generated 16 non-NUL words at about
  `0.61 tok/s`.
- Direct ORT and app loads with default graph optimization failed with
  `std::bad_alloc`; `opt=disabled` is the browser-safe default.

Re-run the public artifact check:

```bash
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
```

Require the cached KV artifacts after republishing:

```bash
python3 scripts/check_hub_artifacts.py --require-kv scasella91/talkie-1930-13b-it-ONNX
```

## Known Limitations

- First load is large and slow because the model is split across external-data
  chunks.
- Browser cache writes are disabled by default to reduce duplicate large
  allocations during cold load.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Cached decoding is the preferred path once `model_kv*` artifacts are present;
  the older full-sequence artifacts remain slower fallbacks.
- The default browser path is full-sequence q4f16, so generation is slow. The
  latest local Chrome smoke measured about `0.61 tok/s` on an M4 Pro.
- Cached KV-cache loading remains experimental in Chrome/WebGPU.
- This repo is public and assumes unauthenticated browser loading. Do not embed
  private Hugging Face tokens in browser code.

## Attribution

Talkie was developed by Alec Radford, Nick Levine, and David Duvenaud. This ONNX
repo builds on the Hugging Face Transformers-format conversion by
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
and the original Talkie project at <https://github.com/talkie-lm/talkie>.

## License

Apache-2.0, matching the source model metadata.
