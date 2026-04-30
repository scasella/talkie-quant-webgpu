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
- llm
- instruction-tuned
- conversational
- talkie
- vintage
- historical
- quantized
- onnxruntime
- onnxruntime-web
- kv-cache
- browser-ai
- browser
- client-side-inference
- edge-ai
- apple-silicon
- macos
- q4f16
- q8
pipeline_tag: text-generation
base_model: lewtun/talkie-1930-13b-it-hf
---

# Talkie 1930 13B IT ONNX WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU quantization of
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
for browser inference with Transformers.js tokenizer/chat-template handling and
direct ONNX Runtime WebGPU cached decoding.

- Live browser demo: <https://scasella.github.io/talkie-quant-webgpu/>
- GitHub runner and export scripts: <https://github.com/scasella/talkie-quant-webgpu>
- Source model revision: `6311dedf518470856a8503f2080bb4b54fcb3323`
- Validated ONNX artifact commit: `631cbea56319f30469aae41af8fbd3078c460b3b`
- Default browser dtype: `q4f16`
- Fallback dtype: `q8`
- Stop token IDs: `[65535, 65536]`
- Best measured browser speed: `3.17 tok/s` reported rolling token latency on
  a MacBook Pro with M4 Pro and 24 GB unified memory

This model keeps the source tokenizer, chat template, generation config, and
Apache-2.0 license metadata. It is not an official Talkie release. The ONNX
artifacts validate outside the browser. The default fast cached q4f16 path has
also passed a Chrome/WebGPU smoke on a MacBook Pro with M4 Pro and 24 GB unified
memory using direct ONNX Runtime WebGPU.

Later model-card-only commits may move this repo's HEAD without changing the
validated ONNX artifacts.

## Badges And Discovery

The model-card metadata is set so the Hugging Face page badges and filters match
what this repo actually ships:

- **License:** `apache-2.0`, matching the source model metadata and repo
  license.
- **Language:** `en`, because Talkie targets English text.
- **Library:** `transformers.js`, because the browser runner uses
  Transformers.js for tokenizer/chat-template handling.
- **Task:** `text-generation`, because the artifact is a decoder-only chat/text
  generation model even though the app uses a custom manual loop instead of
  stock `pipeline("text-generation")`.
- **Base model:** `lewtun/talkie-1930-13b-it-hf`, so Hugging Face links and
  filters this as a quantized derivative of the source model.
- **Discovery tags:** `webgpu`, `onnx`, `onnxruntime-web`, `kv-cache`,
  `browser-ai`, `client-side-inference`, `edge-ai`, `apple-silicon`, `macos`,
  `q4f16`, and `q8`.

## Quantization At A Glance

The default browser artifact is the **fast cached q4f16 ONNX** file: about
**13.0 GB** across **55** smaller external-data chunks. It quantizes query/key
attention projections while leaving value projections unquantized, preserving
top-1 agreement on the smoke prompt and avoiding a full prompt re-run on every
generated token.

| Artifact | Size | Reduction vs source | Notes |
| --- | ---: | ---: | --- |
| BF16 source safetensors | 26.56&nbsp;GB | baseline | `lewtun/talkie-1930-13b-it-hf` |
| Fast cached q4f16 ONNX default | 13.0&nbsp;GB | 51% smaller | Direct ORT KV-cache browser path; q/k quantized, value unquantized |
| Cached q4f16 ONNX fallback | 16.53&nbsp;GB | 38% smaller | KV-cache fallback; q/k/v projections unquantized |
| Cached q8 ONNX fallback | 21.60&nbsp;GB | 19% smaller | KV-cache fallback; K/V projections unquantized |
| Full-sequence q4f16 fallback | 10.58&nbsp;GB | 60% smaller | Smaller download, slower generation |
| Full-sequence q8 fallback | 15.31&nbsp;GB | 42% smaller | Full-prompt fallback path |

The q4f16 files are larger than a theoretical pure 4-bit checkpoint because
ONNX stores scales, metadata, and unquantized tensors. The cached files trade
first-load size for faster steady per-token decoding.

## Performance Journey

This repo started with the browser as the acceptance test. The first validated
path was the smaller full-sequence q4f16 graph: it loaded in Chrome/WebGPU and
generated real text, but every new token reran the whole accumulated prompt. On
the MacBook Pro M4 Pro / 24 GB unified-memory smoke, that measured about
`0.61 tok/s`.

The current default is the result of a few browser-specific turns:

- KV-cache ONNX export so decode can feed only the newest token after prefill.
- A custom Talkie generation loop because this is not a stock Transformers.js
  causal-LM architecture.
- Direct `onnxruntime-web/webgpu` execution for the cached graph, while keeping
  Transformers.js for tokenizer and chat-template handling.
- Manual `input_ids`, `position_ids`, and `past_key_values.*` feeds, with
  `present.*` cache tensors kept on GPU where possible and only logits copied
  back for CPU sampling.
- Browser graph optimization set to `disabled` after default optimization hit
  `std::bad_alloc` during session creation.
- A page-level Hugging Face fetch limiter. A service-worker-only queue still let
  too many long model fetches accumulate and hit browser request timeouts.
- Partial top-k sampling and throttled UI updates to keep JavaScript overhead
  from dominating decode.

The quantization path also had a failed candidate: quantizing all attention
projections was faster-looking on paper but failed reference top-5 validation.
The published fast q4f16 artifact quantizes query/key projections and leaves the
value projection unquantized. That preserved top-1 agreement on the smoke prompt
and produced the default `onnx/model_kv_fast_q4f16.onnx`.

Current browser smoke result: `kv-cache` / `ort-direct` generated 16 non-NUL
words at about `3.17 tok/s` reported rolling token latency and `3.11 tok/s` p50
token latency, over 5x the original full-sequence steady token rate. Cold load
is still the big caveat: about `528.7s` to `Ready` and about `43.1s` TTFT in
the measured run.

## Files

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_kv_fast_q4f16.onnx` | hybrid q4f16 | Default direct ORT cached browser path, about 13.0&nbsp;GB | 55 |
| `onnx/model_kv_q4f16.onnx` | hybrid q4f16 | Conservative cached fallback, 16.53&nbsp;GB | 32 |
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

The app defaults to cached q4f16. The smaller full-sequence q4f16 artifact
remains available as a fallback.

## Transformers.js Notes

Talkie has a custom `model_type`, so stock `pipeline("text-generation")` is not
used here. The browser runner formats messages with the shipped chat template,
tokenizes them, runs an explicit manual generation loop, samples on the CPU,
suppresses token `0`, and stops on token IDs `65535` or `65536`.

Minimal tokenizer and fallback-model loading sketch:

```ts
import { AutoModel, AutoTokenizer } from "@huggingface/transformers";

const repo = "scasella91/talkie-1930-13b-it-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModel.from_pretrained(repo, {
  device: "webgpu",
  dtype: "q4f16"
});
```

Use the GitHub runner for the complete manual generation loop, including direct
ONNX Runtime loading, limited concurrent model fetches, and manual
`past_key_values.*` cache management for the cached files.

## Validation

- Hub artifact validation confirms tokenizer/config files, ONNX files, and all
  q4/q8 external-data chunks, including the additive fast cached q4 artifact.
- The additive fast cached q4f16 artifact validated top-1 against the PyTorch
  full-sequence wrapper with value projection left unquantized. The all-projection
  fast q4 attempt failed top-5 validation and was not made the default.
- The cached q8 fallback validated against the same reference on the CPU
  provider and is kept as a browser/WebGPU fallback.
- Chrome on a MacBook Pro with M4 Pro and 24 GB unified memory loaded cached
  q4f16 with `cache=0`, `opt=disabled`, and `fetches=6`, then generated 16
  non-NUL words with
  `kv-cache` / `ort-direct`, about `3.17 tok/s` reported rolling latency, and
  about `3.11 tok/s` p50 token latency.
- Cold load remains slow: the measured run took about `528.7s` to `Ready` and
  about `43.1s` TTFT.
- Direct ORT and app loads with default graph optimization failed with
  `std::bad_alloc`; `opt=disabled` is the browser-safe default.

Re-run the public artifact check:

```bash
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
```

Require the cached KV artifacts after republishing:

```bash
python3 scripts/check_hub_artifacts.py --require-kv --require-fast-kv-q4 scasella91/talkie-1930-13b-it-ONNX
```

## Known Limitations

- First load is large and slow because the model is split across external-data
  chunks.
- Browser cache writes are disabled by default to reduce duplicate large
  allocations during cold load.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Cached q4f16 is faster after the first token, but first load and first-token
  latency are still slow.
- The displayed tok/sec is a rolling token-latency rate, not a cold-start
  average.
- The older full-sequence artifacts remain slower fallbacks.
- This repo is public and assumes unauthenticated browser loading. Do not embed
  private Hugging Face tokens in browser code.

## Attribution

Talkie was developed by Alec Radford, Nick Levine, and David Duvenaud. This ONNX
repo builds on the Hugging Face Transformers-format conversion by
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
and the original Talkie project at <https://github.com/talkie-lm/talkie>.

## License

Apache-2.0, matching the source model metadata.
