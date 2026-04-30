# Talkie Quant WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU release for
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf).
This repo publishes browser-oriented ONNX artifacts and a static WebGPU runner
for Talkie 1930 13B, using
[Transformers.js](https://huggingface.co/docs/transformers.js/index) for
tokenizer/chat-template handling and direct ONNX Runtime WebGPU for the default
cached graph. There is no backend and no browser-exposed credential.
The demo now defaults to a direct ONNX Runtime WebGPU KV-cache path. On a
MacBook Pro with M4 Pro and 24 GB unified memory, the additive fast cached
q4f16 artifact generated 16 non-NUL words in Chrome/WebGPU at about **3.17
tok/s reported rolling latency** / **3.11 tok/s p50 token latency** after cold
load. First load is still large and slow, and first-token latency is still
called out below.

- Live demo: <https://scasella.github.io/talkie-quant-webgpu/>
- Hugging Face model: <https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX>
- Source model: <https://huggingface.co/lewtun/talkie-1930-13b-it-hf>

## Quantization At A Glance

The browser default is the **fast cached q4f16 ONNX** file: about **13.0 GB**
across **55** smaller external-data chunks. It uses the KV-cache graph and
quantizes the query/key attention projections while leaving the value projection
unquantized for quality. That preserves top-1 agreement on the smoke prompt and
avoids rerunning the full prompt every generated token.

| Artifact | Size | Reduction vs source | Notes |
| --- | ---: | ---: | --- |
| BF16 source safetensors | 26.56&nbsp;GB | baseline | `lewtun/talkie-1930-13b-it-hf` |
| Fast cached q4f16 ONNX default | 13.0&nbsp;GB | 51% smaller | Direct ORT KV-cache browser path; q/k quantized, value unquantized |
| Cached q4f16 ONNX fallback | 16.53&nbsp;GB | 38% smaller | KV-cache fallback; most MatMuls q4, q/k/v projections unquantized |
| Cached q8 ONNX fallback | 21.60&nbsp;GB | 19% smaller | KV-cache fallback; K/V projections unquantized |
| Full-sequence q4f16 fallback | 10.58&nbsp;GB | 60% smaller | Smaller download, slower generation |
| Full-sequence q8 fallback | 15.31&nbsp;GB | 42% smaller | Full-prompt fallback path |

The q4f16 files are larger than a theoretical pure 4-bit checkpoint because
ONNX stores scales, metadata, and unquantized tensors. The cached files trade
first-load size for faster steady per-token decoding.

## Performance Journey

The first working browser release optimized for correctness and availability:
export the custom Talkie architecture to ONNX, quantize the full-sequence graph,
ship tokenizer/config/chat-template files beside it, and prove that Chrome
could stream real non-NUL text. That path worked, but it had the expected
latency problem: each generated token reran the entire accumulated prompt. The
24 GB M4 Pro smoke measured about **0.61 tok/s** on the full-sequence q4f16
artifact.

The faster path came from treating the browser as the system of record, not just
the export. The steps that mattered:

1. Exported KV-cache ONNX graphs so generation could feed only the newest token
   after prompt prefill.
2. Kept a custom generation loop because Talkie is not a stock
   Transformers.js causal-LM architecture.
3. Moved cached execution to direct `onnxruntime-web/webgpu`, while still using
   Transformers.js for tokenizer and chat-template handling.
4. Fed `input_ids`, `position_ids`, and `past_key_values.*` manually; kept
   `present.*` tensors on GPU where possible and copied only logits back to the
   CPU sampler.
5. Disabled ONNX Runtime graph optimization in the browser after default
   optimization repeatedly hit `std::bad_alloc` during session creation.
6. Added a page-level Hugging Face fetch limiter. A service-worker-only queue
   still let too many long-lived model fetches pile up and hit browser request
   timeouts; limiting large ONNX chunk fetches from the page was the practical
   fix.
7. Reworked sampling and UI updates so the browser was not sorting the whole
   vocabulary or repainting on every tiny decode update.

Quantization was also trial and error. A more aggressive fast q4 attempt that
quantized all attention projections failed the reference top-5 validation. The
published fast q4 path quantizes query/key projections but leaves the value
projection unquantized. That version preserved top-1 agreement on the smoke
prompt and produced the current default `model_kv_fast_q4f16.onnx` artifact.

Current measured result: the MacBook Pro M4 Pro / 24 GB unified-memory smoke
reached about **3.17 tok/s reported rolling token latency** and **3.11 tok/s p50
token latency** with `kv-cache` / `ort-direct`, a little over 5x the original
steady token rate. The honest remaining work is cold load and TTFT: the
measured cached run still took about **528.7s to Ready** and about **43.1s to
first token**.

## Why This Exists

Talkie has drawn substantial interest as a 13B instruction model trained on
pre-1931 English. Until this release, there was no ready-to-use browser/WebGPU
quantized artifact for people who wanted to try it locally without setting up a
CUDA machine.

This is not an official Talkie release. It is a community quantization and
browser runner that preserves the source tokenizer, chat template, generation
config, and Apache-2.0 license metadata.

## Try The Demo

Requirements:

- Chrome or another browser with WebGPU enabled
- Enough disk and memory for multi-GB ONNX external-data chunks
- Patience on first load while the browser fetches and compiles the model

Open the live demo:

```text
https://scasella.github.io/talkie-quant-webgpu/
```

The demo defaults to cached q4f16. On the validated MacBook Pro M4 Pro / 24 GB
Chrome smoke, the fast cached q4f16 artifact loaded in about 8.8 minutes,
reached first token in about 43 seconds after sending the prompt, and then
reported about 3.17 tok/s rolling token latency. The smaller full-sequence
q4f16 fallback is still available in the model-path selector.

Run locally:

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`.

## Model Artifacts

Default model repo:

```text
scasella91/talkie-1930-13b-it-ONNX
```

Validated artifact commit:

```text
631cbea56319f30469aae41af8fbd3078c460b3b
```

Later model-card-only commits may move the Hub repo HEAD without changing the
validated ONNX artifacts.

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_kv_fast_q4f16.onnx` | hybrid q4f16 | Default direct ORT cached browser path, about 13.0&nbsp;GB | 55 |
| `onnx/model_kv_q4f16.onnx` | hybrid q4f16 | Conservative cached fallback, 16.53&nbsp;GB | 32 |
| `onnx/model_kv_quantized.onnx` | hybrid q8 | Cached fallback path, 21.60&nbsp;GB | 42 |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Full-sequence fallback, 10.58&nbsp;GB | 22 |
| `onnx/model_quantized.onnx` | q8 | Full-sequence fallback, 15.31&nbsp;GB | 31 |

The app defaults to cached q4f16 and falls back to the older cached/full
artifacts if the direct ORT path fails. The cached path keeps
`past_key_values.*` tensors between tokens so generation avoids rerunning the
full prompt every step.

## Configuration

Point the app at another compatible public ONNX repo or a pinned revision:

```bash
VITE_TALKIE_ONNX_MODEL_ID=scasella91/talkie-1930-13b-it-ONNX \
VITE_TALKIE_ONNX_REVISION=631cbea56319f30469aae41af8fbd3078c460b3b \
npm run dev
```

Force a dtype:

```bash
VITE_TALKIE_ONNX_DTYPE=q8 npm run dev
```

For browser diagnostics, the app also accepts URL parameters without a rebuild:

```text
http://127.0.0.1:5173/?path=full&dtype=q4f16&cache=0
```

`path=cached` selects the KV-cache artifact, `path=full` selects the smaller
full-sequence artifact, and `cache=0` disables Transformers.js browser cache
writes. Cache writes are off by default because these multi-GB external data
files can create extra large browser-side allocations before WebGPU session
creation. The direct ORT path also limits concurrent Hugging Face model fetches;
override with `?fetches=8` or `VITE_TALKIE_FETCH_CONCURRENCY=8` when comparing
network behavior.

The app also defaults ONNX Runtime graph optimization to `disabled` for browser
loads. A Playwright direct-ORT probe reached `Ready` on the full q4f16 artifact
with `opt=disabled`; the same direct probe failed with `std::bad_alloc` at the
default `all` optimization level. Use `?opt=all` only when comparing runtime
failure modes.

Build for GitHub Pages:

```bash
npm run build:pages
```

## Validation

Check the published Hub layout:

```bash
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
```

Require the cached KV artifacts after republishing:

```bash
python3 scripts/check_hub_artifacts.py --require-kv scasella91/talkie-1930-13b-it-ONNX
```

Require the additive fast cached q4 artifact:

```bash
python3 scripts/check_hub_artifacts.py --require-kv --require-fast-kv-q4 scasella91/talkie-1930-13b-it-ONNX
```

Browser smoke test:

```bash
TALKIE_WEB_URL=https://scasella.github.io/talkie-quant-webgpu/ \
TALKIE_WEB_EXPECTED_DTYPE=q4f16 \
TALKIE_WEB_MIN_TOKENS=16 \
python3 scripts/validate_browser_webgpu.py
```

The smoke test launches Chromium with WebGPU, loads the ONNX repo, sends a fixed
prompt, and requires non-NUL streamed text.

Focused load diagnostics:

```bash
TALKIE_WEB_URL=http://127.0.0.1:5173/ \
TALKIE_LOAD_PATH=full \
TALKIE_DTYPE=q4f16 \
TALKIE_BROWSER_CACHE=0 \
npm run diagnose:browser
```

This launches a fresh Playwright Chromium session, records console errors,
network responses, memory snapshots, and the last visible load state, then
writes the trace under `output/playwright/`.

To include the 16-word generation smoke in the same Playwright session:

```bash
TALKIE_WEB_URL=http://127.0.0.1:5173/ \
TALKIE_LOAD_PATH=full \
TALKIE_DTYPE=q4f16 \
TALKIE_BROWSER_CACHE=0 \
TALKIE_DIAG_GENERATE=1 \
TALKIE_WEB_MIN_TOKENS=16 \
npm run diagnose:browser
```

Direct ONNX Runtime loader comparison:

```bash
TALKIE_WEB_URL='http://127.0.0.1:5173/?fetches=6' \
TALKIE_ORT_FILE=model_kv_fast_q4f16.onnx \
TALKIE_ORT_CHUNKS=55 \
TALKIE_ORT_OPT=disabled \
npm run diagnose:ort
```

Repeatable latency benchmark:

```bash
TALKIE_WEB_URL='http://127.0.0.1:5173/?fetches=6' \
TALKIE_BENCH_TARGETS=cached-q4f16 \
TALKIE_BENCH_RUNS=1 \
TALKIE_WEB_MIN_TOKENS=16 \
npm run benchmark:browser
```

Export validation for the cached artifacts requires top-token agreement against
the PyTorch full-sequence wrapper. The additive fast cached q4f16 artifact keeps
the value projection unquantized and validated top-1 against the reference
prompt; the all-projection fast q4 attempt failed top-5 validation and was not
published as the default. The cached q8 fallback validated on the CPU provider
and is kept as a fallback for browser/WebGPU compatibility testing.

Current browser result on a MacBook Pro with M4 Pro and 24 GB unified memory
using Chrome/WebGPU:

- Cached q4f16 direct ORT reached `Ready` in Chromium with `cache=0`,
  `opt=disabled`, and `fetches=6`.
- The same run generated 16 non-NUL words from a fixed prompt with `kv-cache`
  / `ort-direct`, no NUL tokens, about `3.17 tok/s` reported rolling latency,
  and about `3.11 tok/s` p50 token latency.
- Cold load remained large: about `528.7s` to `Ready` and about `43.1s` TTFT
  on the measured run.
- Full-sequence q4f16 remains available as the smaller fallback; the earlier
  smoke generated 16 non-NUL words at about `0.61 tok/s`.
- Direct ORT and app loads with default graph optimization failed during session
  creation with `std::bad_alloc`; `opt=disabled` is the browser-safe default.

## Known Limits

- First load is large and slow because the model is split across ONNX
  external-data files.
- Browser cache writes are disabled by default to reduce duplicate large
  allocations during cold load. Set `VITE_TALKIE_BROWSER_CACHE=1` or open with
  `?cache=1` only when testing cache behavior explicitly.
- The default browser path is cached q4f16 and is faster after the first token,
  but cold load and first-token latency are still slow.
- Transformers.js does not have a native Talkie architecture wrapper. The app
  uses Transformers.js for tokenizer/chat-template handling, direct ONNX Runtime
  for the default cached path, and an `AutoModel` manual-loop fallback instead
  of `pipeline("text-generation")`.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- The first several cached tokens can still be slow while WebGPU compiles and
  warms shapes. The displayed tok/sec is a rolling token-latency rate, not a
  cold-start average.
- The older full-sequence artifacts remain slower fallbacks.
- If Chrome logs 404s for generic filenames such as `onnx/model.onnx`,
  `onnx/model_uint8.onnx`, or `onnx/model_q4.onnx`, the browser is running an
  older app bundle that used Transformers.js dtype probing. The published
  artifacts use explicit filenames such as `onnx/model_kv_fast_q4f16.onnx`.
  Hard refresh the demo, or open it in a fresh profile, so the latest bundle is
  used.

## Advanced: Reproduce ONNX Artifacts

Most users do not need Modal, CUDA, or Hugging Face credentials. The scripts in
`scripts/` are maintainer tools for reproducing or republishing the ONNX model.

Credentials are read from local `.env` and forwarded to Modal as environment
variables. `.env` is ignored by git and excluded from Modal image uploads.

Run a cheap CUDA sanity check first:

```bash
python3 scripts/modal_gpu.py --gpu=T4 --timeout=10 -- \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The full export and quantization path can spend real H100 budget and upload
large Hub artifacts. It is guarded by `TALKIE_ALLOW_EXPENSIVE_EXPORT=1`:

```bash
TALKIE_ALLOW_EXPENSIVE_EXPORT=1 \
python3 scripts/modal_gpu.py --allow-talkie-export --gpu=H100 --memory=524288 --timeout=240 -- \
  python scripts/export_talkie_onnx.py \
    --legacy-export \
    --work-dir /cache/talkie-onnx \
    --output-repo scasella91/talkie-1930-13b-it-ONNX \
    --revision 6311dedf518470856a8503f2080bb4b54fcb3323 \
    --kv-cache
```

The additive fast q4 cached artifact was produced from the same raw KV export
with query/key projection quantization enabled and value projection left
unquantized:

```bash
TALKIE_ALLOW_EXPENSIVE_EXPORT=1 \
python3 scripts/modal_gpu.py --allow-talkie-export --gpu=H100 --memory=524288 --timeout=300 -- \
  python scripts/export_talkie_onnx.py \
    --work-dir /cache/talkie-onnx-fast \
    --kv-cache-fast \
    --skip-q8 \
    --kv-q4-exclude-attn-projections value \
    --external-data-chunk-mib 256
```

## Release Assets

- Hugging Face model-card source: [`docs/huggingface-model-card.md`](./docs/huggingface-model-card.md)
- Outreach copy: [`docs/outreach.md`](./docs/outreach.md)
- Release checklist: [`RELEASE_CHECKLIST.md`](./RELEASE_CHECKLIST.md)
- Security notes: [`SECURITY.md`](./SECURITY.md)

## Attribution

Talkie was developed by Alec Radford, Nick Levine, and David Duvenaud. This repo
depends on the Hugging Face Transformers-format conversion at
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
and the original Talkie project at <https://github.com/talkie-lm/talkie>.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).
