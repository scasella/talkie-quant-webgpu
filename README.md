# Talkie Quant WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU release for
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf).
This repo publishes browser-oriented ONNX artifacts and a static
[Transformers.js](https://huggingface.co/docs/transformers.js/index) WebGPU
loader for Talkie 1930 13B, with no backend and no browser-exposed credentials.
The smaller full-sequence q4f16 path has passed a Chrome/WebGPU smoke on a
24 GB M4 Pro; cached KV-cache artifacts are published but still experimental in
the browser.

- Live demo: <https://scasella.github.io/talkie-quant-webgpu/>
- Hugging Face model: <https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX>
- Source model: <https://huggingface.co/lewtun/talkie-1930-13b-it-hf>

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

The demo defaults to the smaller full-sequence q4f16 artifact. On a 24 GB M4
Pro, Chrome loaded it in about 8 minutes and streamed at about 0.61 tok/s in the
local smoke. Cached q4f16 is available in the model-path selector, but it is
still an experimental load path in Chrome/WebGPU.

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
8353531db9d507d96b8a92f5aceb12ff71b6b753
```

Later model-card-only commits may move the Hub repo HEAD without changing these
ONNX artifacts.

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_kv_q4f16.onnx` | hybrid q4f16 | Preferred cached browser path, 16.53&nbsp;GB | 32 |
| `onnx/model_kv_quantized.onnx` | hybrid q8 | Cached fallback path, 21.60&nbsp;GB | 42 |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Full-sequence fallback, 10.58&nbsp;GB | 22 |
| `onnx/model_quantized.onnx` | q8 | Full-sequence fallback, 15.31&nbsp;GB | 31 |

The app defaults to the smaller full-sequence q4f16 artifact and exposes cached
q4f16 as an explicit option. The cached path keeps `past_key_values.*` tensors
between tokens so generation avoids rerunning the full prompt every step, but it
has not yet passed the browser smoke on a 24 GB M4 Pro.

## Configuration

Point the app at another compatible public ONNX repo or a pinned revision:

```bash
VITE_TALKIE_ONNX_MODEL_ID=scasella91/talkie-1930-13b-it-ONNX \
VITE_TALKIE_ONNX_REVISION=8353531db9d507d96b8a92f5aceb12ff71b6b753 \
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

`path=full` selects the smaller full-sequence artifact, `path=cached` selects
the larger KV-cache artifact, and `cache=0` disables Transformers.js browser
cache writes. Cache writes are off by default because these multi-GB external
data files can create extra large browser-side allocations before WebGPU session
creation.

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
TALKIE_WEB_URL=http://127.0.0.1:5173/ \
TALKIE_ORT_FILE=model_q4f16.onnx \
TALKIE_ORT_CHUNKS=22 \
TALKIE_ORT_OPT=disabled \
npm run diagnose:ort
```

Export validation for the cached artifacts requires top-token agreement against
the PyTorch full-sequence wrapper. The published q4f16 cached artifact validated
on CUDA and CPU providers; the cached q8 fallback validated on the CPU provider
and is kept as a fallback for browser/WebGPU compatibility testing.

Current browser result on a 24 GB M4 Pro Chrome/WebGPU run:

- Full-sequence q4f16 reached `Ready` in Chromium with `cache=0` and ONNX
  Runtime graph optimization disabled.
- The same run generated 16 non-NUL words from a fixed prompt at about
  `0.61 tok/s`.
- Direct ORT and app loads with default graph optimization failed during session
  creation with `std::bad_alloc`; `opt=disabled` is the browser-safe default.
- Cached q4f16 remains experimental in Chrome/WebGPU.

## Known Limits

- First load is large and slow because the model is split across ONNX
  external-data files.
- Browser cache writes are disabled by default to reduce duplicate large
  allocations during cold load. Set `VITE_TALKIE_BROWSER_CACHE=1` or open with
  `?cache=1` only when testing cache behavior explicitly.
- The default browser path is full-sequence q4f16, so generation is slow. The
  latest local Chrome smoke measured about `0.61 tok/s` on an M4 Pro.
- Transformers.js does not have a native Talkie architecture wrapper, so this
  app uses `AutoModel` plus a manual generation loop instead of
  `pipeline("text-generation")`.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Cached decoding is the preferred path once `model_kv*` artifacts are present;
  the older full-sequence artifacts remain slower fallbacks.

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
