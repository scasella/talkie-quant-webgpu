# Talkie Quant WebGPU

Unofficial community q4f16/q8 ONNX + WebGPU release for
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf).
This repo makes Talkie 1930 13B usable from a static browser app with
[Transformers.js](https://huggingface.co/docs/transformers.js/index), no backend,
and no browser-exposed credentials.

- Live demo: <https://scasella.github.io/talkie-quant-webgpu/>
- Hugging Face model: <https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX>
- Source model: <https://huggingface.co/lewtun/talkie-1930-13b-it-hf>

## Quantization At A Glance

The default browser artifact is **10.58 GB**, down from the **26.56 GB** BF16
source weights: about **60% smaller** and **2.5x compressed**. The q8 fallback is
**15.31 GB**, about **42% smaller** than the source.

| Artifact | Size | Reduction vs source | Notes |
| --- | ---: | ---: | --- |
| BF16 source safetensors | 26.56 GB | baseline | `lewtun/talkie-1930-13b-it-hf` |
| q4f16 ONNX default | 10.58 GB | 60% smaller | Nominal q4 weights; roughly 6.4 bits/parameter on disk |
| q8 ONNX fallback | 15.31 GB | 42% smaller | Roughly 9.2 bits/parameter on disk |

The q4f16 file is larger than a theoretical pure 4-bit checkpoint because ONNX
stores scales, metadata, and some unquantized tensors; this artifact also keeps
runtime tensors in float32 where needed for WebGPU stability.

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

Click **Load**, wait for `Ready` and `q4f16`, then send a prompt.

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
addb95a08622583a3017576e9442a9d6853e88c1
```

Later model-card-only commits may move the Hub repo HEAD without changing these
ONNX artifacts.

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Default browser path, 10.58 GB | 10 |
| `onnx/model_quantized.onnx` | q8 | Fallback path, 15.31 GB | 15 |

The app tries `q4f16` first and falls back to `q8` if q4f16 loading fails.
Generation is full-sequence: each new token reruns the accumulated `input_ids`
because this v1 export does not include a KV-cache graph.

## Configuration

Point the app at another compatible public ONNX repo or a pinned revision:

```bash
VITE_TALKIE_ONNX_MODEL_ID=scasella91/talkie-1930-13b-it-ONNX \
VITE_TALKIE_ONNX_REVISION=addb95a08622583a3017576e9442a9d6853e88c1 \
npm run dev
```

Force a dtype:

```bash
VITE_TALKIE_ONNX_DTYPE=q8 npm run dev
```

Build for GitHub Pages:

```bash
npm run build:pages
```

## Validation

Check the published Hub layout:

```bash
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
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

## Known Limits

- First load is large and slow because the model is split across ONNX
  external-data files.
- Browser cache writes may hit quota. That can be noisy but is not fatal if the
  model reaches `Ready`.
- Transformers.js does not have a native Talkie architecture wrapper, so this
  app uses `AutoModel` plus a manual generation loop instead of
  `pipeline("text-generation")`.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Full-sequence decoding is slower than a KV-cache decoder. KV-cache support is
  future work.

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
    --revision 6311dedf518470856a8503f2080bb4b54fcb3323
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
