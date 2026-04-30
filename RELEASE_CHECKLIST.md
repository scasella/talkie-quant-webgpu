# Release Checklist

Use this checklist before publishing or tagging the repo.

## Current Artifact Evidence

- GitHub repo: `scasella/talkie-quant-webgpu`
- Live demo: `https://scasella.github.io/talkie-quant-webgpu/`
- Hub repo: `scasella91/talkie-1930-13b-it-ONNX`
- Validated ONNX artifact commit: `8353531db9d507d96b8a92f5aceb12ff71b6b753`
- Browser default: `onnx/model_q4f16.onnx`
- Cached q4 artifact: `onnx/model_kv_q4f16.onnx`, `32` chunks, `16.53 GB`
- Cached q8 fallback: `onnx/model_kv_quantized.onnx`
- Cached q8 external-data chunks: `42` chunks, `21.60 GB`
- Full-sequence q4 fallback: `onnx/model_q4f16.onnx`, `22` chunks
- Full-sequence q8 fallback: `onnx/model_quantized.onnx`, `31` chunks
- Smoke prompt target: at least `16` streamed non-NUL words over WebGPU
- Current browser gate: passed in Chromium on a 24 GB M4 Pro using
  full-sequence q4f16, `cache=0`, and ONNX Runtime graph optimization disabled.
  Evidence: load reached `Ready` in about 8 minutes and generated 16 non-NUL
  words at about `0.61 tok/s`.

## Required Checks

```bash
npm run build
npm run build:pages
python3 -m py_compile scripts/*.py
python3 scripts/check_hub_artifacts.py --require-kv scasella91/talkie-1930-13b-it-ONNX
TALKIE_WEB_URL=https://scasella.github.io/talkie-quant-webgpu/ TALKIE_WEB_EXPECTED_DTYPE=q4f16 TALKIE_WEB_MIN_TOKENS=16 python3 scripts/validate_browser_webgpu.py
```

## Hygiene

- `.env` remains local and contains no committed token values.
- `.env.example` contains placeholders only.
- `dist/`, `.playwright-mcp/`, `.DS_Store`, `__pycache__/`, ONNX blobs, Modal
  logs, and local cache directories are absent from the source tree.
- Normal users can run the demo with only `npm install` and `npm run dev`.
- Modal and Hugging Face credentials are documented as advanced maintainer-only
  requirements.
- GitHub topics include `talkie`, `webgpu`, `onnx`, `transformers-js`,
  `huggingface`, `quantization`, `llm`, `browser-ai`, and `text-generation`.
- Hugging Face card links the GitHub repo, live demo, source model, and original
  Talkie project.
