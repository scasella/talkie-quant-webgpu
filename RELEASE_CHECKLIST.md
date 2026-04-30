# Release Checklist

Use this checklist before publishing or tagging the repo.

## Current Artifact Evidence

- GitHub repo: `scasella/talkie-quant-webgpu`
- Live demo: `https://scasella.github.io/talkie-quant-webgpu/`
- Hub repo: `scasella91/talkie-1930-13b-it-ONNX`
- Validated ONNX artifact commit: `631cbea56319f30469aae41af8fbd3078c460b3b`
- Browser default: `onnx/model_kv_fast_q4f16.onnx`
- Fast cached q4 artifact: `onnx/model_kv_fast_q4f16.onnx`, `55` chunks,
  about `13.0 GB`
- Cached q4 fallback: `onnx/model_kv_q4f16.onnx`, `32` chunks, `16.53 GB`
- Cached q8 fallback: `onnx/model_kv_quantized.onnx`
- Cached q8 external-data chunks: `42` chunks, `21.60 GB`
- Full-sequence q4 fallback: `onnx/model_q4f16.onnx`, `22` chunks
- Full-sequence q8 fallback: `onnx/model_quantized.onnx`, `31` chunks
- Smoke prompt target: at least `16` streamed non-NUL words over WebGPU
- Current browser gate: passed in Chromium on a 24 GB M4 Pro using cached
  q4f16 direct ORT, `cache=0`, `opt=disabled`, and `fetches=6`. Evidence:
  load reached `Ready` in about `528.7s`, TTFT was about `43.1s`, and the run
  generated 16 non-NUL words with `kv-cache` / `ort-direct`, about `3.17 tok/s`
  reported rolling latency, and about `3.11 tok/s` p50 token latency.
- Full-sequence q4f16 fallback previously generated 16 non-NUL words at about
  `0.61 tok/s`.

## Required Checks

```bash
npm run build
npm run build:pages
python3 -m py_compile scripts/*.py
python3 scripts/check_hub_artifacts.py --require-kv --require-fast-kv-q4 scasella91/talkie-1930-13b-it-ONNX
TALKIE_WEB_URL='http://127.0.0.1:5173/?fetches=6' TALKIE_BENCH_TARGETS=cached-q4f16 TALKIE_BENCH_RUNS=1 TALKIE_WEB_MIN_TOKENS=16 npm run benchmark:browser
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
