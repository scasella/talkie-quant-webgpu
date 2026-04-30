# Release Checklist

Use this checklist before publishing or tagging the repo.

## Current Artifact Evidence

- GitHub repo: `scasella/talkie-quant-webgpu`
- Live demo: `https://scasella.github.io/talkie-quant-webgpu/`
- Hub repo: `scasella91/talkie-1930-13b-it-ONNX`
- Latest GitHub release: `v0.2.0` fast cached WebGPU path
- Validated ONNX artifact commit: `631cbea56319f30469aae41af8fbd3078c460b3b`
- Latest HF model-card update: `0c5e983a8b218c94f21ab1ca9834cdf26500f7cf`
- Latest HF gzip artifact update: `5b0876acd92f3b1cf33ddb4156195e748a7f9fa8`
- Browser default: `onnx/model_kv_fast_q4f16.onnx`
- Fast cached q4 artifact: `onnx/model_kv_fast_q4f16.onnx`, `55` chunks,
  about `13.0 GB`
- Fast cached q4 gzip companions: `onnx/model_kv_fast_q4f16.onnx_data*.gz`,
  `55` chunks, about `8.85 GB` transferred when `compressed=1`
- Cold64 q4 opt-in candidate: `onnx/model_kv_cold64_q4f16.onnx`, `54` chunks,
  about `12.25 GB`
- Cached q4 fallback: `onnx/model_kv_q4f16.onnx`, `32` chunks, `16.53 GB`
- Cached q8 fallback: `onnx/model_kv_quantized.onnx`
- Cached q8 external-data chunks: `42` chunks, `21.60 GB`
- Full-sequence q4 fallback: `onnx/model_q4f16.onnx`, `22` chunks
- Full-sequence q8 fallback: `onnx/model_quantized.onnx`, `31` chunks
- Smoke prompt target: at least `16` streamed non-NUL words over WebGPU
- Current browser gate: passed in Chromium on a 24 GB M4 Pro using cached
  q4f16 direct ORT, `cache=0`, `opt=disabled`, `fetches=4`, and the default
  warmup. Evidence: load reached `Ready` in about `472.4s`, TTFT was about
  `1.1s`, and the run generated 16 non-NUL words with `kv-cache` /
  `ort-direct`, about `3.60 tok/s` reported rolling latency, and about
  `3.58 tok/s` p50 token latency.
- Best no-warmup fetch-concurrency run: `530.6s` Ready / `17.4s` TTFT /
  `3.69 tok/s` with `fetches=4`.
- Cold64 q4 opt-in candidate passed the same browser smoke at about `484.2s`
  Ready / `19.0s` TTFT / `3.67 tok/s`, but did not meet the 2x cold-start
  target, so the default remains `onnx/model_kv_fast_q4f16.onnx`.
- Gzip external-data opt-in passed the browser smoke at about `316.6s` Ready /
  `17.3s` TTFT / `3.61 tok/s` with `compressed=1`, `warmup=0`, `fetches=8`,
  `cache=0`, and `opt=disabled`. It improves cold load by about 40% versus the
  original `528.7s` Ready baseline, but misses the `<=265s` target, so it
  remains opt-in.
- Full-sequence q4f16 fallback previously generated 16 non-NUL words at about
  `0.61 tok/s`.
- Public docs should preserve the performance journey: full-sequence baseline,
  KV-cache export, direct ORT runtime, graph-optimization failure,
  fetch-limiting fix, failed all-projection q4 candidate, and the final
  q/k-quantized value-unquantized fast q4 artifact.

## Required Checks

```bash
npm run build
npm run build:pages
python3 -m py_compile scripts/*.py
python3 scripts/check_hub_artifacts.py --require-kv --require-fast-kv-q4 --expect-model model_kv_cold64_q4f16.onnx scasella91/talkie-1930-13b-it-ONNX
TALKIE_WEB_URL='http://127.0.0.1:5173/?fetches=4' TALKIE_BENCH_TARGETS=cached-q4f16 TALKIE_BENCH_RUNS=1 TALKIE_WEB_MIN_TOKENS=16 TALKIE_DIRECT_WARMUP=1 npm run benchmark:browser
TALKIE_WEB_URL='http://127.0.0.1:5173/?compressed=1&warmup=0&fetches=8' TALKIE_BENCH_TARGETS=cached-q4f16 TALKIE_BENCH_RUNS=1 TALKIE_WEB_MIN_TOKENS=16 TALKIE_DIRECT_WARMUP=0 TALKIE_COMPRESSED_EXTERNAL_DATA=1 npm run benchmark:browser
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
