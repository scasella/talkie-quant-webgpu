# Release Checklist

Use this checklist before publishing or tagging the repo.

## Current Artifact Evidence

- GitHub repo: `scasella/talkie-quant-webgpu`
- Live demo: `https://scasella.github.io/talkie-quant-webgpu/`
- Hub repo: `scasella91/talkie-1930-13b-it-ONNX`
- Validated ONNX artifact commit: `addb95a08622583a3017576e9442a9d6853e88c1`
- Browser default: `onnx/model_q4f16.onnx`
- q4 external-data chunks: `10`
- q8 fallback: `onnx/model_quantized.onnx`
- q8 external-data chunks: `15`
- Smoke prompt target: at least `16` streamed non-NUL tokens over WebGPU

## Required Checks

```bash
npm run build
npm run build:pages
python3 -m py_compile scripts/*.py
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
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
