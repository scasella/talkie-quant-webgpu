# Security

This repo is a static browser demo plus advanced maintainer scripts for
reproducing ONNX artifacts. The default app path does not require secrets.

## Credentials

- Keep `.env` local. It is ignored by git and should never be committed.
- `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `MODAL_TOKEN_ID`, and
  `MODAL_TOKEN_SECRET` are only for advanced export or Hub validation scripts.
- Do not add tokens to `VITE_*` variables. Vite exposes `VITE_*` values to the
  browser bundle.
- Do not publish screenshots, logs, or Modal output that contain tokens, local
  cache paths, or private repo names.

## Public Model Assumption

The static app assumes the ONNX model repo is public. If you point the app at a
private Hugging Face repo, add a server-side authentication layer. Do not embed
Hugging Face tokens in browser code.

## Expensive Operations

The Modal export path can spend real GPU budget and upload large Hub artifacts.
It is guarded by `--allow-talkie-export` and
`TALKIE_ALLOW_EXPENSIVE_EXPORT=1`. Run the cheap CUDA sanity check first and
use explicit timeouts.

## Reporting

If you find a security issue, open a private advisory or contact the maintainer
out of band before publishing details.
