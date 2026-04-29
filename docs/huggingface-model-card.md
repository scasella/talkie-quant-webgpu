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
- Validated ONNX artifact commit: `addb95a08622583a3017576e9442a9d6853e88c1`
- Default browser dtype: `q4f16`
- Fallback dtype: `q8`
- Stop token IDs: `[65535, 65536]`

This model keeps the source tokenizer, chat template, generation config, and
Apache-2.0 license metadata. It is not an official Talkie release.

Later model-card-only commits may move this repo's HEAD without changing the
validated ONNX artifacts.

## Quantization At A Glance

The default browser artifact is **10.58 GB**, down from the **26.56 GB** BF16
source weights: about **60% smaller** and **2.5x compressed**. The q8 fallback is
**15.31 GB**, about **42% smaller** than the source.

| Artifact | Size | Reduction vs source | Notes |
| --- | ---: | ---: | --- |
| BF16 source safetensors | 26.56&nbsp;GB | baseline | `lewtun/talkie-1930-13b-it-hf` |
| q4f16 ONNX default | 10.58&nbsp;GB | 60% smaller | Nominal q4 weights; roughly 6.4 bits/parameter on disk |
| q8 ONNX fallback | 15.31&nbsp;GB | 42% smaller | Roughly 9.2 bits/parameter on disk |

The q4f16 file is larger than a theoretical pure 4-bit checkpoint because ONNX
stores scales, metadata, and some unquantized tensors; this artifact also keeps
runtime tensors in float32 where needed for WebGPU stability.

## Files

| File | Runtime dtype | Use | External chunks |
| --- | --- | --- | ---: |
| `onnx/model_q4f16.onnx` | q4 weights, WebGPU-safe runtime tensors | Default browser path, 10.58&nbsp;GB | 10 |
| `onnx/model_quantized.onnx` | q8 | Fallback path, 15.31&nbsp;GB | 15 |

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

The app loads this repo with `device: "webgpu"` and `dtype: "q4f16"` first, then
falls back to `q8` if q4f16 loading fails.

## Transformers.js Notes

Talkie has a custom `model_type`, so stock `pipeline("text-generation")` is not
used here. The browser runner formats messages with the shipped chat template,
tokenizes them, runs the full accumulated `input_ids` for each new token, samples
on the CPU, suppresses token `0`, and stops on token IDs `65535` or `65536`.

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

Use the GitHub runner for the complete manual generation loop.

## Validation

- Hub artifact validation confirms tokenizer/config files, ONNX files, and all
  q4/q8 external-data chunks.
- The q4f16 browser path has been smoke-tested through Chromium/WebGPU with at
  least 16 streamed non-NUL tokens.
- The export path validates the full-sequence ONNX graph against the source
  PyTorch wrapper before quantization.

Re-run the public artifact check:

```bash
python3 scripts/check_hub_artifacts.py scasella91/talkie-1930-13b-it-ONNX
```

## Known Limitations

- First load is large and slow because the model is split across external-data
  chunks.
- Browser cache writes may hit quota; that is noisy but not necessarily fatal if
  the model reaches `Ready`.
- q4f16 keeps q4 weights but uses float32 runtime tensors in the current browser
  artifact for WebGPU stability.
- Generation is full-sequence and intentionally slower than a KV-cache decoder.
- This repo is public and assumes unauthenticated browser loading. Do not embed
  private Hugging Face tokens in browser code.

## Attribution

Talkie was developed by Alec Radford, Nick Levine, and David Duvenaud. This ONNX
repo builds on the Hugging Face Transformers-format conversion by
[`lewtun/talkie-1930-13b-it-hf`](https://huggingface.co/lewtun/talkie-1930-13b-it-hf)
and the original Talkie project at <https://github.com/talkie-lm/talkie>.

## License

Apache-2.0, matching the source model metadata.
