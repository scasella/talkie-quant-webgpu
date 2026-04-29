/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TALKIE_ONNX_MODEL_ID?: string;
  readonly VITE_TALKIE_ONNX_REVISION?: string;
  readonly VITE_TALKIE_ONNX_DTYPE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
