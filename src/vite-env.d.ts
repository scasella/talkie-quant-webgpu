/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TALKIE_ONNX_MODEL_ID?: string;
  readonly VITE_TALKIE_ONNX_REVISION?: string;
  readonly VITE_TALKIE_ONNX_DTYPE?: string;
  readonly VITE_TALKIE_BROWSER_CACHE?: string;
  readonly VITE_TALKIE_GRAPH_OPTIMIZATION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
