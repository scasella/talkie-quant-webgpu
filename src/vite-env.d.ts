/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TALKIE_ONNX_MODEL_ID?: string;
  readonly VITE_TALKIE_ONNX_REVISION?: string;
  readonly VITE_TALKIE_ONNX_DTYPE?: string;
  readonly VITE_TALKIE_LOAD_PATH?: string;
  readonly VITE_TALKIE_DIRECT_CACHED?: string;
  readonly VITE_TALKIE_FETCH_RETRY?: string;
  readonly VITE_TALKIE_FETCH_LIMITER?: string;
  readonly VITE_TALKIE_FETCH_CONCURRENCY?: string;
  readonly VITE_TALKIE_BROWSER_CACHE?: string;
  readonly VITE_TALKIE_GRAPH_OPTIMIZATION?: string;
}

interface Window {
  __talkieMetrics?: {
    events?: Array<Record<string, unknown>>;
    generation?: Array<{
      mode: string;
      backend: string;
      tokensGenerated: number;
      tokensPerSecond: number;
      lastTokenMs: number;
      timeMs?: number;
    }>;
  };
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
