const DEFAULT_MAX_PARALLEL_MODEL_FETCHES = 4;
const DEFAULT_COMPRESSED_MAX_PARALLEL_MODEL_FETCHES = 8;
const MAX_ATTEMPTS = 4;
const RUNTIME_PARAMS = typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search);
const COMPRESSED_EXTERNAL_DATA_ENABLED =
  RUNTIME_PARAMS.get("compressed") === "1" || import.meta.env.VITE_TALKIE_COMPRESSED_EXTERNAL_DATA === "1";

let installed = false;
let activeFetches = 0;
let progressListener: TalkieFetchProgressListener | null = null;
const pendingFetches: Array<{
  task: () => Promise<Response>;
  resolve: (response: Response) => void;
  reject: (error: unknown) => void;
}> = [];

export interface TalkieFetchProgress {
  url: string;
  loaded: number;
  total?: number;
}

export type TalkieFetchProgressListener = (event: TalkieFetchProgress) => void;

export function setTalkieFetchProgressListener(listener: TalkieFetchProgressListener | null): void {
  progressListener = listener;
}

export function talkieFetchConcurrency(): number {
  return maxParallelFetches();
}

export function installTalkieFetchLimiter(): void {
  if (installed || typeof window === "undefined") return;
  if (import.meta.env.VITE_TALKIE_FETCH_LIMITER === "0") return;
  installed = true;

  const originalFetch = window.fetch.bind(window);
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    if (!isModelFetch(input)) return originalFetch(input, init);
    return enqueueFetch(() => fetchWithRetry(originalFetch, input, init));
  }) as typeof window.fetch;
}

function maxParallelFetches(): number {
  const params = new URLSearchParams(window.location.search);
  const configured = Number(params.get("fetches") || import.meta.env.VITE_TALKIE_FETCH_CONCURRENCY || "");
  if (Number.isFinite(configured) && configured >= 1) return Math.floor(configured);
  return COMPRESSED_EXTERNAL_DATA_ENABLED
    ? DEFAULT_COMPRESSED_MAX_PARALLEL_MODEL_FETCHES
    : DEFAULT_MAX_PARALLEL_MODEL_FETCHES;
}

function isModelFetch(input: RequestInfo | URL): boolean {
  try {
    const url = new URL(input instanceof Request ? input.url : String(input), window.location.href);
    return (
      (url.hostname.includes("huggingface.co") || url.hostname.includes("hf.co")) &&
      url.pathname.includes("/resolve/") &&
      url.pathname.includes("/onnx/")
    );
  } catch {
    return false;
  }
}

function enqueueFetch(task: () => Promise<Response>): Promise<Response> {
  return new Promise((resolve, reject) => {
    pendingFetches.push({ task, resolve, reject });
    pumpFetchQueue();
  });
}

function pumpFetchQueue(): void {
  const limit = maxParallelFetches();
  while (activeFetches < limit && pendingFetches.length > 0) {
    const item = pendingFetches.shift();
    if (!item) return;
    activeFetches += 1;
    item
      .task()
      .then(item.resolve, item.reject)
      .finally(() => {
        activeFetches -= 1;
        pumpFetchQueue();
      });
  }
}

async function fetchWithRetry(
  fetchImpl: typeof window.fetch,
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    try {
      const sourceUrl = fetchUrl(input);
      const compressedUrl = compressedExternalDataUrl(sourceUrl);
      const response = await fetchImpl(compressedUrl ? compressedUrl : cloneFetchInput(input), init);
      if (response.ok || (response.status < 500 && response.status !== 429)) {
        return withProgress(compressedUrl ? decompressGzipResponse(response) : response, sourceUrl);
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(500 * 2 ** attempt);
  }
  throw lastError ?? new Error("Model fetch failed");
}

function withProgress(response: Response, url: string): Response {
  if (!progressListener || !response.ok || !response.body) return response;

  const total = headerNumber(response.headers, "content-length") ?? headerNumber(response.headers, "x-linked-size");
  const reader = response.body.getReader();
  let loaded = 0;

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        progressListener?.({ url, loaded, total: total ?? loaded });
        controller.close();
        return;
      }
      loaded += value.byteLength;
      progressListener?.({ url, loaded, total });
      controller.enqueue(value);
    },
    cancel(reason) {
      return reader.cancel(reason);
    }
  });

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers
  });
}

function headerNumber(headers: Headers, name: string): number | undefined {
  const value = Number(headers.get(name));
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function fetchUrl(input: RequestInfo | URL): string {
  return input instanceof Request ? input.url : String(input);
}

function cloneFetchInput(input: RequestInfo | URL): RequestInfo | URL {
  return input instanceof Request ? input.clone() : input;
}

function compressedExternalDataUrl(url: string): string | null {
  if (!COMPRESSED_EXTERNAL_DATA_ENABLED || typeof DecompressionStream === "undefined") return null;
  if (!/\.onnx_data(?:_\d+)?(?:[?#].*)?$/.test(url)) return null;
  try {
    const parsed = new URL(url);
    parsed.pathname = `${parsed.pathname}.gz`;
    return parsed.toString();
  } catch {
    return `${url}.gz`;
  }
}

function decompressGzipResponse(response: Response): Response {
  if (!response.body) return response;
  const headers = new Headers(response.headers);
  headers.delete("content-length");
  headers.delete("content-encoding");
  return new Response(response.body.pipeThrough(new DecompressionStream("gzip")), {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
