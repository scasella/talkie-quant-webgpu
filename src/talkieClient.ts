import {
  AutoConfig,
  AutoModel,
  AutoTokenizer,
  env,
  LogLevel,
  Tensor as HfTensor
} from "@huggingface/transformers";
import * as ort from "onnxruntime-web/webgpu";
import { waitForTalkieFetchRetry } from "./fetchRetry";
import { installTalkieFetchLimiter, setTalkieFetchProgressListener } from "./fetchLimiter";

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface GenerationSettings {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  topK: number;
}

export interface LoadProgress {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

export interface TalkieSession {
  modelId: string;
  revision: string;
  dtype: string;
  dtypes: string[];
  mode: RuntimeMode;
  modelFileName: string;
  backend: RuntimeBackend;
}

export type RuntimeMode = "kv-cache" | "full-sequence";
export type LoadPath = "full" | "cached";
export type RuntimeBackend = "transformers" | "ort-direct";

export interface GenerationStats {
  mode: RuntimeMode;
  backend: RuntimeBackend;
  tokensGenerated: number;
  tokensPerSecond: number;
  lastTokenMs: number;
}

const DEFAULT_MODEL_ID = "scasella91/talkie-1930-13b-it-ONNX";
const DEFAULT_REVISION = "631cbea56319f30469aae41af8fbd3078c460b3b";
const MAX_CONTEXT_TOKENS = 2048;
const STOP_TOKEN_IDS = new Set([65535, 65536]);
const SUPPRESSED_TOKEN_IDS = new Set([0]);

const RUNTIME_PARAMS = runtimeSearchParams();
const MODEL_ID = RUNTIME_PARAMS.get("model") || import.meta.env.VITE_TALKIE_ONNX_MODEL_ID || DEFAULT_MODEL_ID;
const REVISION = RUNTIME_PARAMS.get("revision") || import.meta.env.VITE_TALKIE_ONNX_REVISION || DEFAULT_REVISION;
const DTYPE_OVERRIDE = RUNTIME_PARAMS.get("dtype") || import.meta.env.VITE_TALKIE_ONNX_DTYPE;
const LOAD_PATH_OVERRIDE = RUNTIME_PARAMS.get("path");
const CACHE_OVERRIDE = RUNTIME_PARAMS.get("cache");
const GRAPH_OPTIMIZATION_LEVEL =
  RUNTIME_PARAMS.get("opt") || import.meta.env.VITE_TALKIE_GRAPH_OPTIMIZATION || "disabled";
const DIRECT_CACHED_ENABLED =
  RUNTIME_PARAMS.get("direct") !== "0" && import.meta.env.VITE_TALKIE_DIRECT_CACHED !== "0";
const BROWSER_CACHE_ENABLED =
  CACHE_OVERRIDE === "1" || (CACHE_OVERRIDE !== "0" && import.meta.env.VITE_TALKIE_BROWSER_CACHE === "1");
const DEFAULT_LOAD_PATH: LoadPath =
  LOAD_PATH_OVERRIDE === "full"
    ? "full"
    : LOAD_PATH_OVERRIDE === "cached"
      ? "cached"
      : import.meta.env.VITE_TALKIE_LOAD_PATH === "full"
        ? "full"
        : "cached";

const ONNX_CHUNK_COUNTS: Record<string, number> = {
  "model_kv_q4f16.onnx": 32,
  "model_kv_quantized.onnx": 42,
  "model_kv_fast_q4f16.onnx": 0,
  "model_kv_fast_quantized.onnx": 0,
  "model_q4f16.onnx": 22,
  "model_quantized.onnx": 31
};

env.logLevel = LogLevel.WARNING;
env.useBrowserCache = BROWSER_CACHE_ENABLED;

type ProgressCallback = (progress: LoadProgress) => void;

interface Runtime {
  tokenizer: any;
  model: LoadedModelRuntime;
  session: TalkieSession;
}

type LoadedModelRuntime = any | DirectOrtModel;

let runtimePromise: Promise<Runtime> | null = null;
let runtimePath: LoadPath | null = null;
let retainedGpuAdapter: any = null;
let retainedGpuDevice: any = null;

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export function modelDefaults(): Pick<TalkieSession, "modelId" | "revision"> {
  return { modelId: MODEL_ID, revision: REVISION };
}

export function browserCacheEnabled(): boolean {
  return BROWSER_CACHE_ENABLED;
}

function runtimeSearchParams(): URLSearchParams {
  if (typeof window === "undefined") return new URLSearchParams();
  return new URLSearchParams(window.location.search);
}

export async function resetTalkieRuntime(): Promise<void> {
  if (!runtimePromise) return;
  const runtime = await runtimePromise.catch(() => null);
  runtimePromise = null;
  runtimePath = null;
  await runtime?.model?.dispose?.();
}

export async function loadTalkieRuntime(onProgress?: ProgressCallback, loadPath?: LoadPath): Promise<Runtime> {
  const requestedPath = loadPath ?? runtimePath ?? DEFAULT_LOAD_PATH;
  if (runtimePromise && runtimePath === requestedPath) return runtimePromise;
  if (runtimePromise && runtimePath !== requestedPath) {
    await resetTalkieRuntime();
  }
  runtimePath = requestedPath;

  runtimePromise = (async () => {
    if (!hasWebGPU()) {
      throw new Error("WebGPU is not available in this browser.");
    }
    await retainWebGpuDevice();

    installTalkieFetchLimiter();
    configureDirectOrtRuntime();

    const progress_callback = (event: LoadProgress) => {
      onProgress?.(event);
    };

    const config = await AutoConfig.from_pretrained(MODEL_ID, {
      revision: REVISION,
      progress_callback
    } as any);
    const dtypes = getAvailableDtypes(config);
    const explicitDtype = DTYPE_OVERRIDE && DTYPE_OVERRIDE !== "auto" ? DTYPE_OVERRIDE : null;
    const hasExplicitDtype = explicitDtype != null;
    const preferred = explicitDtype ? [explicitDtype] : ["q4f16", "q4", "q8", "fp16", "fp32"];
    const dtype = preferred.find((candidate) => dtypes.includes(candidate)) ?? "q4f16";

    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      revision: REVISION,
      progress_callback
    } as any);

    const attempts = buildLoadAttempts(preferred, dtypes, requestedPath, !hasExplicitDtype, config);
    let loaded: LoadedModel | null = null;
    let lastError: unknown = null;
    for (const attempt of attempts) {
      try {
        loaded = await loadModelAttempt(config, progress_callback, attempt);
        break;
      } catch (error) {
        lastError = error;
        const detail = `${attempt.dtype} ${attempt.mode} load failed: ${errorMessage(error)}`;
        console.warn(`[talkie] ${detail}`);
        progress_callback({ status: detail });
      }
    }
    if (!loaded) {
      runtimePromise = null;
      runtimePath = null;
      throw new Error(`Could not load Talkie ONNX model (${requestedPath}). Last error: ${errorMessage(lastError)}`);
    }

    return {
      tokenizer,
      model: loaded.model,
      session: {
        modelId: MODEL_ID,
        revision: REVISION,
        dtype: loaded.dtype,
        dtypes,
        mode: loaded.mode,
        modelFileName: loaded.modelFileName,
        backend: loaded.backend
      }
    };
  })();

  return runtimePromise;
}

function getAvailableDtypes(config: any): string[] {
  const externalData = config?.["transformers.js_config"]?.use_external_data_format ?? {};
  const files = new Set(Object.keys(externalData));
  const dtypes: string[] = [];
  if (
    files.has("model_kv_fast_q4f16.onnx") ||
    files.has("model_kv_q4f16.onnx") ||
    files.has("model_q4f16.onnx")
  ) {
    dtypes.push("q4f16");
  }
  if (
    files.has("model_kv_fast_quantized.onnx") ||
    files.has("model_kv_quantized.onnx") ||
    files.has("model_quantized.onnx")
  ) {
    dtypes.push("q8");
  }
  const configured = config?.["transformers.js_config"]?.dtype;
  if (typeof configured === "string" && !dtypes.includes(configured)) dtypes.push(configured);
  return dtypes.length > 0 ? dtypes : ["q4f16", "q8"];
}

export async function generateTalkieReply(
  messages: ChatMessage[],
  settings: GenerationSettings,
  onText: (text: string) => void,
  signal: AbortSignal,
  onProgress?: ProgressCallback,
  onStats?: (stats: GenerationStats) => void
): Promise<string> {
  const { tokenizer, model, session } = await loadTalkieRuntime(onProgress);
  const prompt = formatTalkieMessages(tokenizer, messages);
  const encoded = await tokenizer(prompt, { add_special_tokens: false });
  let inputIds = tensorToIds(encoded.input_ids);

  const availablePromptTokens = Math.max(1, MAX_CONTEXT_TOKENS - settings.maxNewTokens);
  if (inputIds.length > availablePromptTokens) {
    inputIds = inputIds.slice(inputIds.length - availablePromptTokens);
  }

  if (session.mode === "kv-cache") {
    if (isDirectOrtModel(model)) {
      return generateWithDirectKvCache(model, tokenizer, inputIds, settings, onText, signal, onStats);
    }
    return generateWithKvCache(model, tokenizer, inputIds, settings, onText, signal, onStats);
  }

  return generateFullSequence(model, tokenizer, inputIds, settings, onText, signal, onStats);
}

async function generateFullSequence(
  model: any,
  tokenizer: any,
  inputIds: number[],
  settings: GenerationSettings,
  onText: (text: string) => void,
  signal: AbortSignal,
  onStats?: (stats: GenerationStats) => void
): Promise<string> {
  const startTime = performance.now();

  const generated: number[] = [];
  let lastText = "";
  let lastEmitMs = 0;
  const rateWindow: number[] = [];

  for (let step = 0; step < settings.maxNewTokens; step += 1) {
    if (signal.aborted) break;
    if (inputIds.length >= MAX_CONTEXT_TOKENS) break;

    const tokenStart = performance.now();
    const outputs = await model.forward({ input_ids: idsToTensor(inputIds) });
    const logits = outputs.logits ?? Object.values(outputs)[0];
    const scores = lastTokenScores(logits as any);
    const nextId = sampleToken(scores, settings);
    const tokenMs = performance.now() - tokenStart;

    if (STOP_TOKEN_IDS.has(nextId)) break;

    inputIds.push(nextId);
    generated.push(nextId);
    reportStats("full-sequence", "transformers", generated.length, startTime, tokenMs, rateWindow, onStats);
    ({ lastText, lastEmitMs } = maybeEmitText(
      tokenizer,
      generated,
      lastText,
      onText,
      lastEmitMs,
      step === settings.maxNewTokens - 1
    ));

    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const finalText = decode(tokenizer, generated);
  if (finalText !== lastText) {
    lastText = finalText;
    onText(finalText);
  }
  return lastText;
}

async function generateWithKvCache(
  model: any,
  tokenizer: any,
  promptIds: number[],
  settings: GenerationSettings,
  onText: (text: string) => void,
  signal: AbortSignal,
  onStats?: (stats: GenerationStats) => void
): Promise<string> {
  const startTime = performance.now();
  const generated: number[] = [];
  let lastText = "";
  let lastEmitMs = 0;
  const rateWindow: number[] = [];
  let cache = createEmptyKvCache(model);
  let currentInputIds = promptIds;
  let currentPositionStart = 0;
  let nextPosition = promptIds.length;

  try {
    for (let step = 0; step < settings.maxNewTokens; step += 1) {
      if (signal.aborted) break;
      if (nextPosition >= MAX_CONTEXT_TOKENS) break;

      const tokenStart = performance.now();
      const outputs = await model.forward({
        input_ids: idsToTensor(currentInputIds),
        position_ids: positionIdsToTensor(currentPositionStart, currentInputIds.length),
        ...cache
      });
      cache = await updateKvCache(cache, outputs);

      const logits = outputs.logits ?? Object.values(outputs)[0];
      const scores = lastTokenScores(logits as any);
      const nextId = sampleToken(scores, settings);
      const tokenMs = performance.now() - tokenStart;

      if (STOP_TOKEN_IDS.has(nextId)) break;

      generated.push(nextId);
      reportStats("kv-cache", "transformers", generated.length, startTime, tokenMs, rateWindow, onStats);
      ({ lastText, lastEmitMs } = maybeEmitText(
        tokenizer,
        generated,
        lastText,
        onText,
        lastEmitMs,
        step === settings.maxNewTokens - 1
      ));

      currentInputIds = [nextId];
      currentPositionStart = nextPosition;
      nextPosition += 1;

      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } finally {
    await disposeKvCache(cache);
  }

  const finalText = decode(tokenizer, generated);
  if (finalText !== lastText) {
    lastText = finalText;
    onText(finalText);
  }
  return lastText;
}

async function generateWithDirectKvCache(
  model: DirectOrtModel,
  tokenizer: any,
  promptIds: number[],
  settings: GenerationSettings,
  onText: (text: string) => void,
  signal: AbortSignal,
  onStats?: (stats: GenerationStats) => void
): Promise<string> {
  const startTime = performance.now();
  const generated: number[] = [];
  let lastText = "";
  let lastEmitMs = 0;
  const rateWindow: number[] = [];
  let cache = createEmptyDirectKvCache(model);
  let currentInputIds = promptIds;
  let currentPositionStart = 0;
  let nextPosition = promptIds.length;

  try {
    for (let step = 0; step < settings.maxNewTokens; step += 1) {
      if (signal.aborted) break;
      if (nextPosition >= MAX_CONTEXT_TOKENS) break;

      const tokenStart = performance.now();
      const outputs = await model.session.run(
        {
          input_ids: idsToOrtTensor(currentInputIds),
          position_ids: positionIdsToOrtTensor(currentPositionStart, currentInputIds.length),
          ...cache
        },
        directKvFetches(model)
      );
      cache = updateDirectKvCache(cache, outputs);

      const logits = outputs.logits;
      if (!logits) throw new Error("Direct cached ONNX run did not return logits.");
      const scores = lastTokenScores(logits);
      const nextId = sampleToken(scores, settings);
      const tokenMs = performance.now() - tokenStart;

      if (STOP_TOKEN_IDS.has(nextId)) break;

      generated.push(nextId);
      reportStats("kv-cache", "ort-direct", generated.length, startTime, tokenMs, rateWindow, onStats);
      ({ lastText, lastEmitMs } = maybeEmitText(
        tokenizer,
        generated,
        lastText,
        onText,
        lastEmitMs,
        step === settings.maxNewTokens - 1
      ));

      currentInputIds = [nextId];
      currentPositionStart = nextPosition;
      nextPosition += 1;

      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } finally {
    disposeDirectKvCache(cache);
  }

  const finalText = decode(tokenizer, generated);
  if (finalText !== lastText) {
    lastText = finalText;
    onText(finalText);
  }
  return lastText;
}

export function formatTalkieMessages(tokenizer: any, messages: ChatMessage[]): string {
  try {
    const rendered = tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true
    });
    if (typeof rendered === "string" && rendered.length > 0) return rendered;
  } catch {
    // The published tokenizer carries the template; this keeps local dev usable
    // while the ONNX repo is still being assembled.
  }

  const body = messages
    .map((message) => {
      if (message.role === "system") return `<|system|>${message.content}<|end|>`;
      if (message.role === "assistant") return `<|assistant|>${message.content}<|end|>`;
      return `<|user|>${message.content}<|end|>`;
    })
    .join("");
  return `${body}<|assistant|>`;
}

function tensorToIds(tensor: any): number[] {
  const data = Array.from(tensor.data ?? tensor);
  return data.map((value) => Number(value));
}

function idsToTensor(ids: number[]): HfTensor {
  return new HfTensor("int64", BigInt64Array.from(ids.map((id) => BigInt(id))), [1, ids.length]);
}

function positionIdsToTensor(start: number, length: number): HfTensor {
  return new HfTensor(
    "int64",
    BigInt64Array.from(Array.from({ length }, (_value, index) => BigInt(start + index))),
    [1, length]
  );
}

function idsToOrtTensor(ids: number[]): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.from(ids.map((id) => BigInt(id))), [1, ids.length]);
}

function positionIdsToOrtTensor(start: number, length: number): ort.Tensor {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(Array.from({ length }, (_value, index) => BigInt(start + index))),
    [1, length]
  );
}

function lastTokenScores(logits: any): Float32Array | number[] {
  const dims = logits.dims as number[];
  const vocab = dims[dims.length - 1];
  const sequence = dims.length >= 3 ? dims[dims.length - 2] : 1;
  const start = (sequence - 1) * vocab;
  return Array.prototype.slice.call(logits.data, start, start + vocab);
}

function decode(tokenizer: any, ids: number[]): string {
  try {
    return tokenizer.decode(ids, { skip_special_tokens: true });
  } catch {
    return tokenizer.decode(ids.map(BigInt), { skip_special_tokens: true });
  }
}

function sampleToken(scores: Float32Array | number[], settings: GenerationSettings): number {
  const temperature = Math.max(0.05, settings.temperature);
  const topK = Math.max(1, Math.min(settings.topK, scores.length));
  const topP = Math.min(1, Math.max(0.01, settings.topP));

  const ranked = topKScores(scores, topK);

  if (ranked.length === 0) {
    throw new Error("The model returned no finite, sampleable logits.");
  }

  const maxScore = ranked[0]?.score ?? 0;
  const weighted = ranked.map((entry) => ({
    id: entry.id,
    weight: Math.exp((entry.score - maxScore) / temperature)
  }));
  const totalWeight = weighted.reduce((sum, entry) => sum + entry.weight, 0);
  let cumulative = 0;
  const nucleus: typeof weighted = [];

  for (const entry of weighted) {
    cumulative += entry.weight / totalWeight;
    nucleus.push(entry);
    if (cumulative >= topP) break;
  }

  const sampleTotal = nucleus.reduce((sum, entry) => sum + entry.weight, 0);
  let cursor = Math.random() * sampleTotal;
  for (const entry of nucleus) {
    cursor -= entry.weight;
    if (cursor <= 0) return entry.id;
  }
  return nucleus[nucleus.length - 1]?.id ?? ranked[0]?.id ?? 0;
}

function topKScores(scores: Float32Array | number[], topK: number): Array<{ id: number; score: number }> {
  const ranked: Array<{ id: number; score: number }> = [];
  let floor = Number.NEGATIVE_INFINITY;

  for (let id = 0; id < scores.length; id += 1) {
    if (SUPPRESSED_TOKEN_IDS.has(id)) continue;
    const score = Number(scores[id]);
    if (!Number.isFinite(score)) continue;
    if (ranked.length === topK && score <= floor) continue;

    let insertAt = ranked.length;
    while (insertAt > 0 && ranked[insertAt - 1].score < score) insertAt -= 1;
    ranked.splice(insertAt, 0, { id, score });
    if (ranked.length > topK) ranked.pop();
    floor = ranked[ranked.length - 1]?.score ?? Number.NEGATIVE_INFINITY;
  }

  return ranked;
}

interface LoadAttempt {
  dtype: string;
  mode: RuntimeMode;
  modelFileName: string;
  backend: RuntimeBackend;
  onnxFileName?: string;
}

interface LoadedModel extends LoadAttempt {
  model: LoadedModelRuntime;
}

function buildLoadAttempts(
  preferred: string[],
  dtypes: string[],
  loadPath: LoadPath,
  allowQ8Fallback: boolean,
  config: any
): LoadAttempt[] {
  const candidates = preferred
    .filter((candidate) => candidate === "q4f16" || candidate === "q8" || loadPath === "full")
    .filter((candidate, index, array) => array.indexOf(candidate) === index);
  const available = dtypes.length > 0 ? candidates.filter((candidate) => dtypes.includes(candidate)) : candidates;
  const ordered = allowQ8Fallback && !available.includes("q8") ? [...available, "q8"] : available;
  if (loadPath === "cached") {
    const cachedAttempts = ordered.flatMap((dtype) => directCachedCandidates(dtype, config).map((onnxFileName) => ({
      dtype,
      mode: "kv-cache" as const,
      modelFileName: "model_kv",
      backend: DIRECT_CACHED_ENABLED ? ("ort-direct" as const) : ("transformers" as const),
      onnxFileName
    })));
    const transformerFallback = ordered.map((dtype) => ({
      dtype,
      mode: "kv-cache" as const,
      modelFileName: "model_kv",
      backend: "transformers" as const
    }));
    const fullFallback = ordered.map((dtype) => ({
      dtype,
      mode: "full-sequence" as const,
      modelFileName: "model",
      backend: "transformers" as const
    }));
    return DIRECT_CACHED_ENABLED ? [...cachedAttempts, ...transformerFallback, ...fullFallback] : [...transformerFallback, ...fullFallback];
  }
  return ordered.map((dtype) => ({ dtype, mode: "full-sequence" as const, modelFileName: "model", backend: "transformers" }));
}

async function loadModelAttempt(config: any, progress_callback: ProgressCallback, attempt: LoadAttempt): Promise<LoadedModel> {
  if (attempt.backend === "ort-direct") {
    const model = await loadDirectOrtModel(config, progress_callback, attempt);
    return { ...attempt, model };
  }

  const model = await AutoModel.from_pretrained(MODEL_ID, {
    revision: REVISION,
    config,
    device: "webgpu",
    dtype: attempt.dtype,
    model_file_name: attempt.modelFileName,
    progress_callback,
    session_options: sessionOptionsForAttempt(config, attempt)
  } as any);
  return { ...attempt, model };
}

function sessionOptionsForAttempt(config: any, attempt: LoadAttempt) {
  return {
    graphOptimizationLevel: GRAPH_OPTIMIZATION_LEVEL,
    ...(attempt.mode === "kv-cache" ? cacheSessionOptions(config) : {})
  };
}

function cacheSessionOptions(config: any) {
  const layerCount = Number(config?.num_hidden_layers ?? config?.n_layer ?? 0);
  const preferredOutputLocation: Record<string, "gpu-buffer"> = {};
  for (let layer = 0; layer < layerCount; layer += 1) {
    preferredOutputLocation[`present.${layer}.key`] = "gpu-buffer";
    preferredOutputLocation[`present.${layer}.value`] = "gpu-buffer";
  }
  return { preferredOutputLocation };
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

async function retainWebGpuDevice(): Promise<void> {
  if (retainedGpuDevice) return;
  const onnxWebgpu = (env.backends?.onnx as any)?.webgpu;
  const gpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!onnxWebgpu || !gpu?.requestAdapter) return;

  retainedGpuAdapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!retainedGpuAdapter?.requestDevice) return;

  retainedGpuDevice = await retainedGpuAdapter.requestDevice();
  retainedGpuDevice?.lost?.then?.((info: { reason?: string; message?: string }) => {
    console.warn(`[talkie] WebGPU device lost: ${info?.reason ?? "unknown"} ${info?.message ?? ""}`.trim());
  });
  onnxWebgpu.adapter = retainedGpuAdapter;
  onnxWebgpu.device = retainedGpuDevice;
  configureDirectOrtRuntime();
}

function createEmptyKvCache(model: any): Record<string, HfTensor> {
  const session = model.sessions?.model;
  const metadata = session?.inputMetadata ?? [];
  const config = model.config ?? {};
  const entries: Record<string, HfTensor> = {};

  for (const meta of metadata) {
    if (!meta.name?.startsWith("past_key_values.")) continue;
    const dims = resolveCacheShape(meta.shape ?? [], config);
    const size = dims.reduce((total: number, dim: number) => total * dim, 1);
    entries[meta.name] = new HfTensor(meta.type, zerosForTensorType(meta.type, size), dims);
  }

  if (Object.keys(entries).length === 0) {
    throw new Error("Loaded KV-cache model does not expose past_key_values inputs.");
  }

  return entries;
}

function resolveCacheShape(shape: Array<number | string>, config: any): number[] {
  const heads = Number(config?.num_attention_heads ?? config?.n_head ?? 0);
  const headDim = Number(config?.head_dim ?? 0);
  return shape.map((dim) => {
    if (typeof dim === "number") return dim < 0 ? 0 : dim;
    const name = dim.toLowerCase();
    if (name.includes("batch")) return 1;
    if (name.includes("past") || name.includes("sequence")) return 0;
    if (name.includes("head_dim")) return headDim;
    if (name.includes("head")) return heads;
    return 0;
  });
}

function zerosForTensorType(type: string, size: number): Float32Array | Uint16Array {
  if (type === "float16" || type === "bfloat16") return new Uint16Array(size);
  if (type === "float32" || type === "float") return new Float32Array(size);
  throw new Error(`Unsupported KV-cache tensor type: ${type}`);
}

async function updateKvCache(
  oldCache: Record<string, HfTensor>,
  outputs: Record<string, any>
): Promise<Record<string, HfTensor>> {
  const nextCache: Record<string, HfTensor> = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (!name.startsWith("present.")) continue;
    nextCache[name.replace("present.", "past_key_values.")] = tensor as HfTensor;
  }
  if (Object.keys(nextCache).length === 0) {
    throw new Error("KV-cache model did not return present.* cache tensors.");
  }
  for (const [name, tensor] of Object.entries(oldCache)) {
    if (nextCache[name] === tensor) continue;
    await disposeTensor(tensor);
  }
  return nextCache;
}

async function disposeKvCache(cache: Record<string, HfTensor>): Promise<void> {
  await Promise.all(Object.values(cache).map(disposeTensor));
}

async function disposeTensor(tensor: HfTensor): Promise<void> {
  if ((tensor as any).location === "gpu-buffer") {
    await (tensor as any).dispose?.();
  }
}

function reportStats(
  mode: RuntimeMode,
  backend: RuntimeBackend,
  tokensGenerated: number,
  startTime: number,
  lastTokenMs: number,
  rateWindow: number[],
  onStats?: (stats: GenerationStats) => void
): void {
  rateWindow.push(lastTokenMs);
  if (rateWindow.length > 8) rateWindow.shift();
  const elapsedSec = Math.max(0.001, (performance.now() - startTime) / 1000);
  const rollingTokenMs = median(rateWindow) ?? elapsedSec * 1000;
  const stats = {
    mode,
    backend,
    tokensGenerated,
    tokensPerSecond: 1000 / Math.max(1, rollingTokenMs),
    lastTokenMs
  };
  recordGenerationStats(stats);
  onStats?.(stats);
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function maybeEmitText(
  tokenizer: any,
  generated: number[],
  lastText: string,
  onText: (text: string) => void,
  lastEmitMs: number,
  force = false
): { lastText: string; lastEmitMs: number } {
  const now = performance.now();
  if (!force && generated.length % 4 !== 0 && now - lastEmitMs < 100) return { lastText, lastEmitMs };
  const nextText = decode(tokenizer, generated);
  if (nextText !== lastText) {
    onText(nextText);
    return { lastText: nextText, lastEmitMs: now };
  }
  return { lastText, lastEmitMs };
}

interface DirectOrtModel {
  kind: "direct-ort-kv";
  session: ort.InferenceSession;
  config: any;
  modelFileName: string;
  dtype: string;
  cacheInputNames: string[];
  cacheOutputNames: string[];
  numHeads: number;
  headDim: number;
  dispose: () => Promise<void>;
}

function isDirectOrtModel(model: LoadedModelRuntime): model is DirectOrtModel {
  return model?.kind === "direct-ort-kv";
}

async function loadDirectOrtModel(
  config: any,
  progress_callback: ProgressCallback,
  attempt: LoadAttempt
): Promise<DirectOrtModel> {
  if (!attempt.onnxFileName) throw new Error("Direct cached load requires an ONNX filename.");
  progress_callback({ status: `Loading direct ${attempt.dtype} kv-cache` });
  await waitForTalkieFetchRetry();

  setTalkieFetchProgressListener((event) => {
    progress_callback({
      status: `Fetching direct ${attempt.dtype} ONNX chunks`,
      file: event.url,
      loaded: event.loaded,
      total: event.total
    });
  });

  let session: ort.InferenceSession | null = null;
  try {
    session = await ort.InferenceSession.create(hubOnnxUrl(attempt.onnxFileName), {
      executionProviders: ["webgpu"],
      externalData: externalDataForModel(attempt.onnxFileName, config),
      graphOptimizationLevel: GRAPH_OPTIMIZATION_LEVEL as ort.InferenceSession.SessionOptions["graphOptimizationLevel"],
      preferredOutputLocation: directPreferredOutputLocation(config)
    });
  } finally {
    setTalkieFetchProgressListener(null);
  }

  const cacheInputNames = session.inputNames.filter((name) => name.startsWith("past_key_values."));
  const cacheOutputNames = session.outputNames.filter((name) => name.startsWith("present."));
  if (cacheInputNames.length === 0 || cacheOutputNames.length === 0) {
    await session.release();
    throw new Error("Direct cached ONNX session does not expose past/present KV tensors.");
  }

  return {
    kind: "direct-ort-kv",
    session,
    config,
    modelFileName: attempt.onnxFileName,
    dtype: attempt.dtype,
    cacheInputNames,
    cacheOutputNames,
    numHeads: Number(config?.num_attention_heads ?? config?.n_head ?? 0),
    headDim: Number(config?.head_dim ?? 0),
    dispose: async () => {
      await session.release();
    }
  };
}

function directPreferredOutputLocation(config: any): ort.InferenceSession.SessionOptions["preferredOutputLocation"] {
  const preferredOutputLocation: Record<string, "gpu-buffer"> = {};
  const layerCount = Number(config?.num_hidden_layers ?? config?.n_layer ?? 0);
  for (let layer = 0; layer < layerCount; layer += 1) {
    preferredOutputLocation[`present.${layer}.key`] = "gpu-buffer";
    preferredOutputLocation[`present.${layer}.value`] = "gpu-buffer";
  }
  return preferredOutputLocation;
}

function createEmptyDirectKvCache(model: DirectOrtModel): Record<string, ort.Tensor> {
  const metadata = model.session.inputMetadata ?? [];
  const entries: Record<string, ort.Tensor> = {};

  for (const name of model.cacheInputNames) {
    const meta = metadata.find((item) => item.name === name) as { shape?: Array<number | string>; type?: string } | undefined;
    const dims = resolveCacheShape(meta?.shape ?? [], model.config);
    const type = normalizeOrtTensorType(meta?.type ?? "float32");
    const size = dims.reduce((total, dim) => total * dim, 1);
    entries[name] = new ort.Tensor(type, zerosForOrtTensorType(type, size), dims);
  }

  return entries;
}

function updateDirectKvCache(
  oldCache: Record<string, ort.Tensor>,
  outputs: Record<string, ort.Tensor>
): Record<string, ort.Tensor> {
  const nextCache: Record<string, ort.Tensor> = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (!name.startsWith("present.")) continue;
    nextCache[name.replace("present.", "past_key_values.")] = tensor;
  }
  if (Object.keys(nextCache).length === 0) {
    throw new Error("Direct cached ONNX session did not return present.* tensors.");
  }
  for (const [name, tensor] of Object.entries(oldCache)) {
    if (nextCache[name] !== tensor) tensor.dispose();
  }
  return nextCache;
}

function disposeDirectKvCache(cache: Record<string, ort.Tensor>): void {
  for (const tensor of Object.values(cache)) tensor.dispose();
}

function directKvFetches(model: DirectOrtModel): Record<string, null> {
  return Object.fromEntries(["logits", ...model.cacheOutputNames].map((name) => [name, null]));
}

function normalizeOrtTensorType(type: string): ort.Tensor.Type {
  if (type === "tensor(float16)") return "float16";
  if (type === "tensor(float)") return "float32";
  if (type === "tensor(int64)") return "int64";
  if (type === "float" || type === "float32") return "float32";
  if (type === "float16") return "float16";
  if (type === "int64") return "int64";
  throw new Error(`Unsupported direct ONNX tensor type: ${type}`);
}

function zerosForOrtTensorType(type: ort.Tensor.Type, size: number): Float32Array | Uint16Array | BigInt64Array {
  if (type === "float16") return new Uint16Array(size);
  if (type === "float32") return new Float32Array(size);
  if (type === "int64") return new BigInt64Array(size);
  throw new Error(`Unsupported direct ONNX zero tensor type: ${type}`);
}

function directCachedCandidates(dtype: string, config: any): string[] {
  const standard = cachedOnnxFileName(dtype);
  const fast = dtype === "q8" ? "model_kv_fast_quantized.onnx" : "model_kv_fast_q4f16.onnx";
  return hasExternalDataEntry(config, fast) ? [fast, standard] : [standard];
}

function cachedOnnxFileName(dtype: string): string {
  return dtype === "q8" ? "model_kv_quantized.onnx" : "model_kv_q4f16.onnx";
}

function hasExternalDataEntry(config: any, fileName: string): boolean {
  return Number(config?.["transformers.js_config"]?.use_external_data_format?.[fileName] ?? 0) > 0;
}

function hubOnnxUrl(name: string): string {
  return `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/onnx/${name}`;
}

function externalDataForModel(modelFileName: string, config?: any): Array<{ path: string; data: string }> {
  const configured = Number(config?.["transformers.js_config"]?.use_external_data_format?.[modelFileName] ?? 0);
  const count = configured || ONNX_CHUNK_COUNTS[modelFileName] || 0;
  return Array.from({ length: count }, (_value, index) => {
    const path = index === 0 ? `${modelFileName}_data` : `${modelFileName}_data_${index}`;
    return { path, data: hubOnnxUrl(path) };
  });
}

function configureDirectOrtRuntime(): void {
  ort.env.logLevel = "warning";
  ort.env.wasm.numThreads = 1;
  if (retainedGpuAdapter) ort.env.webgpu.adapter = retainedGpuAdapter;
  if (retainedGpuDevice) ort.env.webgpu.device = retainedGpuDevice;
  ort.env.webgpu.powerPreference = "high-performance";
}

function recordGenerationStats(stats: GenerationStats): void {
  if (typeof window === "undefined") return;
  const target = window as Window & {
    __talkieMetrics?: {
      generation: Array<GenerationStats & { timeMs?: number }>;
    };
  };
  target.__talkieMetrics ??= { generation: [] };
  target.__talkieMetrics.generation.push({ ...stats, timeMs: performance.now() });
}
