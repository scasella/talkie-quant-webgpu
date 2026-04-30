import {
  AutoConfig,
  AutoModel,
  AutoTokenizer,
  env,
  LogLevel,
  ModelRegistry,
  Tensor
} from "@huggingface/transformers";

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
}

export type RuntimeMode = "kv-cache" | "full-sequence";
export type LoadPath = "full" | "cached";

export interface GenerationStats {
  mode: RuntimeMode;
  tokensGenerated: number;
  tokensPerSecond: number;
}

const DEFAULT_MODEL_ID = "scasella91/talkie-1930-13b-it-ONNX";
const DEFAULT_REVISION = "8353531db9d507d96b8a92f5aceb12ff71b6b753";
const MAX_CONTEXT_TOKENS = 2048;
const STOP_TOKEN_IDS = new Set([65535, 65536]);
const SUPPRESSED_TOKEN_IDS = new Set([0]);

const RUNTIME_PARAMS = runtimeSearchParams();
const MODEL_ID = RUNTIME_PARAMS.get("model") || import.meta.env.VITE_TALKIE_ONNX_MODEL_ID || DEFAULT_MODEL_ID;
const REVISION = RUNTIME_PARAMS.get("revision") || import.meta.env.VITE_TALKIE_ONNX_REVISION || DEFAULT_REVISION;
const DTYPE_OVERRIDE = RUNTIME_PARAMS.get("dtype") || import.meta.env.VITE_TALKIE_ONNX_DTYPE;
const CACHE_OVERRIDE = RUNTIME_PARAMS.get("cache");
const GRAPH_OPTIMIZATION_LEVEL =
  RUNTIME_PARAMS.get("opt") || import.meta.env.VITE_TALKIE_GRAPH_OPTIMIZATION || "disabled";
const BROWSER_CACHE_ENABLED =
  CACHE_OVERRIDE === "1" || (CACHE_OVERRIDE !== "0" && import.meta.env.VITE_TALKIE_BROWSER_CACHE === "1");

env.logLevel = LogLevel.WARNING;
env.useBrowserCache = BROWSER_CACHE_ENABLED;

type ProgressCallback = (progress: LoadProgress) => void;

interface Runtime {
  tokenizer: any;
  model: any;
  session: TalkieSession;
}

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
  const requestedPath = loadPath ?? runtimePath ?? "full";
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

    const dtypes = await getAvailableDtypes();
    const explicitDtype = DTYPE_OVERRIDE && DTYPE_OVERRIDE !== "auto" ? DTYPE_OVERRIDE : null;
    const hasExplicitDtype = explicitDtype != null;
    const preferred = explicitDtype ? [explicitDtype] : ["q4f16", "q4", "q8", "fp16", "fp32"];
    const dtype = preferred.find((candidate) => dtypes.includes(candidate)) ?? "q4f16";

    const progress_callback = (event: LoadProgress) => {
      onProgress?.(event);
    };

    const config = await AutoConfig.from_pretrained(MODEL_ID, {
      revision: REVISION,
      progress_callback
    } as any);
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      revision: REVISION,
      progress_callback
    } as any);

    const attempts = buildLoadAttempts(preferred, dtypes, requestedPath, !hasExplicitDtype);
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
        modelFileName: loaded.modelFileName
      }
    };
  })();

  return runtimePromise;
}

async function getAvailableDtypes(): Promise<string[]> {
  try {
    const dtypes = await ModelRegistry.get_available_dtypes(MODEL_ID, { revision: REVISION } as any);
    return Array.isArray(dtypes) ? dtypes : [];
  } catch {
    return [];
  }
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

  for (let step = 0; step < settings.maxNewTokens; step += 1) {
    if (signal.aborted) break;
    if (inputIds.length >= MAX_CONTEXT_TOKENS) break;

    const outputs = await model.forward({ input_ids: idsToTensor(inputIds) });
    const logits = outputs.logits ?? Object.values(outputs)[0];
    const scores = lastTokenScores(logits as any);
    const nextId = sampleToken(scores, settings);

    if (STOP_TOKEN_IDS.has(nextId)) break;

    inputIds.push(nextId);
    generated.push(nextId);
    reportStats("full-sequence", generated.length, startTime, onStats);
    const nextText = decode(tokenizer, generated);
    if (nextText !== lastText) {
      lastText = nextText;
      onText(nextText);
    }

    await new Promise((resolve) => setTimeout(resolve, 0));
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
  let cache = createEmptyKvCache(model);
  let currentInputIds = promptIds;
  let currentPositionStart = 0;
  let nextPosition = promptIds.length;

  try {
    for (let step = 0; step < settings.maxNewTokens; step += 1) {
      if (signal.aborted) break;
      if (nextPosition >= MAX_CONTEXT_TOKENS) break;

      const outputs = await model.forward({
        input_ids: idsToTensor(currentInputIds),
        position_ids: positionIdsToTensor(currentPositionStart, currentInputIds.length),
        ...cache
      });
      cache = await updateKvCache(cache, outputs);

      const logits = outputs.logits ?? Object.values(outputs)[0];
      const scores = lastTokenScores(logits as any);
      const nextId = sampleToken(scores, settings);

      if (STOP_TOKEN_IDS.has(nextId)) break;

      generated.push(nextId);
      reportStats("kv-cache", generated.length, startTime, onStats);
      const nextText = decode(tokenizer, generated);
      if (nextText !== lastText) {
        lastText = nextText;
        onText(nextText);
      }

      currentInputIds = [nextId];
      currentPositionStart = nextPosition;
      nextPosition += 1;

      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } finally {
    await disposeKvCache(cache);
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

function idsToTensor(ids: number[]): Tensor {
  return new Tensor("int64", BigInt64Array.from(ids.map((id) => BigInt(id))), [1, ids.length]);
}

function positionIdsToTensor(start: number, length: number): Tensor {
  return new Tensor(
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

  const ranked = Array.from(scores, (score, id) => ({ id, score: Number(score) }))
    .filter((entry) => Number.isFinite(entry.score) && !SUPPRESSED_TOKEN_IDS.has(entry.id))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

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

interface LoadAttempt {
  dtype: string;
  mode: RuntimeMode;
  modelFileName: string;
}

interface LoadedModel extends LoadAttempt {
  model: any;
}

function buildLoadAttempts(preferred: string[], dtypes: string[], loadPath: LoadPath, allowQ8Fallback: boolean): LoadAttempt[] {
  const candidates = preferred.filter((candidate, index) => preferred.indexOf(candidate) === index);
  const available = dtypes.length > 0 ? candidates.filter((candidate) => dtypes.includes(candidate)) : candidates;
  const ordered = allowQ8Fallback && !available.includes("q8") ? [...available, "q8"] : available;
  if (loadPath === "cached") {
    return ordered.map((dtype) => ({ dtype, mode: "kv-cache" as const, modelFileName: "model_kv" }));
  }
  return ordered.map((dtype) => ({ dtype, mode: "full-sequence" as const, modelFileName: "model" }));
}

async function loadModelAttempt(config: any, progress_callback: ProgressCallback, attempt: LoadAttempt): Promise<LoadedModel> {
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
}

function createEmptyKvCache(model: any): Record<string, Tensor> {
  const session = model.sessions?.model;
  const metadata = session?.inputMetadata ?? [];
  const config = model.config ?? {};
  const entries: Record<string, Tensor> = {};

  for (const meta of metadata) {
    if (!meta.name?.startsWith("past_key_values.")) continue;
    const dims = resolveCacheShape(meta.shape ?? [], config);
    const size = dims.reduce((total: number, dim: number) => total * dim, 1);
    entries[meta.name] = new Tensor(meta.type, zerosForTensorType(meta.type, size), dims);
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
  oldCache: Record<string, Tensor>,
  outputs: Record<string, any>
): Promise<Record<string, Tensor>> {
  const nextCache: Record<string, Tensor> = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    if (!name.startsWith("present.")) continue;
    nextCache[name.replace("present.", "past_key_values.")] = tensor as Tensor;
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

async function disposeKvCache(cache: Record<string, Tensor>): Promise<void> {
  await Promise.all(Object.values(cache).map(disposeTensor));
}

async function disposeTensor(tensor: Tensor): Promise<void> {
  if ((tensor as any).location === "gpu-buffer") {
    await (tensor as any).dispose?.();
  }
}

function reportStats(
  mode: RuntimeMode,
  tokensGenerated: number,
  startTime: number,
  onStats?: (stats: GenerationStats) => void
): void {
  if (!onStats) return;
  const elapsedSec = Math.max(0.001, (performance.now() - startTime) / 1000);
  onStats({
    mode,
    tokensGenerated,
    tokensPerSecond: tokensGenerated / elapsedSec
  });
}
