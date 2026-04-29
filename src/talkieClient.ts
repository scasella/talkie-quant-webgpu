import {
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
}

const DEFAULT_MODEL_ID = "scasella91/talkie-1930-13b-it-ONNX";
const DEFAULT_REVISION = "main";
const MAX_CONTEXT_TOKENS = 2048;
const STOP_TOKEN_IDS = new Set([65535, 65536]);
const SUPPRESSED_TOKEN_IDS = new Set([0]);

const MODEL_ID = import.meta.env.VITE_TALKIE_ONNX_MODEL_ID || DEFAULT_MODEL_ID;
const REVISION = import.meta.env.VITE_TALKIE_ONNX_REVISION || DEFAULT_REVISION;
const DTYPE_OVERRIDE = import.meta.env.VITE_TALKIE_ONNX_DTYPE;

env.logLevel = LogLevel.WARNING;
env.useBrowserCache = true;

type ProgressCallback = (progress: LoadProgress) => void;

interface Runtime {
  tokenizer: any;
  model: any;
  session: TalkieSession;
}

let runtimePromise: Promise<Runtime> | null = null;

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export function modelDefaults(): Pick<TalkieSession, "modelId" | "revision"> {
  return { modelId: MODEL_ID, revision: REVISION };
}

export async function resetTalkieRuntime(): Promise<void> {
  if (!runtimePromise) return;
  const runtime = await runtimePromise.catch(() => null);
  runtimePromise = null;
  await runtime?.model?.dispose?.();
}

export async function loadTalkieRuntime(onProgress?: ProgressCallback): Promise<Runtime> {
  if (runtimePromise) return runtimePromise;

  runtimePromise = (async () => {
    if (!hasWebGPU()) {
      throw new Error("WebGPU is not available in this browser.");
    }

    const dtypes = await getAvailableDtypes();
    const preferred =
      DTYPE_OVERRIDE && DTYPE_OVERRIDE !== "auto"
        ? [DTYPE_OVERRIDE]
        : ["q4f16", "q4", "q8", "fp16", "fp32"];
    const dtype = preferred.find((candidate) => dtypes.includes(candidate)) ?? "q4f16";

    const progress_callback = (event: LoadProgress) => {
      onProgress?.(event);
    };

    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      revision: REVISION,
      progress_callback
    } as any);

    let model: any;
    let loadedDtype = dtype;
    try {
      model = await AutoModel.from_pretrained(MODEL_ID, {
        revision: REVISION,
        device: "webgpu",
        dtype,
        progress_callback
      } as any);
    } catch (error) {
      if (dtype === "q8") throw error;
      loadedDtype = "q8";
      model = await AutoModel.from_pretrained(MODEL_ID, {
        revision: REVISION,
        device: "webgpu",
        dtype: "q8",
        progress_callback
      } as any);
    }

    return {
      tokenizer,
      model,
      session: {
        modelId: MODEL_ID,
        revision: REVISION,
        dtype: loadedDtype,
        dtypes
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
  onProgress?: ProgressCallback
): Promise<string> {
  const { tokenizer, model } = await loadTalkieRuntime(onProgress);
  const prompt = formatTalkieMessages(tokenizer, messages);
  const encoded = await tokenizer(prompt, { add_special_tokens: false });
  let inputIds = tensorToIds(encoded.input_ids);

  const availablePromptTokens = Math.max(1, MAX_CONTEXT_TOKENS - settings.maxNewTokens);
  if (inputIds.length > availablePromptTokens) {
    inputIds = inputIds.slice(inputIds.length - availablePromptTokens);
  }

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
    const nextText = decode(tokenizer, generated);
    if (nextText !== lastText) {
      lastText = nextText;
      onText(nextText);
    }

    await new Promise((resolve) => setTimeout(resolve, 0));
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
