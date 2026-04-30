import * as ort from "onnxruntime-web/webgpu";

const DEFAULT_REPO = "scasella91/talkie-1930-13b-it-ONNX";
const DEFAULT_REVISION = "8353531db9d507d96b8a92f5aceb12ff71b6b753";
const DEFAULT_FILE = "model_q4f16.onnx";
const DEFAULT_CHUNKS: Record<string, number> = {
  "model_kv_q4f16.onnx": 32,
  "model_kv_quantized.onnx": 42,
  "model_q4f16.onnx": 22,
  "model_quantized.onnx": 31
};

const params = new URLSearchParams(window.location.search);
const repo = params.get("repo") || DEFAULT_REPO;
const revision = params.get("revision") || DEFAULT_REVISION;
const file = params.get("file") || DEFAULT_FILE;
const chunkCount = Number(params.get("chunks") || DEFAULT_CHUNKS[file] || 0);
const executionProvider = params.get("ep") || "webgpu";
const threads = Number(params.get("threads") || "1");
const graphOptimizationLevel = (params.get("opt") || "all") as ort.InferenceSession.SessionOptions["graphOptimizationLevel"];

ort.env.logLevel = "warning";
ort.env.wasm.numThreads = threads;
ort.env.wasm.wasmPaths = {
  mjs: "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs",
  wasm: "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm"
};

setText("model", `${repo}@${revision}/onnx/${file}`);
setText("ep", `${executionProvider}, wasm threads ${threads}, opt ${graphOptimizationLevel}`);
setText("chunks", String(chunkCount));

void run();

async function run() {
  try {
    setStatus("checking WebGPU");
    await reportGpuLimits();

    const modelUrl = hubUrl(file);
    const externalData = externalDataNames(file, chunkCount).map((name) => ({
      path: name,
      data: hubUrl(name)
    }));

    log(`modelUrl: ${modelUrl}`);
    log(`externalData: ${externalData.length}`);
    setStatus("creating ONNX Runtime session");
    const started = performance.now();
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: [executionProvider],
      externalData,
      graphOptimizationLevel
    });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setStatus("Ready");
    log(`ready in ${seconds}s`);
    log(`inputs: ${session.inputNames.join(", ")}`);
    log(`outputs: ${session.outputNames.join(", ")}`);
    await session.release();
  } catch (error) {
    setStatus("Error");
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error);
    log(message);
    console.warn(`[ort-direct] ${message}`);
  }
}

async function reportGpuLimits() {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<any> } }).gpu;
  if (!gpu) {
    throw new Error("navigator.gpu is not available");
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter is not available");
  }
  const limits = adapter.limits;
  setText(
    "limits",
    [
      `maxBufferSize=${limits.maxBufferSize}`,
      `maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize}`,
      `maxBufferBindingSize=${limits.maxBufferBindingSize}`
    ].join(", ")
  );
}

function externalDataNames(modelFile: string, count: number): string[] {
  if (count <= 0) return [];
  const prefix = `${modelFile}_data`;
  return Array.from({ length: count }, (_value, index) => (index === 0 ? prefix : `${prefix}_${index}`));
}

function hubUrl(name: string): string {
  return `https://huggingface.co/${repo}/resolve/${revision}/onnx/${name}`;
}

function setStatus(value: string) {
  setText("status", value);
  log(`status: ${value}`);
}

function setText(id: string, value: string) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function log(value: string) {
  const element = document.getElementById("log");
  if (element) element.textContent += `${new Date().toISOString()} ${value}\n`;
}
