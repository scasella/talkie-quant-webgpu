import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const baseUrl = process.env.TALKIE_WEB_URL ?? "http://127.0.0.1:5173/";
const modelFile = process.env.TALKIE_ORT_FILE ?? "model_q4f16.onnx";
const chunks = process.env.TALKIE_ORT_CHUNKS ?? "";
const executionProvider = process.env.TALKIE_ORT_EP ?? "webgpu";
const threads = process.env.TALKIE_ORT_THREADS ?? "1";
const opt = process.env.TALKIE_ORT_OPT ?? "all";
const timeoutMs = Number(process.env.TALKIE_DIAG_TIMEOUT_MS ?? `${15 * 60 * 1000}`);
const pollMs = Number(process.env.TALKIE_DIAG_POLL_MS ?? "15000");
const outDir = process.env.TALKIE_DIAG_OUT_DIR ?? "output/playwright";
const headed = process.env.TALKIE_HEADLESS !== "1";

const url = new URL(baseUrl);
url.pathname = `${url.pathname.replace(/\/$/, "")}/diagnostics/ort-direct.html`;
url.searchParams.set("file", modelFile);
url.searchParams.set("ep", executionProvider);
url.searchParams.set("threads", threads);
url.searchParams.set("opt", opt);
if (chunks) url.searchParams.set("chunks", chunks);

const startedAt = new Date();
const trace = {
  startedAt: startedAt.toISOString(),
  url: url.toString(),
  modelFile,
  chunks,
  executionProvider,
  threads,
  opt,
  timeoutMs,
  console: [],
  pageErrors: [],
  requests: [],
  responses: [],
  snapshots: [],
  result: null
};

await mkdir(outDir, { recursive: true });
const outPath = path.join(outDir, `ort-direct-${startedAt.toISOString().replace(/[:.]/g, "-")}.json`);

const browser = await chromium.launch({
  headless: !headed,
  args: ["--enable-unsafe-webgpu"]
});

try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable").catch(() => null);

  page.setDefaultTimeout(timeoutMs);
  page.on("console", (message) => {
    trace.console.push({
      timeMs: Date.now() - startedAt.getTime(),
      type: message.type(),
      text: message.text()
    });
    const type = message.type();
    if (type === "error" || type === "warning") {
      console.log(`[browser:${type}] ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => {
    trace.pageErrors.push({
      timeMs: Date.now() - startedAt.getTime(),
      message: error.message,
      stack: error.stack
    });
    console.log(`[pageerror] ${error.message}`);
  });
  page.on("request", (request) => {
    const requestUrl = request.url();
    if (!isModelRequest(requestUrl)) return;
    trace.requests.push({
      timeMs: Date.now() - startedAt.getTime(),
      method: request.method(),
      url: requestUrl
    });
  });
  page.on("response", (response) => {
    const responseUrl = response.url();
    if (!isModelRequest(responseUrl)) return;
    const headers = response.headers();
    trace.responses.push({
      timeMs: Date.now() - startedAt.getTime(),
      status: response.status(),
      url: responseUrl,
      contentLength: headers["content-length"] ?? null,
      contentRange: headers["content-range"] ?? null
    });
  });

  await page.goto(url.toString(), { waitUntil: "domcontentloaded" });

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const snapshot = await collectSnapshot(page, cdp, startedAt.getTime());
    trace.snapshots.push(snapshot);
    console.log(`[${Math.round(snapshot.timeMs / 1000)}s] ${snapshot.status}`);
    if (snapshot.status === "Ready" || snapshot.status === "Error") {
      trace.result = snapshot;
      break;
    }
    await page.waitForTimeout(pollMs);
  }

  if (!trace.result) {
    trace.result = {
      timeMs: Date.now() - startedAt.getTime(),
      status: "Timeout"
    };
  }
} finally {
  await browser.close();
  await writeFile(outPath, JSON.stringify(trace, null, 2));
  console.log(`WROTE ${outPath}`);
}

function isModelRequest(url) {
  return url.includes("/resolve/") || url.includes("/onnx/") || url.includes("huggingface.co");
}

async function collectSnapshot(page, cdp, startTimeMs) {
  const status = await page.locator("#status").innerText({ timeout: 1000 }).catch(() => "Unknown");
  const log = await page.locator("#log").innerText({ timeout: 1000 }).catch(() => "");
  const limits = await page.locator("#limits").innerText({ timeout: 1000 }).catch(() => "");
  const heap = await cdp.send("Runtime.getHeapUsage").catch(() => null);
  const metrics = await cdp.send("Performance.getMetrics").catch(() => null);
  return {
    timeMs: Date.now() - startTimeMs,
    status,
    limits,
    logTail: log.slice(-4000),
    heap,
    metrics: summarizeMetrics(metrics)
  };
}

function summarizeMetrics(metrics) {
  if (!metrics?.metrics) return null;
  const keep = new Set(["JSHeapUsedSize", "JSHeapTotalSize", "Nodes", "Documents", "Frames"]);
  return Object.fromEntries(metrics.metrics.filter((metric) => keep.has(metric.name)).map((metric) => [metric.name, metric.value]));
}
