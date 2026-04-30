import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const TARGETS = {
  "full-q4f16": { path: "full", dtype: "q4f16" },
  "cached-q4f16": { path: "cached", dtype: "q4f16" },
  "cached-q8": { path: "cached", dtype: "q8" }
};

const baseUrl = process.env.TALKIE_WEB_URL ?? "http://127.0.0.1:5173/";
const targets = (process.env.TALKIE_BENCH_TARGETS ?? "full-q4f16,cached-q4f16,cached-q8")
  .split(",")
  .map((target) => target.trim())
  .filter(Boolean);
const runs = Number(process.env.TALKIE_BENCH_RUNS ?? "3");
const timeoutMs = Number(process.env.TALKIE_BENCH_TIMEOUT_MS ?? `${45 * 60 * 1000}`);
const pollMs = Number(process.env.TALKIE_BENCH_POLL_MS ?? "2000");
const minWords = Number(process.env.TALKIE_WEB_MIN_TOKENS ?? "16");
const targetTokSec = Number(process.env.TALKIE_BENCH_TARGET_TOK_SEC ?? "1.8");
const cache = process.env.TALKIE_BROWSER_CACHE ?? "0";
const opt = process.env.TALKIE_GRAPH_OPTIMIZATION ?? "disabled";
const direct = process.env.TALKIE_DIRECT_CACHED ?? "1";
const warmup = process.env.TALKIE_DIRECT_WARMUP ?? "1";
const fetches = process.env.TALKIE_FETCH_CONCURRENCY ?? "";
const q4file = process.env.TALKIE_CACHED_Q4_FILE ?? "";
const q8file = process.env.TALKIE_CACHED_Q8_FILE ?? "";
const revision = process.env.TALKIE_ONNX_REVISION ?? "";
const compressed = process.env.TALKIE_COMPRESSED_EXTERNAL_DATA ?? "";
const headed = process.env.TALKIE_HEADLESS !== "1";
const outDir = process.env.TALKIE_BENCH_OUT_DIR ?? "output/playwright";
const prompt =
  process.env.TALKIE_WEB_PROMPT ??
  "In the voice of a 1930 radio host, describe a moonlit train station in one vivid paragraph.";

const startedAt = new Date();
await mkdir(outDir, { recursive: true });
const outPath = path.join(outDir, `browser-latency-${startedAt.toISOString().replace(/[:.]/g, "-")}.json`);

const results = [];
const browser = await chromium.launch({
  headless: !headed,
  args: ["--enable-unsafe-webgpu"]
});

try {
  for (const target of targets) {
    const config = TARGETS[target];
    if (!config) throw new Error(`Unknown benchmark target: ${target}`);
    for (let runIndex = 0; runIndex < runs; runIndex += 1) {
      const result = await runTarget(browser, target, config, runIndex);
      results.push(result);
      console.log(
        `${target} #${runIndex + 1}: ${result.status} load=${formatSeconds(result.loadMs)} ttft=${formatSeconds(
          result.ttftMs
        )} reported=${formatNumber(result.reportedTokSec)} tok/s p50=${formatNumber(
          result.p50TokSec
        )} p90=${formatNumber(result.p90TokSec)} download=${formatSeconds(
          result.phaseSummary?.onnxNetworkMs
        )} session=${formatSeconds(result.phaseSummary?.onnxSessionCreateMs)} bytes=${formatBytes(
          result.phaseSummary?.onnxFetchedBytes
        )}`
      );
    }
  }
} finally {
  await browser.close();
  const report = {
    startedAt: startedAt.toISOString(),
    baseUrl,
    targets,
    runs,
    targetTokSec,
    cache,
    opt,
    direct,
    warmup,
    fetches: fetches || "default",
    q4file: q4file || "default",
    q8file: q8file || "default",
    revision: revision || "default",
    compressed: compressed || "default",
    results,
    summary: summarize(results)
  };
  await writeFile(outPath, JSON.stringify(report, null, 2));
  console.log(`WROTE ${outPath}`);
}

function benchmarkUrl(config) {
  const url = new URL(baseUrl);
  url.searchParams.set("path", config.path);
  url.searchParams.set("dtype", config.dtype);
  url.searchParams.set("cache", cache);
  url.searchParams.set("opt", opt);
  url.searchParams.set("direct", direct);
  url.searchParams.set("warmup", warmup);
  if (fetches) url.searchParams.set("fetches", fetches);
  if (q4file) url.searchParams.set("q4file", q4file);
  if (q8file) url.searchParams.set("q8file", q8file);
  if (revision) url.searchParams.set("revision", revision);
  if (compressed) url.searchParams.set("compressed", compressed);
  return url.toString();
}

async function runTarget(browser, target, config, runIndex) {
  const page = await browser.newPage();
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable").catch(() => null);
  page.setDefaultTimeout(timeoutMs);

  const started = performance.now();
  const samples = [];
  const consoleWarnings = [];
  const network = { requests: [], responses: [] };
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      consoleWarnings.push({ timeMs: performance.now() - started, type: message.type(), text: message.text() });
      console.log(`[browser:${message.type()}] ${message.text()}`);
    }
  });
  page.on("request", (request) => {
    const requestUrl = request.url();
    if (!isModelRequest(requestUrl)) return;
    network.requests.push({
      timeMs: performance.now() - started,
      method: request.method(),
      url: requestUrl,
      file: modelFileName(requestUrl)
    });
  });
  page.on("response", (response) => {
    const responseUrl = response.url();
    if (!isModelRequest(responseUrl)) return;
    const headers = response.headers();
    network.responses.push({
      timeMs: performance.now() - started,
      status: response.status(),
      url: responseUrl,
      file: modelFileName(responseUrl),
      contentLength: numberHeader(headers["content-length"]),
      contentRange: headers["content-range"] ?? null
    });
  });

  try {
    await page.goto(benchmarkUrl(config), { waitUntil: "domcontentloaded" });
    if (!(await page.evaluate(() => Boolean(navigator.gpu)))) {
      throw new Error("navigator.gpu is not available");
    }

    await page.getByRole("button", { name: "Load" }).click();
    const loadResult = await waitForLoad(page, cdp, started, samples);
    if (loadResult.status !== "Ready") {
      const metrics = await readMetrics(page);
      return {
        target,
        runIndex,
        status: loadResult.status,
        error: loadResult.error ?? "load failed",
        loadMs: loadResult.timeMs,
        phaseSummary: summarizePhases(metrics, network),
        metrics,
        network,
        samples,
        consoleWarnings
      };
    }

    await page.locator("label", { hasText: "Max tokens" }).locator("input").fill(String(Math.max(minWords * 2, 24)));
    await page.locator("textarea").fill(prompt);
    await page.getByRole("button", { name: "Send" }).click();
    const generationResult = await waitForGeneration(page, cdp, started, samples);
    const metrics = await readMetrics(page);
    const generation = metrics.generation ?? [];
    const events = metrics.events ?? [];
    const generationStart = events.find((event) => event.kind === "generation-start")?.timeMs;
    const firstToken = generation[0]?.timeMs;
    const tokenLatencies = generation.map((item) => Number(item.lastTokenMs)).filter(Number.isFinite);
    const reportedTokSec = Number(generation.at(-1)?.tokensPerSecond ?? generationResult.tokenRate ?? 0) || null;
    const phaseSummary = summarizePhases(metrics, network);

    return {
      target,
      runIndex,
      status: generationResult.status,
      loadMs: loadResult.timeMs,
      ttftMs:
        typeof generationStart === "number" && typeof firstToken === "number" ? Math.max(0, firstToken - generationStart) : null,
      reportedTokSec,
      p50TokSec: latencyToTokSec(percentile(tokenLatencies, 0.5)),
      p90TokSec: latencyToTokSec(percentile(tokenLatencies, 0.9)),
      wordCount: generationResult.wordCount,
      nulCount: generationResult.nulCount,
      mode: generation.at(-1)?.mode ?? null,
      backend: generation.at(-1)?.backend ?? null,
      phaseSummary,
      metrics,
      network,
      tokenLatencies,
      samples,
      consoleWarnings
    };
  } catch (error) {
    return {
      target,
      runIndex,
      status: "Error",
      error: error instanceof Error ? error.message : String(error),
      network,
      samples,
      consoleWarnings
    };
  } finally {
    await page.close();
  }
}

async function waitForLoad(page, cdp, started, samples) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const sample = await collectSample(page, cdp, started);
    samples.push(sample);
    if (sample.status === "Ready" || sample.status === "Error" || sample.error) return sample;
    await page.waitForTimeout(pollMs);
  }
  return { timeMs: performance.now() - started, status: "Timeout", error: "load timeout" };
}

async function waitForGeneration(page, cdp, started, samples) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const sample = await collectSample(page, cdp, started);
    samples.push(sample);
    if (sample.wordCount >= minWords || sample.status === "Error" || sample.error) return sample;
    await page.waitForTimeout(pollMs);
  }
  return { ...(await collectSample(page, cdp, started)), status: "Timeout", error: "generation timeout" };
}

async function collectSample(page, cdp, started) {
  const bodyText = await page.locator("body").innerText({ timeout: 1000 }).catch(() => "");
  const heap = await cdp.send("Runtime.getHeapUsage").catch(() => null);
  const messages = await page.locator("article.assistant p").allTextContents().catch(() => []);
  const text = messages.at(-1) ?? "";
  const withoutNuls = text.replace(/\u0000/g, "").trim();
  return {
    timeMs: performance.now() - started,
    status: readStatus(bodyText),
    dtype: readStatusValue(bodyText, ["q4f16", "q8", "unloaded"]),
    mode: readStatusValue(bodyText, ["kv-cache", "full-sequence", "not loaded"]),
    backend: readStatusValue(bodyText, ["ort-direct", "transformers", "not loaded"]),
    tokenRate: Number(bodyText.match(/([0-9]+(?:\.[0-9]+)?) tok\/s/)?.[1] ?? 0) || null,
    wordCount: withoutNuls.split(/\s+/).filter(Boolean).length,
    nulCount: [...text].filter((char) => char === "\u0000").length,
    error: (await page.locator(".error").innerText({ timeout: 250 }).catch(() => "")) || null,
    heap
  };
}

async function readMetrics(page) {
  return await page.evaluate(() => window.__talkieMetrics ?? { events: [], load: [], generation: [] });
}

function summarizePhases(metrics, network) {
  const events = metrics.events ?? [];
  const load = metrics.load ?? [];
  const generation = metrics.generation ?? [];
  const event = (items, kind) => items.find((item) => item.kind === kind);
  const duration = (items, startKind, endKind) => {
    const start = event(items, startKind)?.timeMs;
    const end = event(items, endKind)?.timeMs;
    return typeof start === "number" && typeof end === "number" ? Math.max(0, end - start) : null;
  };
  const onnxRequests = network.requests.filter((item) => item.file?.includes(".onnx"));
  const onnxResponses = network.responses.filter((item) => item.file?.includes(".onnx"));
  const firstOnnxRequest = onnxRequests[0] ?? null;
  const lastOnnxRequest = onnxRequests.at(-1) ?? null;
  const firstOnnxResponse = onnxResponses[0] ?? null;
  const lastOnnxResponse = onnxResponses.at(-1) ?? null;
  const onnxFetchedBytes = onnxResponses.reduce((sum, item) => sum + (item.contentLength ?? 0), 0);
  const onnxNetworkMs =
    firstOnnxRequest && lastOnnxResponse ? Math.max(0, lastOnnxResponse.timeMs - firstOnnxRequest.timeMs) : null;
  const onnxSessionCreateMs = duration(load, "onnx-session-create-start", "onnx-session-create-end");
  const generationStart = event(events, "generation-start")?.timeMs;
  const firstToken = generation[0]?.timeMs;
  return {
    appLoadMs: duration(events, "load-start", "load-ready"),
    configMs: duration(load, "config-start", "config-end"),
    tokenizerMs: duration(load, "tokenizer-start", "tokenizer-end"),
    fetchRetryWaitMs: duration(load, "fetch-retry-wait-start", "fetch-retry-wait-end"),
    onnxSessionCreateMs,
    onnxNetworkMs,
    estimatedSessionCompileMs:
      onnxSessionCreateMs != null && onnxNetworkMs != null ? Math.max(0, onnxSessionCreateMs - onnxNetworkMs) : null,
    warmupMs: duration(load, "warmup-start", "warmup-end"),
    ttftMs:
      typeof generationStart === "number" && typeof firstToken === "number" ? Math.max(0, firstToken - generationStart) : null,
    onnxFetchedBytes,
    firstOnnxRequest,
    lastOnnxRequest,
    firstOnnxResponse,
    lastOnnxResponse,
    requestCount: network.requests.length,
    responseCount: network.responses.length
  };
}

function readStatus(text) {
  const statuses = ["Ready", "Error", "Loading", "Idle", "Resetting", "Thinking", "Stopped"];
  return statuses.find((status) => text.includes(status)) ?? "Unknown";
}

function readStatusValue(text, values) {
  return values.find((value) => text.includes(value)) ?? null;
}

function percentile(values, p) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1));
  return sorted[index];
}

function latencyToTokSec(ms) {
  return ms && ms > 0 ? 1000 / ms : null;
}

function isModelRequest(url) {
  return url.includes("huggingface.co") || url.includes("/onnx/") || url.includes("/resolve/");
}

function modelFileName(url) {
  const clean = url.split(/[?#]/)[0];
  const parts = clean.split("/");
  return parts.at(-1) ?? url;
}

function numberHeader(value) {
  const parsed = Number(value ?? "");
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function summarize(items) {
  return Object.fromEntries(
    targets.map((target) => {
      const rows = items.filter((item) => item.target === target);
      const successes = rows.filter((item) => item.status !== "Error" && item.status !== "Timeout");
      return [
        target,
        {
          runs: rows.length,
          successes: successes.length,
          reportedTokSecMedian: median(successes.map((item) => item.reportedTokSec).filter(Number.isFinite)),
          p50TokSecMedian: median(successes.map((item) => item.p50TokSec).filter(Number.isFinite)),
          p90TokSecMedian: median(successes.map((item) => item.p90TokSec).filter(Number.isFinite)),
          targetMet: successes.some(
            (item) => Number(item.reportedTokSec) >= targetTokSec || Number(item.p50TokSec) >= targetTokSec
          )
        }
      ];
    })
  );
}

function median(values) {
  return percentile(values, 0.5);
}

function formatSeconds(ms) {
  return ms == null ? "--" : `${(ms / 1000).toFixed(1)}s`;
}

function formatNumber(value) {
  return value == null || !Number.isFinite(value) ? "--" : Number(value).toFixed(2);
}

function formatBytes(value) {
  return value == null || !Number.isFinite(value) ? "--" : `${(value / 1e9).toFixed(2)}GB`;
}
