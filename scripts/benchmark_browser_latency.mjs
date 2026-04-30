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
        )} p90=${formatNumber(result.p90TokSec)}`
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
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      consoleWarnings.push({ timeMs: performance.now() - started, type: message.type(), text: message.text() });
      console.log(`[browser:${message.type()}] ${message.text()}`);
    }
  });

  try {
    await page.goto(benchmarkUrl(config), { waitUntil: "domcontentloaded" });
    if (!(await page.evaluate(() => Boolean(navigator.gpu)))) {
      throw new Error("navigator.gpu is not available");
    }

    await page.getByRole("button", { name: "Load" }).click();
    const loadResult = await waitForLoad(page, cdp, started, samples);
    if (loadResult.status !== "Ready") {
      return {
        target,
        runIndex,
        status: loadResult.status,
        error: loadResult.error ?? "load failed",
        loadMs: loadResult.timeMs,
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
  return await page.evaluate(() => window.__talkieMetrics ?? { events: [], generation: [] });
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
