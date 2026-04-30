import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const baseUrl = process.env.TALKIE_WEB_URL ?? "http://127.0.0.1:5173/";
const loadPath = process.env.TALKIE_LOAD_PATH ?? "full";
const dtype = process.env.TALKIE_DTYPE ?? "q4f16";
const cache = process.env.TALKIE_BROWSER_CACHE ?? "0";
const timeoutMs = Number(process.env.TALKIE_DIAG_TIMEOUT_MS ?? `${20 * 60 * 1000}`);
const pollMs = Number(process.env.TALKIE_DIAG_POLL_MS ?? "5000");
const outDir = process.env.TALKIE_DIAG_OUT_DIR ?? "output/playwright";
const headed = process.env.TALKIE_HEADLESS !== "1";
const shouldGenerate = process.env.TALKIE_DIAG_GENERATE === "1";
const prompt =
  process.env.TALKIE_WEB_PROMPT ??
  "In the voice of a 1930 radio host, describe a moonlit train station in one vivid paragraph.";
const minTokens = Number(process.env.TALKIE_WEB_MIN_TOKENS ?? "16");

const url = new URL(baseUrl);
url.searchParams.set("path", loadPath);
url.searchParams.set("dtype", dtype);
url.searchParams.set("cache", cache);

const startedAt = new Date();
const trace = {
  startedAt: startedAt.toISOString(),
  url: url.toString(),
  loadPath,
  dtype,
  cache,
  timeoutMs,
  console: [],
  pageErrors: [],
  requests: [],
  responses: [],
  snapshots: [],
  result: null,
  generation: null
};

await mkdir(outDir, { recursive: true });
const outPath = path.join(outDir, `browser-load-${startedAt.toISOString().replace(/[:.]/g, "-")}.json`);

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
  page.on("response", async (response) => {
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
  const hasWebGPU = await page.evaluate(() => Boolean(navigator.gpu));
  if (!hasWebGPU) throw new Error("navigator.gpu is not available");

  await page.getByRole("button", { name: "Load" }).click();

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const snapshot = await collectSnapshot(page, cdp, startedAt.getTime());
    trace.snapshots.push(snapshot);
    console.log(
      `[${Math.round(snapshot.timeMs / 1000)}s] ${snapshot.status} ${snapshot.percent ?? "--"}% ${snapshot.detail ?? ""}`
    );

    if (snapshot.status === "Ready" || snapshot.status === "Error" || snapshot.error) {
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

  if (shouldGenerate && trace.result.status === "Ready") {
    trace.generation = await generateSmoke(page, startedAt.getTime(), deadline);
  }
} finally {
  await browser.close();
  await writeFile(outPath, JSON.stringify(trace, null, 2));
  console.log(`WROTE ${outPath}`);
}

async function generateSmoke(page, startTimeMs, deadline) {
  const maxTokens = Math.max(minTokens * 2, 24);
  await page
    .locator("label", { hasText: "Max tokens" })
    .locator("input")
    .fill(String(maxTokens));
  await page.locator("textarea").fill(prompt);
  await page.getByRole("button", { name: "Send" }).click();

  while (Date.now() < deadline) {
    const sample = await generationSample(page, startTimeMs);
    console.log(`[gen ${Math.round(sample.timeMs / 1000)}s] ${sample.wordCount} words ${sample.status}`);
    if (sample.wordCount >= minTokens || sample.status === "Error") {
      return sample;
    }
    await page.waitForTimeout(pollMs);
  }

  return {
    ...(await generationSample(page, startTimeMs)),
    status: "Timeout"
  };
}

async function generationSample(page, startTimeMs) {
  return await page.evaluate(
    (start) => {
      const bodyText = document.body.innerText;
      const messages = [...document.querySelectorAll("article.assistant p")];
      const text = messages.at(-1)?.textContent ?? "";
      const withoutNuls = text.replace(/\u0000/g, "").trim();
      const words = withoutNuls.split(/\s+/).filter(Boolean);
      const tokenRate = bodyText.match(/([0-9]+(?:\.[0-9]+)?) tok\/s/)?.[1] ?? null;
      const error = document.querySelector(".error")?.textContent ?? null;
      return {
        timeMs: Date.now() - start,
        status: bodyText.includes("Error") ? "Error" : bodyText.includes("Thinking") ? "Thinking" : "Ready",
        text,
        wordCount: words.length,
        nulCount: [...text].filter((char) => char === "\u0000").length,
        tokenRate,
        error
      };
    },
    startTimeMs
  );
}

function isModelRequest(url) {
  return url.includes("/resolve/") || url.includes("/onnx/") || url.includes("huggingface.co");
}

async function collectSnapshot(page, cdp, startTimeMs) {
  const text = await page.locator("body").innerText({ timeout: 1000 }).catch(() => "");
  const errorText = await page.locator(".error").innerText({ timeout: 500 }).catch(() => "");
  const meterText = await page.locator(".meter").innerText({ timeout: 500 }).catch(() => "");
  const heap = await cdp.send("Runtime.getHeapUsage").catch(() => null);
  const metrics = await cdp.send("Performance.getMetrics").catch(() => null);
  const { jsHeapUsed, jsHeapTotal } = await page
    .evaluate(() => {
      const memory = performance.memory;
      return memory
        ? {
            jsHeapUsed: memory.usedJSHeapSize,
            jsHeapTotal: memory.totalJSHeapSize
          }
        : { jsHeapUsed: null, jsHeapTotal: null };
    })
    .catch(() => ({ jsHeapUsed: null, jsHeapTotal: null }));

  return {
    timeMs: Date.now() - startTimeMs,
    status: readStatus(text),
    percent: readPercent(text),
    detail: readMeterDetail(text),
    error: errorText || readError(text),
    meterText,
    bodyTextTail: text.slice(-2000),
    heap,
    metrics: summarizeMetrics(metrics),
    jsHeapUsed,
    jsHeapTotal
  };
}

function readStatus(text) {
  const statuses = ["Ready", "Error", "Loading", "Idle", "Resetting", "Thinking", "Stopped"];
  return statuses.find((status) => text.includes(status)) ?? "Unknown";
}

function readPercent(text) {
  const matches = [...text.matchAll(/(\d+)%/g)].map((match) => Number(match[1]));
  return matches.length > 0 ? matches.at(-1) : null;
}

function readMeterDetail(text) {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.find((line) => line.includes(".onnx")) ?? null;
}

function readError(text) {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.find((line) => /array buffer|allocation|quota|failed/i.test(line)) ?? null;
}

function summarizeMetrics(metrics) {
  if (!metrics?.metrics) return null;
  const keep = new Set(["JSHeapUsedSize", "JSHeapTotalSize", "Nodes", "Documents", "Frames"]);
  return Object.fromEntries(metrics.metrics.filter((metric) => keep.has(metric.name)).map((metric) => [metric.name, metric.value]));
}
