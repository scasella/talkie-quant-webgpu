const RETRYABLE_HOST_PARTS = ["huggingface.co", "hf.co"];
const MAX_ATTEMPTS = 4;
const DEFAULT_MAX_PARALLEL_MODEL_FETCHES = 4;

let activeModelFetches = 0;
let maxParallelModelFetches = DEFAULT_MAX_PARALLEL_MODEL_FETCHES;
const pendingModelFetches = [];

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (!shouldRetry(event.request)) return;
  event.respondWith(enqueueModelFetch(() => fetchWithRetry(event.request)));
});

self.addEventListener("message", (event) => {
  if (event.data?.type !== "talkie-fetch-concurrency") return;
  const configured = Number(event.data.maxParallelModelFetches);
  if (Number.isFinite(configured) && configured >= 1) {
    maxParallelModelFetches = Math.floor(configured);
    pumpModelFetchQueue();
  }
});

function shouldRetry(request) {
  if (request.method !== "GET") return false;
  try {
    const url = new URL(request.url);
    return RETRYABLE_HOST_PARTS.some((part) => url.hostname.includes(part));
  } catch {
    return false;
  }
}

async function fetchWithRetry(request) {
  let lastError = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(request.clone());
      if (response.ok || response.status < 500) return response;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(400 * 2 ** attempt);
  }
  throw lastError ?? new Error("Model fetch failed");
}

function enqueueModelFetch(task) {
  return new Promise((resolve, reject) => {
    pendingModelFetches.push({ task, resolve, reject });
    pumpModelFetchQueue();
  });
}

function pumpModelFetchQueue() {
  while (activeModelFetches < maxParallelModelFetches && pendingModelFetches.length > 0) {
    const item = pendingModelFetches.shift();
    activeModelFetches += 1;
    Promise.resolve()
      .then(item.task)
      .then(item.resolve, item.reject)
      .finally(() => {
        activeModelFetches -= 1;
        pumpModelFetchQueue();
      });
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
