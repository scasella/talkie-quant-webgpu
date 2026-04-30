let serviceWorkerReady: Promise<void> | null = null;
const SERVICE_WORKER_VERSION = "4";
const DEFAULT_MAX_PARALLEL_MODEL_FETCHES = 4;
const DEFAULT_COMPRESSED_MAX_PARALLEL_MODEL_FETCHES = 8;

export function registerTalkieFetchRetry(): void {
  if (!("serviceWorker" in navigator)) return;
  if (import.meta.env.VITE_TALKIE_FETCH_RETRY === "0") return;

  serviceWorkerReady = navigator.serviceWorker
    .register(`${import.meta.env.BASE_URL}talkie-sw.js?v=${SERVICE_WORKER_VERSION}`, { scope: import.meta.env.BASE_URL })
    .then(async (registration) => {
      await registration.update().catch(() => undefined);
      await navigator.serviceWorker.ready;
      if (!navigator.serviceWorker.controller) {
        await waitForController();
      }
      configureControllerFetchConcurrency();
    })
    .catch((error) => {
      console.warn(`[talkie] fetch retry service worker unavailable: ${error instanceof Error ? error.message : error}`);
    });
}

export async function waitForTalkieFetchRetry(): Promise<void> {
  await serviceWorkerReady;
}

function waitForController(): Promise<void> {
  if (navigator.serviceWorker.controller) return Promise.resolve();
  return new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, 1500);
    navigator.serviceWorker.addEventListener(
      "controllerchange",
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true }
    );
  });
}

function configureControllerFetchConcurrency(): void {
  navigator.serviceWorker.controller?.postMessage({
    type: "talkie-fetch-concurrency",
    maxParallelModelFetches: configuredFetchConcurrency()
  });
}

function configuredFetchConcurrency(): number {
  const params = new URLSearchParams(window.location.search);
  const configured = Number(params.get("fetches") || import.meta.env.VITE_TALKIE_FETCH_CONCURRENCY || "");
  if (Number.isFinite(configured) && configured >= 1) return Math.floor(configured);
  const compressed =
    params.get("compressed") === "1" || import.meta.env.VITE_TALKIE_COMPRESSED_EXTERNAL_DATA === "1";
  return compressed ? DEFAULT_COMPRESSED_MAX_PARALLEL_MODEL_FETCHES : DEFAULT_MAX_PARALLEL_MODEL_FETCHES;
}
