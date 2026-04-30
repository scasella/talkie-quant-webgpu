let serviceWorkerReady: Promise<void> | null = null;
const SERVICE_WORKER_VERSION = "2";

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
