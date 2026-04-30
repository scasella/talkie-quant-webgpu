import { spawn } from "node:child_process";

const fetchMatrix = (process.env.TALKIE_FETCH_MATRIX ?? "2,4,6,8,12")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

for (const fetches of fetchMatrix) {
  console.log(`\n=== fetches=${fetches} ===`);
  await runBenchmark(fetches);
}

function runBenchmark(fetches) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["scripts/benchmark_browser_latency.mjs"], {
      stdio: "inherit",
      env: {
        ...process.env,
        TALKIE_FETCH_CONCURRENCY: fetches,
        TALKIE_BENCH_TARGETS: process.env.TALKIE_BENCH_TARGETS ?? "cached-q4f16",
        TALKIE_BENCH_RUNS: process.env.TALKIE_BENCH_RUNS ?? "1"
      }
    });
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`fetches=${fetches} benchmark failed with exit code ${code}`));
    });
    child.on("error", reject);
  });
}
