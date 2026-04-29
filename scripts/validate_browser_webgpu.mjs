import { chromium } from "playwright";

const url = process.env.TALKIE_WEB_URL ?? "http://127.0.0.1:5173/";
const prompt =
  process.env.TALKIE_WEB_PROMPT ??
  "In the voice of a 1930 radio host, describe a moonlit train station in one vivid paragraph.";
const minTokens = Number(process.env.TALKIE_WEB_MIN_TOKENS ?? "16");
const timeoutMs = Number(process.env.TALKIE_WEB_TIMEOUT_MS ?? `${45 * 60 * 1000}`);
const expectedDtype = process.env.TALKIE_WEB_EXPECTED_DTYPE ?? "q4f16";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu"]
});

try {
  const page = await browser.newPage();
  page.setDefaultTimeout(timeoutMs);
  page.on("console", (message) => {
    const type = message.type();
    if (type === "error" || type === "warning") {
      console.log(`[browser:${type}] ${message.text()}`);
    }
  });

  await page.goto(url, { waitUntil: "domcontentloaded" });
  const hasWebGPU = await page.evaluate(() => Boolean(navigator.gpu));
  if (!hasWebGPU) throw new Error("navigator.gpu is not available");

  await page.getByRole("button", { name: "Load" }).click();
  await page.waitForFunction(
    (dtype) => document.body.innerText.includes("Ready") && document.body.innerText.includes(dtype),
    expectedDtype,
    { timeout: timeoutMs }
  );
  console.log(`LOAD_READY ${expectedDtype}`);

  await page
    .locator("label", { hasText: "Max tokens" })
    .locator("input")
    .fill(String(Math.max(minTokens * 2, 24)));
  await page.locator("textarea").fill(prompt);
  await page.getByRole("button", { name: "Send" }).click();

  await page.waitForFunction(
    (minimum) => {
      const messages = [...document.querySelectorAll("article.assistant p")];
      const text = messages.at(-1)?.textContent ?? "";
      const withoutNuls = text.replace(/\u0000/g, "").trim();
      const tokens = withoutNuls.split(/\s+/).filter(Boolean);
      return tokens.length >= minimum;
    },
    minTokens,
    { timeout: timeoutMs }
  );

  const result = await page.evaluate(() => {
    const messages = [...document.querySelectorAll("article.assistant p")];
    const text = messages.at(-1)?.textContent ?? "";
    const withoutNuls = text.replace(/\u0000/g, "").trim();
    return {
      statusText: document.body.innerText,
      text,
      tokenCount: withoutNuls.split(/\s+/).filter(Boolean).length,
      nulCount: [...text].filter((char) => char === "\u0000").length
    };
  });

  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
