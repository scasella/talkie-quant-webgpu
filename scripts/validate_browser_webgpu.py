#!/usr/bin/env python3
"""Validate the Talkie browser runner through Chromium/WebGPU."""

from __future__ import annotations

import json
import os

from playwright.sync_api import sync_playwright


URL = os.environ.get("TALKIE_WEB_URL", "http://127.0.0.1:5173/")
PROMPT = os.environ.get(
    "TALKIE_WEB_PROMPT",
    "In the voice of a 1930 radio host, describe a moonlit train station in one vivid paragraph.",
)
MIN_TOKENS = int(os.environ.get("TALKIE_WEB_MIN_TOKENS", "16"))
TIMEOUT_MS = int(os.environ.get("TALKIE_WEB_TIMEOUT_MS", str(45 * 60 * 1000)))
EXPECTED_DTYPE = os.environ.get("TALKIE_WEB_EXPECTED_DTYPE", "q4f16")


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, args=["--enable-unsafe-webgpu"])
        try:
            page = browser.new_page()
            page.set_default_timeout(TIMEOUT_MS)
            page.on(
                "console",
                lambda message: print(f"[browser:{message.type}] {message.text}")
                if message.type in {"error", "warning"}
                else None,
            )
            page.goto(URL, wait_until="domcontentloaded")
            if not page.evaluate("Boolean(navigator.gpu)"):
                raise RuntimeError("navigator.gpu is not available")

            page.get_by_role("button", name="Load").click()
            page.wait_for_function(
                "expected => document.body.innerText.includes('Ready') && document.body.innerText.includes(expected)",
                arg=EXPECTED_DTYPE,
                timeout=TIMEOUT_MS,
            )
            print(f"LOAD_READY {EXPECTED_DTYPE}", flush=True)

            page.locator("label", has_text="Max tokens").locator("input").fill(str(max(MIN_TOKENS * 2, 24)))
            page.locator("textarea").fill(PROMPT)
            page.get_by_role("button", name="Send").click()
            page.wait_for_function(
                """minimum => {
                  const messages = [...document.querySelectorAll('article.assistant p')];
                  const text = messages.at(-1)?.textContent ?? '';
                  const withoutNuls = text.replace(/\\u0000/g, '').trim();
                  const tokens = withoutNuls.split(/\\s+/).filter(Boolean);
                  return tokens.length >= minimum;
                }""",
                arg=MIN_TOKENS,
                timeout=TIMEOUT_MS,
            )
            result = page.evaluate(
                """() => {
                  const messages = [...document.querySelectorAll('article.assistant p')];
                  const text = messages.at(-1)?.textContent ?? '';
                  const withoutNuls = text.replace(/\\u0000/g, '').trim();
                  return {
                    text,
                    tokenCount: withoutNuls.split(/\\s+/).filter(Boolean).length,
                    nulCount: [...text].filter((char) => char === '\\u0000').length,
                    body: document.body.innerText,
                  };
                }"""
            )
            print(json.dumps(result, indent=2), flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
