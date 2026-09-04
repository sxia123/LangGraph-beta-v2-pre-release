"""Inspect browser loading of http://127.0.0.1:8080/ and capture console errors and network failures."""

import os
import time

from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\sidne\.gemini\antigravity\brain\3b00f35f-1695-4530-a6db-419a27a55d2f"


def debug_browser():
    console_logs = []
    console_errors = []
    page_errors = []
    failed_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        page.on("console", lambda msg: (
            console_errors.append(f"[{msg.type.upper()}] {msg.text}")
            if msg.type in ("error", "warning")
            else console_logs.append(f"[{msg.type.upper()}] {msg.text}")
        ))
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url} -> {req.failure}"))

        print("Navigating to http://127.0.0.1:8080/ ...")
        t0 = time.time()
        try:
            page.goto("http://127.0.0.1:8080/", timeout=30000, wait_until="load")
            load_time = time.time() - t0
            print(f"Page loaded in {load_time:.2f} seconds.")
        except Exception as e:
            print(f"Error navigating: {e}")

        time.sleep(3)

        # Check DOM elements
        chat_textarea = page.query_selector("#chatTextarea")
        status_bullet = page.query_selector("#statusBullet")
        status_title = page.evaluate("() => document.getElementById('statusTitle')?.textContent")
        tools_badge = page.evaluate("() => document.getElementById('toolsBadge')?.textContent")
        memory_badge = page.evaluate("() => document.getElementById('memoryBadge')?.textContent")

        print("\n--- DOM INSPECTION ---")
        print("chatTextarea present:", chat_textarea is not None)
        print("statusBullet present:", status_bullet is not None)
        print("statusTitle text:", status_title)
        print("toolsBadge text:", tools_badge)
        print("memoryBadge text:", memory_badge)

        shot_path = os.path.join(ARTIFACT_DIR, "browser_server_load_debug.png")
        page.screenshot(path=shot_path, full_page=False)
        print(f"\nScreenshot saved to: {shot_path}")

        print("\n--- CONSOLE ERRORS ---")
        for e in console_errors:
            print(" ", e)
        if not console_errors:
            print("  None")

        print("\n--- PAGE ERRORS ---")
        for e in page_errors:
            print(" ", e)
        if not page_errors:
            print("  None")

        print("\n--- FAILED REQUESTS ---")
        for r in failed_requests:
            print(" ", r)
        if not failed_requests:
            print("  None")

        browser.close()


if __name__ == "__main__":
    debug_browser()
