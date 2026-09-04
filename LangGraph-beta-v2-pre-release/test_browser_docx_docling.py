"""Automated Playwright browser test for testing server.py and Docling structured document parsing."""

import os
import sys
import time
from typing import Any, Dict

import requests
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\sidne\.gemini\antigravity\brain\835c8682-8055-4d87-948b-f483cbce4e54"
DOCX_FILE = os.path.abspath("test_enterprise_document_2026.docx")
SERVER_URL = "http://127.0.0.1:8080"


def ensure_artifact_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def test_api_status() -> Dict[str, Any]:
    """Verify backend API endpoints before browser execution."""
    print("Testing /api/status endpoint...")
    res = requests.get(f"{SERVER_URL}/api/status", timeout=10)
    data = res.json()
    print(f"Status OK: {data.get('connected')}, Provider: {data.get('provider')}, Message: {data.get('message')}")

    print("Testing /api/tools endpoint for docling_parse...")
    tools_res = requests.get(f"{SERVER_URL}/api/tools", timeout=10)
    tools_data = tools_res.json()
    tool_names = [t.get("name") for t in tools_data.get("tools", [])]
    has_docling = "docling_parse" in tool_names
    print(f"Available tools ({len(tool_names)}): docling_parse present={has_docling}")

    print("Testing /api/tools/run with docling_parse on generated DOCX...")
    tool_run_res = requests.post(
        f"{SERVER_URL}/api/tools/run",
        json={"tool": "docling_parse", "args": {"file_path": DOCX_FILE}},
        timeout=30,
    )
    run_data = tool_run_res.json()
    print(f"Docling Tool Run Result: ok={run_data.get('ok')}, message_len={len(run_data.get('message', ''))}")

    return {
        "status": data,
        "tools_count": len(tool_names),
        "has_docling": has_docling,
        "docling_run_ok": run_data.get("ok"),
    }


def run_browser_docx_test() -> Dict[str, Any]:
    ensure_artifact_dir()
    results: Dict[str, Any] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print(f"\n[Browser] Navigating to {SERVER_URL}/ ...")
        page.goto(f"{SERVER_URL}/", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#chatTextarea", timeout=10000)
        time.sleep(1)

        # -------------------------------------------------------------
        # 1. Capture Initial UI State
        # -------------------------------------------------------------
        init_shot = os.path.join(ARTIFACT_DIR, "browser_initial_ui.png")
        page.screenshot(path=init_shot, full_page=False)
        print(f"[Browser] Saved initial UI screenshot: {init_shot}")
        results["initial_screenshot"] = init_shot

        # Inspect DOM badges
        status_text = page.evaluate("() => document.getElementById('statusTitle')?.textContent")
        tools_text = page.evaluate("() => document.getElementById('toolsBadge')?.textContent")
        memory_text = page.evaluate("() => document.getElementById('memoryBadge')?.textContent")
        print(f"[Browser] Status Badge: '{status_text}', Tools Badge: '{tools_text}', Memory Badge: '{memory_text}'")

        # -------------------------------------------------------------
        # 2. Select Chart Architecture Pipeline
        # -------------------------------------------------------------
        print("[Browser] Selecting 'Chart Architecture' workflow...")
        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="chart"]', timeout=5000)
        page.click('button[data-pipeline="chart"]')
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 3. Stage the Generated DOCX File
        # -------------------------------------------------------------
        print(f"[Browser] Staging enterprise DOCX file: {DOCX_FILE}")
        page.set_input_files("#fileInput", DOCX_FILE)
        page.wait_for_selector(".file-stage-chip", timeout=5000)
        time.sleep(0.5)

        staged_shot = os.path.join(ARTIFACT_DIR, "browser_docx_staged.png")
        page.screenshot(path=staged_shot, full_page=False)
        print(f"[Browser] Saved staged file screenshot: {staged_shot}")
        results["staged_screenshot"] = staged_shot

        # -------------------------------------------------------------
        # 4. Fill Prompt & Submit
        # -------------------------------------------------------------
        prompt = (
            "Decipher test_enterprise_document_2026.docx using IBM Docling structured parsing "
            "and visualize the technical architecture and media ingestion pipeline with an interactive Mermaid diagram."
        )
        print(f"[Browser] Entering prompt: {prompt}")
        page.fill("#chatTextarea", prompt)
        time.sleep(0.5)

        print("[Browser] Clicking Send...")
        page.click("#sendBtn")
        time.sleep(1)

        # -------------------------------------------------------------
        # 5. Wait for Response Streaming & Completion
        # -------------------------------------------------------------
        print("[Browser] Waiting for assistant response to stream and complete...")
        page.wait_for_selector(".message-row.assistant", timeout=20000)
        page.wait_for_function("() => document.getElementById('sendBtn')?.style.display !== 'none'", timeout=60000)
        time.sleep(3)

        # Scroll to bottom
        page.evaluate("() => document.getElementById('chatViewport').scrollTo(0, document.getElementById('chatViewport').scrollHeight)")
        time.sleep(1)

        # -------------------------------------------------------------
        # 6. Capture Final Result Screenshots
        # -------------------------------------------------------------
        result_shot = os.path.join(ARTIFACT_DIR, "browser_docx_result.png")
        page.screenshot(path=result_shot, full_page=False)
        print(f"[Browser] Saved full result screenshot: {result_shot}")
        results["result_screenshot"] = result_shot

        # Close-up of assistant response
        card_shot = os.path.join(ARTIFACT_DIR, "browser_docx_card.png")
        last_msg = page.locator(".message-row.assistant").last
        last_msg.screenshot(path=card_shot)
        print(f"[Browser] Saved assistant card screenshot: {card_shot}")
        results["card_screenshot"] = card_shot

        # Verify page content
        content = page.content()
        has_docling_mention = "docling" in content.lower() or "docling" in page.inner_text(".message-row.assistant").lower()
        has_mermaid = "mermaid" in content.lower() or "<svg" in content
        has_enterprise = "Enterprise" in content or "Architecture" in content or "Subsystem" in content
        print(f"[Browser] Content checks: Docling detected={has_docling_mention}, Mermaid/SVG={has_mermaid}, Enterprise Content={has_enterprise}")

        results["has_docling_mention"] = has_docling_mention
        results["has_mermaid"] = has_mermaid
        results["has_enterprise"] = has_enterprise

        browser.close()

    return results


if __name__ == "__main__":
    api_info = test_api_status()
    browser_info = run_browser_docx_test()

    print("\n================ TEST SUMMARY ================")
    print("API Status:", api_info)
    print("Browser Test Results:", browser_info)
    print("==============================================")
    sys.exit(0)
