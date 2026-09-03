"""Playwright browser test for testing Chart Pipeline Excel deciphering and charting UI."""

import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\sidne\.gemini\antigravity\brain\3b00f35f-1695-4530-a6db-419a27a55d2f"
EXCEL_FILE = os.path.abspath("mock_enterprise_financial_q1_2026.xlsx")


def run_browser_test():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("Navigating to http://127.0.0.1:8080/ ...")
        page.goto("http://127.0.0.1:8080/", wait_until="networkidle")
        page.wait_for_selector("#chatTextarea", timeout=10000)
        time.sleep(1)

        # Screenshot 1: Initial UI State
        initial_shot = os.path.join(ARTIFACT_DIR, "browser_initial_ui.png")
        page.screenshot(path=initial_shot, full_page=False)
        print(f"Captured initial screenshot: {initial_shot}")

        # Select 'Chart Architecture' workflow
        print("Selecting 'Chart Architecture' workflow...")
        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="chart"]', timeout=5000)
        page.click('button[data-pipeline="chart"]')
        time.sleep(0.5)

        # Attach the Excel spreadsheet
        print(f"Staging file attachment: {EXCEL_FILE}")
        page.set_input_files("#fileInput", EXCEL_FILE)
        page.wait_for_selector(".file-stage-chip", timeout=5000)
        time.sleep(1)

        # Fill prompt
        prompt_text = "Please decipher this Excel spreadsheet and visualize the Q1 revenue trends and performance with charts."
        print(f"Entering prompt: {prompt_text}")
        page.fill("#chatTextarea", prompt_text)
        time.sleep(0.5)

        # Submit
        print("Submitting chat query...")
        page.click("#sendBtn")

        # Wait for assistant response to stream and complete
        print("Waiting for response streaming to finish...")
        page.wait_for_selector(".message-row.assistant", timeout=15000)
        page.wait_for_selector(".message-row.assistant .markdown-body h3", timeout=20000)
        time.sleep(3)  # Allow Mermaid and Markdown rendering to finalize

        # Scroll down
        page.evaluate("() => document.getElementById('chatViewport').scrollTo(0, document.getElementById('chatViewport').scrollHeight)")
        time.sleep(1)

        # Screenshot 2: Full UI with Completed Result
        result_shot = os.path.join(ARTIFACT_DIR, "browser_chart_result.png")
        page.screenshot(path=result_shot, full_page=False)
        print(f"Captured final result screenshot: {result_shot}")

        # Screenshot 3: Close-up of the Assistant Response Card
        card_shot = os.path.join(ARTIFACT_DIR, "browser_chart_card.png")
        last_msg = page.locator(".message-row.assistant").last
        last_msg.screenshot(path=card_shot)
        print(f"Captured assistant card screenshot: {card_shot}")

        # Check content in the page
        content = page.content()
        has_exec_summary = "Executive Summary" in content or "Variance" in content or "Workbook" in content
        has_mermaid = "mermaid" in content.lower() or "<svg" in content
        print(f"Content checks: Executive Summary detected={has_exec_summary}, Mermaid/SVG detected={has_mermaid}")

        browser.close()
        return has_exec_summary


if __name__ == "__main__":
    success = run_browser_test()
    print("Browser Test Finished. Success:", success)
