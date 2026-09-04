"""Playwright automated browser test testing fabricated XLSX and PPTX files against Chart Pipeline and Direct Chat."""

import os
import time

from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\sidne\.gemini\antigravity\brain\3b00f35f-1695-4530-a6db-419a27a55d2f"
XLSX_FILE = os.path.abspath("test_global_sales_q1_2026.xlsx")
PPTX_FILE = os.path.abspath("test_ai_product_roadmap_2026.pptx")


def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("Navigating to http://127.0.0.1:8080/ ...")
        page.goto("http://127.0.0.1:8080/", wait_until="networkidle")
        page.wait_for_selector("#chatTextarea", timeout=10000)
        time.sleep(1)

        # -----------------------------------------------------------------
        # TEST 1: Chart Architecture with Fabricated Excel Spreadsheet
        # -----------------------------------------------------------------
        print("\n--- TEST 1: Chart Architecture with Excel ---")
        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="chart"]', timeout=5000)
        page.click('button[data-pipeline="chart"]')
        time.sleep(0.5)

        print(f"Staging Excel file: {XLSX_FILE}")
        page.set_input_files("#fileInput", XLSX_FILE)
        page.wait_for_selector(".file-stage-chip", timeout=5000)
        time.sleep(0.5)

        prompt1 = "Decipher test_global_sales_q1_2026.xlsx and visualize the Q1 revenue trends and variance with charts."
        page.fill("#chatTextarea", prompt1)
        time.sleep(0.5)
        page.click("#sendBtn")

        print("Waiting for Chart Pipeline response to complete...")
        page.wait_for_selector(".message-row.assistant", timeout=15000)
        page.wait_for_function("() => !document.getElementById('sendBtn').disabled", timeout=25000)
        time.sleep(3)

        # Scroll to view chart
        page.evaluate("() => document.getElementById('chatViewport').scrollTo(0, document.getElementById('chatViewport').scrollHeight)")
        time.sleep(1)

        xlsx_shot = os.path.join(ARTIFACT_DIR, "browser_chart_xlsx_result.png")
        page.screenshot(path=xlsx_shot, full_page=False)
        print(f"Captured Excel chart result: {xlsx_shot}")

        # -----------------------------------------------------------------
        # TEST 2: Chart Architecture with Fabricated PowerPoint Slideshow
        # -----------------------------------------------------------------
        print("\n--- TEST 2: Chart Architecture with PowerPoint Slideshow ---")
        print(f"Staging PowerPoint file: {PPTX_FILE}")
        page.set_input_files("#fileInput", PPTX_FILE)
        page.wait_for_selector(".file-stage-chip", timeout=5000)
        time.sleep(0.5)

        prompt2 = "Review test_ai_product_roadmap_2026.pptx and visualize the strategic roadmap timeline."
        page.fill("#chatTextarea", prompt2)
        time.sleep(0.5)
        page.click("#sendBtn")

        print("Waiting for PowerPoint slideshow response to complete...")
        page.wait_for_function("() => document.querySelectorAll('.message-row.assistant').length >= 2", timeout=15000)
        page.wait_for_function("() => !document.getElementById('sendBtn').disabled", timeout=25000)
        time.sleep(3)

        page.evaluate("() => document.getElementById('chatViewport').scrollTo(0, document.getElementById('chatViewport').scrollHeight)")
        time.sleep(1)

        pptx_shot = os.path.join(ARTIFACT_DIR, "browser_chart_pptx_result.png")
        page.screenshot(path=pptx_shot, full_page=False)
        print(f"Captured PowerPoint chart result: {pptx_shot}")

        # -----------------------------------------------------------------
        # TEST 3: Direct Chat Pipeline
        # -----------------------------------------------------------------
        print("\n--- TEST 3: Direct Chat Pipeline ---")
        # Click new chat
        new_chat_btn = page.query_selector("#newChatBtn")
        if new_chat_btn:
            new_chat_btn.click()
            time.sleep(0.5)

        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="direct"]', timeout=5000)
        page.click('button[data-pipeline="direct"]')
        time.sleep(0.5)

        prompt3 = "Analyze the operational architecture and key components of our multi-agent platform."
        page.fill("#chatTextarea", prompt3)
        time.sleep(0.5)
        page.click("#sendBtn")

        print("Waiting for Direct Chat response to complete...")
        page.wait_for_selector(".message-row.assistant", timeout=15000)
        page.wait_for_function("() => !document.getElementById('sendBtn').disabled", timeout=20000)
        time.sleep(2)

        page.evaluate("() => document.getElementById('chatViewport').scrollTo(0, document.getElementById('chatViewport').scrollHeight)")
        time.sleep(1)

        direct_shot = os.path.join(ARTIFACT_DIR, "browser_direct_chat_test.png")
        page.screenshot(path=direct_shot, full_page=False)
        print(f"Captured Direct Chat result: {direct_shot}")

        browser.close()
        print("\nAll browser UI tests finished successfully!")


if __name__ == "__main__":
    run_tests()

