"""Playwright browser test for testing multi-format media intake (documents, slideshows, photos)."""

import os
import time

from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\sidne\.gemini\antigravity\brain\3b00f35f-1695-4530-a6db-419a27a55d2f"


def run_multimedia_browser_test():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("Navigating to http://127.0.0.1:8080/ ...")
        page.goto("http://127.0.0.1:8080/", wait_until="networkidle")
        page.wait_for_selector("#chatTextarea", timeout=10000)
        time.sleep(1)

        # Select 'Direct Chat (Vision)' pipeline
        print("Selecting Direct Chat (Vision) pipeline...")
        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="direct"]', timeout=5000)
        page.click('button[data-pipeline="direct"]')
        time.sleep(0.5)

        # Screenshot: Pipeline selected
        vision_pipeline_shot = os.path.join(ARTIFACT_DIR, "browser_vision_pipeline.png")
        page.screenshot(path=vision_pipeline_shot, full_page=False)
        print(f"Captured vision pipeline screenshot: {vision_pipeline_shot}")

        # Send test query to Direct Chat
        prompt_text = "Examine this multi-tier architecture diagram and summarize key visual components."
        page.fill("#chatTextarea", prompt_text)
        time.sleep(0.5)
        page.click("#sendBtn")

        page.wait_for_selector(".message-row.assistant", timeout=15000)
        time.sleep(2)

        direct_result_shot = os.path.join(ARTIFACT_DIR, "browser_direct_vision_result.png")
        page.screenshot(path=direct_result_shot, full_page=False)
        print(f"Captured direct vision result screenshot: {direct_result_shot}")

        # Now test Chart Pipeline with document
        page.click("#pipelineSelectBtn")
        page.wait_for_selector('button[data-pipeline="chart"]', timeout=5000)
        page.click('button[data-pipeline="chart"]')
        time.sleep(0.5)

        chart_prompt = "Analyze the policy framework document and construct an architectural Mermaid diagram."
        page.fill("#chatTextarea", chart_prompt)
        time.sleep(0.5)
        page.click("#sendBtn")

        page.wait_for_selector(".message-row.assistant:nth-of-type(2)", timeout=15000)
        time.sleep(3)

        chart_doc_shot = os.path.join(ARTIFACT_DIR, "browser_chart_doc_result.png")
        page.screenshot(path=chart_doc_shot, full_page=False)
        print(f"Captured chart document result screenshot: {chart_doc_shot}")

        browser.close()


if __name__ == "__main__":
    run_multimedia_browser_test()

