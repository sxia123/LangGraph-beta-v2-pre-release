"""
Root workspace entry point for LangGraph Multi-Agent Studio Server.
Automatically delegates to the server module inside the nested project folder.
"""
import os
import sys

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LangGraph-beta-v2-pre-release")
if os.path.exists(PROJECT_DIR):
    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)

from server import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True, app_dir=PROJECT_DIR)
