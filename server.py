"""Root workspace entry point for LangGraph Multi-Agent Studio Server.

Automatically delegates to the server module inside the nested project folder.
"""
import importlib.util
import os
import sys

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LangGraph-beta-v2-pre-release")
if os.path.exists(PROJECT_DIR):
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)

_inner_server_path = os.path.join(PROJECT_DIR, "server.py")
if os.path.isfile(_inner_server_path):
    _spec = importlib.util.spec_from_file_location("server", _inner_server_path)
    if _spec and _spec.loader:
        _inner_server = importlib.util.module_from_spec(_spec)
        sys.modules["server"] = _inner_server
        _spec.loader.exec_module(_inner_server)
        globals().update({k: v for k, v in _inner_server.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True, app_dir=PROJECT_DIR)

