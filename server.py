"""Root entrypoint to run server from repository root directory."""
import importlib.util
import os
import sys

_sub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LangGraph-beta-v2-pre-release")
if _sub_dir not in sys.path:
    sys.path.insert(0, _sub_dir)

_inner_server_path = os.path.join(_sub_dir, "server.py")
if os.path.isfile(_inner_server_path):
    _spec = importlib.util.spec_from_file_location("server", _inner_server_path)
    if _spec and _spec.loader:
        _inner_server = importlib.util.module_from_spec(_spec)
        sys.modules["server"] = _inner_server
        _spec.loader.exec_module(_inner_server)
        globals().update({k: v for k, v in _inner_server.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True, app_dir=_sub_dir)
