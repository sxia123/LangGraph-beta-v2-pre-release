import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app, create_direct_chat_graph
from src.core.local_llm import LocalLLMClient, LocalLLMConfig
from src.core.memory_store import clear_memories, get_tool_checkpoints
from src.core.tool_loader import get_tool_loader


@pytest.fixture(autouse=True)
def clean_memory_db():
    clear_memories()
    yield
    clear_memories()


def test_api_tools_list_endpoint():
    client = TestClient(app)
    response = client.get("/api/tools")
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    assert data.get("count") >= 8
    tool_names = data.get("tool_names", [])
    assert "web_search" in tool_names
    assert "python_repl" in tool_names
    assert "math_eval" in tool_names
    assert "wikipedia" in tool_names
    assert "arxiv" in tool_names


def test_api_tools_run_math_eval():
    client = TestClient(app)
    run_id = "test_run_math_eval_101"
    response = client.post(
        "/api/tools/run",
        json={
            "tool": "math_eval",
            "args": {"expression": "25 * 4 + 10"},
            "run_id": run_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    assert "110" in str(data.get("message"))
    assert data.get("checkpoint_id") is not None
    assert data.get("post_checkpoint_id") is not None

    # Verify SQLite checkpoint recorded
    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="math_eval")
    assert len(checkpoints) == 2
    assert checkpoints[0]["metadata"]["checkpoint_type"] == "pre_tool_execution"
    assert checkpoints[1]["metadata"]["checkpoint_type"] == "post_tool_execution"


def test_api_tools_run_python_repl():
    client = TestClient(app)
    run_id = "test_run_python_repl_102"
    response = client.post(
        "/api/tools/run",
        json={
            "tool": "python_repl",
            "args": {"code": "print('hello from web ui tool')"},
            "run_id": run_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    assert "hello from web ui tool" in str(data.get("message"))


def test_api_tools_checkpoints_endpoint():
    client = TestClient(app)
    run_id = "test_run_cp_lookup_103"
    # Execute a tool to populate checkpoints
    client.post(
        "/api/tools/run",
        json={
            "tool": "math_eval",
            "args": {"expression": "100 / 5"},
            "run_id": run_id,
        },
    )

    response = client.get(f"/api/tools/checkpoints?run_id={run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    assert data.get("count") >= 2
    cps = data.get("checkpoints", [])
    assert any(cp.get("event") == "checkpoint_before_math_eval" for cp in cps)
    assert any(cp.get("event") == "checkpoint_after_math_eval" for cp in cps)


def test_direct_chat_tool_execution():
    llm = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    loader = get_tool_loader(llm)
    assert loader is not None

    graph = create_direct_chat_graph(llm)
    run_id = "test_run_direct_tool_104"
    state_input = {
        "run_id": run_id,
        "messages": [
            {
                "id": "msg_1",
                "role": "user",
                "content": "Calculate math_eval expression: 20 * 5",
            }
        ],
        "user_input": "Calculate math_eval expression: 20 * 5",
        "final_response": "",
        "agent_thoughts": [],
    }

    result = graph.invoke(state_input)
    assert result.get("final_response")
    thoughts = result.get("agent_thoughts", [])
    assert len(thoughts) >= 1
    # Check that tool execution was recorded in thoughts
    assert any("Tool: math_eval" in t.get("agent", "") for t in thoughts)

    # Check checkpoints in database
    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="math_eval")
    assert len(checkpoints) >= 2


def test_api_models_endpoint():
    client = TestClient(app)
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    models = data.get("models", [])
    assert len(models) >= 1
    assert "Qwen3.8-27B-oQ6-mtp" in models or any("qwen" in m.lower() for m in models)


def test_api_models_select_endpoint():
    client = TestClient(app)
    target_model = "Qwen2.5-Coder-32B-Instruct"
    response = client.post("/api/models/select", json={"model_name": target_model})
    assert response.status_code == 200
    data = response.json()
    assert data.get("ok") is True
    assert data.get("model_name") == target_model

    # Check status endpoint reflects it
    status_res = client.get("/api/status")
    assert status_res.status_code == 200
    assert status_res.json().get("model_name") == target_model

