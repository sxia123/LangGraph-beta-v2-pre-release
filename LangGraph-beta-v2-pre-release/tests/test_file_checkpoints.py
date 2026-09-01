import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app
from src.core.checkpointer import (
    checkpoint_after_file_change,
    checkpoint_before_file_change,
    compute_file_sha256,
    get_default_checkpointer,
    get_file_checkpoints,
    rollback_file_checkpoint,
)
from src.core.local_llm import LocalLLMClient, LocalLLMConfig
from src.core.memory_store import clear_memories
from src.core.tool_loader import ToolLoader


@pytest.fixture(autouse=True)
def clean_memory_db():
    clear_memories()
    yield
    clear_memories()



def test_langgraph_default_checkpointer_integration():
    """Verify LangGraph StateGraph compiles and stores state checkpoints."""
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class SimpleState(TypedDict):
        count: int

    def step_node(state: SimpleState) -> Dict[str, Any]:
        return {"count": state.get("count", 0) + 1}

    builder = StateGraph(SimpleState)
    builder.add_node("step", step_node)
    builder.set_entry_point("step")
    builder.set_finish_point("step")

    checkpointer = get_default_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = "thread_test_chk_001"
    config = {"configurable": {"thread_id": thread_id}}

    res = graph.invoke({"count": 10}, config=config)
    assert res["count"] == 11

    # Verify state snapshot was captured by LangGraph checkpointer
    state = graph.get_state(config)
    assert state is not None
    assert state.values["count"] == 11


def test_checkpoint_before_and_after_file_change():
    """Verify file snapshot creation before and after file changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "sample.txt")
        original_content = "Hello, Original Content 12345!"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        run_id = "run_file_test_001"
        pre_cp = checkpoint_before_file_change(
            file_path=test_file,
            run_id=run_id,
            metadata={"reason": "test_pre_change"},
        )

        assert pre_cp["ok"] is True
        assert pre_cp["exists_before"] is True
        assert pre_cp["size_before"] == len(original_content)
        assert pre_cp["sha256_before"] == compute_file_sha256(test_file)
        checkpoint_id = pre_cp["checkpoint_id"]

        # Modify file
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Modified content!")

        post_cp = checkpoint_after_file_change(
            file_path=test_file,
            pre_checkpoint_id=checkpoint_id,
            run_id=run_id,
        )
        assert post_cp["ok"] is True
        assert post_cp["exists_after"] is True

        # Query checkpoints
        cps = get_file_checkpoints(file_path=test_file, run_id=run_id)
        assert len(cps) >= 2


def test_rollback_file_checkpoint_restore_content():
    """Verify rolling back a modified file to its previous pre-change snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "rollback_test.txt")
        initial_text = "Important initial configuration."

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(initial_text)

        pre_cp = checkpoint_before_file_change(
            file_path=test_file,
            run_id="run_rollback_001",
        )
        checkpoint_id = pre_cp["checkpoint_id"]

        # Overwrite file with destructive content
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Corrupted or broken content!")

        # Execute rollback
        rb_res = rollback_file_checkpoint(checkpoint_id)
        assert rb_res["ok"] is True
        assert rb_res["action"] == "restored_content"

        # Verify content restored
        with open(test_file, "r", encoding="utf-8") as f:
            restored = f.read()
        assert restored == initial_text


def test_rollback_file_checkpoint_remove_new_file():
    """Verify rolling back a newly created file removes it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_file = os.path.join(tmpdir, "brand_new_file.txt")
        assert not os.path.exists(new_file)

        pre_cp = checkpoint_before_file_change(
            file_path=new_file,
            run_id="run_rollback_new_001",
        )
        assert pre_cp["exists_before"] is False

        # Create file
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("Created new content")
        assert os.path.exists(new_file)

        # Rollback
        rb_res = rollback_file_checkpoint(pre_cp["checkpoint_id"])
        assert rb_res["ok"] is True
        assert rb_res["action"] == "removed_new_file"
        assert not os.path.exists(new_file)


def test_tool_loader_file_write_and_edit_checkpoints():
    """Verify ToolLoader file_write and file_edit create pre-change checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "tool_managed_file.txt")
        tools = ToolLoader()
        run_id = "run_tool_loader_file_001"

        # 1. file_write
        res = tools.run(
            "file_write",
            run_id=run_id,
            file_path=test_file,
            content="Version 1.0 content\nLine 2\n",
        )
        assert res["ok"] is True
        assert os.path.exists(test_file)

        # 2. file_edit
        edit_res = tools.run(
            "file_edit",
            run_id=run_id,
            file_path=test_file,
            target="Version 1.0",
            replacement="Version 2.0",
        )
        assert edit_res["ok"] is True
        with open(test_file, "r", encoding="utf-8") as f:
            assert "Version 2.0 content" in f.read()

        # Check that checkpoints were recorded in memory
        cps = get_file_checkpoints(file_path=test_file, run_id=run_id)
        assert len(cps) >= 2


def test_tool_loader_python_repl_auto_file_checkpoint():
    """Verify python_repl automatically creates pre-change file checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = os.path.join(tmpdir, "repl_target.txt").replace("\\", "/")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("Original REPL content")

        tools = ToolLoader()
        run_id = "run_repl_auto_cp_001"

        code = f"""
with open('{target_file}', 'w') as f:
    f.write('Overwritten by Python REPL')
"""
        res = tools.run("python_repl", run_id=run_id, code=code)
        assert res["ok"] is True

        cps = get_file_checkpoints(file_path=target_file, run_id=run_id)
        assert len(cps) >= 1
        assert any(cp.get("metadata", {}).get("inferred_from_code") for cp in cps)


def test_api_file_checkpoints_and_rollback_endpoints():
    """Verify REST API endpoints for querying and rolling back file checkpoints."""
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = os.path.join(tmpdir, "api_test.json").replace("\\", "/")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write('{"initial": true}')

        run_id = "run_api_chk_009"
        pre_cp = checkpoint_before_file_change(file_path=target_path, run_id=run_id)
        checkpoint_id = pre_cp["checkpoint_id"]

        # Modify file
        with open(target_path, "w", encoding="utf-8") as f:
            f.write('{"modified": true}')

        # 1. GET /api/checkpoints/files
        get_res = client.get(f"/api/checkpoints/files?run_id={run_id}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["ok"] is True
        assert data["count"] >= 1
        assert any(c["id"] == checkpoint_id for c in data["checkpoints"])

        # 2. POST /api/checkpoints/rollback
        rb_res = client.post("/api/checkpoints/rollback", json={"checkpoint_id": checkpoint_id})
        assert rb_res.status_code == 200
        rb_data = rb_res.json()
        assert rb_data["ok"] is True
        assert rb_data["action"] == "restored_content"

        with open(target_path, "r", encoding="utf-8") as f:
            restored = f.read()
        assert '{"initial": true}' in restored


def test_chart_pipeline_compiled_with_checkpointer():
    """Verify chart_pipeline executes with LangGraph checkpointer."""
    from src.agents.chart_pipeline import create_chart_pipeline_graph

    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_chart_pipeline_graph(client)

    run_id = "run_chart_chk_verify_001"
    config = {"configurable": {"thread_id": run_id}}
    initial_input = {
        "run_id": run_id,
        "user_input": "Test task with checkpointer",
        "current_step": "intake",
        "messages": [],
        "agent_thoughts": [],
    }

    res = graph.invoke(initial_input, config=config)
    assert res is not None
