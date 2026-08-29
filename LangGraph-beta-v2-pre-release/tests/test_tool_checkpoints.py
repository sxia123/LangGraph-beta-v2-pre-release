import pytest

from src.core.memory_store import clear_memories, get_tool_checkpoints
from src.core.tool_loader import ToolLoader, checkpoint_tool


@pytest.fixture(autouse=True)
def clean_memory_db():
    clear_memories()
    yield
    clear_memories()


def test_tool_loader_pre_and_post_checkpoints():
    loader = ToolLoader()
    loader.register("add_numbers", lambda a, b: a + b, description="Add two numbers")

    run_id = "run_test_addition_001"
    res = loader.run("add_numbers", run_id=run_id, a=5, b=10)

    assert res["ok"] is True
    assert res["checkpoint_id"] is not None
    assert res["post_checkpoint_id"] is not None
    assert res["tool"] == "add_numbers"

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="add_numbers")
    assert len(checkpoints) == 2

    # Pre-tool checkpoint
    pre_cp = checkpoints[0]
    assert pre_cp["event"] == "checkpoint_before_add_numbers"
    assert pre_cp["input"] == {"a": 5, "b": 10}
    assert pre_cp["metadata"]["checkpoint_type"] == "pre_tool_execution"
    assert pre_cp["id"] == res["checkpoint_id"]

    # Post-tool checkpoint
    post_cp = checkpoints[1]
    assert post_cp["event"] == "checkpoint_after_add_numbers"
    assert post_cp["metadata"]["checkpoint_type"] == "post_tool_execution"
    assert post_cp["metadata"]["ok"] is True
    assert post_cp["metadata"]["pre_checkpoint_id"] == res["checkpoint_id"]
    assert "duration_s" in post_cp["metadata"]
    assert post_cp["id"] == res["post_checkpoint_id"]


def test_tool_loader_error_checkpoint():
    loader = ToolLoader()

    def faulty_tool(x: int):
        raise ValueError("Simulated tool computation failure")

    loader.register("faulty_tool", faulty_tool, description="A tool that fails")

    run_id = "run_test_error_002"
    res = loader.run("faulty_tool", run_id=run_id, x=42)

    assert res["ok"] is False
    assert res["checkpoint_id"] is not None
    assert res["post_checkpoint_id"] is not None
    assert "Simulated tool computation failure" in res["message"]

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="faulty_tool")
    assert len(checkpoints) == 2

    pre_cp = checkpoints[0]
    assert pre_cp["event"] == "checkpoint_before_faulty_tool"
    assert pre_cp["metadata"]["checkpoint_type"] == "pre_tool_execution"

    err_cp = checkpoints[1]
    assert err_cp["event"] == "checkpoint_error_faulty_tool"
    assert err_cp["metadata"]["checkpoint_type"] == "tool_error"
    assert err_cp["metadata"]["ok"] is False
    assert err_cp["metadata"]["pre_checkpoint_id"] == res["checkpoint_id"]
    assert "Simulated tool computation failure" in err_cp["metadata"]["error"]


def test_tool_loader_type_error_checkpoint():
    loader = ToolLoader()
    loader.register("strict_args", lambda required_param: f"Got {required_param}")

    run_id = "run_test_type_err_003"
    # Call without required_param
    res = loader.run("strict_args", run_id=run_id)

    assert res["ok"] is False
    assert res["checkpoint_id"] is not None
    assert res["post_checkpoint_id"] is not None

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="strict_args")
    assert len(checkpoints) == 2
    assert checkpoints[1]["event"] == "checkpoint_error_strict_args"
    assert checkpoints[1]["metadata"]["checkpoint_type"] == "tool_error"


def test_checkpoint_tool_decorator():
    run_id = "run_test_decorator_004"

    @checkpoint_tool(name="custom_multiply", run_id=run_id)
    def multiply(x: int, y: int) -> int:
        return x * y

    result = multiply(4, 7)
    assert result == 28

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="custom_multiply")
    assert len(checkpoints) == 2
    assert checkpoints[0]["event"] == "checkpoint_before_custom_multiply"
    assert checkpoints[1]["event"] == "checkpoint_after_custom_multiply"
    assert checkpoints[1]["metadata"]["ok"] is True


def test_checkpoint_tool_decorator_exception():
    run_id = "run_test_decorator_err_005"

    @checkpoint_tool(name="failing_dec_tool", run_id=run_id)
    def bad_func():
        raise RuntimeError("Decorated function crashed")

    with pytest.raises(RuntimeError, match="Decorated function crashed"):
        bad_func()

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="failing_dec_tool")
    assert len(checkpoints) == 2
    assert checkpoints[0]["event"] == "checkpoint_before_failing_dec_tool"
    assert checkpoints[1]["event"] == "checkpoint_error_failing_dec_tool"
    assert checkpoints[1]["metadata"]["ok"] is False


def test_builtin_math_eval_checkpointing():
    loader = ToolLoader()
    run_id = "run_test_math_006"

    res = loader.run("math_eval", run_id=run_id, expression="10 + 20 * 3")
    assert res["ok"] is True
    assert res["message"] == "70"
    assert res["checkpoint_id"] is not None
    assert res["post_checkpoint_id"] is not None

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="math_eval")
    assert len(checkpoints) == 2
    assert checkpoints[0]["event"] == "checkpoint_before_math_eval"
    assert checkpoints[1]["event"] == "checkpoint_after_math_eval"


def test_tool_checkpoint_disabled():
    loader = ToolLoader()
    loader.register("simple_echo", lambda text: text)

    res = loader.run("simple_echo", text="hello", create_checkpoint=False)
    assert res["ok"] is True
    assert res["checkpoint_id"] is None
    assert res["post_checkpoint_id"] is None

    checkpoints = get_tool_checkpoints(tool_name="simple_echo")
    assert len(checkpoints) == 0


def test_get_tool_checkpoints_filtering():
    loader = ToolLoader()
    loader.register("tool_alpha", lambda: "alpha")
    loader.register("tool_beta", lambda: "beta")

    loader.run("tool_alpha", run_id="run_A")
    loader.run("tool_beta", run_id="run_B")

    cps_a = get_tool_checkpoints(run_id="run_A")
    assert len(cps_a) == 2
    assert all("tool_alpha" in cp["event"] for cp in cps_a)

    cps_b = get_tool_checkpoints(run_id="run_B")
    assert len(cps_b) == 2
    assert all("tool_beta" in cp["event"] for cp in cps_b)

    pre_cps = get_tool_checkpoints(checkpoint_type="pre_tool_execution")
    assert len(pre_cps) >= 2
    assert all(cp["metadata"]["checkpoint_type"] == "pre_tool_execution" for cp in pre_cps)


def test_chart_pipeline_execute_node_checkpoints():
    from src.agents.chart_pipeline import create_chart_pipeline_graph
    from src.core.local_llm import LocalLLMClient, LocalLLMConfig

    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_chart_pipeline_graph(client)

    run_id = "run_chart_tool_test_007"
    initial_input = {
        "run_id": run_id,
        "user_input": "Calculate 15 + 25",
        "current_step": "execute",
        "intake_status": "APPROVED",
        "action_payload": {
            "target_action": "evaluate_math",
            "tool": "math_eval",
            "tool_args": {"expression": "15 + 25"},
        },
        "messages": [],
        "agent_thoughts": [],
    }

    res = graph.invoke(initial_input)
    assert "40" in res.get("execution_result", "")

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="math_eval")
    assert len(checkpoints) == 2
    assert checkpoints[0]["event"] == "checkpoint_before_math_eval"
    assert checkpoints[1]["event"] == "checkpoint_after_math_eval"
    assert checkpoints[0]["metadata"]["tool"] == "math_eval"
    assert checkpoints[1]["metadata"]["ok"] is True


def test_supervisor_researcher_node_tool_checkpoints():
    from src.agents.multi_agent_supervisor import create_multi_agent_supervisor_graph
    from src.core.local_llm import LocalLLMClient, LocalLLMConfig

    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_multi_agent_supervisor_graph(client)

    run_id = "run_supervisor_tool_test_008"
    initial_input = {
        "run_id": run_id,
        "current_task": "search for quantum computing breakthroughs",
        "messages": [{"role": "user", "content": "search for quantum computing breakthroughs"}],
        "agent_thoughts": [],
    }

    graph.invoke(initial_input)

    checkpoints = get_tool_checkpoints(run_id=run_id, tool_name="web_search")
    assert len(checkpoints) == 2
    assert checkpoints[0]["event"] == "checkpoint_before_web_search"
    assert checkpoints[1]["event"] == "checkpoint_after_web_search"
    assert checkpoints[0]["metadata"]["tool"] == "web_search"



