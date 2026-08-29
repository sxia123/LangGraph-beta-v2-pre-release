from src.agents import (
    create_chart_pipeline_graph,
    create_claims_triage_graph,
    create_code_review_team_graph,
    create_master_pipeline_graph,
    create_multi_agent_supervisor_graph,
    create_solution_review_team_graph,
)
from src.core.local_llm import LocalLLMClient, LocalLLMConfig


def test_local_llm_reasoning_extraction():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))

    # Test <think> parsing
    res = client._parse_response(
        "<think>Step 1: analyze task\nStep 2: compute result</think>Final Output"
    )
    assert res.thought == "Step 1: analyze task\nStep 2: compute result"
    assert res.content == "Final Output"

    # Test reasoning parameter
    res2 = client._parse_response("Direct response", reasoning="Thinking deeply about edge cases")
    assert res2.thought == "Thinking deeply about edge cases"
    assert res2.content == "Direct response"

    # Test combined tags and reasoning
    res3 = client._parse_response(
        "<reasoning>Internal thoughts</reasoning>Done", reasoning="Initial reasoning"
    )
    assert res3.thought is not None
    assert "Initial reasoning" in res3.thought
    assert "Internal thoughts" in res3.thought
    assert res3.content == "Done"


def test_code_review_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_code_review_team_graph(client)
    res = graph.invoke({
        "task": "Write a binary search function in Python",
        "code": "",
        "review": "",
        "approved": False,
        "revision_count": 0,
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1
    assert any("Developer" in t.get("agent", "") for t in thoughts)


def test_solution_review_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_solution_review_team_graph(client)
    res = graph.invoke({
        "task": "Draft proposal review",
        "solution": "",
        "review": "",
        "approved": False,
        "revision_count": 0,
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1
    assert any("Solution" in t.get("agent", "") for t in thoughts)


def test_supervisor_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_multi_agent_supervisor_graph(client)
    res = graph.invoke({
        "current_task": "Write a python script to parse CSV data",
        "next_agent": "supervisor",
        "research_output": "",
        "coder_output": "",
        "critic_feedback": "",
        "final_response": "",
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1
    assert any("Supervisor" in t.get("agent", "") for t in thoughts)


def test_claims_triage_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_claims_triage_graph(client)
    res = graph.invoke({
        "claim_input": "Water leak caused damaged equipment",
        "current_step": "step_1_classification",
        "classification_details": None,
        "severity_assessment": None,
        "action_plan": None,
        "final_response": "",
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1


def test_chart_pipeline_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_chart_pipeline_graph(client)
    res = graph.invoke({
        "user_input": "Analyze system performance chart",
        "current_step": "intake",
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1


def test_master_pipeline_thoughts_emission():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    graph = create_master_pipeline_graph(client)
    res = graph.invoke({
        "user_input": "Review enterprise backup plan",
        "current_step": "pipeline_start",
        "triage_details": None,
        "supervisor_details": None,
        "review_details": None,
        "final_response": "",
        "agent_thoughts": [],
        "messages": [],
    })
    thoughts = res.get("agent_thoughts", [])
    assert len(thoughts) >= 1

