import os
import pytest
from src.agents.chart_pipeline import create_chart_pipeline_graph
from src.core.local_llm import LocalLLMClient, LocalLLMConfig


@pytest.fixture
def mock_chart_graph():
    os.environ["LLM_PROVIDER"] = "mock"
    client = LocalLLMClient(LocalLLMConfig(provider="mock"))
    return create_chart_pipeline_graph(client)


def test_routing_with_foreign_file_on_disk(mock_chart_graph):
    """If input references an existing foreign spreadsheet file, route to the spreadsheet pipeline."""
    nodes = []
    final_output = ""
    for chunk in mock_chart_graph.stream(
        {
            "user_input": "Please decipher mock_enterprise_financial_q1_2026.xlsx and show charts",
            "current_step": "intake",
        }
    ):
        for node, upd in chunk.items():
            nodes.append(node)
            if "final_response" in upd:
                final_output = upd["final_response"]

    assert "foreign_file_router" in nodes
    assert "spreadsheet_specialist" in nodes
    assert "spreadsheet_verify" in nodes
    # Must NOT run the original specialist web search pipeline
    assert "tier0_checks" not in nodes
    assert "tier05_web_verify" not in nodes
    # Check that output contains deciphered tables and Mermaid chart
    assert "Executive Summary" in final_output
    assert "```mermaid" in final_output


def test_routing_with_foreign_file_attachment_in_state(mock_chart_graph):
    """If input includes a foreign file in state['files'], route to spreadsheet pipeline."""
    nodes = []
    for chunk in mock_chart_graph.stream(
        {
            "user_input": "Analyze company sales data",
            "files": [{"filename": "company_sales_q1_2026.xlsx", "content": "company_sales_q1_2026.xlsx"}],
            "current_step": "intake",
        }
    ):
        for node in chunk.keys():
            nodes.append(node)

    assert "foreign_file_router" in nodes
    assert "spreadsheet_specialist" in nodes
    assert "spreadsheet_verify" in nodes
    assert "tier0_checks" not in nodes


def test_routing_without_foreign_file_keeps_original_pipeline(mock_chart_graph):
    """If input does NOT include any foreign file, keep with the original pipeline."""
    nodes = []
    for chunk in mock_chart_graph.stream(
        {
            "user_input": "Explain how transformer attention heads work in modern LLMs",
            "current_step": "intake",
        }
    ):
        for node in chunk.keys():
            nodes.append(node)

    assert "foreign_file_router" in nodes
    # Must follow original pipeline
    assert "specialist" in nodes
    assert "tier0_checks" in nodes
    assert "tier05_web_verify" in nodes
    assert "tier1_verify" in nodes
    assert "revisions" in nodes
    assert "tier2_verify" in nodes
    # Must NOT route to spreadsheet pipeline
    assert "spreadsheet_specialist" not in nodes
    assert "spreadsheet_verify" not in nodes

