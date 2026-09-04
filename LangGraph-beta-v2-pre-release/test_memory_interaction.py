"""Tests whether the pipeline and direct chat interact with memory."""

from server import create_direct_chat_graph
from src.agents.chart_pipeline import create_chart_pipeline_graph
from src.core.local_llm import LocalLLMClient, LocalLLMConfig
from src.core.memory_store import search_memories


def test_direct_chat_memory_recall():
    prompt = "What was the revenue and variance from our earlier test_global_sales_q1_2026.xlsx spreadsheet?"
    mem_matches = search_memories(prompt, limit=4)
    print(f"Direct Chat Search found {len(mem_matches)} memories for prompt.")

    recalled_facts = []
    for mem in mem_matches:
        r_val = mem.get("result")
        if r_val:
            recalled_facts.append(f"- [{mem.get('event', 'memory')}]: {str(r_val)[:120]}")
    context_header = "\n[Recalled Memory Context:\n" + "\n".join(recalled_facts) + "\n]" if recalled_facts else ""
    effective_prompt = f"{prompt}{context_header}"

    client = LocalLLMClient(LocalLLMConfig(provider="mock"))
    graph = create_direct_chat_graph(client)
    state = {
        "messages": [{"role": "user", "content": effective_prompt}],
        "run_id": "test_direct_run",
    }
    config = {"configurable": {"thread_id": "test_direct_thread"}}
    for chunk in graph.stream(state, config=config):
        for node, upd in chunk.items():
            if "messages" in upd:
                print("\nDirect Chat Final Response:")
                print(upd["messages"][-1]["content"])


def test_chart_pipeline_memory_recall():
    prompt = "Synthesize performance based on earlier test_global_sales_q1_2026.xlsx"
    mem_matches = search_memories(prompt, limit=4)
    print(f"\nChart Pipeline Search found {len(mem_matches)} memories.")

    recalled_facts = []
    for mem in mem_matches:
        r_val = mem.get("result")
        if r_val:
            recalled_facts.append(f"- [{mem.get('event', 'memory')}]: {str(r_val)[:120]}")
    context_header = "\n[Recalled Memory Context:\n" + "\n".join(recalled_facts) + "\n]" if recalled_facts else ""
    effective_prompt = f"{prompt}{context_header}"

    client = LocalLLMClient(LocalLLMConfig(provider="mock"))
    graph = create_chart_pipeline_graph(client)
    state = {
        "user_input": effective_prompt,
        "current_step": "intake",
    }
    nodes = []
    config = {"configurable": {"thread_id": "test_chart_thread"}}
    for chunk in graph.stream(state, config=config):
        for node, upd in chunk.items():
            nodes.append(node)
            if "final_response" in upd:
                print("\nChart Pipeline Final Response snippet:")
                print(upd["final_response"][:200])
    print("\nChart Pipeline Executed Nodes:", " -> ".join(nodes))


if __name__ == "__main__":
    test_direct_chat_memory_recall()
    test_chart_pipeline_memory_recall()
