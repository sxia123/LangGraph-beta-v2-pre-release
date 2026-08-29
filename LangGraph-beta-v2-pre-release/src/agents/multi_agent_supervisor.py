import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient
from src.core.soul_loader import load_soul
from src.core.tool_loader import get_tool_loader


class MultiAgentState(TypedDict, total=False):
    run_id: Optional[str]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_task: str
    next_agent: str
    research_output: str
    coder_output: str
    critic_feedback: str
    final_response: str
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]



def _get_task(state: MultiAgentState) -> str:
    task = state.get("current_task", "")
    if not task:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                task = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                task = getattr(last_msg, "content", "")
    return task


def create_multi_agent_supervisor_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(MultiAgentState)

    # 1. SUPERVISOR ROUTER NODE
    def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
        task = _get_task(state)
        soul_prompt = load_soul("supervisor")
        prompt = f"""{soul_prompt}

Task: "{task}"
State Summary:
- Research Done: {"Yes" if state.get("research_output") else "No"}
- Solution/Draft Generated: {"Yes" if state.get("coder_output") else "No"}
- Critic Approved: {"Yes" if state.get("critic_feedback") else "No"}"""

        res = llm_client.generate_completion(
            prompt, messages=state.get("messages", []), max_tokens=15, agent="supervisor"
        )

        decision = res.content.strip().lower()

        target = "writer"
        if "researcher" in decision:
            target = "researcher"
        elif "coder" in decision:
            target = "coder"
        elif "critic" in decision:
            target = "critic"
        elif "finish" in decision or "writer" in decision:
            target = "writer"
        else:
            task_lower = task.lower()
            if not state.get("research_output") and ("find" in task_lower or "research" in task_lower):
                target = "researcher"
            elif not state.get("coder_output"):
                target = "coder"
            elif not state.get("critic_feedback"):
                target = "critic"

        thought = (
            res.thought
            or f"Supervisor evaluated state and routed control to [{target.upper()}] node."
        )

        return {
            "next_agent": target,
            "agent_thoughts": [
                {"agent": "Supervisor", "thought": thought, "timestamp": time.strftime("%H:%M:%S")}
            ],
        }

    # 2. RESEARCHER NODE
    def researcher_node(state: MultiAgentState) -> Dict[str, Any]:
        task = _get_task(state)

        # 1. Perform live DuckDuckGo Web Search via ToolLoader with checkpointing
        tools = get_tool_loader(llm_client)
        tool_res = tools.run(
            "web_search",
            run_id=state.get("run_id"),
            metadata={"agent": "researcher"},
            query=task,
            max_results=5,
            max_chars=8000,
        )
        if tool_res.get("ok"):
            search_context = str(tool_res.get("message", ""))
        else:
            raw_search = llm_client.search_web(task, max_results=5, max_tokens=12000)
            search_context = llm_client.build_search_context(raw_search, max_chars=8000)


        # 2. Pass findings to LLM for synthesis
        soul_prompt = load_soul("researcher")
        prompt = f"""{soul_prompt}

Task: "{task}"
Live DuckDuckGo Search Context:
{search_context}"""

        res = llm_client.generate_completion(
            prompt,
            messages=[],
            available_tools=["web_search"],
            max_tokens=2048,
            agent="researcher",
        )

        research_summary = f"{res.content.strip()}\n\n{search_context}"

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Researcher Agent",
            "role": "assistant",
            "content": research_summary,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "research_output": research_summary,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Researcher",
                    "thought": res.thought or f"Executed live DuckDuckGo search for '{task}' and synthesized context.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 3. CODER NODE (Primary Solution & Content Generator)
    def coder_node(state: MultiAgentState) -> Dict[str, Any]:
        task = _get_task(state)
        soul_prompt = load_soul("coder")
        prompt = f"""{soul_prompt}

Task: "{task}"
Research Context: {state.get("research_output", "N/A")}
Critic Feedback: {state.get("critic_feedback", "None")}"""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="coder"
        )
        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Coder Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "coder_output": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Coder",
                    "thought": res.thought or "Generated solution response.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4. CRITIC NODE
    def critic_node(state: MultiAgentState) -> Dict[str, Any]:
        soul_prompt = load_soul("critic")
        prompt = f"""{soul_prompt}

Content to Audit:\n{state.get("coder_output", "")}"""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=1024, agent="critic"
        )

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Critic Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "critic_feedback": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Critic",
                    "thought": res.thought or "Audited solution response.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 5. WRITER NODE
    def writer_node(state: MultiAgentState) -> Dict[str, Any]:
        task = _get_task(state)
        soul_prompt = load_soul("writer")
        prompt = f"""{soul_prompt}

Task: "{task}"
Draft Solution: {state.get("coder_output", "N/A")}"""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="writer"
        )

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Writer Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "final_response": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Writer",
                    "thought": res.thought or "Synthesized final deliverable.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # ROUTER CONDITIONAL FUNCTION
    def route_supervisor(state: MultiAgentState) -> str:
        next_agent = state.get("next_agent", "writer")
        if next_agent in ["researcher", "coder", "critic", "writer"]:
            return next_agent
        return END

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "critic": "critic",
            "writer": "writer",
            END: END,
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("critic", "supervisor")
    workflow.add_edge("writer", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_supervisor_graph = create_multi_agent_supervisor_graph(LocalLLMClient())
