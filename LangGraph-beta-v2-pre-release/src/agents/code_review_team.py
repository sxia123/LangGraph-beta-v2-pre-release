import operator
import time
from typing import Annotated, Any, Dict, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient
from src.core.soul_loader import load_soul


class CodeReviewState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    task: str
    code: str
    review: str
    approved: bool
    revision_count: int
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]


def _get_task(state: CodeReviewState) -> str:
    task = state.get("task", "")
    if not task:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                task = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                task = getattr(last_msg, "content", "")
    return task


def create_code_review_team_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(CodeReviewState)

    def developer_node(state: CodeReviewState) -> Dict[str, Any]:
        existing_code = state.get("code", "")
        if existing_code and state.get("revision_count", 0) == 0:
            return {"revision_count": 1}

        task = _get_task(state)
        soul = load_soul("coder", fallback_prompt="You are Lead Software Developer.")
        prompt = f"""{soul}\nTask: "{task}"."""
        if existing_code:
            prompt += f"\nExisting Code to revise:\n{existing_code}\nReviewer Feedback: {state.get('review', '')}"

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=1024, agent="coder"
        )

        msg = {
            "id": f"dev_{int(time.time() * 1000)}",
            "sender": "Developer Node",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        rev_num = state.get("revision_count", 0) + 1
        thought = (
            res.thought
            or f"Developed implementation (Revision {rev_num}) satisfying requirements."
        )

        return {
            "code": res.content,
            "messages": [msg],
            "revision_count": rev_num,
            "agent_thoughts": [
                {
                    "agent": f"Developer (Rev {rev_num})",
                    "thought": thought,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    def reviewer_node(state: CodeReviewState) -> Dict[str, Any]:
        soul = load_soul("tier0_auditor", fallback_prompt="You are Code Auditor.")
        prompt = f"""{soul}\nReview code:\n{state.get("code", "")}\nOutput APPROVED if valid."""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=256, agent="tier0_auditor"
        )
        is_approved = "APPROVED" in res.content.upper()

        msg = {
            "id": f"rev_{int(time.time() * 1000)}",
            "sender": "Reviewer Node",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        verdict = "APPROVED" if is_approved else "REVISION REQUIRED"
        thought = res.thought or f"Audited implementation code. Verdict: [{verdict}]."

        return {
            "review": res.content,
            "approved": is_approved,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Code Auditor",
                    "thought": thought,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    def route_review(state: CodeReviewState) -> str:
        if state.get("approved") or state.get("revision_count", 0) >= 3:
            return END
        return "developer"

    workflow.add_node("developer", developer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "developer")
    workflow.add_edge("developer", "reviewer")
    workflow.add_conditional_edges("reviewer", route_review, {"developer": "developer", END: END})

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_code_review_graph = create_code_review_team_graph(LocalLLMClient())
