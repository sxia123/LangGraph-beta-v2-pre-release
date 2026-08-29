import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents.claims_triage_team import ClaimsTriageState, create_claims_triage_graph
from src.agents.multi_agent_supervisor import MultiAgentState, create_multi_agent_supervisor_graph
from src.agents.solution_review_team import SolutionReviewState, create_solution_review_team_graph
from src.core.local_llm import LocalLLMClient


class MasterPipelineState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_input: str
    current_step: str
    triage_details: Optional[Dict[str, Any]]
    supervisor_details: Optional[Dict[str, Any]]
    review_details: Optional[Dict[str, Any]]
    final_response: str
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]


def create_master_pipeline_graph(llm_client: LocalLLMClient):
    triage_subgraph = create_claims_triage_graph(llm_client)
    supervisor_subgraph = create_multi_agent_supervisor_graph(llm_client)
    review_subgraph = create_solution_review_team_graph(llm_client)

    workflow = StateGraph(MasterPipelineState)

    # 1. CLAIMS TRIAGE STAGE NODE
    def triage_stage_node(state: MasterPipelineState) -> Dict[str, Any]:
        messages = state.get("messages") or []
        user_input = state.get("user_input")
        if not user_input and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                user_input = getattr(last_msg, "content", "")
        prompt = user_input or "Execute master pipeline for task."

        triage_input: ClaimsTriageState = {
            "messages": messages,
            "claim_input": prompt,
            "current_step": "step_1_classification",
            "classification_details": None,
            "severity_assessment": None,
            "action_plan": None,
            "final_response": "",
            "agent_thoughts": [],
        }

        triage_res = triage_subgraph.invoke(triage_input)

        class_details = triage_res.get("classification_details") or {}
        severity_details = triage_res.get("severity_assessment") or {}
        category = class_details.get("category", "General")
        severity_lvl = severity_details.get("level", "MEDIUM")

        msg = {
            "id": f"master_triage_{int(time.time() * 1000)}",
            "sender": "Master Pipeline [Stage 1: Claims Triage]",
            "role": "assistant",
            "content": f"✅ **Stage 1 (Triage) Complete**\nCategory: {category}\nSeverity: {severity_lvl}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "triage_complete",
            "triage_details": {
                "classification": triage_res.get("classification_details"),
                "severity": triage_res.get("severity_assessment"),
            },
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Master Pipeline",
                    "thought": "Stage 1: Claims Triage executed successfully.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 2. SUPERVISOR STAGE NODE
    def supervisor_stage_node(state: MasterPipelineState) -> Dict[str, Any]:
        triage_info = state.get("triage_details", {})
        task_prompt = f"User Request: {state.get('user_input')}\nTriage Context: {triage_info}"

        supervisor_input: MultiAgentState = {
            "messages": state.get("messages", []),
            "current_task": task_prompt,
            "next_agent": "supervisor",
            "research_output": "",
            "coder_output": "",
            "critic_feedback": "",
            "final_response": "",
            "agent_thoughts": [],
        }

        supervisor_res = supervisor_subgraph.invoke(supervisor_input)

        solution = supervisor_res.get("final_response") or supervisor_res.get("coder_output") or supervisor_res.get("research_output") or "Solution synthesized."

        msg = {
            "id": f"master_sup_{int(time.time() * 1000)}",
            "sender": "Master Pipeline [Stage 2: Supervisor Team]",
            "role": "assistant",
            "content": f"✅ **Stage 2 (Supervisor Workflow) Complete**\n\n{solution}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        sub_thoughts = supervisor_res.get("agent_thoughts") or []
        stage_thought = {
            "agent": "Master Pipeline",
            "thought": f"Stage 2: Multi-agent supervisor team completed task with {len(sub_thoughts)} thought steps.",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "supervisor_complete",
            "supervisor_details": {
                "research": supervisor_res.get("research_output"),
                "draft": supervisor_res.get("coder_output"),
                "critic": supervisor_res.get("critic_feedback"),
                "solution": solution,
            },
            "messages": [msg],
            "agent_thoughts": sub_thoughts + [stage_thought],
        }

    # 3. REVIEWER STAGE NODE
    def reviewer_stage_node(state: MasterPipelineState) -> Dict[str, Any]:
        sup_details = state.get("supervisor_details") or {}
        solution_to_review = (
            sup_details.get("solution")
            or state.get("user_input", "")
        )

        review_input: SolutionReviewState = {
            "messages": state.get("messages", []),
            "task": state.get("user_input", ""),
            "solution": solution_to_review,
            "review": "",
            "approved": False,
            "revision_count": 0,
        }

        review_res = review_subgraph.invoke(review_input)

        is_approved = review_res.get("approved", True)
        review_text = review_res.get("review", "Approved without changes.")

        triage_details = state.get("triage_details") or {}
        severity_info = triage_details.get("severity") or {}
        severity_lvl = severity_info.get("level", "NORMAL")

        final_content = (
            f"### Final Master Pipeline Result\n\n"
            f"**Triage Summary:** {severity_lvl} Severity\n\n"
            f"**Generated Solution:**\n{solution_to_review}\n\n"
            f"**Quality Auditor Review:** {'APPROVED' if is_approved else 'REVISION SUGGESTED'}\n"
            f"{review_text}"
        )

        msg = {
            "id": f"master_rev_{int(time.time() * 1000)}",
            "sender": "Master Pipeline [Stage 3: Quality Review]",
            "role": "assistant",
            "content": final_content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "pipeline_complete",
            "review_details": {
                "approved": is_approved,
                "review": review_text,
            },
            "final_response": final_content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Master Pipeline",
                    "thought": f"Stage 3: Review stage completed with status [{'APPROVED' if is_approved else 'REVISION'}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    workflow.add_node("claims_triage_stage", triage_stage_node)
    workflow.add_node("supervisor_stage", supervisor_stage_node)
    workflow.add_node("reviewer_stage", reviewer_stage_node)

    workflow.add_edge(START, "claims_triage_stage")
    workflow.add_edge("claims_triage_stage", "supervisor_stage")
    workflow.add_edge("supervisor_stage", "reviewer_stage")
    workflow.add_edge("reviewer_stage", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_master_graph = create_master_pipeline_graph(LocalLLMClient())
