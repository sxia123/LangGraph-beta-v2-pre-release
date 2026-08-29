import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient
from src.core.soul_loader import load_soul


class ClaimsTriageState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    claim_input: str
    current_step: str
    classification_details: Optional[Dict[str, Any]]
    severity_assessment: Optional[Dict[str, Any]]
    action_plan: Optional[Dict[str, Any]]
    final_response: str
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]


def _get_claim_input(state: ClaimsTriageState) -> str:
    claim = state.get("claim_input", "")
    if not claim:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                claim = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                claim = getattr(last_msg, "content", "")
    return claim


def create_claims_triage_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(ClaimsTriageState)

    def classifier_node(state: ClaimsTriageState) -> Dict[str, Any]:
        claim = _get_claim_input(state)
        soul = load_soul("intake", fallback_prompt="You are Step 1 Classifier in Claims Triage.")
        prompt = f"""{soul}\nCategorize claim: "{claim}"."""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=256, agent="intake"
        )

        category = "Product Defect & Safety"
        claim_lower = claim.lower()
        if "billing" in claim_lower or "charge" in claim_lower:
            category = "Billing & Subscription Discrepancy"
        elif any(k in claim_lower for k in ["hack", "unauthorized", "security"]):
            category = "System Outage & Security Incident"

        classification = {
            "category": category,
            "confidence": 0.95,
            "financial_amount": "$500 - $1,000",
        }

        msg = {
            "id": f"msg_class_{int(time.time() * 1000)}",
            "sender": "Step 1: Claim Classifier Agent",
            "role": "assistant",
            "content": f"### Step 1: Claim Classified\n**Category**: {category}\n**Confidence**: 95%\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "step_1_classified",
            "classification_details": classification,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Claim Classifier (Step 1)",
                    "thought": res.thought or f"Classified claim into [{category}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    def severity_filter_node(state: ClaimsTriageState) -> Dict[str, Any]:
        claim = _get_claim_input(state)
        soul = load_soul("tier0_auditor", fallback_prompt="You are Step 2 Severity Filter.")
        prompt = f"""{soul}\nAssess severity for: "{claim}"."""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=256, agent="tier0_auditor"
        )

        claim_lower = claim.lower()
        level = "MEDIUM"
        sla = "24 Hours"
        if any(k in claim_lower for k in ["fire", "injury", "50k", "$75,000"]):
            level = "CRITICAL"
            sla = "1 Hour Immediate"
        elif any(k in claim_lower for k in ["breach", "intrusion"]):
            level = "HIGH"
            sla = "4 Hours Urgent"

        severity = {"level": level, "sla_urgency": sla}

        msg = {
            "id": f"msg_sev_{int(time.time() * 1000)}",
            "sender": "Step 2: Severity Filter Agent",
            "role": "assistant",
            "content": f"### Step 2: Severity Filtered\n**Level**: **{level}**\n**SLA**: {sla}\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "step_2_filtered",
            "severity_assessment": severity,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Severity Filter (Step 2)",
                    "thought": res.thought or f"Filtered severity to [{level}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    def resolution_handler_node(state: ClaimsTriageState) -> Dict[str, Any]:
        soul = load_soul("specialist", fallback_prompt="You are Step 3 Resolution Handler.")
        prompt = f"""{soul}\nProvide resolution for claim."""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=512, agent="specialist"
        )

        sev = state.get("severity_assessment") or {}
        level = sev.get("level", "MEDIUM") if isinstance(sev, dict) else "MEDIUM"
        assigned_team = "Standard Support Team"
        if level == "CRITICAL":
            assigned_team = "Executive Response Team"
        elif level == "HIGH":
            assigned_team = "Tier 3 Incident Team"

        msg = {
            "id": f"msg_res_{int(time.time() * 1000)}",
            "sender": "Step 3: Action & Resolution Agent",
            "role": "assistant",
            "content": f"### Step 3: Resolution Dispatched\n**Assigned Team**: {assigned_team}\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "current_step": "step_3_resolved",
            "final_response": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Resolution Handler (Step 3)",
                    "thought": res.thought or f"Dispatched resolution to team [{assigned_team}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("severity_filter", severity_filter_node)
    workflow.add_node("resolution_handler", resolution_handler_node)

    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "severity_filter")
    workflow.add_edge("severity_filter", "resolution_handler")
    workflow.add_edge("resolution_handler", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_claims_triage_graph = create_claims_triage_graph(LocalLLMClient())
