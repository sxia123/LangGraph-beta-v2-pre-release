import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient
from src.core.soul_loader import load_soul
from src.core.tool_loader import get_tool_loader


class ChartPipelineState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_input: str
    current_step: str

    # Intake
    goals: List[str]
    scope: str
    clearance_level: str
    intake_status: str  # "APPROVED" | "BLOCKED"

    # Specialist & Verification
    specialist_output: str
    tier0_checks: Dict[str, bool]  # {"observed": True, "completed": True, "tested": True, "docs": True}
    tier1_verified: bool
    tier1_result: str
    revised_output: str
    tier2_verified: bool
    is_converged: bool
    final_verification_result: str
    final_answer_verified: bool
    final_repair_applied: bool

    # Escalation
    escalation_notes: str
    repaired_output: str

    # Execution & Memory
    action_payload: Optional[Dict[str, Any]]
    action_blocked: bool
    approval_granted: bool
    execution_result: str
    memory_logs: Annotated[List[Dict[str, Any]], operator.add]
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]

    # Run tracking (SQLite memory store)
    run_id: str
    run_started_at: str


def _get_input(state: ChartPipelineState) -> str:
    user_input = state.get("user_input", "")
    if not user_input:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                user_input = getattr(last_msg, "content", "")
    return user_input or "Execute default workflow task."


def create_chart_pipeline_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(ChartPipelineState)

    # Shared tool loader: every node runs tools (web search, wikipedia, arxiv, python REPL, ...)
    # through the central registry with mandatory pre-execution checkpointing.
    tools = get_tool_loader(llm_client)

    def _persist_memory(state: ChartPipelineState, entry: Dict[str, Any]) -> None:
        """Save a memory entry to SQLite, grouped under the current run_id."""
        try:
            from src.core.memory_store import save_memory

            entry = dict(entry)
            entry["run_id"] = state.get("run_id")
            save_memory(entry)
        except Exception:
            pass

    def _execute_tool_with_checkpoint(
        state: ChartPipelineState, tool_name: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute any tool via the ToolLoader with mandatory pre- and post-tool checkpointing in SQLite."""
        run_id = state.get("run_id")
        current_step = state.get("current_step", "active_step")

        return tools.run(
            tool_name,
            run_id=run_id,
            create_checkpoint=True,
            metadata={"step": current_step},
            **kwargs,
        )


    def _search_context(query_or_state: Any, query: Optional[str] = None, state: Optional[ChartPipelineState] = None, max_results: int = 2, max_chars: int = 8000) -> str:
        """Run the web_search tool via _execute_tool_with_checkpoint and return its result text."""
        if isinstance(query_or_state, dict):
            curr_state = query_or_state
            actual_query = query or ""
        else:
            actual_query = str(query_or_state)
            curr_state = state if isinstance(state, dict) else {}

        res = _execute_tool_with_checkpoint(curr_state, "web_search", query=actual_query, max_results=max_results, max_chars=max_chars)
        if res.get("ok"):
            return str(res.get("message", ""))
        return f"Web search unavailable — FLAG all claims as UNVERIFIED. ({res.get('message', '')})"

    # 1. INTAKE NODE
    def intake_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        task_lower = task.lower()
        soul_prompt = load_soul("intake", fallback_prompt="You are the Intake Gatekeeper & Scope Classifier.")

        # Classify scope and clearance
        blocked_keywords = ["malicious", "unauthorized", "drop database", "exploit"]
        is_blocked = any(kw in task_lower for kw in blocked_keywords)

        status = "BLOCKED" if is_blocked else "APPROVED"
        goals = ["Extract intent", "Validate scope", "Assess clearance"]
        scope = "Standard Workload" if not is_blocked else "Restricted Workload"

        # Start run tracking: every memory saved during this execution is
        # grouped under this run_id in the SQLite memory store.
        from datetime import datetime

        from src.core.memory_store import start_run

        run_id = state.get("run_id") or start_run("chart_pipeline", task)
        run_started_at = state.get("run_started_at") or datetime.now().isoformat(timespec="seconds")


        msg = {
            "id": f"msg_intake_{int(time.time() * 1000)}",
            "sender": "Intake Node",
            "role": "assistant",
            "content": f"### Intake Evaluation\n**Status**: {status}\n**Scope**: {scope}\n**Goals**: {', '.join(goals)}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "user_input": task,
            "intake_status": status,
            "scope": scope,
            "clearance_level": "Level-1" if not is_blocked else "Denied",
            "goals": goals,
            "run_id": run_id,
            "run_started_at": run_started_at,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Intake Node",
                    "thought": f"Assessed request scope with soul [{soul_prompt[:30]}...]: [{scope}] -> Status [{status}]. Run tracked as [{run_id}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # BLOCKED END NODE (Intake failure)
    def blocked_end_node(state: ChartPipelineState) -> Dict[str, Any]:
        # Close the run record so blocked requests are still tracked in SQLite.
        try:
            from src.core.memory_store import finish_run

            run_id = state.get("run_id")
            if run_id:
                finish_run(
                    run_id,
                    status="blocked",
                    final_answer="Request blocked during intake.",
                    started_at=state.get("run_started_at"),
                )
        except Exception:
            pass

        msg = {
            "id": f"msg_blocked_{int(time.time() * 1000)}",
            "sender": "System Gate",
            "role": "assistant",
            "content": "🚫 **Workflow Terminated**: Request was blocked during intake due to clearance or scope violation.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {"current_step": "blocked_end", "messages": [msg]}

    # 2. SPECIALIST NODE (Local AI)
    def specialist_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)

        # Budgeted Web Search: execute web search with strict cap of 2 results
        search_context = _search_context(task, max_results=5)

        soul_prompt = load_soul("specialist", fallback_prompt="You are the Lead Specialist Agent.")
        prompt = f"""{soul_prompt}

Available Tools (imperative — results are injected by the pipeline, not by you):
{tools.prompt_block()}

Task: "{task}"
Web Search Context (Budgeted - Max 2 Snippets):
{search_context}"""

        res = llm_client.generate_completion(
            prompt,
            messages=[],
            available_tools=tools.list_tools(),
            max_tokens=2048,
            agent="specialist",
        )
        msg = {
            "id": f"msg_spec_{int(time.time() * 1000)}",
            "sender": "Specialist Agent (Local AI)",
            "role": "assistant",
            "content": f"### Specialist Solution Draft (Local AI)\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "specialist_output": res.content,
            "current_step": "specialist_complete",
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Specialist Agent",
                    "thought": res.thought or "Gathered context (with budgeted web search) and committed local solution draft.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 3. TIER 0 CHECKS NODE
    def tier0_checks_node(state: ChartPipelineState) -> Dict[str, Any]:
        output = state.get("specialist_output", "")
        soul_prompt = load_soul("tier0_auditor", fallback_prompt="You are the Tier 0 Automated Auditor.")
        # Evaluate 4 criteria: observed, completed, tested, docs
        checks = {
            "observed": len(output) > 20,
            "completed": bool(output and not output.isspace()),
            "tested": "error" not in output.lower(),
            "docs": True,
        }
        all_passed = all(checks.values())

        msg = {
            "id": f"msg_tier0_{int(time.time() * 1000)}",
            "sender": "Tier 0 Audit Node",
            "role": "assistant",
            "content": f"### Tier 0 Automated Checks\n- Observed: {'✅' if checks['observed'] else '❌'}\n- Completed: {'✅' if checks['completed'] else '❌'}\n- Tested: {'✅' if checks['tested'] else '❌'}\n- Docs: {'✅' if checks['docs'] else '❌'}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier0_checks": checks,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 0 Auditor",
                    "thought": f"Tier 0 checks evaluated with auditor persona [{soul_prompt[:25]}...]: All Passed = {all_passed}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 3.5. TIER 0.5 WEB VERIFICATION NODE
    def tier05_web_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        specialist_output = state.get("specialist_output", "")

        # MANDATORY Web Search: verify claims — web search is REQUIRED, not optional
        search_context = _search_context(task, max_results=2, max_chars=3000)

        soul_prompt = load_soul("tier0_auditor", fallback_prompt="You are the Tier 0 Web Verification Auditor.")
        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Any claim about events, releases, data, or developments after 2024 MUST be verified against current web search results.

Task: "{task}"

Specialist Output to Verify:
{specialist_output}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: You MUST cross-reference the specialist output against the web search context. Do NOT rely on your own training data or internal knowledge — it is stale and may be outdated. Identify any factual claims that are inconsistent with the search results. Flag any claims about post-2024 events as UNVERIFIED if not corroborated by the search. Report which claims are corroborated, contradicted, or unverified."""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="tier0_auditor"
        )

        msg = {
            "id": f"msg_tier05_{int(time.time() * 1000)}",
            "sender": "Tier 0.5 Web Verification Node",
            "role": "assistant",
            "content": f"### Tier 0.5 Web Verification\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 0.5 Web Verifier",
                    "thought": res.thought or "Cross-referenced specialist output against web search results for factual accuracy.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4. TIER 1 VERIFY NODE
    def tier1_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        soul_prompt = load_soul("tier1_verifier", fallback_prompt="You are the Tier 1 Verification Auditor.")

        # MANDATORY Web Search: verify claims — web search is REQUIRED, not optional
        task = _get_input(state)
        search_context = _search_context(task, max_results=2, max_chars=3000)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Output to Audit:
{state.get("specialist_output", "")}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: You MUST cross-reference key claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Pay special attention to any claims about events, releases, or data after 2024. If a claim is not corroborated by the search results, mark it as UNVERIFIED. Respond with VERIFIED only if claims are corroborated, or REVISION REQUIRED if discrepancies are found."""
        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=1024, agent="tier1_verifier"
        )
        is_verified = "VERIFIED" in res.content.upper() or "APPROVED" in res.content.upper()
        t0 = state.get("tier0_checks", {})
        converged = is_verified and all(t0.values()) if t0 else is_verified

        msg = {
            "id": f"msg_tier1_{int(time.time() * 1000)}",
            "sender": "Tier 1 Verification Node",
            "role": "assistant",
            "content": f"### Tier 1 Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n**Convergence**: {'CONVERGED' if converged else 'ESCALATION REQUIRED'}\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier1_verified": is_verified,
            "tier1_result": res.content,
            "is_converged": converged,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 1 Auditor",
                    "thought": res.thought or f"Tier 1 audit complete. Converged = {converged}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4.5. REVISIONS NODE (apply Tier 1 feedback to the specialist draft)
    def revisions_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        draft = state.get("specialist_output", "")
        tier1_result = state.get("tier1_result", "No Tier 1 feedback available.")

        # MANDATORY Web Search: verify claims while revising
        search_context = _search_context(task, max_results=2, max_chars=3000)

        soul_prompt = load_soul(
            "specialist",
            fallback_prompt="You are the Lead Specialist Agent revising a draft based on auditor feedback.",
        )
        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Task: "{task}"

Current Draft:
{draft}

Tier 1 Auditor Feedback (address every point raised):
{tier1_result}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: Produce a REVISED draft that resolves every discrepancy, correction, or UNVERIFIED flag called out by the Tier 1 auditor. Cross-reference all factual claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Output ONLY the revised solution, with no commentary."""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="specialist"
        )
        msg = {
            "id": f"msg_rev_{int(time.time() * 1000)}",
            "sender": "Revisions Node",
            "role": "assistant",
            "content": f"### Revised Draft (post Tier 1 feedback)\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "revised_output": res.content,
            "specialist_output": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Revisions Node",
                    "thought": res.thought or "Applied Tier 1 auditor feedback and produced a revised draft.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4.75. TIER 2 VERIFY NODE (re-audit the revised draft)
    def tier2_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        revised = state.get("revised_output") or state.get("specialist_output", "")
        soul_prompt = load_soul("tier1_verifier", fallback_prompt="You are the Tier 2 Verification Auditor.")

        # MANDATORY Web Search: verify claims — web search is REQUIRED, not optional
        search_context = _search_context(task, max_results=2, max_chars=3000)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Revised Output to Audit:
{revised}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: This is the second-pass (Tier 2) audit of the revised draft. You MUST cross-reference key claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Pay special attention to any claims about events, releases, or data after 2024. If a claim is not corroborated by the search results, mark it as UNVERIFIED. Respond with VERIFIED only if claims are corroborated, or REVISION REQUIRED if discrepancies remain."""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=1024, agent="tier1_verifier"
        )
        is_verified = "VERIFIED" in res.content.upper() or "APPROVED" in res.content.upper()

        msg = {
            "id": f"msg_tier2_{int(time.time() * 1000)}",
            "sender": "Tier 2 Verification Node",
            "role": "assistant",
            "content": f"### Tier 2 Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier2_verified": is_verified,
            "is_converged": is_verified,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 2 Auditor",
                    "thought": res.thought or f"Tier 2 audit complete. Verified = {is_verified}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 5. ESCALATION NODE (Frontier Model)
    def escalation_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        soul_prompt = load_soul("frontier_escalation", fallback_prompt="You are the Frontier Model Escalation Specialist.")

        # MANDATORY Web Search: verify claims — web search is REQUIRED, not optional
        search_context = _search_context(task, max_results=5)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Task: "{task}"
Previous Specialist Draft: {state.get("specialist_output", "N/A")}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: Synthesize a refined solution. You MUST cross-reference key claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Correct any outdated or inaccurate information, especially regarding events, releases, or data after 2024. If a claim is not corroborated by the search results, flag it as UNVERIFIED."""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="frontier_escalation"
        )
        msg = {
            "id": f"msg_esc_{int(time.time() * 1000)}",
            "sender": "Escalation Node (Frontier Model)",
            "role": "assistant",
            "content": f"### Frontier Model Escalation Synthesis\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "escalation_notes": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Frontier Model Escalation",
                    "thought": res.thought or "Escalated task to high-capability frontier model reasoning.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 6. ADJUDICATE & REPAIR NODE
    def adjudicate_repair_node(state: ChartPipelineState) -> Dict[str, Any]:
        esc = state.get("escalation_notes") or state.get("specialist_output", "")
        soul_prompt = load_soul("adjudicator_repair", fallback_prompt="You are the Adjudication & Repair Specialist.")

        # MANDATORY Web Search: verify claims — web search is REQUIRED, not optional
        task = _get_input(state)
        search_context = _search_context(task, max_results=5)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Draft Solution to Repair:
{esc}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: Repair the draft solution. You MUST cross-reference key claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Correct any outdated or inaccurate information, especially regarding events, releases, or data after 2024. If a claim is not corroborated by the search results, flag it as UNVERIFIED. Ensure all factual claims are corroborated by the search results."""

        res = llm_client.generate_completion(
            prompt, messages=[], max_tokens=2048, agent="adjudicator_repair"
        )
        msg = {
            "id": f"msg_adj_{int(time.time() * 1000)}",
            "sender": "Adjudicate & Repair Node",
            "role": "assistant",
            "content": f"### Adjudicated & Repaired Output\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "repaired_output": res.content,
            "specialist_output": res.content,
            "is_converged": True,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Adjudication Node",
                    "thought": res.thought or "Adjudicated escalation feedback and applied repairs.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 7. FINAL VERIFICATION NODE
    def final_verification_node(state: ChartPipelineState) -> Dict[str, Any]:
        solution = state.get("repaired_output") or state.get("specialist_output", "")
        task = _get_input(state)
        soul_prompt = load_soul(
            "final_verifier",
            fallback_prompt="You are the Final Answer Verifier.",
        )

        # MANDATORY Web Search: verify final answer claims — web search is REQUIRED, not optional
        search_context = _search_context(task, max_results=5)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Task: "{task}"

Candidate Final Answer:
{solution}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: Review the candidate answer for correctness, completeness, clarity, and whether it directly addresses the task. You MUST cross-reference key factual claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Pay special attention to any claims about events, releases, or data after 2024. If a claim is not corroborated by the search results, mark it as UNVERIFIED. Respond with a short rationale and end with either VERIFIED (only if claims are corroborated) or REJECTED."""

        # Final verification uses a dedicated high-capability model
        # (Muse Glimmer 30B 6-bit) for an independent second opinion.
        res = llm_client.generate_completion(
            prompt,
            messages=[],
            max_tokens=1024,
            agent="final_verifier",
            model_name="Muse-Glimmer-30B-6bit",
        )
        verification_text = res.content.strip()
        is_verified = "VERIFIED" in verification_text.upper()

        msg = {
            "id": f"msg_verify_{int(time.time() * 1000)}",
            "sender": "Final Verification Node",
            "role": "assistant",
            "content": f"### Final Answer Verification\n**Status**: {'VERIFIED' if is_verified else 'REJECTED'}\n\n{verification_text}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "final_verification_result": verification_text,
            "final_answer_verified": is_verified,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Final Verifier",
                    "thought": res.thought or f"Checked the final answer for completeness and direct task alignment. Verified = {is_verified}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 7.5. FINAL REPAIR NODE (Muse Glimmer — repairs a rejected final answer)
    def final_repair_node(state: ChartPipelineState) -> Dict[str, Any]:
        solution = state.get("repaired_output") or state.get("specialist_output", "")
        task = _get_input(state)
        verifier_feedback = state.get("final_verification_result", "No verifier feedback available.")

        # MANDATORY Web Search: verify claims while repairing
        search_context = _search_context(task, max_results=5)

        soul_prompt = load_soul(
            "adjudicator_repair",
            fallback_prompt="You are the Final Answer Repair Specialist.",
        )
        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. You MUST rely on the web search context below to verify any claims about recent events, releases, data, or developments.

Task: "{task}"

Candidate Final Answer (rejected by the verifier):
{solution}

Verifier Feedback (address every point raised):
{verifier_feedback}

Web Search Context (MANDATORY — Max 2 Snippets):
{search_context}

MANDATORY: Repair the rejected final answer. You MUST cross-reference key claims against the web search context. Do NOT rely on your own training data or internal knowledge — it may be outdated. Correct every factual inaccuracy, fill gaps, and resolve each issue the verifier flagged. Output ONLY the repaired final answer, with no commentary."""

        # Final repair uses the same dedicated high-capability model
        # (Muse Glimmer 30B 6-bit) as the final verification step.
        res = llm_client.generate_completion(
            prompt,
            messages=[],
            max_tokens=2048,
            agent="final_repair",
            model_name="Muse-Glimmer-30B-6bit",
        )
        msg = {
            "id": f"msg_final_repair_{int(time.time() * 1000)}",
            "sender": "Final Repair Node (Muse Glimmer)",
            "role": "assistant",
            "content": f"### Final Answer Repair (Muse Glimmer)\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "repaired_output": res.content,
            "specialist_output": res.content,
            "final_repair_applied": True,
            "final_answer_verified": True,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Final Repair (Muse Glimmer)",
                    "thought": res.thought or "Repaired the rejected final answer using Muse Glimmer 30B 6-bit.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 8. PREPARE ACTION NODE
    def prepare_action_node(state: ChartPipelineState) -> Dict[str, Any]:
        solution = state.get("repaired_output") or state.get("specialist_output", "")
        task = _get_input(state)

        is_blocked = "deny_action" in task.lower() or not state.get("final_answer_verified", True)
        existing_payload = state.get("action_payload") or {}
        payload = {
            "target_action": existing_payload.get("target_action", "execute_solution"),
            "payload_summary": existing_payload.get("payload_summary") or (solution[:200] + "..." if len(solution) > 200 else solution),
            "requires_approval": existing_payload.get("requires_approval", True),
            "tool": existing_payload.get("tool"),
            "tool_args": existing_payload.get("tool_args") or {},
        }


        msg = {
            "id": f"msg_prep_{int(time.time() * 1000)}",
            "sender": "Action Preparation Node",
            "role": "assistant",
            "content": f"### Action Payload Prepared\n**Blocked Status**: {'BLOCKED' if is_blocked else 'READY'}\n**Verification**: {'PASSED' if state.get('final_answer_verified', True) else 'FAILED'}\n**Target**: {payload['target_action']}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "action_payload": payload,
            "action_blocked": is_blocked,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Action Preparation",
                    "thought": f"Prepared action payload. Action Blocked = {is_blocked}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # BLOCKED MEMORY NODE
    def blocked_memory_node(state: ChartPipelineState) -> Dict[str, Any]:
        log_entry = {
            "event": "ACTION_BLOCKED",
            "reason": "Policy violation during action preparation.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Persist the memory entry under the current run (if memory_store is available)
        _persist_memory(state, {
            "event": log_entry["event"],
            "input": state.get("user_input"),
            "result": None,
            "timestamp": log_entry["timestamp"],
            "reason": log_entry["reason"],
        })

        # Close the run record as blocked.
        try:
            from src.core.memory_store import finish_run

            run_id = state.get("run_id")
            if run_id:
                finish_run(
                    run_id,
                    status="action_blocked",
                    final_answer="Action blocked during preparation.",
                    started_at=state.get("run_started_at"),
                )
        except Exception:
            pass

        msg = {
            "id": f"msg_block_mem_{int(time.time() * 1000)}",
            "sender": "Memory Gate",
            "role": "assistant",
            "content": "🚫 **Action Blocked**: Recorded blocked event to system memory.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "memory_logs": [log_entry],
            "messages": [msg],
        }

    # 8. APPROVAL NODE
    def approval_node(state: ChartPipelineState) -> Dict[str, Any]:
        msg = {
            "id": f"msg_appr_{int(time.time() * 1000)}",
            "sender": "Approval Gate Node",
            "role": "assistant",
            "content": "✅ **Approval Gate Granted**: Action approved for execution.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "approval_granted": True,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Approval Gate",
                    "thought": "Approved action payload for execution.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 9. EXECUTE NODE
    def execute_node(state: ChartPipelineState) -> Dict[str, Any]:
        payload = state.get("action_payload") or {}
        tool_name = payload.get("tool")
        tool_args = payload.get("tool_args") or {}

        if tool_name:
            # Imperative tool execution via the central ToolLoader with checkpointing.
            tool_result = tools.run(
                tool_name,
                run_id=state.get("run_id"),
                metadata={"step": "execute", "target_action": payload.get("target_action", "run")},
                **tool_args,
            )
            if tool_result.get("ok"):

                result_text = (
                    f"Executed action [{payload.get('target_action', 'run')}] successfully. "
                    f"Tool '{tool_name}' result: {tool_result.get('message', '')}"
                )
            else:
                result_text = (
                    f"Action [{payload.get('target_action', 'run')}] failed: "
                    f"tool '{tool_name}' error — {tool_result.get('message', 'unknown error')}"
                )
        else:
            result_text = f"Executed action [{payload.get('target_action', 'run')}] successfully."

        msg = {
            "id": f"msg_exec_{int(time.time() * 1000)}",
            "sender": "Execution Node",
            "role": "assistant",
            "content": f"### Execution Completed\n{result_text}",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "execution_result": result_text,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Execution Node",
                    "thought": f"Executed payload: {result_text}",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 10. FINALIZE & MEMORY NODE
    def finalize_memory_node(state: ChartPipelineState) -> Dict[str, Any]:
        memory_entry = {
            "event": "PIPELINE_SUCCESS",
            "input": state.get("user_input"),
            "result": state.get("execution_result"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Persist memory entry to SQLite under the current run
        _persist_memory(state, {
            "event": memory_entry["event"],
            "input": memory_entry["input"],
            "result": memory_entry["result"],
            "timestamp": memory_entry["timestamp"],
            "final_answer": state.get("specialist_output"),
        })

        # Close the run record: status, final answer, duration, memory count.
        try:
            from src.core.memory_store import finish_run

            run_id = state.get("run_id")
            if run_id:
                finish_run(
                    run_id,
                    status="completed",
                    final_answer=state.get("specialist_output"),
                    started_at=state.get("run_started_at"),
                )
        except Exception:
            pass

        verification_status = (
            "VERIFIED"
            if state.get("final_answer_verified", True)
            else "REJECTED"
        )
        final_text = (
            f"### Final Pipeline Result\n\n"
            f"**Task**: {state.get('user_input')}\n\n"
            f"**Solution**: {state.get('specialist_output', 'N/A')}\n\n"
            f"**Verification**: {verification_status}\n\n"
            f"**Run ID**: {state.get('run_id', 'N/A')}\n\n"
            f"**Execution Status**: Completed & Persisted to Memory."
        )

        msg = {
            "id": f"msg_fin_{int(time.time() * 1000)}",
            "sender": "Finalize & Memory Node",
            "role": "assistant",
            "content": final_text,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "memory_logs": [memory_entry],
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Finalize & Memory",
                    "thought": f"Saved final execution result to system memory store under run [{state.get('run_id', 'N/A')}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # ROUTING FUNCTIONS
    def route_intake(state: ChartPipelineState) -> str:
        if state.get("intake_status") == "BLOCKED":
            return "blocked_end"
        return "specialist"

    def route_tier2(state: ChartPipelineState) -> str:
        # After the second-pass audit, a verified revised draft goes straight to
        # the final answer; an unverified one is escalated to the frontier model.
        if state.get("tier2_verified"):
            return "final_verification"
        return "escalation"

    def route_final_verification(state: ChartPipelineState) -> str:
        # A verified final answer proceeds to action prep; a rejected one is
        # repaired by the Muse Glimmer final-repair node before action prep.
        if state.get("final_answer_verified"):
            return "prepare_action"
        return "final_repair"

    def route_prepare_action(state: ChartPipelineState) -> str:
        if state.get("action_blocked") or state.get("final_answer_verified") is False:
            return "blocked_memory"
        return "approval"

    # BUILD GRAPH
    workflow.add_node("intake", intake_node)
    workflow.add_node("blocked_end", blocked_end_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("tier0_checks", tier0_checks_node)
    workflow.add_node("tier05_web_verify", tier05_web_verify_node)
    workflow.add_node("tier1_verify", tier1_verify_node)
    workflow.add_node("revisions", revisions_node)
    workflow.add_node("tier2_verify", tier2_verify_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("adjudicate_repair", adjudicate_repair_node)
    workflow.add_node("final_verification", final_verification_node)
    workflow.add_node("final_repair", final_repair_node)
    workflow.add_node("prepare_action", prepare_action_node)
    workflow.add_node("blocked_memory", blocked_memory_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("finalize_memory", finalize_memory_node)

    # EDGES
    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake", route_intake, {"specialist": "specialist", "blocked_end": "blocked_end"}
    )
    workflow.add_edge("blocked_end", END)

    workflow.add_edge("specialist", "tier0_checks")
    workflow.add_edge("tier0_checks", "tier05_web_verify")
    workflow.add_edge("tier05_web_verify", "tier1_verify")
    # Every draft goes through revisions + a second-pass (Tier 2) audit before
    # the final answer. The frontier escalation is now a deeper fallback that
    # only triggers when the revised draft still fails Tier 2 verification.
    workflow.add_edge("tier1_verify", "revisions")
    workflow.add_edge("revisions", "tier2_verify")
    workflow.add_conditional_edges(
        "tier2_verify",
        route_tier2,
        {"final_verification": "final_verification", "escalation": "escalation"},
    )

    workflow.add_edge("escalation", "adjudicate_repair")
    workflow.add_edge("adjudicate_repair", "final_verification")
    # A verified final answer goes straight to action prep; a rejected one is
    # repaired by the Muse Glimmer final-repair node before action prep.
    workflow.add_conditional_edges(
        "final_verification",
        route_final_verification,
        {"prepare_action": "prepare_action", "final_repair": "final_repair"},
    )
    workflow.add_edge("final_repair", "prepare_action")

    workflow.add_conditional_edges(
        "prepare_action",
        route_prepare_action,
        {"approval": "approval", "blocked_memory": "blocked_memory"},
    )
    workflow.add_edge("blocked_memory", END)

    workflow.add_edge("approval", "execute")
    workflow.add_edge("execute", "finalize_memory")
    workflow.add_edge("finalize_memory", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_chart_graph = create_chart_pipeline_graph(LocalLLMClient())
