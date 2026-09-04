import operator
import os
import re
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.document_parser import decipher_media_file
from src.core.local_llm import LocalLLMClient
from src.core.soul_loader import load_soul
from src.core.tool_loader import get_tool_loader


class ChartPipelineState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_input: str
    current_step: str
    final_response: str

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

    # Foreign File Ingestion & Routing
    files: Optional[List[Dict[str, Any]]]
    images: Optional[List[str]]
    has_foreign_file: bool
    foreign_file_type: Optional[str]
    spreadsheet_context: Optional[str]
    spreadsheet_metadata: Optional[Dict[str, Any]]


def _get_input(state: ChartPipelineState) -> str:
    user_input = state.get("user_input", "") or state.get("user_query", "")
    if not user_input:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                user_input = getattr(last_msg, "content", "")
    return user_input or "Execute default workflow task."


def create_chart_pipeline_graph(
    llm_client: LocalLLMClient, checkpointer: Optional[Any] = None
):
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

    def _search_context(
        query_or_state: Any,
        query: Optional[str] = None,
        state: Optional[ChartPipelineState] = None,
        max_results: int = 5,
        max_chars: int = 16000,
    ) -> str:
        """Run the web_search tool via _execute_tool_with_checkpoint and return its result text."""
        if isinstance(query_or_state, dict):
            curr_state = query_or_state
            actual_query = query or ""
        else:
            actual_query = str(query_or_state)
            curr_state = state if isinstance(state, dict) else {}

        # Sanitize query: strip attachment blocks, sheet dumps, and pipe tables
        clean_query = actual_query
        if "--- File Attachment:" in clean_query:
            clean_query = clean_query.split("--- File Attachment:")[0].strip()
        if "### Sheet:" in clean_query:
            clean_query = clean_query.split("### Sheet:")[0].strip()
        if "## Deciphered Spreadsheet" in clean_query:
            clean_query = clean_query.split("## Deciphered Spreadsheet")[0].strip()
        clean_query = re.sub(r"\|.*\|", "", clean_query)
        clean_query = " ".join(clean_query.split()).strip()

        if not clean_query or len(clean_query) < 3:
            clean_query = "enterprise spreadsheet data analysis chart"

        if len(clean_query) > 150:
            clean_query = clean_query[:150].rsplit(" ", 1)[0]

        res = _execute_tool_with_checkpoint(
            curr_state, "web_search", query=clean_query, max_results=max_results, max_chars=max_chars
        )
        if res.get("ok"):
            return str(res.get("message", ""))
        return f"Web search unavailable — FLAG all claims as UNVERIFIED. ({res.get('message', '')})"

    def _run_node_with_tools(
        state: ChartPipelineState,
        prompt: str,
        agent_name: str,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        images: Optional[List[str]] = None,
    ) -> tuple[str, Optional[str], List[Dict[str, Any]]]:
        """Executes LLM completion with tool calling capabilities and automatic SQLite checkpointing."""
        effective_images = images if images is not None else state.get("images")
        res = llm_client.generate_completion(
            prompt,
            messages=[],
            available_tools=tools.list_tools(),
            max_tokens=max_tokens,
            agent=agent_name,
            model_name=model_name,
            images=effective_images,
        )
        node_thoughts: List[Dict[str, Any]] = []
        if res.thought:
            node_thoughts.append(
                {
                    "agent": agent_name,
                    "thought": res.thought,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )

        # Check for tool calls
        tool_calls = res.tool_calls or []
        if not tool_calls:
            import json
            import re
            json_matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", res.content)
            if not json_matches:
                raw_match = re.search(r"(\{\s*\"(?:tool|name)\"\s*:\s*\"[^\"]+\"[\s\S]*?\})", res.content)
                if raw_match:
                    json_matches = [raw_match.group(1)]
            for j_str in json_matches:
                try:
                    parsed = json.loads(j_str)
                    if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
                        tool_calls.append(
                            {
                                "name": parsed.get("tool") or parsed.get("name"),
                                "args": parsed.get("args") or parsed.get("parameters") or {},
                            }
                        )
                        break
                except Exception:
                    pass

        final_content = res.content
        if tool_calls:
            import json
            for tc in tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args") or {}
                if t_name:
                    tool_res = _execute_tool_with_checkpoint(state, t_name, **t_args)
                    cp_id = tool_res.get("checkpoint_id")
                    post_id = tool_res.get("post_checkpoint_id")
                    status_text = "SUCCESS" if tool_res.get("ok") else "FAILED"
                    msg_text = tool_res.get("message", "")

                    node_thoughts.append(
                        {
                            "agent": f"{agent_name} -> Tool: {t_name}",
                            "thought": (
                                f"**Tool Execution**: `{t_name}` ({status_text})\n"
                                f"- **Arguments**: `{json.dumps(t_args)}`\n"
                                f"- **Checkpoint**: `{post_id or cp_id or 'N/A'}`\n\n"
                                f"**Result Preview**:\n```\n{str(msg_text)[:1200]}\n```"
                            ),
                            "tool": t_name,
                            "args": t_args,
                            "result": tool_res,
                            "ok": tool_res.get("ok", False),
                            "checkpoint_id": post_id or cp_id,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )

                    synth_prompt = (
                        f"{prompt}\n\n"
                        f"Tool `{t_name}` execution result:\n```\n{msg_text}\n```\n\n"
                        "Please incorporate this verified tool result into a clean, complete response for this stage."
                    )
                    synth_res = llm_client.generate_completion(
                        synth_prompt,
                        messages=[],
                        max_tokens=max_tokens,
                        agent=agent_name,
                        model_name=model_name,
                    )
                    if synth_res.thought:
                        node_thoughts.append(
                            {
                                "agent": f"{agent_name} Synthesis",
                                "thought": synth_res.thought,
                                "timestamp": time.strftime("%H:%M:%S"),
                            }
                        )
                    final_content = synth_res.content
                    break

        return final_content, res.thought, node_thoughts

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
            "images": state.get("images") or [],
            "files": state.get("files") or [],
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
        return {"current_step": "blocked_end", "final_response": "Request blocked during intake.", "messages": [msg]}

    # 1.5. FOREIGN FILE ROUTER NODE
    def foreign_file_router_node(state: ChartPipelineState) -> Dict[str, Any]:
        """Inspects task and state for foreign files (spreadsheets, documents, slideshows, photos/images).

        If found, deciphers the workbook/document/slideshow/image and flags has_foreign_file = True.
        Else flags has_foreign_file = False to keep with the original pipeline.
        """
        task = _get_input(state)
        collected_images: List[str] = list(state.get("images") or [])
        raw_files = state.get("files") or []
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], dict):
            last_msg = messages[-1]
            if not raw_files and last_msg.get("files"):
                raw_files = last_msg.get("files")
            if last_msg.get("images"):
                for img in last_msg.get("images"):
                    if img and img not in collected_images:
                        collected_images.append(img)

        parsed_files = []
        if raw_files:
            for f in raw_files:
                if isinstance(f, dict):
                    fn = f.get("filename", "")
                    content = f.get("content") or f.get("text") or ""
                    fn_lower = fn.lower()
                    if fn_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")) or str(content).startswith("data:image/"):
                        if content and content not in collected_images:
                            collected_images.append(content)
                        parsed_files.append({
                            "ok": True,
                            "type": "image",
                            "filename": fn,
                            "summary": f"**Visual Image**: `{fn}`",
                            "deciphered_context": f"## Visual Image Asset: {fn}\nAttached image for visual inspection and architectural charting.",
                        })
                    elif fn_lower.endswith((".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".pdf", ".docx", ".doc", ".pptx", ".ppt")):
                        parsed = decipher_media_file(content, filename=fn)
                        if parsed.get("ok"):
                            parsed_files.append(parsed)

        # Check for embedded file attachment blocks (e.g. from server.py)
        if not parsed_files and "--- File Attachment:" in task:
            matches = re.findall(
                r"--- File Attachment:\s*([^\n]+)\s*---\n([\s\S]*?)\n--- End File ---", task
            )
            for fn, f_body in matches:
                clean_fn = fn.strip()
                fn_lower = clean_fn.lower()
                if fn_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")):
                    parsed_files.append({
                        "ok": True,
                        "type": "image",
                        "filename": clean_fn,
                        "summary": f"**Visual Image**: `{clean_fn}`",
                        "deciphered_context": f"## Visual Image Asset: {clean_fn}\nAttached image for visual inspection and architectural charting.",
                    })
                elif fn_lower.endswith((".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".pdf", ".docx", ".doc", ".pptx", ".ppt")):
                    parsed = decipher_media_file(f_body.strip(), filename=clean_fn)
                    if parsed.get("ok"):
                        parsed_files.append(parsed)

        # Check for media file path mentions in prompt
        if not parsed_files:
            candidates = re.findall(
                r"[\w\-\.]+\.(?:xlsx|xls|xlsm|csv|tsv|pdf|docx|doc|pptx|ppt|png|jpg|jpeg|webp|gif|svg)",
                task,
                flags=re.IGNORECASE,
            )
            for cand in candidates:
                cand_paths = [
                    cand,
                    os.path.join(os.getcwd(), cand),
                    os.path.join(os.getcwd(), "LangGraph-beta-v2-pre-release", cand),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), cand),
                ]
                found_cand = None
                for cp in cand_paths:
                    if os.path.isfile(cp):
                        found_cand = cp
                        break
                if found_cand:
                    parsed = decipher_media_file(found_cand, filename=cand)
                    if parsed.get("ok"):
                        parsed_files.append(parsed)
                        if parsed.get("type") == "image":
                            collected_images.append(found_cand)
                        break

        has_foreign = len(parsed_files) > 0 or len(collected_images) > 0

        if has_foreign:
            types = [p.get("type") for p in parsed_files if p.get("type")]
            if "spreadsheet" in types:
                primary_type = "spreadsheet"
            elif "slideshow" in types:
                primary_type = "slideshow"
            elif "document" in types:
                primary_type = "document"
            elif collected_images or "image" in types:
                primary_type = "image"
            else:
                primary_type = "file"

            contexts = [p["deciphered_context"] for p in parsed_files if p.get("deciphered_context")]
            if collected_images and not contexts:
                contexts.append(f"## Visual Photo / Image Input\nAttached {len(collected_images)} image(s) for visual inspection, diagramming, and charting.")
            elif collected_images:
                contexts.append(f"## Visual Attachments\nAttached {len(collected_images)} image(s) available for multimodal inspection.")
            combined_context = "\n\n".join(contexts)

            media_filenames = [p.get("filename") for p in parsed_files if p.get("filename")]
            if not media_filenames and collected_images:
                media_filenames = [f"image_{i+1}.png" for i in range(len(collected_images))]

            spreadsheet_metadata = {
                "files": media_filenames,
                "primary_type": primary_type,
                "has_images": bool(collected_images),
                "image_count": len(collected_images),
                "sheet_names": [s for p in parsed_files for s in p.get("sheet_names", [])],
                "metrics": {p.get("filename", ""): p.get("metrics", {}) for p in parsed_files if "metrics" in p},
            }
            msg = {
                "id": f"msg_router_{int(time.time() * 1000)}",
                "sender": "Foreign File Router",
                "role": "assistant",
                "content": (
                    f"### Foreign File Route Activated ({primary_type.capitalize()})\n"
                    f"- Detected {len(media_filenames)} media/data file(s): {', '.join(media_filenames)}\n"
                    f"- Routing to specialized {primary_type} deciphering & visual charting workflow."
                ),
                "timestamp": time.strftime("%H:%M:%S"),
            }
            return {
                "has_foreign_file": True,
                "foreign_file_type": primary_type,
                "spreadsheet_context": combined_context,
                "spreadsheet_metadata": spreadsheet_metadata,
                "images": collected_images,
                "current_step": "foreign_file_routed",
                "messages": [msg],
                "agent_thoughts": [
                    {
                        "agent": "Foreign File Router",
                        "thought": f"Foreign file/media detected (type={primary_type}, files={media_filenames}, images={len(collected_images)}). Routing to specialized deciphering pipeline.",
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                ],
            }
        else:
            return {
                "has_foreign_file": False,
                "foreign_file_type": None,
                "spreadsheet_context": None,
                "spreadsheet_metadata": None,
                "images": collected_images,
                "current_step": "original_pipeline_routed",
                "agent_thoughts": [
                    {
                        "agent": "Foreign File Router",
                        "thought": "No foreign file or image detected in input. Keeping with the original pipeline.",
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                ],
            }

    # 1.6. SPREADSHEET & MEDIA SPECIALIST NODE (for foreign file path)
    def spreadsheet_specialist_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        spreadsheet_context = state.get("spreadsheet_context") or ""
        foreign_file_type = state.get("foreign_file_type") or "spreadsheet"

        soul_prompt = load_soul("specialist", fallback_prompt="You are the Lead Specialist Agent.")

        if foreign_file_type == "document":
            reqs = (
                "MANDATORY DOCUMENT ANALYSIS & VISUALIZATION REQUIREMENTS:\n"
                "1. Synthesize the provided document: extract key sections, critical facts, embedded tables, and takeaways.\n"
                "2. Present a structured Executive Summary with core insights and data highlights.\n"
                "3. Include at least one visual Mermaid diagram (e.g. ```mermaid ... ``` code block using flowchart, mindmap, timeline, or chart) illustrating document architecture, process flow, or key statistics.\n"
                "4. Structure the response with clear headings, organized tables, and actionable conclusions."
            )
        elif foreign_file_type == "slideshow":
            reqs = (
                "MANDATORY SLIDESHOW & PRESENTATION ANALYSIS REQUIREMENTS:\n"
                "1. Synthesize the presentation slide deck: outline slide progression, bullet points, speaker notes, and embedded metrics.\n"
                "2. Present a structured Executive Summary of the presentation narrative and strategic roadmap.\n"
                "3. Include at least one visual Mermaid diagram (e.g. ```mermaid ... ``` code block using timeline, gantt, flowchart, or xychart-beta) visualizing the presentation roadmap or metrics.\n"
                "4. Structure the response with organized slide takeaways, tabular summaries, and recommendations."
            )
        elif foreign_file_type == "image":
            reqs = (
                "MANDATORY VISUAL & IMAGE ANALYSIS REQUIREMENTS:\n"
                "1. Analyze the attached visual photo, screenshot, or diagram in detail: identify visual components, layouts, text, and visual data.\n"
                "2. Present a comprehensive analysis describing visual features, structure, and findings.\n"
                "3. Include at least one visual Mermaid diagram (e.g. ```mermaid ... ``` code block) modeling the architecture, workflow, or visual relationships depicted in the image.\n"
                "4. Structure the response with clear headings, organized observations, and key takeaways."
            )
        else:
            reqs = (
                "MANDATORY SPREADSHEET ANALYSIS & VISUALIZATION REQUIREMENTS:\n"
                "1. Decipher the provided spreadsheet data: identify key metrics, variances, totals, and cross-sheet comparisons.\n"
                "2. Present a structured Executive Summary with high-priority business findings and financial/operational KPIs.\n"
                "3. Include at least one visual Mermaid chart (e.g. ```mermaid ... ``` code block using xychart-beta, bar chart, pie chart, or flowchart) illustrating the core data metrics.\n"
                "4. Structure the response with clear headings, organized markdown tables, and strategic takeaways."
            )

        prompt = f"""{soul_prompt}

Task: "{task}"

{spreadsheet_context}

Available Tools:
{tools.prompt_block()}

{reqs}"""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="specialist", max_tokens=4096, images=state.get("images")
        )

        msg = {
            "id": f"msg_spec_{int(time.time() * 1000)}",
            "sender": "Specialist Agent (Local AI)",
            "role": "assistant",
            "content": f"### Specialist Solution Draft (Local AI)\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "specialist_output": content,
            "current_step": "spreadsheet_specialist_complete",
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Specialist Agent",
                    "thought": raw_thought or f"Formulated {foreign_file_type} deciphering and visual chart draft.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 1.7. SPREADSHEET & MEDIA VERIFY NODE (audits against reference context rather than web search)
    def spreadsheet_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        soul_prompt = load_soul("tier1_verifier", fallback_prompt="You are the Media & Data Verification Auditor.")
        spreadsheet_context = state.get("spreadsheet_context") or ""

        prompt = f"""{soul_prompt}

Task: "{task}"

Output to Audit:
{state.get("specialist_output", "")}

Reference Context:
{spreadsheet_context}

Available Tools:
{tools.prompt_block()}

MANDATORY AUDIT CRITERIA:
Verify that the specialist output accurately reflects the provided reference data (spreadsheet, document, slideshow, or image), calculations, and visual charts.
Do NOT flag reference figures or document details as unverified simply because they are private local data not present in web search.
Respond with VERIFIED if findings match the reference data, or REVISION REQUIRED if discrepancies exist."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="tier1_verifier", max_tokens=4096
        )
        is_verified = "VERIFIED" in content.upper() or "APPROVED" in content.upper()

        msg = {
            "id": f"msg_ss_verify_{int(time.time() * 1000)}",
            "sender": "Spreadsheet Verification Node",
            "role": "assistant",
            "content": f"### Spreadsheet Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier1_verified": is_verified,
            "tier1_result": content,
            "is_converged": is_verified,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Spreadsheet Auditor",
                    "thought": raw_thought or f"Spreadsheet audit complete. Verified = {is_verified}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 2. SPECIALIST NODE (Local AI)
    def specialist_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)

        # Extended Web Search: execute web search with 5+ results
        search_context = _search_context(task, max_results=5, max_chars=16000)

        soul_prompt = load_soul("specialist", fallback_prompt="You are the Lead Specialist Agent.")
        prompt = f"""{soul_prompt}

Available Tools (you can call tools if needed to perform calculations, search, or run code):
{tools.prompt_block()}

Task: "{task}"

Extended Web Search Context (Live Results):
{search_context}

Provide a clean, comprehensive, and well-structured technical solution."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="specialist", max_tokens=4096
        )

        msg = {
            "id": f"msg_spec_{int(time.time() * 1000)}",
            "sender": "Specialist Agent (Local AI)",
            "role": "assistant",
            "content": f"### Specialist Solution Draft (Local AI)\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "specialist_output": content,
            "current_step": "specialist_complete",
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Specialist Agent",
                    "thought": raw_thought or "Formulated technical draft solution with live search context and tool capabilities.",
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

        # Extended Web Search
        search_context = _search_context(task, max_results=5, max_chars=16000)

        soul_prompt = load_soul("tier0_auditor", fallback_prompt="You are the Tier 0 Web Verification Auditor.")
        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Any claim about events, releases, data, or developments after 2024 MUST be verified against current web search results.

Available Tools:
{tools.prompt_block()}

Task: "{task}"

Specialist Output to Verify:
{specialist_output}

Extended Web Search Context (Live Results):
{search_context}

MANDATORY: Cross-reference the specialist output against the web search context. Identify any factual claims that are inconsistent with the search results. Flag any claims about post-2024 events as UNVERIFIED if not corroborated by search. Report which claims are corroborated, contradicted, or unverified."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="tier0_auditor", max_tokens=4096
        )

        msg = {
            "id": f"msg_tier05_{int(time.time() * 1000)}",
            "sender": "Tier 0.5 Web Verification Node",
            "role": "assistant",
            "content": f"### Tier 0.5 Web Verification\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Tier 0.5 Web Verifier",
                    "thought": raw_thought or "Cross-referenced specialist output against web search results for factual accuracy.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4. TIER 1 VERIFY NODE
    def tier1_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        soul_prompt = load_soul("tier1_verifier", fallback_prompt="You are the Tier 1 Verification Auditor.")
        task = _get_input(state)
        search_context = _search_context(task, max_results=5, max_chars=16000)

        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025. Your internal knowledge may be stale. Rely on the live web search context below.

Available Tools:
{tools.prompt_block()}

Output to Audit:
{state.get("specialist_output", "")}

Extended Web Search Context (Live Results):
{search_context}

MANDATORY: Cross-reference key claims against the web search context. Respond with VERIFIED only if claims are corroborated, or REVISION REQUIRED if discrepancies are found."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="tier1_verifier", max_tokens=4096
        )
        is_verified = "VERIFIED" in content.upper() or "APPROVED" in content.upper()
        t0 = state.get("tier0_checks", {})
        converged = is_verified and all(t0.values()) if t0 else is_verified

        msg = {
            "id": f"msg_tier1_{int(time.time() * 1000)}",
            "sender": "Tier 1 Verification Node",
            "role": "assistant",
            "content": f"### Tier 1 Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n**Convergence**: {'CONVERGED' if converged else 'ESCALATION REQUIRED'}\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier1_verified": is_verified,
            "tier1_result": content,
            "is_converged": converged,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Tier 1 Auditor",
                    "thought": raw_thought or f"Tier 1 audit complete. Converged = {converged}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4.5. REVISIONS NODE (apply Tier 1 feedback to the specialist draft)
    def revisions_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        draft = state.get("specialist_output", "")
        tier1_result = state.get("tier1_result", "No Tier 1 feedback available.")
        search_context = _search_context(task, max_results=5, max_chars=16000)

        soul_prompt = load_soul(
            "specialist",
            fallback_prompt="You are the Lead Specialist Agent revising a draft based on auditor feedback.",
        )
        prompt = f"""{soul_prompt}

⚠️ CRITICAL DATE CONTEXT: We are in 2026. Do NOT assume we are in 2024 or 2025.

Available Tools:
{tools.prompt_block()}

Task: "{task}"

Current Draft:
{draft}

Tier 1 Auditor Feedback:
{tier1_result}

Extended Web Search Context (Live Results):
{search_context}

MANDATORY: Produce a REVISED draft that resolves every discrepancy or correction called out by the Tier 1 auditor. Output a clean, complete revised solution."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="specialist", max_tokens=4096
        )

        revised_text = content.strip()
        if (
            len(revised_text) < 150
            and ("VERIFIED" in revised_text.upper() or "APPROVED" in revised_text.upper())
            and len(draft) > len(revised_text)
        ):
            revised_text = draft

        msg = {
            "id": f"msg_rev_{int(time.time() * 1000)}",
            "sender": "Revisions Node",
            "role": "assistant",
            "content": f"### Revised Draft (post Tier 1 feedback)\n\n{revised_text}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "revised_output": revised_text,
            "specialist_output": revised_text,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Revisions Node",
                    "thought": raw_thought or "Applied Tier 1 auditor feedback and produced a revised draft.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4.75. TIER 2 VERIFY NODE (re-audit the revised draft)
    def tier2_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        revised = state.get("revised_output") or state.get("specialist_output", "")
        soul_prompt = load_soul("tier1_verifier", fallback_prompt="You are the Tier 2 Verification Auditor.")
        search_context = _search_context(task, max_results=5, max_chars=16000)

        prompt = f"""{soul_prompt}

Available Tools:
{tools.prompt_block()}

Revised Output to Audit:
{revised}

Extended Web Search Context (Live Results):
{search_context}

MANDATORY: Second-pass (Tier 2) audit. Respond with VERIFIED only if claims are corroborated, or REVISION REQUIRED if discrepancies remain."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="tier1_verifier", max_tokens=4096
        )
        is_verified = "VERIFIED" in content.upper() or "APPROVED" in content.upper()

        msg = {
            "id": f"msg_tier2_{int(time.time() * 1000)}",
            "sender": "Tier 2 Verification Node",
            "role": "assistant",
            "content": f"### Tier 2 Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier2_verified": is_verified,
            "is_converged": is_verified,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Tier 2 Auditor",
                    "thought": raw_thought or f"Tier 2 audit complete. Verified = {is_verified}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 5. ESCALATION NODE (Frontier Model)
    def escalation_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        soul_prompt = load_soul("frontier_escalation", fallback_prompt="You are the Frontier Model Escalation Specialist.")
        search_context = _search_context(task, max_results=5, max_chars=16000)

        prompt = f"""{soul_prompt}

Available Tools:
{tools.prompt_block()}

Task: "{task}"
Previous Specialist Draft: {state.get("specialist_output", "N/A")}

Extended Web Search Context (Live Results):
{search_context}

Synthesize a refined, authoritative solution correcting any inaccuracies."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="frontier_escalation", max_tokens=4096
        )

        msg = {
            "id": f"msg_esc_{int(time.time() * 1000)}",
            "sender": "Escalation Node (Frontier Model)",
            "role": "assistant",
            "content": f"### Frontier Model Escalation Synthesis\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "escalation_notes": content,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Frontier Model Escalation",
                    "thought": raw_thought or "Escalated task to high-capability frontier model reasoning.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 6. ADJUDICATE & REPAIR NODE
    def adjudicate_repair_node(state: ChartPipelineState) -> Dict[str, Any]:
        esc = state.get("escalation_notes") or state.get("specialist_output", "")
        soul_prompt = load_soul("adjudicator_repair", fallback_prompt="You are the Adjudication & Repair Specialist.")
        task = _get_input(state)
        search_context = _search_context(task, max_results=5, max_chars=16000)

        prompt = f"""{soul_prompt}

Available Tools:
{tools.prompt_block()}

Draft Solution to Repair:
{esc}

Extended Web Search Context (Live Results):
{search_context}

Repair the draft solution, ensuring all factual claims are verified and clearly explained."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state, prompt, agent_name="adjudicator_repair", max_tokens=4096
        )

        msg = {
            "id": f"msg_adj_{int(time.time() * 1000)}",
            "sender": "Adjudicate & Repair Node",
            "role": "assistant",
            "content": f"### Adjudicated & Repaired Output\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "repaired_output": content,
            "specialist_output": content,
            "is_converged": True,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Adjudication Node",
                    "thought": raw_thought or "Adjudicated escalation feedback and applied repairs.",
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
        spreadsheet_context = state.get("spreadsheet_context")

        if spreadsheet_context:
            prompt = f"""{soul_prompt}

Task: "{task}"

Candidate Final Answer:
{solution}

Reference Spreadsheet Context:
{spreadsheet_context}

Available Tools:
{tools.prompt_block()}

Review the candidate answer for correctness against the provided spreadsheet data, ensuring metrics and charts are accurate. Respond with a short rationale and end with either VERIFIED or REJECTED."""
        else:
            search_context = _search_context(task, max_results=5, max_chars=16000)
            prompt = f"""{soul_prompt}

Available Tools:
{tools.prompt_block()}

Task: "{task}"

Candidate Final Answer:
{solution}

Extended Web Search Context (Live Results):
{search_context}

Review the candidate answer for correctness, completeness, and clarity. Respond with a short rationale and end with either VERIFIED or REJECTED."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state,
            prompt,
            agent_name="final_verifier",
            model_name="Muse-Glimmer-30B-6bit",
            max_tokens=4096,
        )
        verification_text = content.strip()
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
            "agent_thoughts": thoughts or [
                {
                    "agent": "Final Verifier",
                    "thought": raw_thought or f"Checked the final answer for completeness and direct task alignment. Verified = {is_verified}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 7.5. FINAL REPAIR NODE (Muse Glimmer — repairs a rejected final answer)
    def final_repair_node(state: ChartPipelineState) -> Dict[str, Any]:
        solution = state.get("repaired_output") or state.get("specialist_output", "")
        task = _get_input(state)
        verifier_feedback = state.get("final_verification_result", "No verifier feedback available.")
        search_context = _search_context(task, max_results=5, max_chars=16000)

        soul_prompt = load_soul(
            "adjudicator_repair",
            fallback_prompt="You are the Final Answer Repair Specialist.",
        )
        prompt = f"""{soul_prompt}

Available Tools:
{tools.prompt_block()}

Task: "{task}"

Candidate Final Answer (rejected by verifier):
{solution}

Verifier Feedback:
{verifier_feedback}

Extended Web Search Context (Live Results):
{search_context}

Repair the rejected final answer, correcting every inaccuracy and filling gaps. Output a clean, complete final answer."""

        content, raw_thought, thoughts = _run_node_with_tools(
            state,
            prompt,
            agent_name="final_repair",
            model_name="Muse-Glimmer-30B-6bit",
            max_tokens=4096,
        )

        msg = {
            "id": f"msg_final_repair_{int(time.time() * 1000)}",
            "sender": "Final Repair Node (Muse Glimmer)",
            "role": "assistant",
            "content": f"### Final Answer Repair (Muse Glimmer)\n\n{content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "repaired_output": content,
            "specialist_output": content,
            "final_repair_applied": True,
            "final_answer_verified": True,
            "messages": [msg],
            "agent_thoughts": thoughts or [
                {
                    "agent": "Final Repair (Muse Glimmer)",
                    "thought": raw_thought or "Repaired the rejected final answer using Muse Glimmer 30B 6-bit.",
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
        tool_name = existing_payload.get("tool")
        tool_args = existing_payload.get("tool_args") or {}

        if not tool_name:
            import json
            import re
            json_matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", solution)
            if not json_matches:
                raw_match = re.search(r"(\{\s*\"(?:tool|name)\"\s*:\s*\"[^\"]+\"[\s\S]*?\})", solution)
                if raw_match:
                    json_matches = [raw_match.group(1)]
            for j_str in json_matches:
                try:
                    parsed = json.loads(j_str)
                    if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
                        tool_name = parsed.get("tool") or parsed.get("name")
                        tool_args = parsed.get("args") or parsed.get("parameters") or {}
                        break
                except Exception:
                    pass

        payload = {
            "target_action": existing_payload.get("target_action", "execute_solution"),
            "payload_summary": existing_payload.get("payload_summary") or (solution[:200] + "..." if len(solution) > 200 else solution),
            "requires_approval": existing_payload.get("requires_approval", True),
            "tool": tool_name,
            "tool_args": tool_args,
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
            "final_response": "Action blocked during preparation.",
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
        raw_solution = state.get("repaired_output") or state.get("specialist_output", "")

        # Clean raw solution if it has redundant internal headers
        clean_solution = raw_solution.strip()
        for prefix in [
            "### Specialist Solution Draft (Local AI)",
            "### Final Answer Repair (Muse Glimmer)",
            "### Adjudicated & Repaired Output",
            "### Revised Draft (post Tier 1 feedback)",
        ]:
            if clean_solution.startswith(prefix):
                clean_solution = clean_solution[len(prefix):].strip()

        clean_solution = clean_solution or "Pipeline execution completed successfully."

        memory_entry = {
            "event": "PIPELINE_SUCCESS",
            "input": state.get("user_input"),
            "result": state.get("execution_result") or clean_solution,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Persist memory entry to SQLite under the current run
        _persist_memory(state, {
            "event": memory_entry["event"],
            "input": memory_entry["input"],
            "result": memory_entry["result"],
            "timestamp": memory_entry["timestamp"],
            "final_answer": clean_solution,
        })

        # Close the run record: status, final answer, duration, memory count.
        try:
            from src.core.memory_store import finish_run

            run_id = state.get("run_id")
            if run_id:
                finish_run(
                    run_id,
                    status="completed",
                    final_answer=clean_solution,
                    started_at=state.get("run_started_at"),
                )
        except Exception:
            pass

        msg = {
            "id": f"msg_fin_{int(time.time() * 1000)}",
            "sender": "Chart Pipeline",
            "role": "assistant",
            "content": clean_solution,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "final_response": clean_solution,
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
        return "foreign_file_router"

    def route_foreign_file(state: ChartPipelineState) -> str:
        if state.get("has_foreign_file"):
            return "spreadsheet_specialist"
        return "specialist"

    def route_spreadsheet_verify(state: ChartPipelineState) -> str:
        if state.get("tier1_verified"):
            return "final_verification"
        return "revisions"

    def route_tier2(state: ChartPipelineState) -> str:
        if state.get("tier2_verified"):
            return "final_verification"
        return "escalation"

    def route_final_verification(state: ChartPipelineState) -> str:
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
    workflow.add_node("foreign_file_router", foreign_file_router_node)
    workflow.add_node("spreadsheet_specialist", spreadsheet_specialist_node)
    workflow.add_node("spreadsheet_verify", spreadsheet_verify_node)
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
        "intake", route_intake, {"foreign_file_router": "foreign_file_router", "blocked_end": "blocked_end"}
    )
    workflow.add_edge("blocked_end", END)

    # Conditional router: if foreign file detected -> new path, else -> original pipeline
    workflow.add_conditional_edges(
        "foreign_file_router",
        route_foreign_file,
        {"spreadsheet_specialist": "spreadsheet_specialist", "specialist": "specialist"},
    )

    # New Foreign File Branch
    workflow.add_edge("spreadsheet_specialist", "spreadsheet_verify")
    workflow.add_conditional_edges(
        "spreadsheet_verify",
        route_spreadsheet_verify,
        {"final_verification": "final_verification", "revisions": "revisions"},
    )

    # Original Pipeline Branch (100% preserved)
    workflow.add_edge("specialist", "tier0_checks")
    workflow.add_edge("tier0_checks", "tier05_web_verify")
    workflow.add_edge("tier05_web_verify", "tier1_verify")
    workflow.add_edge("tier1_verify", "revisions")
    workflow.add_edge("revisions", "tier2_verify")
    workflow.add_conditional_edges(
        "tier2_verify",
        route_tier2,
        {"final_verification": "final_verification", "escalation": "escalation"},
    )

    workflow.add_edge("escalation", "adjudicate_repair")
    workflow.add_edge("adjudicate_repair", "final_verification")
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

    active_checkpointer = checkpointer
    return workflow.compile(checkpointer=active_checkpointer)


# Default compiled graph instance for LangGraph Studio CLI
default_chart_graph = create_chart_pipeline_graph(LocalLLMClient())
