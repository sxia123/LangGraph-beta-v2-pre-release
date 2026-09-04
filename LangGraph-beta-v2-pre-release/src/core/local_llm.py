import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from src.core.web_search import format_search_results, perform_web_search


def parse_agent_model_map() -> Dict[str, str]:
    """Parse environment-configured per-agent model overrides.

    Example values:
      AGENT_MODEL_MAP='supervisor:gpt-4,critic:gpt-4o'
      AGENT_MODEL_MAP='{"supervisor": "gpt-4", "critic": "gpt-4o"}'
    """
    raw = os.getenv("AGENT_MODEL_MAP", "")
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k).lower(): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    result: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[key.strip().lower()] = value.strip()
    return result


class Message(BaseModel):
    id: str
    sender: str
    role: str
    content: str
    timestamp: str
    images: Optional[List[str]] = None


class LLMResponse(BaseModel):
    content: str
    thought: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LocalLLMConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "omlx"))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "Qwen3.8-27B-oQ6-mtp")
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    ollama_model_name: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:9b")
    )
    enable_ollama_fallback: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("1", "true", "yes")
    )
    ollama_timeout: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "60"))
    )
    agent_models: Dict[str, str] = Field(default_factory=parse_agent_model_map)
    temperature: float = 0.2
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    # agent_models maps lowercase agent names to model names for per-agent overrides.
    # These values are used by LocalLLMClient._resolve_model_name().


class LocalLLMClient:
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        if config is None:
            config = LocalLLMConfig()
        self.config = config

    def search_web(
        self,
        query: str,
        max_results: int = 5,
        expand_queries: bool = True,
        max_tokens: int = 12000,
    ) -> str:
        """Executes enhanced DuckDuckGo search and returns formatted markdown results.

        Wrapped in a hard wall-clock timeout so a hung/slow DuckDuckGo request
        can never block the pipeline (and the SSE stream) indefinitely.
        """
        if self.config.provider == "mock":
            return f"### Mock Web Search Results for '{query}'\n- Comprehensive overview of {query}\n- Key findings and domain facts for pipeline state."

        try:
            timeout = int(os.getenv("WEB_SEARCH_TIMEOUT", "45"))

        except (TypeError, ValueError):
            timeout = 45

        result_box: Dict[str, Any] = {}

        def _run() -> None:
            try:
                results = perform_web_search(
                    query,
                    max_results=max_results,
                    expand_queries=expand_queries,
                    max_tokens=max_tokens,
                )
                result_box["ok"] = True
                result_box["text"] = format_search_results(results, max_tokens=max_tokens)
            except Exception as err:  # noqa: BLE001
                result_box["ok"] = False
                result_box["error"] = str(err)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout)

        if worker.is_alive():
            logging.warning("Web search timed out after %ss for query: %s", timeout, query)
            return f"Web search timed out after {timeout}s — FLAG all claims as UNVERIFIED."

        if not result_box.get("ok", False):
            logging.warning("Web search failed for query %s: %s", query, result_box.get("error"))
            return "Web search unavailable — FLAG all claims as UNVERIFIED."

        return result_box.get("text", "No web search results found.")

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k) and v is not None:
                setattr(self.config, k, v)

    def build_search_context(self, search_text: str, max_chars: Optional[int] = None) -> str:
        """Preserve a larger chunk of search context for agent prompts without overloading them."""
        if not search_text:
            return "No web search results found."

        limit = max_chars or int(os.getenv("AGENT_SEARCH_CONTEXT_CHARS", "8000"))
        if len(search_text) <= limit:
            return search_text

        return (
            f"{search_text[:limit].rstrip()}\n\n[Context truncated to {limit} characters; "
            f"the remaining search results were omitted to keep the prompt compact.]"
        )

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _get_openai_url(self, endpoint: str) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/{endpoint.lstrip('/')}"
        return f"{base}/v1/{endpoint.lstrip('/')}"

    def _resolve_model_name(
        self, agent: Optional[str] = None, model_name: Optional[str] = None
    ) -> str:
        """Choose the most specific model name available.

        Priority:
        1. Explicit request-level model_name
        2. Agent-specific override from config.agent_models
        3. Default config.model_name
        """
        if model_name:
            return model_name
        if agent:
            lookup = self.config.agent_models.get(agent.lower())
            if lookup:
                return lookup
        return self.config.model_name

    def _normalize_ollama_model(self, model_name: Optional[str]) -> str:
        """Normalizes model aliases like 'qwen3.5-9b' or 'qwen3.5' to actual Ollama model tags."""
        if not model_name:
            return self.config.ollama_model_name or "qwen3.5:9b"
        m_clean = model_name.strip()
        m_lower = m_clean.lower()
        if any(alias in m_lower for alias in ("qwen3.5", "qwen35", "qwen-3.5", "qwen_3.5")):
            return "qwen3.5:9b"
        if "2.5-coder" in m_lower or "coder" in m_lower:
            return "qwen2.5-coder:7b"
        if "2.5-7b" in m_lower or "2.5:7b" in m_lower:
            return "qwen2.5:7b"
        if ":" in m_clean:
            return m_clean
        return self.config.ollama_model_name or "qwen3.5:9b"

    def ping(self) -> Dict[str, Any]:
        if self.config.provider == "mock":
            return {
                "ok": True,
                "message": "Mock engine active (Offline visual simulation)",
                "models": [
                    "Qwen3.8-27B-oQ6-mtp",
                    "qwen3.5:9b",
                    "Qwen2.5-VL-72B-Instruct",
                    "Qwen2.5-VL-7B-Instruct",
                    "Qwen2.5-Coder-32B-Instruct",
                    "mock-llama3.2",
                    "mock-deepseek-r1",
                ],
            }

        headers = self._get_headers()
        base = self.config.base_url.rstrip("/")
        detected_models: List[str] = []
        primary_connected = False
        ollama_connected = False
        ollama_models: List[str] = []

        # 1. Probe Ollama endpoint
        ollama_base = (self.config.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        try:
            res_ollama = requests.get(f"{ollama_base}/api/tags", headers=headers, timeout=3)
            if res_ollama.status_code == 200:
                ollama_connected = True
                ollama_models = [
                    m.get("name")
                    for m in res_ollama.json().get("models", [])
                    if isinstance(m, dict) and m.get("name")
                ]
        except Exception:
            pass

        if self.config.provider == "ollama":
            if ollama_connected:
                return {
                    "ok": True,
                    "message": f"Connected to Ollama ({len(ollama_models)} models detected)",
                    "models": ollama_models or [self.config.ollama_model_name],
                    "provider": "ollama",
                }
            return {
                "ok": False,
                "message": f"Could not connect to Ollama at {ollama_base}",
                "models": [],
                "provider": "ollama",
            }

        # 2. Probe OpenAI / oMLX / vLLM / LM Studio models endpoints
        urls_to_try = [
            self._get_openai_url("models"),
            f"{base}/models",
            f"{base}/v1/models",
        ]
        seen_urls: List[str] = []
        for u in urls_to_try:
            if u not in seen_urls:
                seen_urls.append(u)

        for models_url in seen_urls:
            try:
                res = requests.get(models_url, headers=headers, timeout=3)
                if res.status_code == 200:
                    payload = res.json()
                    raw_list = (
                        payload.get("data") or payload.get("models") or []
                        if isinstance(payload, dict)
                        else payload
                    )
                    if isinstance(raw_list, list):
                        for item in raw_list:
                            if isinstance(item, dict):
                                m_id = item.get("id") or item.get("name") or item.get("model")
                                if m_id:
                                    detected_models.append(str(m_id))
                            elif isinstance(item, str):
                                detected_models.append(item)
                    if detected_models:
                        primary_connected = True
                        break
            except Exception:
                continue

        all_models = list(dict.fromkeys(detected_models + ollama_models))

        if primary_connected:
            return {
                "ok": True,
                "message": f"Connected to {self.config.provider} ({len(detected_models)} models available, Ollama fallback {'ready' if ollama_connected else 'offline'})",
                "models": all_models or [self.config.model_name],
                "provider": self.config.provider,
                "ollama_available": ollama_connected,
            }

        if ollama_connected:
            return {
                "ok": True,
                "message": f"Primary {self.config.provider} offline; Ollama fallback active ({len(ollama_models)} models detected: {', '.join(ollama_models)})",
                "models": ollama_models,
                "provider": "ollama (fallback)",
                "ollama_available": True,
                "fallback_active": True,
            }

        if self.config.model_name:
            return {
                "ok": False,
                "message": f"Could not connect to {self.config.base_url} or Ollama at {ollama_base}",
                "models": [self.config.model_name],
            }

        return {"ok": False, "message": f"Could not list models from {self.config.base_url}"}

    def generate_completion(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        agent: Optional[str] = None,
        model_name: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        if self.config.provider == "mock":
            return self._generate_mock_response(
                system_prompt, messages, available_tools, agent=agent, images=images
            )

        try:
            effective_model = self._resolve_model_name(agent=agent, model_name=model_name)
            # If images are provided at the call level, inject them into the last user message if not already present
            prepared_messages = list(messages)
            if images and prepared_messages:
                last_idx = len(prepared_messages) - 1
                for idx in range(last_idx, -1, -1):
                    if prepared_messages[idx].get("role") == "user":
                        existing_imgs = prepared_messages[idx].get("images") or []
                        prepared_messages[idx] = {
                            **prepared_messages[idx],
                            "images": list(set(existing_imgs + images)),
                        }
                        break
            elif images and not prepared_messages:
                prepared_messages = [{"role": "user", "content": system_prompt, "images": images}]

            if self.config.provider == "ollama":
                target_ollama_model = self._normalize_ollama_model(
                    effective_model or self.config.ollama_model_name
                )
                return self._call_ollama_api(
                    system_prompt,
                    prepared_messages,
                    model_name=target_ollama_model,
                    max_tokens=max_tokens,
                )
            else:
                try:
                    return self._call_openai_compatible_api(
                        system_prompt,
                        prepared_messages,
                        model_name=effective_model,
                        max_tokens=max_tokens,
                    )
                except Exception as primary_err:
                    if self.config.enable_ollama_fallback:
                        try:
                            target_ollama_model = self._normalize_ollama_model(
                                effective_model
                                if ("qwen" in str(effective_model).lower() or ":" in str(effective_model))
                                else self.config.ollama_model_name
                            )
                            ollama_res = self._call_ollama_api(
                                system_prompt,
                                prepared_messages,
                                model_name=target_ollama_model,
                                max_tokens=max_tokens,
                                base_url=self.config.ollama_base_url,
                            )
                            fallback_tag = f"Executed via Ollama Fallback ({target_ollama_model})"
                            ollama_res.thought = (
                                f"[{fallback_tag}] {ollama_res.thought}"
                                if ollama_res.thought
                                else fallback_tag
                            )
                            return ollama_res
                        except Exception:
                            pass
                    raise primary_err
        except Exception as err:
            mock_res = self._generate_mock_response(system_prompt, messages, available_tools)
            mock_res.content = f"[Notice: Local LLM fallback ({str(err)}). Showing simulated response]\n\n{mock_res.content}"
            return mock_res

    def _call_ollama_api(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        model_name: str,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> LLMResponse:
        resolved_model = self._normalize_ollama_model(
            model_name or self.config.ollama_model_name or "qwen3.5:9b"
        )
        if messages:
            formatted = [{"role": "system", "content": system_prompt}]
            for m in messages:
                content = m.get("content", "")
                msg_images = m.get("images") or []
                msg_entry: Dict[str, Any] = {"role": m.get("role", "user")}
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    msg_entry["content"] = " ".join(text_parts)
                else:
                    msg_entry["content"] = str(content)

                if msg_images:
                    cleaned_imgs = []
                    for img in msg_images:
                        if "," in img:
                            cleaned_imgs.append(img.split(",", 1)[1])
                        else:
                            cleaned_imgs.append(img)
                    msg_entry["images"] = cleaned_imgs
                formatted.append(msg_entry)
        else:
            formatted = [{"role": "user", "content": system_prompt}]

        target_base = (
            base_url
            or self.config.ollama_base_url
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        opts = {"temperature": self.config.temperature}
        num_tokens = max_tokens or self.config.max_tokens
        if num_tokens:
            opts["num_predict"] = num_tokens

        timeout_sec = self.config.ollama_timeout or 60
        res = requests.post(
            f"{target_base}/api/chat",
            headers=self._get_headers(),
            json={
                "model": resolved_model,
                "messages": formatted,
                "stream": False,
                "options": opts,
            },
            timeout=timeout_sec,
        )
        res.raise_for_status()
        msg_obj = res.json().get("message", {})
        raw = msg_obj.get("content", "")
        reasoning = (
            msg_obj.get("reasoning_content")
            or msg_obj.get("thought")
            or msg_obj.get("thinking")
        )
        return self._parse_response(raw, reasoning=reasoning)

    def _call_openai_compatible_api(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        model_name: str,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        if messages:
            formatted = [{"role": "system", "content": system_prompt}]
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                msg_images = m.get("images") or []
                if isinstance(content, list):
                    formatted.append({"role": role, "content": content})
                elif msg_images and role == "user":
                    # Build OpenAI/oMLX vision content array
                    parts: List[Dict[str, Any]] = []
                    if content:
                        parts.append({"type": "text", "text": str(content)})
                    for img in msg_images:
                        url_val = (
                            img
                            if (img.startswith("data:") or img.startswith("http"))
                            else f"data:image/jpeg;base64,{img}"
                        )
                        parts.append({"type": "image_url", "image_url": {"url": url_val}})
                    formatted.append({"role": role, "content": parts})
                else:
                    formatted.append({"role": role, "content": content})
        else:
            formatted = [{"role": "user", "content": system_prompt}]

        url = self._get_openai_url("chat/completions")
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": formatted,
            "temperature": self.config.temperature,
        }
        num_tokens = max_tokens or self.config.max_tokens
        if num_tokens:
            payload["max_tokens"] = num_tokens

        current_max_tokens = max_tokens or self.config.max_tokens or 1024
        for attempt in range(2):
            res = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )
            res.raise_for_status()
            data = res.json()
            choices = data.get("choices", [])
            if not choices:
                return self._parse_response("")

            choice = choices[0]
            msg_obj = choice.get("message", {}) if choices else {}
            raw = msg_obj.get("content", "")
            reasoning = (
                msg_obj.get("reasoning_content")
                or msg_obj.get("reasoning")
                or msg_obj.get("thought")
            )
            finish_reason = choice.get("finish_reason")
            if finish_reason != "length" or attempt >= 1:
                return self._parse_response(raw, reasoning=reasoning)

            current_max_tokens = max(1024, min(16384, int(current_max_tokens) * 2))
            payload["max_tokens"] = current_max_tokens

        return self._parse_response("")

    def _parse_response(self, raw: str, reasoning: Optional[str] = None) -> LLMResponse:
        thought = reasoning.strip() if reasoning and isinstance(reasoning, str) else None
        content = raw or ""

        # Extract <think>...</think>, <thought>...</thought>, or <reasoning>...</reasoning>
        think_patterns = [
            r"<think>([\s\S]*?)</think>",
            r"<thought>([\s\S]*?)</thought>",
            r"<reasoning>([\s\S]*?)</reasoning>",
        ]
        for pattern in think_patterns:
            think_match = re.search(pattern, content, re.IGNORECASE)
            if think_match:
                extracted = think_match.group(1).strip()
                thought = f"{thought}\n\n{extracted}".strip() if thought else extracted
                content = re.sub(pattern, "", content, flags=re.IGNORECASE).strip()

        tool_calls = []
        json_matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if not json_matches:
            raw_match = re.search(r"(\{\s*\"(?:tool|name)\"\s*:\s*\"[^\"]+\"[\s\S]*?\})", content)
            if raw_match:
                json_matches = [raw_match.group(1)]

        for j_str in json_matches:
            try:
                parsed = json.loads(j_str)
                if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
                    tool_calls.append(
                        {
                            "id": f"call_{int(time.time() * 1000)}",
                            "name": parsed.get("tool") or parsed.get("name"),
                            "args": parsed.get("args") or parsed.get("parameters") or {},
                        }
                    )
            except Exception:
                pass

        return LLMResponse(
            content=content, thought=thought, tool_calls=tool_calls if tool_calls else None
        )

    def _generate_mock_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        available_tools: Optional[List[str]] = None,
        agent: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        time.sleep(0.3)
        last_user = ""
        has_tool_result = False
        has_images = bool(images)
        for m in messages:
            if isinstance(m, dict):
                content_val = str(m.get("content", ""))
                role = m.get("role")
                if m.get("images"):
                    has_images = True
                if "Tool [" in content_val or "Tool Result" in content_val or role == "tool":
                    has_tool_result = True
                elif role == "user" and not last_user:
                    last_user = content_val
        if not last_user:
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    content_val = str(m.get("content", ""))
                    if not ("Tool [" in content_val or "Tool Result" in content_val):
                        last_user = content_val
                        break
        if not last_user and system_prompt:
            last_user = system_prompt

        sys_lower = system_prompt.lower()
        user_lower = last_user.lower()
        agent_name = (agent or "").lower()

        if agent_name in ["specialist", "frontier_escalation", "adjudicator_repair"]:
            pass  # Proceed directly to specialist response below
        elif (
            agent_name in ["tier0_auditor", "tier1_verifier", "final_verifier", "verifier", "auditor", "critic"]
            or any(k in sys_lower for k in ["you are the tier 0", "you are the tier 1", "you are the final answer verifier", "mandatory: cross-reference", "mandatory audit criteria"])
        ):
            if any(k in sys_lower or k in user_lower for k in ["spreadsheet", "sheet", "document", "slideshow", "slide", "image", "photo"]):
                return LLMResponse(
                    content="VERIFIED - All reference metrics, document takeaways, slide roadmaps, and visual charts match the provided inputs.",
                    thought="Audited specialist draft against provided reference context; confirmed calculations, architecture, and visual charts are consistent.",
                )
            return LLMResponse(
                content="VERIFIED - All factual claims have been corroborated by search context.",
                thought="Reviewed candidate answer against reference context and verified all claims.",
            )

        if "specialist" in sys_lower or agent_name == "specialist":
            # 1. Slideshow & presentation analysis
            if "mandatory slideshow" in sys_lower or any(k in user_lower for k in [".pptx", ".ppt", "deciphered slideshow"]):
                content_text = (
                    "### Executive Summary\n"
                    "Synthesis of the slide presentation outlines strategic roadmap milestones, quarterly achievements, and resource allocation plans.\n\n"
                    "### Slide Deck Breakdown & Agenda\n"
                    "- **Slide 1 (Executive Summary)**: Q1 revenue targets exceeded across major AI platform segments.\n"
                    "- **Slide 2 (Architecture Roadmap)**: Scaling agentic pipelines and persistent memory infrastructure.\n"
                    "- **Slide 3 (Execution Plan)**: Cross-functional deployment with enterprise audit checkpoints.\n\n"
                    "### Presentation Roadmap & Timeline\n"
                    "```mermaid\n"
                    "timeline\n"
                    "    title Q1 Presentation Strategic Roadmap\n"
                    "    section Milestone 1\n"
                    "        Week 1-2 : Intake & Multi-format Parsing\n"
                    "    section Milestone 2\n"
                    "        Week 3-4 : Verification Gate Convergence\n"
                    "    section Milestone 3\n"
                    "        Week 5-6 : Enterprise Deployment\n"
                    "```\n\n"
                    "### Strategic Takeaways\n"
                    "1. Align operational roadmaps directly with quarterly slide deck objectives.\n"
                    "2. Maintain slide milestone pacing to meet project delivery schedules."
                )
                return LLMResponse(
                    content=content_text,
                    thought="Synthesized presentation deck and created Mermaid timeline chart.",
                )

            # 2. Spreadsheet analysis & variance charts
            if "mandatory spreadsheet" in sys_lower or any(k in user_lower for k in [".xlsx", ".xls", ".csv", ".tsv", "deciphered spreadsheet analysis"]):
                content_text = (
                    "### Executive Summary\n"
                    "Analysis of the provided spreadsheet dataset reveals strong operational and financial performance across reporting business units.\n\n"
                    "### Key Metrics & Deciphered Variance\n"
                    "- **Total Actual Revenue**: $20,640,000 against an $18,200,000 target (+13.4% overall portfolio overachievement).\n"
                    "- **Top Growth Driver**: Autonomous Agents SDK and AI Enterprise Suite exhibited the highest YoY adoption rates.\n"
                    "- **Operating Discipline**: Expenses stayed strictly within forecast thresholds.\n\n"
                    "### Segment Performance Breakdown\n"
                    "| Region / Segment | Target ($) | Actual ($) | Variance ($) | Status |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| North America Enterprise | $4,500,000 | $5,250,000 | +$750,000 | Exceeded |\n"
                    "| North America Mid-Market | $2,200,000 | $2,480,000 | +$280,000 | Exceeded |\n"
                    "| Europe / UK Financial AI | $3,800,000 | $3,650,000 | -$150,000 | On Track |\n"
                    "| Asia Pacific Agents SDK | $1,900,000 | $2,420,000 | +$520,000 | Exceeded |\n"
                    "| Global Strategic Accounts | $5,000,000 | $5,950,000 | +$950,000 | Exceeded |\n\n"
                    "### Visual Performance Chart\n"
                    "```mermaid\n"
                    "xychart-beta\n"
                    '    title "Q1 Performance: Target vs Actual Revenue ($M)"\n'
                    '    x-axis ["NA Ent", "NA Mid", "EU Fin", "APAC SDK", "Global"]\n'
                    '    y-axis "Revenue ($M)" 0 --> 7\n'
                    "    bar [4.5, 2.2, 3.8, 1.9, 5.0]\n"
                    "    bar [5.25, 2.48, 3.65, 2.42, 5.95]\n"
                    "```\n\n"
                    "### Strategic Takeaways\n"
                    "1. Expand high-density compute clusters to sustain regional AI platform demand.\n"
                    "2. Replicate the APAC go-to-market playbook across emerging enterprise accounts."
                )
                return LLMResponse(
                    content=content_text,
                    thought="Deciphered multi-sheet spreadsheet dataset, synthesized variance calculations, and constructed interactive Mermaid performance charts.",
                )
            # 3. Document analysis & diagramming
            if "mandatory document" in sys_lower or any(k in user_lower for k in [".pdf", ".docx", "deciphered word document", "deciphered pdf document"]):
                content_text = (
                    "### Executive Summary\n"
                    "Comprehensive synthesis of the provided document highlights key strategic initiatives, governance controls, and operational benchmarks.\n\n"
                    "### Key Document Highlights & Policies\n"
                    "- **Core Strategy**: Multi-agent verification protocols established across all production deployment layers.\n"
                    "- **Risk Controls**: Deterministic Tier 0 sanity checks and multi-agent consensus mandated before execution.\n"
                    "- **Performance Target**: Maintain sub-second routing with complete SQLite checkpointing.\n\n"
                    "### Document Architecture Diagram\n"
                    "```mermaid\n"
                    "graph TD\n"
                    "    A[Document Intake] --> B[Multi-Tier Verification]\n"
                    "    B --> C[Audit Review]\n"
                    "    C --> D[Approved Release]\n"
                    "```\n\n"
                    "### Strategic Takeaways\n"
                    "1. Enforce documented compliance boundaries across all automated workflows.\n"
                    "2. Implement regular audits against reference documentation standards."
                )
                return LLMResponse(
                    content=content_text,
                    thought="Synthesized document sections and created architectural Mermaid diagram.",
                )

            # 4. Photos & Image visual analysis
            if "mandatory visual" in sys_lower or has_images or any(k in user_lower for k in ["visual image asset", ".png", ".jpg", ".jpeg", "visual photo"]):
                content_text = (
                    "### Executive Summary\n"
                    "Visual analysis of the attached image/photo reveals structured workflow components, key relationships, and performance data.\n\n"
                    "### Visual Observations & Feature Map\n"
                    "- **Visual Clarity**: High-contrast layout with distinct process nodes and operational hierarchies.\n"
                    "- **Key Focal Points**: Center workflow transitions connecting intake to verification and action execution.\n"
                    "- **Metrics & Indicators**: Color-coded indicators confirm optimal system status.\n\n"
                    "### Modeled Visual Architecture\n"
                    "```mermaid\n"
                    "flowchart LR\n"
                    "    VisualInput[Visual Media / Photo] --> FeatureExtraction[Feature & Layout Extraction]\n"
                    "    FeatureExtraction --> DiagramModeling[Diagram Modeling]\n"
                    "    DiagramModeling --> VerifiedOutput[Verified Chart Output]\n"
                    "```\n\n"
                    "### Strategic Takeaways\n"
                    "1. Integrate multimodal vision analysis with deterministic verification pipelines.\n"
                    "2. Generate visual Mermaid models for complex diagrammatic photo inputs."
                )
                return LLMResponse(
                    content=content_text,
                    thought="Analyzed visual image assets and generated Mermaid workflow diagram.",
                )

            return LLMResponse(
                content=f"### Solution Strategy\nEngineered specialized resolution for task:\n- Input: {last_user[:60]}\n- Architecture verified.",
                thought=f"Drafted comprehensive technical solution for '{last_user[:40]}'.",
            )

        # Direct Chat / Vision with images
        if has_images or "image / photo attachment" in user_lower or "photo" in user_lower:
            return LLMResponse(
                content=(
                    "### Visual Analysis & Interpretation\n"
                    "Based on the visual analysis of the attached image:\n\n"
                    "- **Subject & Layout**: The image presents a multi-tier technical architecture and performance dashboard.\n"
                    "- **Key Visual Elements**: Shows data flow pipelines, verification nodes, and performance metrics.\n"
                    "- **Status**: All visual structures and indicators are consistent with high-integrity operational standards."
                ),
                thought="Analyzed visual elements, layout structure, and optical contents from attached image.",
            )

        # Handle tool calling when available_tools provided and not yet executed
        if available_tools and not has_tool_result:
            expr_match = re.search(r"(\d+(?:\s*[\+\-\*\/\^]\s*\d+)+)", last_user)
            if "math_eval" in available_tools and (expr_match or any(op in user_lower for op in ["math_eval", "calculate", "calc"])):
                clean_expr = expr_match.group(1).strip() if expr_match else "2 + 2"
                return LLMResponse(
                    content=f'```json\n{{"tool": "math_eval", "args": {{"expression": "{clean_expr}"}}}}\n```',
                    thought=f"Request requires mathematical evaluation. Calling math_eval with expression: {clean_expr}",
                    tool_calls=[{"id": f"call_{int(time.time()*1000)}", "name": "math_eval", "args": {"expression": clean_expr}}],
                )
            if "python_repl" in available_tools and any(kw in user_lower for kw in ["python_repl", "python", "script", "run code"]):
                py_code = 'print("Executed successfully via Python REPL")'
                return LLMResponse(
                    content=f'```json\n{{"tool": "python_repl", "args": {{"code": "{py_code}"}}}}\n```',
                    thought="Request requires Python execution. Calling python_repl.",
                    tool_calls=[{"id": f"call_{int(time.time()*1000)}", "name": "python_repl", "args": {"code": py_code}}],
                )
            if "web_search" in available_tools and any(kw in user_lower for kw in ["search", "find", "lookup", "who is", "what is"]):
                return LLMResponse(
                    content=f'```json\n{{"tool": "web_search", "args": {{"query": "{last_user[:50]}"}}}}\n```',
                    thought=f"Request requires live web search. Calling web_search for query: {last_user[:50]}",
                    tool_calls=[{"id": f"call_{int(time.time()*1000)}", "name": "web_search", "args": {"query": last_user[:50]}}],
                )

        if has_tool_result:
            return LLMResponse(
                content=f"Here is the synthesized answer for '{last_user}':\n\nBased on verified findings, the request has been processed successfully with accurate information.",
                thought="Synthesized clean final answer from verified tool execution output.",
            )

        if "supervisor" in sys_lower:
            if "critic approved: yes" in sys_lower:
                return LLMResponse(
                    content="writer",
                    thought="All quality criteria passed. Routing to Writer for final synthesis.",
                )
            if "solution/draft generated: yes" in sys_lower:
                return LLMResponse(
                    content="critic",
                    thought="Draft solution generated. Routing to Critic for validation and edge-case audit.",
                )
            if "research done: yes" in sys_lower:
                return LLMResponse(
                    content="coder",
                    thought="Domain research completed. Routing to Coder to implement solution.",
                )
            if any(k in user_lower for k in ["search", "find", "research"]):
                return LLMResponse(
                    content="researcher",
                    thought="User request requires live factual knowledge. Delegating research to DuckDuckGo search.",
                )
            if any(k in user_lower for k in ["code", "script", "python", "react"]):
                return LLMResponse(
                    content="coder",
                    thought="Technical development task identified. Assigning to Lead Developer node.",
                )
            return LLMResponse(
                content="FINISH",
                thought="Task objectives achieved. Finalizing multi-agent execution pipeline.",
            )

        if "researcher" in sys_lower:
            return LLMResponse(
                content=f"### Research Summary\nAnalyzed context for: '{last_user}'.\n- Local inference active.\n- LangGraph state channel verified.",
                thought=f"Parsed user query '{last_user[:40]}...', queried DuckDuckGo, and synthesized verified findings.",
            )

        if "coder" in sys_lower or "developer" in sys_lower:
            return LLMResponse(
                content=f"```python\n# Solution generated by Coder Node (Python LangGraph)\ndef execute_task():\n    return {{'status': 'success', 'input': '{last_user}'}}\n```",
                thought="Designed modular, type-safe Python implementation satisfying requirements.",
            )

        if "critic" in sys_lower or "reviewer" in sys_lower or "auditor" in sys_lower:
            return LLMResponse(
                content="### Review\n- ✅ Type Safety & Correctness\n- ✅ Performance & Modularity\n\nVerdict: APPROVED.",
                thought="Performed thorough code audit for edge cases, error handling, and type safety.",
            )

        if "specialist" in sys_lower:
            return LLMResponse(
                content=f"### Solution Strategy\nEngineered specialized resolution for task:\n- Input: {last_user[:60]}\n- Architecture verified.",
                thought="Analyzed task specification and formulated optimal technical solution.",
            )

        if "recalled memory context:" in user_lower or "recalled memory context:" in sys_lower:
            raw_text = last_user if "recalled memory context:" in user_lower else system_prompt
            mem_lines = []
            for line in raw_text.splitlines():
                if line.strip().startswith("- [") and "]" in line:
                    mem_lines.append(line.strip())
            summary_bullet = "\n".join(mem_lines[:3]) if mem_lines else "- Retrieved historical records from SQLite."
            cleaned_query = last_user.split("[Recalled")[0].strip() if "[Recalled" in last_user else last_user[:50]
            return LLMResponse(
                content=(
                    f"### Contextual Recall & Memory Response\n"
                    f"Based on recalled memory from previous sessions:\n\n"
                    f"{summary_bullet}\n\n"
                    f"**Analysis**: The historical context directly informs the current query: '{cleaned_query}'."
                ),
                thought="Detected recalled memory context in user prompt; incorporated historical facts into final synthesized response.",
            )

        return LLMResponse(
            content=f"Processed response for task: {last_user[:50]}...",
            thought="Analyzed user query context, evaluated relevant factors, and formulated final response.",
        )
