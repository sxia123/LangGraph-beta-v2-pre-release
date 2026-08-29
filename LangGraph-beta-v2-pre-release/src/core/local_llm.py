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

    def ping(self) -> Dict[str, Any]:
        if self.config.provider == "mock":
            return {
                "ok": True,
                "message": "Mock engine active (Offline visual simulation)",
                "models": ["mock-llama3.2", "mock-qwen2.5-coder", "mock-deepseek-r1"],
            }

        try:
            if self.config.provider == "ollama":
                base = self.config.base_url.rstrip("/")
                res = requests.get(f"{base}/api/tags", headers=self._get_headers(), timeout=5)
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    return {
                        "ok": True,
                        "message": f"Connected to Ollama ({len(models)} models detected)",
                        "models": models,
                    }

            models_url = self._get_openai_url("models")
            res = requests.get(models_url, headers=self._get_headers(), timeout=5)
            if res.status_code == 200:
                models = [m.get("id") for m in res.json().get("data", [])]
                return {
                    "ok": True,
                    "message": f"Connected to {self.config.provider} ({len(models)} models available)",
                    "models": models,
                }
            return {"ok": False, "message": f"Server returned HTTP {res.status_code}"}
        except Exception as err:
            return {
                "ok": False,
                "message": f"Could not connect to {self.config.base_url}: {str(err)}",
            }

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
            return self._generate_mock_response(system_prompt, messages, available_tools)

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
                return self._call_ollama_api(
                    system_prompt, prepared_messages, model_name=effective_model, max_tokens=max_tokens
                )
            else:
                return self._call_openai_compatible_api(
                    system_prompt,
                    prepared_messages,
                    model_name=effective_model,
                    max_tokens=max_tokens,
                )
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
    ) -> LLMResponse:
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

        base = self.config.base_url.rstrip("/")
        opts = {"temperature": self.config.temperature}
        num_tokens = max_tokens or self.config.max_tokens
        if num_tokens:
            opts["num_predict"] = num_tokens

        res = requests.post(
            f"{base}/api/chat",
            headers=self._get_headers(),
            json={
                "model": model_name,
                "messages": formatted,
                "stream": False,
                "options": opts,
            },
            timeout=30,
        )
        res.raise_for_status()
        msg_obj = res.json().get("message", {})
        raw = msg_obj.get("content", "")
        reasoning = msg_obj.get("reasoning_content") or msg_obj.get("thought")
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

            current_max_tokens = max(1024, min(4096, int(current_max_tokens) * 2))
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
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "tool" in parsed or "name" in parsed:
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
    ) -> LLMResponse:
        time.sleep(0.3)
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = m.get("content", "")
                break
        if not last_user and system_prompt:
            last_user = system_prompt

        sys_lower = system_prompt.lower()
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
            if any(k in last_user.lower() for k in ["search", "find", "research"]):
                return LLMResponse(
                    content="researcher",
                    thought="User request requires live factual knowledge. Delegating research to DuckDuckGo search.",
                )
            if any(k in last_user.lower() for k in ["code", "script", "python", "react"]):
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

        return LLMResponse(
            content=f"Processed response for task: {last_user[:50]}...",
            thought="Analyzed user query context, evaluated relevant factors, and formulated final response.",
        )
