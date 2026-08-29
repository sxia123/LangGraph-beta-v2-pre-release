import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text extractor that strips scripts/styles and preserves paragraph breaks."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = False
        self._skip_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Any]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)
            self._skip = True
            return
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack and tag == self._skip_stack[-1]:
            self._skip_stack.pop()
            self._skip = bool(self._skip_stack)
            return
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if data and not data.isspace():
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(part.strip() for part in self.parts if part and part.strip())


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _estimate_token_count(text) <= max_tokens:
        return text

    words = re.findall(r"\S+", text)
    if not words:
        return ""

    truncated: List[str] = []
    count = 0
    for word in words:
        token_count = max(1, len(re.findall(r"\w+|[^\w\s]", word)))
        if count + token_count > max_tokens:
            break
        truncated.append(word)
        count += token_count

    return " ".join(truncated).strip()


def _extract_text_from_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    text = parser.get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url_content(url: str, max_tokens: int = 8000) -> str:
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        html = response.text
        if not html:
            return ""
        text = _extract_text_from_html(html)
        return _truncate_to_token_limit(text, max_tokens)
    except Exception as err:
        logger.debug("Failed to fetch URL content for %s: %s", url, err)
        return ""


def _build_search_queries(query: str, expand_queries: bool = True) -> List[str]:
    base = query.strip()
    if not base:
        return []

    queries = [base]
    if expand_queries:
        query_lower = base.lower()
        if "latest" not in query_lower:
            queries.append(f"{base} latest")
        if "example" not in query_lower and "tutorial" not in query_lower:
            queries.append(f"{base} example")
        if "documentation" not in query_lower and "docs" not in query_lower:
            queries.append(f"{base} documentation")

    return queries


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or item.get("name") or "").strip()
        url = str(item.get("href") or item.get("link") or item.get("url") or "").strip()
        if not url and not title:
            continue

        key = (url or title).lower()
        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(
            {
                "title": title or "Untitled",
                "href": url or "#",
                "body": str(item.get("body") or item.get("snippet") or item.get("text") or "").strip(),
            }
        )
    return deduped


def perform_web_search(
    query: str,
    max_results: int = 5,
    expand_queries: bool = True,
    max_tokens: int = 8000,
) -> List[Dict[str, Any]]:
    """Executes DuckDuckGo text search, fetches page content, and trims to an 8k-token budget."""
    if not query or not query.strip():
        return []

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        search_queries = _build_search_queries(query, expand_queries=expand_queries)
        combined_results: List[Dict[str, Any]] = []
        for search_query in search_queries:
            try:
                results = list(
                    DDGS().text(
                        search_query,
                        max_results=max_results,
                        region=os.getenv("WEB_SEARCH_REGION", "wt-wt"),
                        safesearch=os.getenv("WEB_SEARCH_SAFESAFE", "moderate"),
                    )
                )
            except TypeError:
                results = list(DDGS().text(search_query, max_results=max_results))

            combined_results.extend(results)
            if len(combined_results) >= max_results * 2:
                break

        deduped = _dedupe_results(combined_results)[:max_results]
        for item in deduped:
            url = str(item.get("href") or item.get("link") or item.get("url") or "").strip()
            if url:
                content = _fetch_url_content(url, max_tokens=max_tokens)
                if content:
                    item["content"] = content
                else:
                    item["content"] = str(item.get("body") or item.get("snippet") or "").strip()
        return deduped
    except Exception as err:
        logger.error(f"DuckDuckGo web search error for query '{query}': {err}")
        return []


def format_search_results(results: List[Dict[str, Any]], max_tokens: int = 8000) -> str:
    """Formats DuckDuckGo search result objects into markdown text with an 8k-token budget."""
    if not results:
        return "No web search results found."

    formatted_sections: List[str] = ["### Live Web Search Results\n"]
    used_tokens = _estimate_token_count("\n".join(formatted_sections))

    for idx, item in enumerate(results, start=1):
        title = str(item.get("title", "Untitled")).strip()
        url = str(item.get("href") or item.get("link") or item.get("url") or "#").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        content = str(item.get("content") or "").strip()

        summary = body or "No summary available."
        if content:
            summary = f"{summary}\n\nExtracted Content: {content}" if summary else f"Extracted Content: {content}"

        entry = f"{idx}. [{title}]({url})\n   {summary}\n"
        entry_tokens = _estimate_token_count(entry)
        if used_tokens + entry_tokens > max_tokens:
            remaining = max(0, max_tokens - used_tokens)
            if remaining <= 0:
                break
            entry = _truncate_to_token_limit(entry, remaining)
            if not entry:
                break
            formatted_sections.append(entry)
            break

        formatted_sections.append(entry)
        used_tokens += entry_tokens

    return "\n".join(formatted_sections)
