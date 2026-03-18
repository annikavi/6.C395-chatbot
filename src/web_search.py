"""
Web search for MIT-specific context, used by the two-pass pipeline in chat.py.

Search engine priority:
  1. Tavily  — set TAVILY_API_KEY in .env (purpose-built for LLM context injection)
  2. DuckDuckGo — fallback, no key required (pip install ddgs)
"""

from __future__ import annotations

import os

try:
    from tavily import TavilyClient as _TC
    _TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
    _tavily = _TC(api_key=_TAVILY_KEY) if _TAVILY_KEY else None
except ImportError:
    _tavily = None

try:
    from ddgs import DDGS as _DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False


def _tavily_search(query: str) -> str:
    if not _tavily:
        return ""
    try:
        resp = _tavily.search(query=query, search_depth="basic", max_results=3, include_answer=True)
        answer = resp.get("answer", "")
        if answer:
            return answer[:500]
        snippets = [r.get("content", "") for r in resp.get("results", []) if r.get("content")]
        return " ".join(snippets)[:500]
    except Exception:
        return ""


def _ddg_search(query: str) -> str:
    if not _HAS_DDGS:
        return ""
    try:
        with _DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        snippets = [r["body"] for r in results if r.get("body")]
        return " ".join(snippets)[:500]
    except Exception:
        return ""


def web_search(query: str) -> str:
    """Run a web search and return a short snippet (max 500 chars)."""
    return _tavily_search(query) or _ddg_search(query)
