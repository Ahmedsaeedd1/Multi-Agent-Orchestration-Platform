"""
Web tools for agents: ``web_search`` and ``web_fetch``.
"""

import os
import logging

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

_MAX_FETCH_CHARS = 10_000


def web_search(query: str) -> str:
    """
    Search the web using Tavily.  Returns a summary of results as a string.
    Requires TAVILY_API_KEY in .env.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in .env"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No search results found."

        lines = [f"Top {len(results)} results for: {query}", ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", "")
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.error("web_search failed: %s", e)
        return f"Error performing web search: {e}"


def web_fetch(url: str) -> str:
    """
    Fetch the text content of a URL via HTTP GET.  Returns up to
    10 000 characters of plain text.
    """
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type and "json" not in content_type:
            return f"Unsupported content-type: {content_type}"

        text = resp.text
        if len(text) > _MAX_FETCH_CHARS:
            text = text[:_MAX_FETCH_CHARS] + "\n\n... (truncated)"
        return text
    except httpx.TimeoutException:
        return "Error: request timed out after 15s"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        logger.error("web_fetch failed for %s: %s", url, e)
        return f"Error fetching {url}: {e}"